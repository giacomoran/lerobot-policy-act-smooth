#!/usr/bin/env python
"""Asynchronous ACTSmooth policy evaluation with Real-Time Chunking (RTC).

This script runs asynchronous policy inference with Real-Time Chunking,
where the policy is conditioned on an action prefix (the first `delay` steps of
the action chunk being executed during inference). This enables smoother chunk
transitions without discontinuities.

**ACTSmooth only** - requires max_delay >= 1.

Architecture:
- Actor thread: runs at fps_observation, switches to new chunks when available, passes action prefix
- Inference thread: triggered by actor, runs inference with action prefix
- Shared state: current observation, pending chunk, inference status, RTC prefix data

Usage:
    python scripts/eval/eval_async_rtc.py \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem5A460829821 \
        --robot.cameras="{wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
        --robot.id=arm_follower_0 \
        --policy.path=outputs/model/pretrained_model \
        --fps_policy=10 \
        --fps_observation=30 \
        --display_data=true
"""

import logging
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pprint import pformat
from threading import Event, Lock, Thread
from typing import Optional

import numpy as np
import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.rl.process import ProcessSignalHandler
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig  # noqa: F401
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun
from utils_rerun import DiscardTracker, LatencyTracker, log_rerun_data

# Import ACTSmooth config for type registration
from lerobot_policy_act_smooth import ACTSmoothConfig  # noqa: F401

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class EvalAsyncRTCConfig:
    """Configuration for asynchronous policy evaluation with Real-Time Chunking."""

    robot: RobotConfig
    policy: PreTrainedConfig | None = None

    # Control parameters
    fps_policy: int = 10
    fps_observation: int = 30
    episode_time_s: float = 60.0

    # Override n_action_steps from policy config (None = use policy default)
    n_action_steps: int | None = None

    # Remaining actions threshold: trigger inference when remaining actions drops below this
    threshold_remaining_actions: int = 2

    # Display and feedback
    display_data: bool = False
    play_sounds: bool = True

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError("A policy must be provided via --policy.path=...")

        if self.fps_observation % self.fps_policy != 0:
            raise ValueError(
                f"fps_observation ({self.fps_observation}) must be a multiple of fps_policy ({self.fps_policy})"
            )
        if self.fps_observation < self.fps_policy:
            raise ValueError(f"fps_observation ({self.fps_observation}) must be >= fps_policy ({self.fps_policy})")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# ============================================================================
# Action Chunk
# ============================================================================


@dataclass
class ActionChunk:
    """A chunk of actions with associated timing information for RTC.

    The policy produces a chunk of actions from an observation and an action
    prefix. The predicted action chunk corresponds to timesteps:
    `[timestep_obs + delay, timestep_obs + delay + chunk_size)`.

    Attributes:
        actions: Action tensor [n_actions, action_dim] as ABSOLUTE joint positions
        timestep_obs: Reference execution timestep for action indexing
        delay: Number of prefix steps used; actions[0] executes at timestep_obs + delay
        idx_chunk: Inference counter that generated this chunk (for logging)
    """

    actions: torch.Tensor
    timestep_obs: int
    delay: int
    idx_chunk: int

    def action_at(self, timestep: int) -> torch.Tensor | None:
        """Get action for the given timestep."""
        timestep_first_action = self.timestep_obs + self.delay
        idx = timestep - timestep_first_action
        if 0 <= idx < len(self.actions):
            return self.actions[idx]
        return None

    def count_remaining_actions_from(self, timestep: int) -> int:
        """Get number of actions remaining after the given timestep."""
        timestep_first_action = self.timestep_obs + self.delay
        idx = timestep + 1 - timestep_first_action
        return max(0, len(self.actions) - idx)

    def action_prefix_at(self, timestep: int, length_max: int) -> tuple[torch.Tensor | None, int]:
        """Get action prefix for RTC inference starting after the given timestep.

        Args:
            timestep: Current timestep (new observation time). Prefix will contain
                actions at timesteps [timestep, timestep + delay - 1].
            length_max: Maximum prefix length (e.g., max_delay)

        Returns:
            Tuple of (action_prefix, delay) where:
            - action_prefix: Tensor [1, delay, action_dim] or None if delay=0
            - delay: Actual prefix length used
        """
        timestep_first_action = self.timestep_obs + self.delay
        idx_start = timestep - timestep_first_action
        if idx_start < 0:
            return None, 0

        count_remaining_including_current = len(self.actions) - idx_start
        delay = min(length_max, count_remaining_including_current)

        if delay > 0:
            action_prefix = self.actions[idx_start : idx_start + delay]
            action_prefix = action_prefix.unsqueeze(0)
            return action_prefix, delay
        else:
            return None, 0


# ============================================================================
# Shared State
# ============================================================================


@dataclass
class State:
    """Centralized state for robot operations and thread communication."""

    lock: Optional[Lock] = None

    timestep: int = 0
    dict_obs: dict | None = None

    action_chunk_active: ActionChunk | None = None
    action_chunk_pending: ActionChunk | None = None

    timestep_inference_requested: int | None = None
    dict_obs_inference_requested: dict | None = None

    event_inference_requested: Event | None = None
    event_shutdown: Event | None = None

    def __post_init__(self):
        self.lock = Lock()
        self.event_inference_requested = Event()
        self.event_shutdown = Event()


# ============================================================================
# Actor Thread
# ============================================================================


def thread_actor_fn(
    robot,
    state: State,
    action_names: list[str],
    postprocessor: PolicyProcessorPipeline,
    cfg: EvalAsyncRTCConfig,
    tracker_discard: DiscardTracker,
) -> None:
    """Actor thread: executes actions from current chunk at target fps."""
    fps_target = cfg.fps_observation
    duration_s_frame_target = 1.0 / fps_target
    count_executed_actions = 0
    action_chunk_active: ActionChunk | None = None

    num_frames_per_control_frame = cfg.fps_observation // cfg.fps_policy

    robot_action_last_executed = None
    action_interp_start: torch.Tensor | None = None
    action_interp_end: torch.Tensor | None = None
    is_interpolation_ready: bool = False

    try:
        timestep = 0
        idx_frame = 0
        while not state.event_shutdown.is_set():
            ts_start_frame = time.perf_counter()

            is_control_frame = (idx_frame % num_frames_per_control_frame) == 0

            dict_obs = robot.get_observation()

            if is_control_frame:
                with state.lock:
                    state.timestep = timestep
                    state.dict_obs = dict_obs.copy()

                    if state.action_chunk_pending:
                        action_chunk_pending = state.action_chunk_pending
                        if action_chunk_pending.action_at(timestep) is not None:
                            # Track discarded actions (old chunk remaining + new chunk skipped)
                            # For RTC: first action is at timestep_obs + delay
                            timestep_first_action = action_chunk_pending.timestep_obs + action_chunk_pending.delay
                            n_discarded = max(0, timestep - timestep_first_action)
                            if action_chunk_active is not None:
                                n_discarded += action_chunk_active.count_remaining_actions_from(timestep)

                            if n_discarded > 0:
                                tracker_discard.record(n_discarded, log_to_rerun=cfg.display_data)

                            action_chunk_active = action_chunk_pending
                            state.action_chunk_active = action_chunk_active
                            state.action_chunk_pending = None
                            logging.info(
                                f"[ACTOR] Switched to chunk #{action_chunk_active.idx_chunk} "
                                f"(timestep_obs={action_chunk_active.timestep_obs}, delay={action_chunk_active.delay}, "
                                f"discarded={n_discarded})"
                            )

                    action = None
                    count_remaining_actions = None
                    if action_chunk_active is not None:
                        action = action_chunk_active.action_at(timestep)
                        if action is not None:
                            count_remaining_actions = action_chunk_active.count_remaining_actions_from(timestep)

                    can_request_inference = (
                        state.timestep_inference_requested is None and state.action_chunk_pending is None
                    )
                    has_action = action is not None
                    below_threshold = has_action and count_remaining_actions <= cfg.threshold_remaining_actions
                    should_request_inference = can_request_inference and (not has_action or below_threshold)

                    if should_request_inference:
                        state.timestep_inference_requested = timestep
                        state.dict_obs_inference_requested = dict_obs.copy()
                        state.event_inference_requested.set()

                    idx_chunk = action_chunk_active.idx_chunk if action_chunk_active else -1

                if action is not None:
                    action = postprocessor(action.unsqueeze(0))  # Unnormalize
                    action = action.squeeze(0).cpu()
                    robot_action = {name: float(action[i]) for i, name in enumerate(action_names)}
                    robot.send_action(robot_action)
                    count_executed_actions += 1
                    robot_action_last_executed = robot_action

                    action_interp_start = action.clone()
                    timestep_next = timestep + 1
                    action_next = action_chunk_active.action_at(timestep_next)
                    # If pending chunk has action at next timestep, we'll switch to it, so use its action
                    if state.action_chunk_pending is not None:
                        action_next_pending = state.action_chunk_pending.action_at(timestep_next)
                        if action_next_pending is not None:
                            action_next = action_next_pending

                    if action_next is not None:
                        action_interp_end = postprocessor(action_next.unsqueeze(0)).squeeze(0).cpu()
                    else:
                        action_interp_end = action_interp_start.clone()
                    is_interpolation_ready = True

                logging.info(
                    f"[ACTOR] timestep={timestep} | chunk={idx_chunk} | "
                    f"remaining={count_remaining_actions if count_remaining_actions is not None else -1} | "
                    f"count={count_executed_actions}"
                )

                timestep += 1
            else:
                if is_interpolation_ready and action_interp_start is not None:
                    idx_within_period = idx_frame % num_frames_per_control_frame
                    t = idx_within_period / num_frames_per_control_frame

                    action_interp = (1.0 - t) * action_interp_start + t * action_interp_end
                    robot_action_interp = {name: float(action_interp[i]) for i, name in enumerate(action_names)}
                    robot.send_action(robot_action_interp)
                    robot_action_last_executed = robot_action_interp

            if cfg.display_data:
                log_rerun_data(
                    timestep=timestep,
                    idx_chunk=idx_chunk if is_control_frame else None,
                    observation=dict_obs,
                    action=robot_action_last_executed if is_control_frame else None,
                )

            idx_frame += 1

            duration_s_frame = time.perf_counter() - ts_start_frame
            precise_sleep(max(0, (duration_s_frame_target - duration_s_frame) - 0.001))

    except Exception as e:
        logging.error(f"[ACTOR] Fatal exception: {e}")
        traceback.print_exc()
        state.event_shutdown.set()
        sys.exit(1)

    logging.info(f"[ACTOR] Thread shutting down. Total actions executed: {count_executed_actions}")


# ============================================================================
# Inference Thread
# ============================================================================


def thread_inference_fn(
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    state: State,
    device: torch.device,
    cfg: EvalAsyncRTCConfig,
    n_action_steps: int,
    motor_names: list[str],
    camera_names: list[str],
    tracker_latency: LatencyTracker,
    robot_type: str,
    max_delay: int,
) -> None:
    """Inference thread: waits for signal, runs inference with RTC, creates new action chunk."""
    use_amp = policy.config.use_amp
    idx_chunk = 0

    try:
        while not state.event_shutdown.is_set():
            if not state.event_inference_requested.wait(timeout=1.0):
                continue
            state.event_inference_requested.clear()

            with state.lock:
                timestep = state.timestep_inference_requested
                dict_obs = state.dict_obs_inference_requested
                action_chunk_active = state.action_chunk_active

                if timestep is None or dict_obs is None:
                    state.timestep_inference_requested = None
                    state.dict_obs_inference_requested = None
                    continue

            if action_chunk_active is not None:
                action_prefix_absolute, delay = action_chunk_active.action_prefix_at(timestep, max_delay)
            else:
                action_prefix_absolute = None
                delay = 0

            if state.event_shutdown.is_set():
                break

            # Build observation frame with state AND images (state must go through preprocessing)
            observation_frame = {}
            array_proprio_obs = [dict_obs[motor_name] for motor_name in motor_names]
            observation_frame["observation.state"] = np.array(array_proprio_obs, dtype=np.float32)
            for cam_name in camera_names:
                observation_frame[f"observation.images.{cam_name}"] = dict_obs[cam_name]

            ts_start_inference = time.perf_counter()

            with (
                torch.inference_mode(),
                torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
            ):
                observation = prepare_observation_for_inference(observation_frame, device, None, robot_type)
                observation = preprocessor(observation)

                actions = policy.predict_action_chunk(
                    observation,
                    action_prefix=action_prefix_absolute,
                )

                actions = actions[:, :n_action_steps, :]

            duration_ms_inference = (time.perf_counter() - ts_start_inference) * 1000

            if state.event_shutdown.is_set():
                break

            idx_chunk += 1
            tracker_latency.record(duration_ms_inference, log_to_rerun=cfg.display_data)

            with state.lock:
                state.action_chunk_pending = ActionChunk(
                    actions=actions.squeeze(0),
                    timestep_obs=timestep,
                    delay=delay,
                    idx_chunk=idx_chunk,
                )
                state.timestep_inference_requested = None
                state.dict_obs_inference_requested = None

            logging.info(
                f"[INFERENCE] chunk={idx_chunk} | duration_ms={duration_ms_inference:.1f}ms | "
                f"timestep={timestep} | delay={delay} | timestep_action_start={timestep + delay}"
            )

    except Exception as e:
        logging.error(f"[INFERENCE] Fatal exception: {e}")
        traceback.print_exc()
        state.event_shutdown.set()
        sys.exit(1)


# ============================================================================
# Main Entry Point
# ============================================================================


@parser.wrap()
def main(cfg: EvalAsyncRTCConfig) -> None:
    """Main entry point for asynchronous policy evaluation with RTC."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="eval_async_rtc")

    name_device = cfg.policy.device if cfg.policy.device else "auto"
    device = get_safe_torch_device(name_device)
    logging.info(f"Using device: {device}")

    logging.info("Creating robot...")
    robot = make_robot_from_config(cfg.robot)

    logging.info(f"Loading policy from {cfg.policy.pretrained_path}...")
    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(cfg.policy.pretrained_path)
    policy.to(device)

    max_delay = getattr(policy.config, "max_delay", 0)
    if max_delay < 1:
        raise ValueError(f"eval_async_rtc requires ACTSmooth policy with max_delay >= 1, got {max_delay}")
    logging.info(f"Policy max_delay: {max_delay}")

    device_override = {"device": str(device)}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={"device_processor": device_override},
        postprocessor_overrides={"device_processor": device_override},
    )

    tracker_latency = LatencyTracker()
    tracker_discard = DiscardTracker()

    ProcessSignalHandler(use_threads=True, display_pid=False)

    keyboard_listener = None
    events_keyboard = {}
    if not is_headless():
        keyboard_listener, events_keyboard = init_keyboard_listener()
        logging.info("Press ESC to terminate episode early")

    thread_inference = None
    thread_actor = None
    ts_start_episode = None
    state = None

    try:
        logging.info("Connecting to robot...")
        robot.connect()
        log_say("Robot connected", cfg.play_sounds)

        log_say("Starting evaluation", cfg.play_sounds)

        logging.info(
            f"Starting episode (max {cfg.episode_time_s}s at {cfg.fps_policy} fps policy, "
            f"{cfg.fps_observation} fps command)"
        )
        logging.info(f"Remaining actions threshold: {cfg.threshold_remaining_actions}")

        policy.reset()
        preprocessor.reset()
        tracker_latency.reset()
        tracker_discard.reset()

        state = State()

        if cfg.display_data:
            tracker_latency.setup_rerun()
            tracker_discard.setup_rerun()

        action_names = list(robot.action_features.keys())
        motor_names = [k for k in robot.observation_features if robot.observation_features[k] is float]
        camera_names = [k for k in robot.observation_features if isinstance(robot.observation_features[k], tuple)]

        robot_type = robot.robot_type

        policy_n_action_steps = getattr(policy.config, "n_action_steps", 1)
        n_action_steps = cfg.n_action_steps if cfg.n_action_steps is not None else policy_n_action_steps
        logging.info(f"Using n_action_steps: {n_action_steps}")

        # === WARMUP INFERENCE ===
        # Run one inference pass to trigger JIT compilation and CUDA kernel loading
        logging.info("Running warmup inference...")
        dict_obs_warmup = robot.get_observation()
        observation_frame_warmup = {}
        array_proprio_warmup = [dict_obs_warmup[motor_name] for motor_name in motor_names]
        observation_frame_warmup["observation.state"] = np.array(array_proprio_warmup, dtype=np.float32)
        for cam_name in camera_names:
            observation_frame_warmup[f"observation.images.{cam_name}"] = dict_obs_warmup[cam_name]

        ts_start_warmup = time.perf_counter()
        use_amp = policy.config.use_amp
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
        ):
            observation_warmup = prepare_observation_for_inference(observation_frame_warmup, device, None, robot_type)
            observation_warmup = preprocessor(observation_warmup)
            _ = policy.predict_action_chunk(observation_warmup, action_prefix=None)
        duration_ms_warmup = (time.perf_counter() - ts_start_warmup) * 1000
        logging.info(f"Warmup inference: {duration_ms_warmup:.1f}ms")

        # Reset preprocessor after warmup (it may have internal state)
        preprocessor.reset()

        thread_inference = Thread(
            target=thread_inference_fn,
            args=(
                policy,
                preprocessor,
                state,
                device,
                cfg,
                n_action_steps,
                motor_names,
                camera_names,
                tracker_latency,
                robot_type,
                max_delay,
            ),
            daemon=True,
            name="Inference",
        )
        thread_inference.start()

        thread_actor = Thread(
            target=thread_actor_fn,
            args=(robot, state, action_names, postprocessor, cfg, tracker_discard),
            daemon=True,
            name="Actor",
        )
        thread_actor.start()

        ts_start_episode = time.perf_counter()

        while not state.event_shutdown.is_set():
            duration_s_episode = time.perf_counter() - ts_start_episode

            if duration_s_episode >= cfg.episode_time_s:
                break

            if events_keyboard.get("exit_early", False):
                logging.info("Terminating episode early (ESC pressed)")
                events_keyboard["exit_early"] = False
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        logging.info("Interrupted by user")

    finally:
        if state is not None:
            state.event_shutdown.set()
            state.event_inference_requested.set()

        if keyboard_listener is not None and not is_headless():
            logging.info("Stopping keyboard listener...")
            keyboard_listener.stop()
            keyboard_listener = None
            events_keyboard.clear()

        logging.info("Waiting for inference thread to finish...")
        if thread_inference is not None:
            if thread_inference.is_alive():
                thread_inference.join(timeout=2.0)
                if thread_inference.is_alive():
                    logging.warning("Inference thread did not finish within timeout")
                else:
                    logging.info("Inference thread finished")
            else:
                logging.info("Inference thread already finished")

        logging.info("Waiting for actor thread to finish...")
        if thread_actor is not None:
            if thread_actor.is_alive():
                thread_actor.join(timeout=2.0)
                if thread_actor.is_alive():
                    logging.warning("Actor thread did not finish within timeout")
                else:
                    logging.info("Actor thread finished")
            else:
                logging.info("Actor thread already finished")

        if ts_start_episode is not None and state is not None:
            duration_s_episode = time.perf_counter() - ts_start_episode
            logging.info(f"Episode completed in {duration_s_episode:.1f}s")

            if cfg.display_data:
                tracker_latency.log_summary_to_rerun()

            stats_latency = tracker_latency.get_stats()
            if stats_latency:
                logging.info(
                    f"Inference latency: mean={stats_latency['mean']:.1f}ms, "
                    f"std={stats_latency['std']:.1f}ms, p95={stats_latency['p95']:.1f}ms"
                )

            stats_discard = tracker_discard.get_stats()
            if stats_discard:
                logging.info(
                    f"Discarded actions: total={stats_discard['count_total']}, "
                    f"mean={stats_discard['mean']:.1f}/chunk, switches={stats_discard['count_switches']}"
                )
            else:
                logging.info("Discarded actions: total=0")

        # Run homing sequence to safely park the arm before disconnecting
        if robot and robot.is_connected:
            logging.info("Running homing sequence...")
            try:
                import sys
                from pathlib import Path

                sys.path.insert(0, str(Path(__file__).parent.parent))
                from shared.homing import run_homing_sequence

                run_homing_sequence(robot, enable_rerun_logging=False)
            except Exception as e:
                logging.error(f"Homing failed: {e}")

            logging.info("Disconnecting robot...")
            robot.disconnect()

        log_say("Done", cfg.play_sounds, blocking=True)


if __name__ == "__main__":
    main()
