#!/usr/bin/env python
"""Asynchronous policy evaluation with threshold-based chunk switching.

This script runs policy inference asynchronously in a separate thread while the
robot continues executing actions from the current chunk. When a new chunk becomes
available, it switches to it entirely - no merging of leftover actions is needed.

Works with both ACT and ACTSmooth policies (no RTC prefix conditioning).

Architecture:
- Actor thread: runs at fps_observation, switches to new chunks when available
- Inference thread: triggered by actor when action count drops below threshold
- Shared state: current observation, pending chunk, inference status

Usage:
    python scripts/eval/eval_async_discard.py \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem5A460829821 \
        --robot.cameras="{wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
        --robot.id=arm_follower_0 \
        --policy.path=outputs/model/pretrained_model \
        --fps_policy=10 \
        --fps_interpolation=30 \
        --fps_observation=60 \
        --display_data=true
"""

import logging
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from threading import Event, Lock, Thread
from typing import Optional

import numpy as np
import rerun as rr
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

# Import policy configs for type registration
from lerobot_policy_act_smooth import ACTSmoothConfig  # noqa: F401

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class EvalAsyncDiscardConfig:
    """Configuration for asynchronous policy evaluation with chunk switching."""

    robot: RobotConfig
    policy: PreTrainedConfig | None = None

    # Control parameters
    fps_policy: int = 10
    fps_observation: int = 60
    fps_interpolation: int | None = None  # None = fps_policy (no interpolation)
    episode_time_s: float = 60.0

    # Override n_action_steps from policy config (None = use policy default)
    n_action_steps: int | None = None

    # Remaining actions threshold: trigger inference when remaining actions drops below this
    threshold_remaining_actions: int = 2

    # Display and recording
    display_data: bool = False
    path_recording: str = ""
    play_sounds: bool = True

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError("A policy must be provided via --policy.path=...")

        if self.path_recording:
            Path(self.path_recording).parent.mkdir(parents=True, exist_ok=True)

        if self.fps_interpolation is None:
            self.fps_interpolation = self.fps_policy

        if self.fps_observation < self.fps_interpolation:
            raise ValueError(
                f"fps_observation ({self.fps_observation}) must be >= fps_interpolation ({self.fps_interpolation})"
            )
        if self.fps_interpolation < self.fps_policy:
            raise ValueError(f"fps_interpolation ({self.fps_interpolation}) must be >= fps_policy ({self.fps_policy})")
        if self.fps_observation % self.fps_interpolation != 0:
            raise ValueError(
                f"fps_observation ({self.fps_observation}) must be a multiple of fps_interpolation ({self.fps_interpolation})"
            )
        if self.fps_interpolation % self.fps_policy != 0:
            raise ValueError(
                f"fps_interpolation ({self.fps_interpolation}) must be a multiple of fps_policy ({self.fps_policy})"
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# ============================================================================
# Action Chunk
# ============================================================================


@dataclass
class ActionChunk:
    """A chunk of actions with associated timing information.

    Attributes:
        actions: Action tensor [n_actions, action_dim]
        timestep_obs: Timestep when observation was taken; actions[0] executes at timestep_obs
        idx_chunk: Inference counter that generated this chunk (for logging)
    """

    actions: torch.Tensor
    timestep_obs: int
    idx_chunk: int

    def action_at(self, timestep: int) -> torch.Tensor | None:
        """Get action for the given timestep."""
        idx = timestep - self.timestep_obs
        if 0 <= idx < len(self.actions):
            return self.actions[idx]
        return None

    def cnt_actions_remaining_from(self, timestep: int) -> int:
        """Get number of actions remaining from the given timestep."""
        idx = timestep - self.timestep_obs
        return max(0, len(self.actions) - idx)


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

    # Final counts from actor thread (set on shutdown)
    cnt_frames_observation: int = 0
    cnt_frames_interpolation: int = 0
    cnt_actions_executed: int = 0

    event_inference_requested: Event | None = None
    event_shutdown: Event | None = None

    def __post_init__(self):
        self.lock = Lock()
        self.event_inference_requested = Event()
        self.event_shutdown = Event()


# ============================================================================
# Actor Thread — Pure Logic
# ============================================================================


@dataclass
class ArgsStepActorControlFrame:
    """Arguments for step_actor_control_frame."""

    timestep: int
    action_chunk_active: ActionChunk | None
    action_chunk_pending: ActionChunk | None
    is_inference_requested: bool
    threshold_remaining_actions: int


@dataclass
class OutputStepActorControlFrame:
    """Pure output of a single actor control frame decision."""

    action: torch.Tensor | None  # Action to send (raw, pre-postprocess)
    action_chunk_active: ActionChunk | None  # Updated active chunk (after potential switch)
    action_chunk_pending_consumed: bool  # Whether pending was consumed
    cnt_actions_remaining: int  # Actions remaining in active chunk
    cnt_actions_discarded: int  # Actions discarded during switch
    should_request_inference: bool
    action_interp_target: torch.Tensor | None  # Next action for interpolation endpoint


def step_actor_control_frame(args: ArgsStepActorControlFrame) -> OutputStepActorControlFrame:
    """Pure decision logic for one actor control frame.

    Handles chunk switching, action selection, inference triggering,
    and interpolation target computation — no I/O or state mutation.

    Eager-switch: pending chunk is checked FIRST (preferred over active).
    """
    action_chunk_active = args.action_chunk_active
    action_chunk_pending_consumed = False
    cnt_actions_discarded = 0

    # 1. Eager switch: prefer pending chunk over active
    action = None
    if args.action_chunk_pending is not None:
        action = args.action_chunk_pending.action_at(args.timestep)
        if action is not None:
            # Discard formula: skipped actions in pending + remaining in active
            cnt_actions_discarded = args.timestep - args.action_chunk_pending.timestep_obs
            if action_chunk_active is not None:
                cnt_actions_discarded += action_chunk_active.cnt_actions_remaining_from(args.timestep)
            action_chunk_active = args.action_chunk_pending
            action_chunk_pending_consumed = True

    # 2. Fall back to active chunk if pending didn't yield an action
    if action is None and action_chunk_active is not None:
        action = action_chunk_active.action_at(args.timestep)

    has_unconsumed_pending = args.action_chunk_pending is not None and not action_chunk_pending_consumed
    can_request_inference = not args.is_inference_requested and not has_unconsumed_pending

    cnt_actions_remaining = action_chunk_active.cnt_actions_remaining_from(args.timestep) if action_chunk_active else 0

    # 3. No action available — request inference and return early
    if action is None:
        return OutputStepActorControlFrame(
            action=None,
            action_chunk_active=action_chunk_active,
            action_chunk_pending_consumed=action_chunk_pending_consumed,
            cnt_actions_remaining=cnt_actions_remaining,
            cnt_actions_discarded=cnt_actions_discarded,
            should_request_inference=can_request_inference,
            action_interp_target=None,
        )

    # --- From here, action and action_chunk_active are guaranteed non-None ---

    # 4. Inference trigger: strict < (not <=)
    should_request_inference = can_request_inference and cnt_actions_remaining < args.threshold_remaining_actions

    # 5. Interpolation target: next timestep's action
    timestep_next = args.timestep + 1
    action_next = action_chunk_active.action_at(timestep_next)
    if has_unconsumed_pending:
        action_next_from_pending = args.action_chunk_pending.action_at(timestep_next)
        if action_next_from_pending is not None:
            action_next = action_next_from_pending
    # No next action: hold last action as interpolation target
    action_interp_target = action_next if action_next is not None else action.clone()

    return OutputStepActorControlFrame(
        action=action,
        action_chunk_active=action_chunk_active,
        action_chunk_pending_consumed=action_chunk_pending_consumed,
        cnt_actions_remaining=cnt_actions_remaining,
        cnt_actions_discarded=cnt_actions_discarded,
        should_request_inference=should_request_inference,
        action_interp_target=action_interp_target,
    )


# ============================================================================
# Actor Thread
# ============================================================================


def thread_actor_fn(
    robot,
    state: State,
    action_names: list[str],
    postprocessor: PolicyProcessorPipeline,
    cfg: EvalAsyncDiscardConfig,
    tracker_discard: DiscardTracker,
) -> None:
    """Actor thread: executes actions from current chunk at target fps."""
    duration_s_frame_target = 1.0 / cfg.fps_observation
    cnt_actions_executed = 0
    cnt_frames_observation = 0
    cnt_frames_interpolation = 0
    action_chunk_active: ActionChunk | None = None

    cnt_frames_per_control_frame = cfg.fps_observation // cfg.fps_policy
    cnt_frames_per_interpolation_frame = cfg.fps_observation // cfg.fps_interpolation

    dict_action_last_executed = None
    action_interpolation_start: torch.Tensor | None = None
    action_interpolation_end: torch.Tensor | None = None
    is_interpolation_ready: bool = False

    try:
        timestep = 0
        idx_frame = 0
        while not state.event_shutdown.is_set():
            ts_start_frame = time.perf_counter()

            is_control_frame = (idx_frame % cnt_frames_per_control_frame) == 0
            is_interpolation_frame = (idx_frame % cnt_frames_per_interpolation_frame) == 0

            # --- Control frame ---
            if is_control_frame:
                dict_obs = robot.get_observation()
                with state.lock:
                    state.timestep = timestep
                    state.dict_obs = dict_obs.copy()

                    output = step_actor_control_frame(
                        ArgsStepActorControlFrame(
                            timestep=timestep,
                            action_chunk_active=action_chunk_active,
                            action_chunk_pending=state.action_chunk_pending,
                            is_inference_requested=state.timestep_inference_requested is not None,
                            threshold_remaining_actions=cfg.threshold_remaining_actions,
                        )
                    )

                    action = output.action
                    action_chunk_active = output.action_chunk_active
                    action_chunk_pending_consumed = output.action_chunk_pending_consumed
                    cnt_actions_remaining = output.cnt_actions_remaining
                    cnt_actions_discarded = output.cnt_actions_discarded
                    should_request_inference = output.should_request_inference
                    action_interp_target = output.action_interp_target

                    if action_chunk_pending_consumed:
                        if cnt_actions_discarded > 0:
                            tracker_discard.record(cnt_actions_discarded, log_to_rerun=cfg.display_data)
                        state.action_chunk_active = action_chunk_active
                        state.action_chunk_pending = None
                        logging.info(
                            f"[ACTOR] Switched to idx_chunk={action_chunk_active.idx_chunk} "
                            f"(timestep_obs={action_chunk_active.timestep_obs}, "
                            f"cnt_actions_discarded={cnt_actions_discarded})"
                        )

                    if should_request_inference:
                        state.timestep_inference_requested = timestep
                        state.dict_obs_inference_requested = dict_obs.copy()
                        state.event_inference_requested.set()

                    idx_chunk = action_chunk_active.idx_chunk if action_chunk_active else -1

                if action is not None:
                    action = postprocessor(action.unsqueeze(0))  # Unnormalize
                    action = action.squeeze(0).cpu()
                    dict_action = {name: float(action[i]) for i, name in enumerate(action_names)}
                    robot.send_action(dict_action)
                    cnt_actions_executed += 1
                    dict_action_last_executed = dict_action

                    action_interpolation_start = action.clone()
                    if action_interp_target is not None:
                        action_interpolation_end = postprocessor(action_interp_target.unsqueeze(0)).squeeze(0).cpu()
                    else:
                        action_interpolation_end = action_interpolation_start.clone()
                    is_interpolation_ready = True

                logging.info(
                    f"[ACTOR] timestep={timestep} | chunk={idx_chunk} | "
                    f"actions_remaining={cnt_actions_remaining} | count={cnt_actions_executed}"
                )

                timestep += 1

            # --- Interpolation frame ---
            if not is_control_frame and is_interpolation_frame:
                dict_obs_motors = robot.bus.sync_read("Present_Position")
                dict_obs = {f"{motor}.pos": val for motor, val in dict_obs_motors.items()}

                if is_interpolation_ready and action_interpolation_start is not None:
                    idx_within_period = idx_frame % cnt_frames_per_control_frame
                    t = idx_within_period / cnt_frames_per_control_frame

                    action_interpolation = (1.0 - t) * action_interpolation_start + t * action_interpolation_end
                    dict_action_interpolation = {
                        name: float(action_interpolation[i]) for i, name in enumerate(action_names)
                    }
                    robot.send_action(dict_action_interpolation)
                    dict_action_last_executed = dict_action_interpolation
                    cnt_frames_interpolation += 1

            # --- Observation-only frame ---
            if not is_control_frame and not is_interpolation_frame:
                dict_obs_motors = robot.bus.sync_read("Present_Position")
                dict_obs = {f"{motor}.pos": val for motor, val in dict_obs_motors.items()}

            # --- Logging ---
            if cfg.display_data:
                log_rerun_data(
                    timestep=timestep,
                    idx_frame=idx_frame,
                    idx_chunk=idx_chunk if is_control_frame else None,
                    observation=dict_obs,
                    action=dict_action_last_executed if (is_control_frame or is_interpolation_frame) else None,
                )

            cnt_frames_observation += 1
            idx_frame += 1

            duration_s_frame = time.perf_counter() - ts_start_frame
            precise_sleep(max(0, (duration_s_frame_target - duration_s_frame) - 0.001))

    except Exception as e:
        logging.error(f"[ACTOR] Fatal exception: {e}")
        traceback.print_exc()
        state.event_shutdown.set()
        sys.exit(1)

    state.cnt_frames_observation = cnt_frames_observation
    state.cnt_frames_interpolation = cnt_frames_interpolation
    state.cnt_actions_executed = cnt_actions_executed
    state.timestep = timestep
    logging.info(f"[ACTOR] Thread shutting down. timestep={timestep}, actions executed: {cnt_actions_executed}")


# ============================================================================
# Inference Thread
# ============================================================================


def thread_inference_fn(
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    state: State,
    device: torch.device,
    cfg: EvalAsyncDiscardConfig,
    n_action_steps: int,
    motor_names: list[str],
    camera_names: list[str],
    tracker_latency: LatencyTracker,
    robot_type: str,
) -> None:
    """Inference thread: waits for signal, runs inference, creates new action chunk."""
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

                if timestep is None or dict_obs is None:
                    state.timestep_inference_requested = None
                    state.dict_obs_inference_requested = None
                    continue

            if state.event_shutdown.is_set():
                break

            # Build observation frame
            observation_frame = {}
            array_proprio_obs = [dict_obs[motor_name] for motor_name in motor_names]
            observation_frame["observation.state"] = np.array(array_proprio_obs, dtype=np.float32)
            for cam_name in camera_names:
                observation_frame[f"observation.images.{cam_name}"] = dict_obs[cam_name]

            logging.info(f"[INFERENCE] Starting idx_chunk={idx_chunk + 1} | timestep={timestep}")
            ts_start_inference = time.perf_counter()

            with (
                torch.inference_mode(),
                torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
            ):
                observation = prepare_observation_for_inference(observation_frame, device, None, robot_type)
                observation = preprocessor(observation)

                # Use predict_action_chunk without prefix (works for both ACT and ACTSmooth)
                actions = policy.predict_action_chunk(observation)
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
                    idx_chunk=idx_chunk,
                )
                state.timestep_inference_requested = None
                state.dict_obs_inference_requested = None

            logging.info(
                f"[INFERENCE] idx_chunk={idx_chunk} | duration_ms={duration_ms_inference:.1f}ms | timestep={timestep}"
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
def main(cfg: EvalAsyncDiscardConfig) -> None:
    """Main entry point for asynchronous policy evaluation."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="eval_async_discard")
        if cfg.path_recording:
            rr.save(cfg.path_recording)
            logging.info(f"Recording to {cfg.path_recording}")

    name_device = cfg.policy.device if cfg.policy.device else "auto"
    device = get_safe_torch_device(name_device)
    logging.info(f"Using device: {device}")

    logging.info("Creating robot...")
    robot = make_robot_from_config(cfg.robot)

    logging.info(f"Loading policy from {cfg.policy.pretrained_path}...")
    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(cfg.policy.pretrained_path)
    policy.to(device)

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
            f"{cfg.fps_interpolation} fps interpolation, {cfg.fps_observation} fps observation)"
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
        array_proprio_obs_warmup = [dict_obs_warmup[motor_name] for motor_name in motor_names]
        observation_frame_warmup["observation.state"] = np.array(array_proprio_obs_warmup, dtype=np.float32)
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
            _ = policy.predict_action_chunk(observation_warmup)
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

            if duration_s_episode > 0:
                fps_observation_real = state.cnt_frames_observation / duration_s_episode
                # timestep counts policy-rate control frames (one per control frame)
                fps_interpolation_real = (state.cnt_frames_interpolation + state.timestep) / duration_s_episode
                fps_policy_real = state.timestep / duration_s_episode
                logging.info(
                    f"Real FPS: observation={fps_observation_real:.1f} (target={cfg.fps_observation}), "
                    f"interpolation={fps_interpolation_real:.1f} (target={cfg.fps_interpolation}), "
                    f"policy={fps_policy_real:.1f} (target={cfg.fps_policy})"
                )

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
