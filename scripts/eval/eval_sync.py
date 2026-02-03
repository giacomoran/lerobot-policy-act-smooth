#!/usr/bin/env python
"""Synchronous policy evaluation baseline.

This script runs policy inference synchronously: get observation, run inference
to get a full action chunk, execute all actions, then repeat.

Works with both ACT and ACTSmooth policies (no RTC prefix conditioning).

Usage:
    python scripts/eval/eval_sync.py \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem5A460829821 \
        --robot.id=arm_follower_0 \
        --robot.cameras="{wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
        --policy.path=outputs/model/pretrained_model \
        --fps=30 \
        --episode_time_s=60 \
        --display_data=true
"""

import logging
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pprint import pformat

import numpy as np
import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig  # noqa: F401
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun

from utils_rerun import LatencyTracker, log_rerun_data

# Import policy configs for type registration
from lerobot_policy_act_smooth import ACTSmoothConfig  # noqa: F401


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class EvalSyncConfig:
    """Configuration for synchronous policy evaluation."""

    robot: RobotConfig
    policy: PreTrainedConfig | None = None

    # Control parameters
    fps: int = 30
    episode_time_s: float = 60.0

    # Override n_action_steps from policy config (None = use policy default)
    n_action_steps: int | None = None

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

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# ============================================================================
# Inference Helper
# ============================================================================


def run_inference_chunk(
    observation_frame: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    device: torch.device,
    n_action_steps: int,
    robot_type: str,
) -> tuple[torch.Tensor, float]:
    """Run policy inference and return action chunk with timing.

    Args:
        observation_frame: Raw observation dict with numpy arrays
        policy: The policy to run inference with
        preprocessor: Pipeline for observation preprocessing
        device: Torch device for inference
        n_action_steps: Number of actions to return from the chunk
        robot_type: Robot type identifier

    Returns:
        Tuple of (action_chunk, inference_time_ms) where:
        - action_chunk: [batch, n_action_steps, action_dim] tensor
        - inference_time_ms: Time taken for inference in milliseconds
    """
    ts_start = time.perf_counter()

    use_amp = policy.config.use_amp
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        observation = prepare_observation_for_inference(observation_frame, device, None, robot_type)
        observation = preprocessor(observation)

        # Use predict_action_chunk without prefix (works for both ACT and ACTSmooth)
        actions = policy.predict_action_chunk(observation)
        actions = actions[:, :n_action_steps, :]

    duration_ms_inference = (time.perf_counter() - ts_start) * 1000
    return actions, duration_ms_inference


# ============================================================================
# Main Evaluation Loop
# ============================================================================


def run_episode_sync(
    robot,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    events: dict,
    cfg: EvalSyncConfig,
    device: torch.device,
    tracker_latency: LatencyTracker,
) -> None:
    """Run a single episode with synchronous inference.

    Control flow:
    1. Get observation
    2. Run inference to get full action chunk (n_action_steps actions)
    3. Execute all actions at target fps
    4. Repeat until episode ends
    """
    logging.info(f"Starting episode (max {cfg.episode_time_s}s at {cfg.fps} fps)")
    logging.info("Press ESC to terminate episode early")

    policy.reset()
    preprocessor.reset()
    tracker_latency.reset()

    if cfg.display_data:
        tracker_latency.setup_rerun()

    action_names = list(robot.action_features.keys())
    motor_names = [k for k in robot.observation_features if robot.observation_features[k] is float]
    camera_names = [k for k in robot.observation_features if isinstance(robot.observation_features[k], tuple)]

    robot_type = robot.robot_type

    policy_n_action_steps = getattr(policy.config, "n_action_steps", 1)
    n_action_steps = cfg.n_action_steps if cfg.n_action_steps is not None else policy_n_action_steps
    logging.info(f"Using n_action_steps: {n_action_steps}")

    duration_s_frame_target = 1.0 / cfg.fps

    # === WARMUP INFERENCE ===
    # Run one inference pass to trigger JIT compilation and CUDA kernel loading
    logging.info("Running warmup inference...")
    dict_obs_warmup = robot.get_observation()
    observation_frame_warmup = {}
    state_values_warmup = [dict_obs_warmup[motor_name] for motor_name in motor_names]
    observation_frame_warmup["observation.state"] = np.array(state_values_warmup, dtype=np.float32)
    for cam_name in camera_names:
        observation_frame_warmup[f"observation.images.{cam_name}"] = dict_obs_warmup[cam_name]
    _, duration_ms_warmup = run_inference_chunk(
        observation_frame=observation_frame_warmup,
        policy=policy,
        preprocessor=preprocessor,
        device=device,
        n_action_steps=n_action_steps,
        robot_type=robot_type,
    )
    logging.info(f"Warmup inference: {duration_ms_warmup:.1f}ms")

    ts_start_episode = time.perf_counter()
    idx_chunk = 0
    count_total_actions = 0

    while True:
        duration_s_episode = time.perf_counter() - ts_start_episode

        if events.get("exit_early") or events.get("stop_recording"):
            events["exit_early"] = False
            break
        if duration_s_episode >= cfg.episode_time_s:
            logging.info("Episode time limit reached")
            break

        # === INFERENCE PHASE ===
        dict_obs = robot.get_observation()

        observation_frame = {}
        state_values = [dict_obs[motor_name] for motor_name in motor_names]
        observation_frame["observation.state"] = np.array(state_values, dtype=np.float32)
        for cam_name in camera_names:
            observation_frame[f"observation.images.{cam_name}"] = dict_obs[cam_name]

        action_chunk, duration_ms_inference = run_inference_chunk(
            observation_frame=observation_frame,
            policy=policy,
            preprocessor=preprocessor,
            device=device,
            n_action_steps=n_action_steps,
            robot_type=robot_type,
        )

        tracker_latency.record(duration_ms_inference, log_to_rerun=cfg.display_data)

        logging.info(f"Chunk {idx_chunk}: inference={duration_ms_inference:.1f}ms, executing {n_action_steps} actions")

        # === EXECUTION PHASE ===
        for idx_action in range(n_action_steps):
            ts_start_frame = time.perf_counter()

            if events.get("exit_early") or events.get("stop_recording"):
                break

            if time.perf_counter() - ts_start_episode >= cfg.episode_time_s:
                break

            tensor_action = action_chunk[:, idx_action, :]  # (1, action_dim)
            tensor_action = postprocessor(tensor_action)  # Unnormalize
            tensor_action = tensor_action.squeeze(0).cpu()  # (action_dim,)
            robot_action = {name: float(tensor_action[i]) for i, name in enumerate(action_names)}
            robot.send_action(robot_action)
            count_total_actions += 1

            if cfg.display_data:
                dict_obs_vis = robot.get_observation()
                log_rerun_data(
                    timestep=count_total_actions,
                    idx_chunk=idx_chunk,
                    observation=dict_obs_vis,
                    action=robot_action,
                )

            duration_s_frame = time.perf_counter() - ts_start_frame
            duration_s_sleep = duration_s_frame_target - duration_s_frame
            if duration_s_sleep > 0:
                precise_sleep(duration_s_sleep)

        idx_chunk += 1

    duration_s_total = time.perf_counter() - ts_start_episode
    fps_actual = count_total_actions / duration_s_total if duration_s_total > 0 else 0
    logging.info(f"Episode complete: {count_total_actions} actions in {duration_s_total:.1f}s ({fps_actual:.1f} fps)")

    if cfg.display_data:
        tracker_latency.log_summary_to_rerun()

    stats_latency = tracker_latency.get_stats()
    if stats_latency:
        logging.info(
            f"Inference latency: mean={stats_latency['mean']:.1f}ms, "
            f"std={stats_latency['std']:.1f}ms, p95={stats_latency['p95']:.1f}ms"
        )


# ============================================================================
# Main Entry Point
# ============================================================================


@parser.wrap()
def main(cfg: EvalSyncConfig) -> None:
    """Main entry point for synchronous policy evaluation."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="eval_sync")

    device = get_safe_torch_device(cfg.policy.device if cfg.policy.device else "auto")
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

    listener, events = init_keyboard_listener()

    tracker_latency = LatencyTracker()

    try:
        logging.info("Connecting to robot...")
        robot.connect()
        log_say("Robot connected", cfg.play_sounds)

        log_say("Starting policy evaluation", cfg.play_sounds)
        run_episode_sync(
            robot=robot,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            events=events,
            cfg=cfg,
            device=device,
            tracker_latency=tracker_latency,
        )

        log_say("Episode finished", cfg.play_sounds, blocking=True)

    except KeyboardInterrupt:
        logging.info("Interrupted by user")

    finally:
        # Run homing sequence to safely park the arm before disconnecting
        if robot.is_connected:
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

        if not is_headless() and listener:
            listener.stop()

        log_say("Done", cfg.play_sounds, blocking=True)


if __name__ == "__main__":
    main()
