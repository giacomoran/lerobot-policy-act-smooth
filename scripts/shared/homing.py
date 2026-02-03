#!/usr/bin/env python3
"""
Homing script for SO-101 Follower arm.

Safely moves the robot from any arbitrary position to the home position
using linear interpolation for smooth trajectories. Related joints move
simultaneously to reduce total homing time.

Usage:
    python scripts/shared/homing.py --port=/dev/tty.usbmodem5A460829821
"""

import argparse
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.visualization_utils import log_rerun_data

logger = logging.getLogger(__name__)

# Type alias for observation callback
ObservationCallback = Callable[[dict, dict | None], None]  # (observation, action) -> None

# Configuration
FPS_DEFAULT = 30

# Maximum velocity per joint (degrees per second)
MAX_VELOCITY_DEG_PER_SEC = 50.0

# Threshold to consider a joint "at target" (in degrees)
POSITION_TOLERANCE = 1.0  # degrees

# Home position for all joints (in degrees)
JOINT_ACTION_HOME = {
    "shoulder_pan.pos": -5,
    "shoulder_lift.pos": -110,
    "elbow_flex.pos": 100,
    "wrist_flex.pos": 64,
    "wrist_roll.pos": 0,
    "gripper.pos": 5,
}

# Safe intermediate height for lifting arm (degrees)
# This is a safe position to lift the arm before other movements
SAFE_HEIGHT = {
    "shoulder_pan.pos": -5,  # Use home position for shoulder_pan
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
}

# Gripper open position (degrees)
GRIPPER_OPEN = 40.0


@dataclass
class JointTrajectory:
    """Represents a linear interpolation trajectory for a single joint."""

    joint_name: str
    start_pos: float
    target_pos: float
    start_time: float
    duration: float

    def position_at(self, t: float) -> float:
        """Compute position at normalized time t in [0, 1] using linear interpolation."""
        if t <= 0.0:
            return self.start_pos
        if t >= 1.0:
            return self.target_pos

        # Linear interpolation: p(t) = (1-t)*p0 + t*p1
        return (1.0 - t) * self.start_pos + t * self.target_pos

    def is_complete(self, current_time: float) -> bool:
        """Check if trajectory is complete."""
        return current_time >= self.start_time + self.duration


def compute_trajectory_duration(start_pos: float, target_pos: float, loop_period: float) -> float:
    """Compute trajectory duration based on distance and max velocity."""
    distance = abs(target_pos - start_pos)
    if distance < POSITION_TOLERANCE:
        return 0.0
    duration = distance / MAX_VELOCITY_DEG_PER_SEC
    return max(duration, loop_period)  # At least one loop period


def get_joint_positions(observation: dict) -> dict[str, float]:
    """Extract joint positions from observation dictionary."""
    return {k.replace(".pos", ""): v for k, v in observation.items() if k.endswith(".pos")}


def move_joint_to_target(
    robot: SO101Follower,
    current_positions: dict[str, float],
    joint_name: str,
    target_pos: float,
    start_time: float,
    enable_rerun_logging: bool = True,
    fps: int = FPS_DEFAULT,
    callback_observation: ObservationCallback | None = None,
) -> tuple[bool, dict[str, float]]:
    """
    Move a single joint to target using linear interpolation.

    Returns:
        (is_complete, updated_positions): Whether movement is complete and updated positions
    """
    return move_joints_to_target(
        robot,
        current_positions,
        {joint_name: target_pos},
        start_time,
        enable_rerun_logging,
        fps,
        callback_observation,
    )


def move_joints_to_target(
    robot: SO101Follower,
    current_positions: dict[str, float],
    joint_targets: dict[str, float],
    start_time: float,
    enable_rerun_logging: bool = True,
    fps: int = FPS_DEFAULT,
    callback_observation: ObservationCallback | None = None,
) -> tuple[bool, dict[str, float]]:
    """
    Move multiple joints to their targets simultaneously using linear interpolation.
    Runs in open loop mode - observations are read for logging/visualization only
    and do not influence the actions. Actions are computed purely from time-based
    interpolation.

    Args:
        robot: The robot instance
        current_positions: Current joint positions (keys without .pos suffix)
        joint_targets: Dictionary mapping joint names to target positions
        start_time: Start time for the trajectory
        enable_rerun_logging: Whether to log data to rerun for visualization
        fps: Control loop frequency in frames per second
        callback_observation: Optional callback called with (observation, action) at each step

    Returns:
        (is_complete, updated_positions): Whether movement is complete and updated positions
    """
    loop_period = 1.0 / fps

    # Get all joint names from robot to ensure we handle all joints consistently
    all_joint_names = list(robot.bus.motors.keys())

    # Track last commanded action for all joints
    # Initialize from current positions - this will be updated as we send commands
    last_action = {}
    for joint_name in all_joint_names:
        action_key = f"{joint_name}.pos"
        last_action[action_key] = current_positions.get(joint_name, 0.0)

    # Create trajectories for all joints
    # Use last_action (last commanded) as start position for smooth transitions
    trajectories = {}
    max_duration = 0.0

    for joint_name, target_pos in joint_targets.items():
        action_key = f"{joint_name}.pos"
        start_pos = last_action.get(action_key, current_positions.get(joint_name, 0.0))

        # Check if already at target
        if abs(start_pos - target_pos) >= POSITION_TOLERANCE:
            duration = compute_trajectory_duration(start_pos, target_pos, loop_period)
            if duration > 0.0:
                trajectories[joint_name] = JointTrajectory(
                    joint_name=joint_name,
                    start_pos=start_pos,
                    target_pos=target_pos,
                    start_time=start_time,
                    duration=duration,
                )
                max_duration = max(max_duration, duration)

    # If no trajectories needed, return early
    if not trajectories:
        return True, current_positions

    def build_action(elapsed: float, use_targets: bool = False) -> dict[str, float]:
        """Build action dictionary for all joints."""
        action = {}
        for joint_name in all_joint_names:
            action_key = f"{joint_name}.pos"

            if joint_name in trajectories:
                # Moving joint: interpolate along trajectory
                traj = trajectories[joint_name]
                if use_targets:
                    # Use final target position
                    action[action_key] = traj.target_pos
                else:
                    # Interpolate along trajectory
                    # Ensure elapsed is non-negative and properly clamped
                    elapsed_clamped = max(0.0, elapsed)
                    t = elapsed_clamped / traj.duration if traj.duration > 0 else 1.0
                    t = min(t, 1.0)  # Clamp to [0, 1]
                    action[action_key] = traj.position_at(t)
                last_action[action_key] = action[action_key]
            else:
                # Non-moving joint: use last commanded position
                action[action_key] = last_action[action_key]

        return action

    # Send initial action to ensure continuity (matches trajectory start positions)
    initial_action = build_action(0.0)
    robot.send_action(initial_action)
    obs = robot.get_observation()  # Read observation for logging only
    if enable_rerun_logging:
        log_rerun_data(observation=obs, action=initial_action)
    if callback_observation is not None:
        callback_observation(obs, initial_action)

    # Execute trajectories simultaneously in open loop
    loop_start = time.perf_counter()
    while True:
        current_time = time.time()
        elapsed = current_time - start_time

        # Check if all trajectories are complete
        if elapsed >= max_duration:
            # Final step: send target positions
            action = build_action(elapsed, use_targets=True)
            robot.send_action(action)
            obs = robot.get_observation()  # Read observation for logging only
            if enable_rerun_logging:
                log_rerun_data(observation=obs, action=action)
            if callback_observation is not None:
                callback_observation(obs, action)
            break

        # Build and send interpolated action
        action = build_action(elapsed)
        robot.send_action(action)
        obs = robot.get_observation()  # Read observation for logging only (not used for control)
        if enable_rerun_logging:
            log_rerun_data(observation=obs, action=action)
        if callback_observation is not None:
            callback_observation(obs, action)

        # Maintain loop frequency
        elapsed_loop = time.perf_counter() - loop_start
        if elapsed_loop < loop_period:
            time.sleep(loop_period - elapsed_loop)
        loop_start = time.perf_counter()

    # Return final commanded positions (not observed)
    final_positions = {}
    for joint_name in all_joint_names:
        action_key = f"{joint_name}.pos"
        final_positions[joint_name] = last_action[action_key]

    return True, final_positions


def move_to_safe_height(
    robot: SO101Follower,
    current_positions: dict[str, float] | None = None,
    enable_rerun_logging: bool = True,
    fps: int = FPS_DEFAULT,
    callback_observation: ObservationCallback | None = None,
) -> dict[str, float]:
    """
    Move the robot to safe height by centering shoulder_pan and lifting shoulder_lift and elbow_flex.

    This is useful before performing other movements to avoid collisions.
    Runs in open loop mode - observations are read for logging only and do not influence actions.

    Args:
        robot: The robot instance (must be connected)
        current_positions: Optional current joint positions (commanded, not observed).
                         If None, will get from observation. Use this to ensure continuity
                         when transitioning from other control loops.
        enable_rerun_logging: Whether to log data to rerun for visualization.
        fps: Control loop frequency in frames per second.
        callback_observation: Optional callback called with (observation, action) at each step.

    Returns:
        Updated joint positions (commanded, not observed) after moving to safe height
    """
    if current_positions is None:
        current_obs = robot.get_observation()
        current_positions = get_joint_positions(current_obs)

    joint_targets = {}
    for joint_name in ["shoulder_pan", "shoulder_lift", "elbow_flex"]:
        joint_key = f"{joint_name}.pos"
        if joint_key in SAFE_HEIGHT:
            joint_targets[joint_name] = SAFE_HEIGHT[joint_key]

    _, current_positions = move_joints_to_target(
        robot, current_positions, joint_targets, time.time(), enable_rerun_logging, fps, callback_observation
    )

    return current_positions


def run_homing_sequence(
    robot: SO101Follower,
    current_positions: dict[str, float] | None = None,
    enable_rerun_logging: bool = True,
    fps: int = FPS_DEFAULT,
    callback_observation: ObservationCallback | None = None,
) -> dict[str, float]:
    """
    Run the homing sequence to safely move the robot to the home position.
    Runs in open loop mode - observations are read at each time step for logging/visualization
    but do not influence the actions. Actions are computed purely from time-based interpolation.

    Args:
        robot: The robot instance (must be connected)
        current_positions: Optional current joint positions (commanded, not observed).
                         If None, will get from observation. Use this to ensure continuity
                         when transitioning from other control loops.
        enable_rerun_logging: Whether to log data to rerun for visualization.
        fps: Control loop frequency in frames per second.
        callback_observation: Optional callback called with (observation, action) at each step.

    Returns:
        Final joint positions (commanded, not observed) after homing
    """
    # Get initial position
    if current_positions is None:
        current_obs = robot.get_observation()
        current_positions = get_joint_positions(current_obs)
        if enable_rerun_logging:
            log_rerun_data(observation=current_obs)
        if callback_observation is not None:
            callback_observation(current_obs, None)
    else:
        # Still log observation for visualization, but use provided positions for control
        current_obs = robot.get_observation()
        if enable_rerun_logging:
            log_rerun_data(observation=current_obs)
        if callback_observation is not None:
            callback_observation(current_obs, None)

    start_time = time.time()

    # Step 1: Lift arm to safe height
    current_positions = move_to_safe_height(
        robot,
        current_positions=current_positions,
        enable_rerun_logging=enable_rerun_logging,
        fps=fps,
        callback_observation=callback_observation,
    )

    # Step 2: Open gripper (release any objects)
    _, current_positions = move_joint_to_target(
        robot,
        current_positions,
        "gripper",
        GRIPPER_OPEN,
        time.time(),
        enable_rerun_logging,
        fps,
        callback_observation,
    )

    # Step 3: Move wrist joints and close gripper together (while in lifted position)
    joint_targets = {}
    for joint_name in ["wrist_flex", "wrist_roll", "gripper"]:
        joint_key = f"{joint_name}.pos"
        if joint_key in JOINT_ACTION_HOME:
            joint_targets[joint_name] = JOINT_ACTION_HOME[joint_key]
    _, current_positions = move_joints_to_target(
        robot, current_positions, joint_targets, time.time(), enable_rerun_logging, fps, callback_observation
    )

    # Step 4: Lower to final home position (move shoulder_lift and elbow_flex together)
    joint_targets = {}
    for joint_name in ["shoulder_lift", "elbow_flex"]:
        joint_key = f"{joint_name}.pos"
        if joint_key in JOINT_ACTION_HOME:
            joint_targets[joint_name] = JOINT_ACTION_HOME[joint_key]
    _, current_positions = move_joints_to_target(
        robot, current_positions, joint_targets, time.time(), enable_rerun_logging, fps, callback_observation
    )

    # Verify final position
    final_obs = robot.get_observation()
    final_positions = get_joint_positions(final_obs)
    if enable_rerun_logging:
        log_rerun_data(observation=final_obs)

    elapsed_time = time.time() - start_time
    max_error = 0.0
    for joint, pos in final_positions.items():
        target_key = f"{joint}.pos"
        if target_key in JOINT_ACTION_HOME:
            error = JOINT_ACTION_HOME[target_key] - pos
            max_error = max(max_error, abs(error))
    logger.info(f"Homing complete in {elapsed_time:.1f}s (max error: {max_error:.2f})")

    return final_positions


def main() -> int:
    """Run the homing sequence on the follower arm."""
    parser = argparse.ArgumentParser(description="SO-101 Homing Utility")
    parser.add_argument(
        "--port", type=str, required=True, help="Robot serial port (e.g., /dev/tty.usbmodem5A460829821)"
    )
    parser.add_argument("--id", type=str, default="arm_follower_0", help="Robot ID")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logger.info(f"Port: {args.port}")
    logger.info(f"Robot ID: {args.id}")

    # Setup robot
    robot_config = SO101FollowerConfig(
        port=args.port,
        id=args.id,
        use_degrees=True,
        max_relative_target=None,  # We handle safety ourselves with velocity limits
    )
    robot = SO101Follower(robot_config)

    # Connect to robot
    logger.info("Connecting to robot...")
    try:
        robot.connect()
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        return 1

    if not robot.is_connected:
        logger.error("Robot is not connected!")
        return 1

    logger.info("Robot connected successfully")

    try:
        # Run homing sequence
        run_homing_sequence(robot, enable_rerun_logging=False)
    except Exception as e:
        logger.error(f"Homing failed: {e}")
        robot.disconnect()
        return 1

    # Disconnect
    logger.info("Disconnecting robot...")
    robot.disconnect()
    logger.info("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
