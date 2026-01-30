#!/bin/bash
#
# Record episodes with the SO101 follower arm using the leader arm for teleoperation.
#
# Usage:
#   ./record.sh          # Start fresh recording
#   ./record.sh --resume # Resume recording on existing dataset
#
# NOTE: The configuration below is specific to Giacomo's setup.
# You will need to substitute your own values:
#   - robot.port / teleop.port: USB ports for your arms (find with `lerobot-find-port`)
#   - robot.id / teleop.id: IDs for your arms
#   - robot.cameras: Camera indices depend on your system (find with `lerobot-find-cameras`)

# Check for --resume flag
RESUME_FLAG=""
if [[ "$1" == "--resume" ]]; then
    RESUME_FLAG="--resume=true"
fi

CAMERAS="{
  wrist: { type: opencv, index_or_path: 1, width: 480, height: 640, fps: 30, rotation: -90 },
  top:   { type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30 }
}"

lerobot-record \
    $RESUME_FLAG \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A460829821 \
    --robot.id=arm_follower_0 \
    --robot.cameras="$CAMERAS" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5A460824651 \
    --teleop.id=arm_leader_0 \
    --display_data=true \
    --dataset.repo_id=giacomoran/lerobot_policy_act_smooth_30fps \
    --dataset.num_episodes=5 \
    --dataset.single_task="Pick up the cube and place it in the target location" \
    --dataset.fps=30 \
    --dataset.push_to_hub=True
