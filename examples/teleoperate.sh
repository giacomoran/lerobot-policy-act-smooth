#!/bin/bash
#
# Teleoperate the SO101 follower arm using the leader arm.
#
# NOTE: The configuration below is specific to Giacomo's setup.
# You will need to substitute your own values:
#   - robot.port / teleop.port: USB ports for your arms (find with `lerobot-find-port`)
#   - robot.id / teleop.id: IDs for your arms
#   - robot.cameras: Camera indices depend on your system (find with `lerobot-find-cameras`)

CAMERAS="{
  wrist: { type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30 },
  top:   { type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30 }
}"

lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5A460829821 \
  --robot.id=arm_follower_0 \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5A460824651 \
  --teleop.id=arm_leader_0 \
  --display_data=true
