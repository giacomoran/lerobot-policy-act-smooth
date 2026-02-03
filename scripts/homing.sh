#!/bin/bash
#
# Run homing sequence to move the SO101 arm to its home position.

python scripts/shared/homing.py --port=/dev/tty.usbmodem5A460829821 --id=arm_follower_0
