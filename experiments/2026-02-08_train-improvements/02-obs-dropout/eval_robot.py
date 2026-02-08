#!/usr/bin/env python
"""Robot eval wrapper for observation-dropout checkpoints.

Obs dropout is training-only (proprioception zeroed with probability p_drop_obs
during training). At inference the model runs unchanged, so this script simply
delegates to eval_async_smooth.

Usage (p=0.1):
    python experiments/2026-02-08_train-improvements/02-obs-dropout/eval_robot.py \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem5A460829821 \
        --robot.id=arm_follower_0 \
        --robot.cameras="{ wrist: { type: opencv, index_or_path: 1, width: 480, height: 640, fps: 30, rotation: -90 }, top: { type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30 } }" \
        --policy.path=experiments/2026-02-08_train-improvements/02-obs-dropout/checkpoints/p_drop_0.1/checkpoints/030000/pretrained_model \
        --fps_policy=30 \
        --fps_interpolation=30 \
        --fps_observation=60 \
        --episode_time_s=30 \
        --display_data=true

Usage (p=0.3):
    python experiments/2026-02-08_train-improvements/02-obs-dropout/eval_robot.py \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem5A460829821 \
        --robot.id=arm_follower_0 \
        --robot.cameras="{ wrist: { type: opencv, index_or_path: 1, width: 480, height: 640, fps: 30, rotation: -90 }, top: { type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30 } }" \
        --policy.path=experiments/2026-02-08_train-improvements/02-obs-dropout/checkpoints/p_drop_0.3/checkpoints/030000/pretrained_model \
        --fps_policy=30 \
        --fps_interpolation=30 \
        --fps_observation=60 \
        --episode_time_s=30 \
        --display_data=true
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "eval"))

from eval_async_smooth import main

main()
