#!/usr/bin/env python
"""Robot eval wrapper for relative-prediction checkpoint.

Monkey-patches ACTSmoothPolicy.predict_action_chunk to apply the
relative-action transform (subtract anchor before model, add back after),
then delegates to eval_async_smooth.main().

Usage:
    python experiments/2026-02-08_train-improvements/03-relative-prediction/eval_robot.py \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem5A460829821 \
        --robot.cameras="..." \
        --robot.id=arm_follower_0 \
        --policy.path=experiments/2026-02-08_train-improvements/03-relative-prediction/checkpoints/030000/pretrained_model \
        --fps_policy=10 \
        --fps_interpolation=30 \
        --fps_observation=60 \
        --display_data=true
"""

import sys
from pathlib import Path

import torch
from torch import Tensor

# Add scripts/eval to path so eval_async_smooth can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "eval"))

from lerobot_policy_act_smooth.modeling_act_smooth import (
    ACTSmoothPolicy,
    INFERENCE_ACTION,
    INFERENCE_ACTION_IS_PAD,
    OBS_IMAGES,
)


def predict_action_chunk_relative(self, batch: dict[str, Tensor]) -> Tensor:
    """predict_action_chunk with relative-action transform."""
    assert INFERENCE_ACTION in batch
    assert INFERENCE_ACTION_IS_PAD in batch

    self.train(False)

    if self.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

    # Anchor = action at the observation timestep (index length_prefix_past)
    length_prefix_past = self.config.length_prefix_past
    action_anchor = batch[INFERENCE_ACTION][:, length_prefix_past : length_prefix_past + 1]

    batch = dict(batch)
    batch[INFERENCE_ACTION] = batch[INFERENCE_ACTION] - action_anchor

    actions_relative = self.model(batch)[0]
    actions = actions_relative + action_anchor
    return actions


# Patch
ACTSmoothPolicy.predict_action_chunk = predict_action_chunk_relative

# Delegate to eval_async_smooth
from eval_async_smooth import main

main()
