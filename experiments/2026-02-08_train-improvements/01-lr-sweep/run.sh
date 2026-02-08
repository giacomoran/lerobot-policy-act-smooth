#!/bin/bash
# 01-lr-sweep: Train from scratch for 5K steps at different learning rates.
# Purpose: Find the best LR for from-scratch training before running longer experiments.
set -e

DIR_EXPERIMENT="$(cd "$(dirname "$0")" && pwd)"
DIR_OUTPUT="${DIR_EXPERIMENT}/checkpoints"
ID_REPO_DATASET="giacomoran/lerobot_policy_act_smooth_30fps"

STEPS=5000
SAVE_FREQ=5000
BATCH_SIZE=32

INPUT_FEATURES='{"observation.state": {"shape": [6], "type": "STATE"}, "observation.images.wrist": {"shape": [3, 640, 480], "type": "VISUAL"}, "observation.images.top": {"shape": [3, 480, 640], "type": "VISUAL"}}'

for LR in 1e-5 3e-5 5e-5 1e-4 3e-4; do
    echo "========================================"
    echo "=== LR sweep: lr=${LR}, ${STEPS} steps ==="
    echo "========================================"

    lerobot-train \
        --policy.type=act_smooth \
        --dataset.repo_id="${ID_REPO_DATASET}" \
        --output_dir="${DIR_OUTPUT}/lr_${LR}" \
        --policy.input_features="${INPUT_FEATURES}" \
        --steps="${STEPS}" \
        --save_freq="${SAVE_FREQ}" \
        --batch_size="${BATCH_SIZE}" \
        --policy.optimizer_lr="${LR}" \
        --policy.optimizer_lr_backbone="${LR}" \
        --policy.chunk_size=30 \
        --policy.n_action_steps=30 \
        --policy.use_vae=false \
        --policy.length_prefix_past=4 \
        --policy.length_prefix_future=2 \
        --policy.device=cuda \
        --policy.push_to_hub=false \
        --wandb.enable=false \
        --num_workers=8

    echo "=== LR ${LR} done ==="
done

echo ""
echo "=== LR Sweep Complete ==="
echo "Checkpoints saved to ${DIR_OUTPUT}/"
echo ""
echo "Compare final training losses across LRs to pick the best one."
echo "Then update 02-obs-dropout/run.sh and 03-relative-prediction/run.sh with the winning LR."
