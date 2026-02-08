#!/bin/bash
# 03-relative-prediction: Train from scratch with relative action prediction.
# All actions (prefix + targets) are transformed to deltas relative to a_{t_0}.
# This architecturally enforces C0 continuity by anchoring predictions.
#
# Code change: Copies modified modeling_act_smooth.py into the installed package,
# then restores originals after training.
set -e

DIR_EXPERIMENT="$(cd "$(dirname "$0")" && pwd)"
DIR_REPO="$HOME/lerobot-policy-act-smooth"
DIR_SRC="${DIR_REPO}/src/lerobot_policy_act_smooth"
ID_REPO_DATASET="giacomoran/lerobot_policy_act_smooth_30fps"

# --- UPDATE THIS after LR sweep ---
LR="3e-5"

STEPS=30000
SAVE_FREQ=10000
BATCH_SIZE=32

INPUT_FEATURES='{"observation.state": {"shape": [6], "type": "STATE"}, "observation.images.wrist": {"shape": [3, 640, 480], "type": "VISUAL"}, "observation.images.top": {"shape": [3, 480, 640], "type": "VISUAL"}}'

echo "=== 03-relative-prediction: Setup ==="

# Back up original source
cp "${DIR_SRC}/modeling_act_smooth.py" "${DIR_SRC}/modeling_act_smooth.py.bak"

# Install modified source (config unchanged, only modeling)
cp "${DIR_EXPERIMENT}/src/modeling_act_smooth.py" "${DIR_SRC}/modeling_act_smooth.py"

# Cleanup function
cleanup() {
    echo "=== Restoring original source ==="
    mv "${DIR_SRC}/modeling_act_smooth.py.bak" "${DIR_SRC}/modeling_act_smooth.py"
}
trap cleanup EXIT

echo "=== 03-relative-prediction: Training ==="

lerobot-train \
    --policy.type=act_smooth \
    --dataset.repo_id="${ID_REPO_DATASET}" \
    --output_dir="${DIR_EXPERIMENT}/checkpoints" \
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

echo "=== 03-relative-prediction: Done ==="
