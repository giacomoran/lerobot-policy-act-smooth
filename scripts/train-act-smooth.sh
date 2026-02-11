#!/bin/bash
#
# Train ACTSmooth policy on the lerobot_policy_act_smooth datasets.
#
# Datasets:
#   - giacomoran/lerobot_policy_act_smooth_10fps
#   - giacomoran/lerobot_policy_act_smooth_30fps
#
# Models:
#   1. p1f2 @ 10fps  — baseline, (2-1)/10 = 100ms latency budget
#   2. p4f2 @ 30fps  — baseline, (2-1)/30 = 33ms latency budget
#   3. p1f4 @ 10fps  — +150ms delay testing, (4-1)/10 = 300ms latency budget
#   4. p4f8 @ 30fps  — +150ms delay testing, (8-1)/30 = 233ms latency budget

set -e

DIR_BASE="/workspace"
INPUT_FEATURES='{"observation.state": {"shape": [6], "type": "STATE"}, "observation.images.wrist": {"shape": [3, 640, 480], "type": "VISUAL"}, "observation.images.top": {"shape": [3, 480, 640], "type": "VISUAL"}}'

STEPS=30000
SAVE_FREQ=10000

# =============================================================================
# 1. ACTSmooth-p1f2 @ 10fps
# =============================================================================
# length_prefix_future=2: needed for interpolation at 30fps.
#   t_0 is at the observation timestep (already executing), so only d-1 actions
#   absorb inference latency. With d=2: (2-1)/10 = 100ms > ~35ms inference latency.
#   d=1 would leave no interpolation target while inference runs.
echo "=== [1/4] Training 10fps p1f2 ==="

lerobot-train \
    --policy.type=act_smooth \
    --dataset.repo_id=giacomoran/lerobot_policy_act_smooth_10fps \
    --policy.repo_id=giacomoran/lerobot_policy_act_smooth_10fps_smooth_p1f2 \
    --output_dir="${DIR_BASE}/lerobot_policy_act_smooth_10fps_smooth_p1f2" \
    --policy.input_features="${INPUT_FEATURES}" \
    --steps=${STEPS} \
    --save_freq=${SAVE_FREQ} \
    --batch_size=32 \
    --policy.optimizer_lr=3e-5 \
    --policy.optimizer_lr_backbone=3e-5 \
    --policy.chunk_size=10 \
    --policy.n_action_steps=10 \
    --policy.use_vae=false \
    --policy.length_prefix_past=1 \
    --policy.length_prefix_future=2 \
    --policy.use_action_relative=true \
    --policy.device=cuda \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --policy.push_to_hub=true \
    --num_workers=8

echo "=== Resuming 10fps p1f2 for extra checkpoints ==="

lerobot-train \
    --resume=true \
    --config_path="${DIR_BASE}/lerobot_policy_act_smooth_10fps_smooth_p1f2/checkpoints/030000/pretrained_model/train_config.json" \
    --output_dir="${DIR_BASE}/lerobot_policy_act_smooth_10fps_smooth_p1f2" \
    --steps=30003 \
    --save_freq=1

# =============================================================================
# 2. ACTSmooth-p4f2 @ 30fps
# =============================================================================
# length_prefix_future=2: sufficient without interpolation (already at 30fps).
#   With d=2: (2-1)/30 = 33ms, close to ~35ms inference latency but OK since
#   no interpolation is needed at native fps.
echo "=== [2/4] Training 30fps p4f2 ==="

lerobot-train \
    --policy.type=act_smooth \
    --dataset.repo_id=giacomoran/lerobot_policy_act_smooth_30fps \
    --policy.repo_id=giacomoran/lerobot_policy_act_smooth_30fps_smooth_p4f2 \
    --output_dir="${DIR_BASE}/lerobot_policy_act_smooth_30fps_smooth_p4f2" \
    --policy.input_features="${INPUT_FEATURES}" \
    --steps=${STEPS} \
    --save_freq=${SAVE_FREQ} \
    --batch_size=32 \
    --policy.optimizer_lr=3e-5 \
    --policy.optimizer_lr_backbone=3e-5 \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --policy.use_vae=false \
    --policy.length_prefix_past=4 \
    --policy.length_prefix_future=2 \
    --policy.use_action_relative=true \
    --policy.device=cuda \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --policy.push_to_hub=true \
    --num_workers=8

echo "=== Resuming 30fps p4f2 for extra checkpoints ==="

lerobot-train \
    --resume=true \
    --config_path="${DIR_BASE}/lerobot_policy_act_smooth_30fps_smooth_p4f2/checkpoints/030000/pretrained_model/train_config.json" \
    --output_dir="${DIR_BASE}/lerobot_policy_act_smooth_30fps_smooth_p4f2" \
    --steps=30003 \
    --save_freq=1

# =============================================================================
# 3. ACTSmooth-p1f4 @ 10fps  (for +150ms injected delay testing)
# =============================================================================
# length_prefix_future=4: larger prefix to absorb +150ms injected delay.
#   With d=4: (4-1)/10 = 300ms > ~35ms + 150ms = 185ms total latency.
echo "=== [3/4] Training 10fps p1f4 ==="

lerobot-train \
    --policy.type=act_smooth \
    --dataset.repo_id=giacomoran/lerobot_policy_act_smooth_10fps \
    --policy.repo_id=giacomoran/lerobot_policy_act_smooth_10fps_smooth_p1f4 \
    --output_dir="${DIR_BASE}/lerobot_policy_act_smooth_10fps_smooth_p1f4" \
    --policy.input_features="${INPUT_FEATURES}" \
    --steps=${STEPS} \
    --save_freq=${SAVE_FREQ} \
    --batch_size=32 \
    --policy.optimizer_lr=3e-5 \
    --policy.optimizer_lr_backbone=3e-5 \
    --policy.chunk_size=10 \
    --policy.n_action_steps=10 \
    --policy.use_vae=false \
    --policy.length_prefix_past=1 \
    --policy.length_prefix_future=4 \
    --policy.use_action_relative=true \
    --policy.device=cuda \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --policy.push_to_hub=true \
    --num_workers=8

echo "=== Resuming 10fps p1f4 for extra checkpoints ==="

lerobot-train \
    --resume=true \
    --config_path="${DIR_BASE}/lerobot_policy_act_smooth_10fps_smooth_p1f4/checkpoints/030000/pretrained_model/train_config.json" \
    --output_dir="${DIR_BASE}/lerobot_policy_act_smooth_10fps_smooth_p1f4" \
    --steps=30003 \
    --save_freq=1

# =============================================================================
# 4. ACTSmooth-p4f8 @ 30fps  (for +150ms injected delay testing)
# =============================================================================
# length_prefix_future=8: larger prefix to absorb +150ms injected delay.
#   With d=8: (8-1)/30 = 233ms > ~35ms + 150ms = 185ms total latency.
echo "=== [4/4] Training 30fps p4f8 ==="

lerobot-train \
    --policy.type=act_smooth \
    --dataset.repo_id=giacomoran/lerobot_policy_act_smooth_30fps \
    --policy.repo_id=giacomoran/lerobot_policy_act_smooth_30fps_smooth_p4f8 \
    --output_dir="${DIR_BASE}/lerobot_policy_act_smooth_30fps_smooth_p4f8" \
    --policy.input_features="${INPUT_FEATURES}" \
    --steps=${STEPS} \
    --save_freq=${SAVE_FREQ} \
    --batch_size=32 \
    --policy.optimizer_lr=3e-5 \
    --policy.optimizer_lr_backbone=3e-5 \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --policy.use_vae=false \
    --policy.length_prefix_past=4 \
    --policy.length_prefix_future=8 \
    --policy.use_action_relative=true \
    --policy.device=cuda \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --policy.push_to_hub=true \
    --num_workers=8

echo "=== Resuming 30fps p4f8 for extra checkpoints ==="

lerobot-train \
    --resume=true \
    --config_path="${DIR_BASE}/lerobot_policy_act_smooth_30fps_smooth_p4f8/checkpoints/030000/pretrained_model/train_config.json" \
    --output_dir="${DIR_BASE}/lerobot_policy_act_smooth_30fps_smooth_p4f8" \
    --steps=30003 \
    --save_freq=1

echo "=== Training complete ==="

# --- Compress outputs ---
# Include:
# - pretrained_model from all checkpoints
# - training_state from last checkpoint only
echo "=== Compressing outputs ==="
cd "${DIR_BASE}"

DIRS_OUTPUT="lerobot_policy_act_smooth_10fps_smooth_p1f2 lerobot_policy_act_smooth_30fps_smooth_p4f2 lerobot_policy_act_smooth_10fps_smooth_p1f4 lerobot_policy_act_smooth_30fps_smooth_p4f8"
ARGS_TAR=""

for DIR in ${DIRS_OUTPUT}; do
    if [ -d "${DIR}/checkpoints" ]; then
        # Add pretrained_model from all checkpoints
        for CKPT in $(ls -1 "${DIR}/checkpoints" | sort -n); do
            if [ -d "${DIR}/checkpoints/${CKPT}/pretrained_model" ]; then
                ARGS_TAR="${ARGS_TAR} ${DIR}/checkpoints/${CKPT}/pretrained_model"
            fi
        done
        # Add training_state from last checkpoint only
        CKPT_LATEST=$(ls -1 "${DIR}/checkpoints" | sort -n | tail -1)
        if [ -n "${CKPT_LATEST}" ] && [ -d "${DIR}/checkpoints/${CKPT_LATEST}/training_state" ]; then
            ARGS_TAR="${ARGS_TAR} ${DIR}/checkpoints/${CKPT_LATEST}/training_state"
        fi
    fi
done

if [ -n "${ARGS_TAR}" ]; then
    tar -czvf lerobot_policy_act_smooth_smooth.tar.gz ${ARGS_TAR}
    echo "Done! Archive: ${DIR_BASE}/lerobot_policy_act_smooth_smooth.tar.gz"
else
    echo "ERROR: No checkpoints found to compress!"
    exit 1
fi

# To transfer and uncompress on local machine:
#   Remote:  croc send lerobot_policy_act_smooth_smooth.tar.gz
#   Local:   croc <code>
#            tar -xzvf lerobot_policy_act_smooth_smooth.tar.gz -C /path/to/destination
