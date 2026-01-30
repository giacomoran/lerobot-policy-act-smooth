#!/bin/bash
#
# Train vanilla ACT policy on the lerobot_policy_act_smooth datasets.
#
# Datasets:
#   - giacomoran/lerobot_policy_act_smooth_10fps
#   - giacomoran/lerobot_policy_act_smooth_30fps

set -e

DIR_BASE="/workspace"

# --- 10fps dataset ---
echo "=== Training on lerobot_policy_act_smooth_10fps ==="

lerobot-train \
    --policy.type=act \
    --dataset.repo_id=giacomoran/lerobot_policy_act_smooth_10fps \
    --policy.repo_id=giacomoran/lerobot_policy_act_smooth_10fps_vanilla \
    --output_dir="${DIR_BASE}/lerobot_policy_act_smooth_10fps_vanilla" \
    --policy.input_features='{"observation.state": {"shape": [6], "type": "STATE"}, "observation.images.wrist": {"shape": [3, 640, 480], "type": "VISUAL"}, "observation.images.top": {"shape": [3, 480, 640], "type": "VISUAL"}}' \
    --steps=20000 \
    --save_freq=9000 \
    --batch_size=32 \
    --policy.optimizer_lr=3e-5 \
    --policy.optimizer_lr_backbone=3e-5 \
    --policy.chunk_size=10 \
    --policy.n_action_steps=10 \
    --policy.use_vae=false \
    --policy.device=cuda \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --policy.push_to_hub=true \
    --num_workers=8

# --- 30fps dataset ---
echo "=== Training on lerobot_policy_act_smooth_30fps ==="

lerobot-train \
    --policy.type=act \
    --dataset.repo_id=giacomoran/lerobot_policy_act_smooth_30fps \
    --policy.repo_id=giacomoran/lerobot_policy_act_smooth_30fps_vanilla \
    --output_dir="${DIR_BASE}/lerobot_policy_act_smooth_30fps_vanilla" \
    --policy.input_features='{"observation.state": {"shape": [6], "type": "STATE"}, "observation.images.wrist": {"shape": [3, 640, 480], "type": "VISUAL"}, "observation.images.top": {"shape": [3, 480, 640], "type": "VISUAL"}}' \
    --steps=20000 \
    --save_freq=9000 \
    --batch_size=32 \
    --policy.optimizer_lr=3e-5 \
    --policy.optimizer_lr_backbone=3e-5 \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --policy.use_vae=false \
    --policy.device=cuda \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --policy.push_to_hub=true \
    --num_workers=8

echo "=== Training complete ==="

# --- Compress outputs ---
# Include:
# - pretrained_model from all checkpoints
# - training_state from last checkpoint only
echo "=== Compressing outputs ==="
cd "${DIR_BASE}"

DIRS_OUTPUT="lerobot_policy_act_smooth_10fps_vanilla lerobot_policy_act_smooth_30fps_vanilla"
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
    tar -czvf lerobot_policy_act_smooth_vanilla.tar.gz ${ARGS_TAR}
    echo "Done! Archive: ${DIR_BASE}/lerobot_policy_act_smooth_vanilla.tar.gz"
else
    echo "ERROR: No checkpoints found to compress!"
    exit 1
fi

# To transfer and uncompress on local machine:
#   Remote:  croc send lerobot_policy_act_smooth_vanilla.tar.gz
#   Local:   croc <code>
#            tar -xzvf lerobot_policy_act_smooth_vanilla.tar.gz -C /path/to/destination
