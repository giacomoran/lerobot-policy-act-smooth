#!/bin/bash
#
# Ablation training runs for ACTSmooth on the 30fps dataset.
#
# Ablations:
#   - p4f2_norelative: no relative action representation (use_action_relative=false)
#   - p0f2: no past action prefix (length_prefix_past=0)

set -e

DIR_BASE="/workspace"

# --- Ablation: no relative actions ---
echo "=== Training 30fps p4f2_norelative (no relative actions) ==="

lerobot-train \
    --policy.type=act_smooth \
    --dataset.repo_id=giacomoran/lerobot_policy_act_smooth_30fps \
    --policy.repo_id=giacomoran/lerobot_policy_act_smooth_30fps_smooth_p4f2_norelative \
    --output_dir="${DIR_BASE}/lerobot_policy_act_smooth_30fps_smooth_p4f2_norelative" \
    --policy.input_features='{"observation.state": {"shape": [6], "type": "STATE"}, "observation.images.wrist": {"shape": [3, 640, 480], "type": "VISUAL"}, "observation.images.top": {"shape": [3, 480, 640], "type": "VISUAL"}}' \
    --steps=30000 \
    --save_freq=10000 \
    --batch_size=32 \
    --policy.optimizer_lr=3e-5 \
    --policy.optimizer_lr_backbone=3e-5 \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --policy.use_vae=false \
    --policy.length_prefix_past=4 \
    --policy.length_prefix_future=2 \
    --policy.use_action_relative=false \
    --policy.device=cuda \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --policy.push_to_hub=true \
    --num_workers=8

echo "=== Resuming p4f2_norelative to add extra checkpoints ==="

lerobot-train \
    --resume=true \
    --config_path="${DIR_BASE}/lerobot_policy_act_smooth_30fps_smooth_p4f2_norelative/checkpoints/030000/pretrained_model/train_config.json" \
    --output_dir="${DIR_BASE}/lerobot_policy_act_smooth_30fps_smooth_p4f2_norelative" \
    --steps=30003 \
    --save_freq=1

# --- Ablation: no past prefix ---
echo "=== Training 30fps p0f2 (no past action prefix) ==="

lerobot-train \
    --policy.type=act_smooth \
    --dataset.repo_id=giacomoran/lerobot_policy_act_smooth_30fps \
    --policy.repo_id=giacomoran/lerobot_policy_act_smooth_30fps_smooth_p0f2 \
    --output_dir="${DIR_BASE}/lerobot_policy_act_smooth_30fps_smooth_p0f2" \
    --policy.input_features='{"observation.state": {"shape": [6], "type": "STATE"}, "observation.images.wrist": {"shape": [3, 640, 480], "type": "VISUAL"}, "observation.images.top": {"shape": [3, 480, 640], "type": "VISUAL"}}' \
    --steps=30000 \
    --save_freq=10000 \
    --batch_size=32 \
    --policy.optimizer_lr=3e-5 \
    --policy.optimizer_lr_backbone=3e-5 \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --policy.use_vae=false \
    --policy.length_prefix_past=0 \
    --policy.length_prefix_future=2 \
    --policy.use_action_relative=true \
    --policy.device=cuda \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --policy.push_to_hub=true \
    --num_workers=8

echo "=== Resuming p0f2 to add extra checkpoints ==="

lerobot-train \
    --resume=true \
    --config_path="${DIR_BASE}/lerobot_policy_act_smooth_30fps_smooth_p0f2/checkpoints/030000/pretrained_model/train_config.json" \
    --output_dir="${DIR_BASE}/lerobot_policy_act_smooth_30fps_smooth_p0f2" \
    --steps=30003 \
    --save_freq=1

echo "=== Ablation training complete ==="

# --- Compress outputs ---
echo "=== Compressing outputs ==="
cd "${DIR_BASE}"

DIRS_OUTPUT="lerobot_policy_act_smooth_30fps_smooth_p4f2_norelative lerobot_policy_act_smooth_30fps_smooth_p0f2"
ARGS_TAR=""

for DIR in ${DIRS_OUTPUT}; do
    if [ -d "${DIR}/checkpoints" ]; then
        for CKPT in $(ls -1 "${DIR}/checkpoints" | sort -n); do
            if [ -d "${DIR}/checkpoints/${CKPT}/pretrained_model" ]; then
                ARGS_TAR="${ARGS_TAR} ${DIR}/checkpoints/${CKPT}/pretrained_model"
            fi
        done
        CKPT_LATEST=$(ls -1 "${DIR}/checkpoints" | sort -n | tail -1)
        if [ -n "${CKPT_LATEST}" ] && [ -d "${DIR}/checkpoints/${CKPT_LATEST}/training_state" ]; then
            ARGS_TAR="${ARGS_TAR} ${DIR}/checkpoints/${CKPT_LATEST}/training_state"
        fi
    fi
done

if [ -n "${ARGS_TAR}" ]; then
    tar -czvf lerobot_policy_act_smooth_ablations.tar.gz ${ARGS_TAR}
    echo "Done! Archive: ${DIR_BASE}/lerobot_policy_act_smooth_ablations.tar.gz"
else
    echo "ERROR: No checkpoints found to compress!"
    exit 1
fi
