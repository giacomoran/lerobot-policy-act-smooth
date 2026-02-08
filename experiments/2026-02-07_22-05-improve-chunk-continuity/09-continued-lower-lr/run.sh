#!/bin/bash
# 09-continued-lower-lr: Continue training for 10K more steps at 10x lower LR.
# No code changes - just lower learning rate (3e-6 vs 3e-5).
# Uses config_path without resume to get fresh optimizer with new LR,
# while pretrained_path in config still loads model weights.
set -e

DIR_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$(dirname "$0")/../env.sh"
DIR_EXPERIMENT="$(cd "$(dirname "$0")" && pwd)"
PATH_CHECKPOINT="${DIR_ROOT}/outputs/lerobot_policy_act_smooth_30fps_smooth_p4f2/checkpoints/050003"
ID_REPO_DATASET="giacomoran/lerobot_policy_act_smooth_30fps"

STEPS_FINETUNE=10000

echo "=== 09-continued-lower-lr: Fine-tune at lr=3e-6 for ${STEPS_FINETUNE} steps ==="

# --- Train ---
# Don't use --resume=true: that loads optimizer state (with old lr=3e-5).
# Instead, load config (which includes pretrained_path) and override lr.
echo "Training: ${STEPS_FINETUNE} steps at lr=3e-6..."
lerobot-train \
    --config_path="${DIR_EXPERIMENT}/train_config_lr3e6.json" \
    --output_dir="${DIR_EXPERIMENT}/checkpoints" \
    --steps="${STEPS_FINETUNE}" \
    --save_freq=$((STEPS_FINETUNE + 1)) \
    --batch_size=8 \
    --num_workers=4 \
    --optimizer.lr=3e-6 \
    --policy.optimizer_lr=3e-6 \
    --policy.optimizer_lr_backbone=3e-6 \
    --policy.push_to_hub=false \
    --wandb.enable=false

# --- Eval ---
echo "Running offline eval..."
PATH_POLICY="${DIR_EXPERIMENT}/checkpoints/checkpoints/last/pretrained_model"
python "${DIR_ROOT}/scripts/eval/eval_offline_replay.py" \
    --path_policy="${PATH_POLICY}" \
    --id_repo_dataset="${ID_REPO_DATASET}" \
    --indices_episode=0,1,2 \
    --path_output="${DIR_EXPERIMENT}/data/lower_lr"

# Copy results
echo "=== Results ===" > "${DIR_EXPERIMENT}/report.md"
echo "" >> "${DIR_EXPERIMENT}/report.md"
echo '```' >> "${DIR_EXPERIMENT}/report.md"
cat "${DIR_EXPERIMENT}/data/lower_lr_results.txt" >> "${DIR_EXPERIMENT}/report.md"
echo '```' >> "${DIR_EXPERIMENT}/report.md"

cp "${DIR_EXPERIMENT}"/data/*trajectories.png "${DIR_EXPERIMENT}/figures/" 2>/dev/null || true
cp "${DIR_EXPERIMENT}"/data/*accelerations.png "${DIR_EXPERIMENT}/figures/" 2>/dev/null || true

echo "=== 09-continued-lower-lr: Done ==="
