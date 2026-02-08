#!/bin/bash
# 02-boundary-loss: Add C0/C1 continuity loss at prefix->chunk boundary.
# Fine-tune from step 50003 for 3000 more steps.
set -e

DIR_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$(dirname "$0")/../env.sh"
DIR_EXPERIMENT="$(cd "$(dirname "$0")" && pwd)"
DIR_SRC_ORIGINAL="${DIR_ROOT}/src/lerobot_policy_act_smooth"
DIR_SRC_EXPERIMENT="${DIR_EXPERIMENT}/src"
PATH_CHECKPOINT="${DIR_ROOT}/outputs/lerobot_policy_act_smooth_30fps_smooth_p4f2/checkpoints/050003"
ID_REPO_DATASET="giacomoran/lerobot_policy_act_smooth_30fps"

STEPS_BASE=50003
STEPS_FINETUNE=3000
STEPS_TOTAL=$((STEPS_BASE + STEPS_FINETUNE))

echo "=== 02-boundary-loss: Fine-tune with boundary loss ==="

# --- Swap source ---
echo "Backing up original source..."
cp "${DIR_SRC_ORIGINAL}/configuration_act_smooth.py" "${DIR_EXPERIMENT}/configuration_act_smooth.py.bak"
cp "${DIR_SRC_ORIGINAL}/modeling_act_smooth.py" "${DIR_EXPERIMENT}/modeling_act_smooth.py.bak"

echo "Installing experiment source..."
cp "${DIR_SRC_EXPERIMENT}/configuration_act_smooth.py" "${DIR_SRC_ORIGINAL}/configuration_act_smooth.py"
cp "${DIR_SRC_EXPERIMENT}/modeling_act_smooth.py" "${DIR_SRC_ORIGINAL}/modeling_act_smooth.py"

# Trap to restore on exit (success or failure)
restore_source() {
    echo "Restoring original source..."
    cp "${DIR_EXPERIMENT}/configuration_act_smooth.py.bak" "${DIR_SRC_ORIGINAL}/configuration_act_smooth.py"
    cp "${DIR_EXPERIMENT}/modeling_act_smooth.py.bak" "${DIR_SRC_ORIGINAL}/modeling_act_smooth.py"
    rm -f "${DIR_EXPERIMENT}/configuration_act_smooth.py.bak" "${DIR_EXPERIMENT}/modeling_act_smooth.py.bak"
}
trap restore_source EXIT

# --- Train ---
echo "Training: ${STEPS_FINETUNE} steps from step ${STEPS_BASE}..."
lerobot-train \
    --resume=true \
    --config_path="${PATH_CHECKPOINT}/pretrained_model/train_config.json" \
    --output_dir="${DIR_EXPERIMENT}/checkpoints" \
    --steps="${STEPS_TOTAL}" \
    --save_freq=$((STEPS_TOTAL + 1)) \
    --batch_size=8 \
    --num_workers=4 \
    --policy.push_to_hub=false \
    --wandb.enable=false

# --- Eval ---
echo "Running offline eval..."
PATH_POLICY="${DIR_EXPERIMENT}/checkpoints/checkpoints/last/pretrained_model"
python "${DIR_ROOT}/scripts/eval/eval_offline_replay.py" \
    --path_policy="${PATH_POLICY}" \
    --id_repo_dataset="${ID_REPO_DATASET}" \
    --indices_episode=0,1,2 \
    --path_output="${DIR_EXPERIMENT}/data/boundary_loss"

# Copy results
echo "=== Results ===" > "${DIR_EXPERIMENT}/report.md"
echo "" >> "${DIR_EXPERIMENT}/report.md"
echo '```' >> "${DIR_EXPERIMENT}/report.md"
cat "${DIR_EXPERIMENT}/data/boundary_loss_results.txt" >> "${DIR_EXPERIMENT}/report.md"
echo '```' >> "${DIR_EXPERIMENT}/report.md"

cp "${DIR_EXPERIMENT}"/data/*trajectories.png "${DIR_EXPERIMENT}/figures/" 2>/dev/null || true
cp "${DIR_EXPERIMENT}"/data/*accelerations.png "${DIR_EXPERIMENT}/figures/" 2>/dev/null || true

echo "=== 02-boundary-loss: Done ==="
