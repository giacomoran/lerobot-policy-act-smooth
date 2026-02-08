#!/bin/bash
# 03-no-proprioception: Remove observation.state from input_features.
# Train from scratch (architecture change breaks weight transfer).
set -e

DIR_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$(dirname "$0")/../env.sh"
DIR_EXPERIMENT="$(cd "$(dirname "$0")" && pwd)"
DIR_SRC_ORIGINAL="${DIR_ROOT}/src/lerobot_policy_act_smooth"
DIR_SRC_EXPERIMENT="${DIR_EXPERIMENT}/src"
ID_REPO_DATASET="giacomoran/lerobot_policy_act_smooth_30fps"

STEPS=5000

echo "=== 03-no-proprioception: Train from scratch without observation.state ==="

# --- Swap source (if experiment has modified source) ---
HAS_MODIFIED_SRC=false
if [ -f "${DIR_SRC_EXPERIMENT}/configuration_act_smooth.py" ]; then
    HAS_MODIFIED_SRC=true
    echo "Backing up original source..."
    cp "${DIR_SRC_ORIGINAL}/configuration_act_smooth.py" "${DIR_EXPERIMENT}/configuration_act_smooth.py.bak"
    cp "${DIR_SRC_ORIGINAL}/modeling_act_smooth.py" "${DIR_EXPERIMENT}/modeling_act_smooth.py.bak"

    echo "Installing experiment source..."
    cp "${DIR_SRC_EXPERIMENT}/configuration_act_smooth.py" "${DIR_SRC_ORIGINAL}/configuration_act_smooth.py"
    cp "${DIR_SRC_EXPERIMENT}/modeling_act_smooth.py" "${DIR_SRC_ORIGINAL}/modeling_act_smooth.py"

    restore_source() {
        echo "Restoring original source..."
        cp "${DIR_EXPERIMENT}/configuration_act_smooth.py.bak" "${DIR_SRC_ORIGINAL}/configuration_act_smooth.py"
        cp "${DIR_EXPERIMENT}/modeling_act_smooth.py.bak" "${DIR_SRC_ORIGINAL}/modeling_act_smooth.py"
        rm -f "${DIR_EXPERIMENT}/configuration_act_smooth.py.bak" "${DIR_EXPERIMENT}/modeling_act_smooth.py.bak"
    }
    trap restore_source EXIT
fi

# --- Train from scratch (no resume -- architecture change) ---
echo "Training: ${STEPS} steps from scratch..."
lerobot-train \
    --policy.type=act_smooth \
    --dataset.repo_id="${ID_REPO_DATASET}" \
    --output_dir="${DIR_EXPERIMENT}/checkpoints" \
    --policy.input_features='{"observation.images.wrist": {"shape": [3, 640, 480], "type": "VISUAL"}, "observation.images.top": {"shape": [3, 480, 640], "type": "VISUAL"}}' \
    --steps="${STEPS}" \
    --save_freq=$((STEPS + 1)) \
    --batch_size=8 \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --policy.length_prefix_past=4 \
    --policy.length_prefix_future=2 \
    --policy.use_vae=false \
    --policy.optimizer_lr=3e-5 \
    --policy.optimizer_lr_backbone=3e-5 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.repo_id=local/no_proprio \
    --wandb.enable=false \
    --num_workers=4

# --- Eval ---
echo "Running offline eval..."
PATH_POLICY="${DIR_EXPERIMENT}/checkpoints/checkpoints/last/pretrained_model"
python "${DIR_ROOT}/scripts/eval/eval_offline_replay.py" \
    --path_policy="${PATH_POLICY}" \
    --id_repo_dataset="${ID_REPO_DATASET}" \
    --indices_episode=0,1,2 \
    --path_output="${DIR_EXPERIMENT}/data/no_proprio"

# Copy results
echo "=== Results ===" > "${DIR_EXPERIMENT}/report.md"
echo "" >> "${DIR_EXPERIMENT}/report.md"
echo '```' >> "${DIR_EXPERIMENT}/report.md"
cat "${DIR_EXPERIMENT}/data/no_proprio_results.txt" >> "${DIR_EXPERIMENT}/report.md"
echo '```' >> "${DIR_EXPERIMENT}/report.md"

cp "${DIR_EXPERIMENT}"/data/*trajectories.png "${DIR_EXPERIMENT}/figures/" 2>/dev/null || true
cp "${DIR_EXPERIMENT}"/data/*accelerations.png "${DIR_EXPERIMENT}/figures/" 2>/dev/null || true

echo "=== 03-no-proprioception: Done ==="
