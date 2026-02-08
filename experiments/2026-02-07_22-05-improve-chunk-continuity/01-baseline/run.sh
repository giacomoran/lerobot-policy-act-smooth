#!/bin/bash
# 01-baseline: Run offline eval on current checkpoint. No training.
set -e

DIR_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$(dirname "$0")/../env.sh"
DIR_EXPERIMENT="$(cd "$(dirname "$0")" && pwd)"
PATH_POLICY="${DIR_ROOT}/outputs/lerobot_policy_act_smooth_30fps_smooth_p4f2/checkpoints/last/pretrained_model"
ID_REPO_DATASET="giacomoran/lerobot_policy_act_smooth_30fps"

echo "=== 01-baseline: Offline eval on current checkpoint ==="
echo "Policy: ${PATH_POLICY}"
echo "Dataset: ${ID_REPO_DATASET}"

python "${DIR_ROOT}/scripts/eval/eval_offline_replay.py" \
    --path_policy="${PATH_POLICY}" \
    --id_repo_dataset="${ID_REPO_DATASET}" \
    --indices_episode=0,1,2 \
    --path_output="${DIR_EXPERIMENT}/data/baseline"

# Copy results to report
echo "=== Results ===" > "${DIR_EXPERIMENT}/report.md"
echo "" >> "${DIR_EXPERIMENT}/report.md"
echo '```' >> "${DIR_EXPERIMENT}/report.md"
cat "${DIR_EXPERIMENT}/data/baseline_results.txt" >> "${DIR_EXPERIMENT}/report.md"
echo '```' >> "${DIR_EXPERIMENT}/report.md"

# Copy figures
cp "${DIR_EXPERIMENT}"/data/*trajectories.png "${DIR_EXPERIMENT}/figures/" 2>/dev/null || true
cp "${DIR_EXPERIMENT}"/data/*accelerations.png "${DIR_EXPERIMENT}/figures/" 2>/dev/null || true

echo "=== 01-baseline: Done ==="
echo "Results: ${DIR_EXPERIMENT}/data/baseline_results.txt"
echo "Report:  ${DIR_EXPERIMENT}/report.md"
