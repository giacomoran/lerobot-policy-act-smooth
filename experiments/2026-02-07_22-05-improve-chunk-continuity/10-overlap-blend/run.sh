#!/bin/bash
# 10-overlap-blend: Inference-time overlap blending between consecutive chunks.
# No training -- uses baseline 50K checkpoint. Tests multiple advance values.
set -e

DIR_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$(dirname "$0")/../env.sh"
DIR_EXPERIMENT="$(cd "$(dirname "$0")" && pwd)"
PATH_POLICY="${DIR_ROOT}/outputs/lerobot_policy_act_smooth_30fps_smooth_p4f2/checkpoints/050003/pretrained_model"
ID_REPO_DATASET="giacomoran/lerobot_policy_act_smooth_30fps"

echo "=== 10-overlap-blend: Inference-time overlap blending ==="

# --- Eval with different advance values ---
# advance=30 (no overlap, same as baseline but verified)
echo "Running eval: advance=30 (no overlap, baseline check)..."
python "${DIR_ROOT}/scripts/eval/eval_offline_replay.py" \
    --path_policy="${PATH_POLICY}" \
    --id_repo_dataset="${ID_REPO_DATASET}" \
    --indices_episode=0,1,2 \
    --advance=30 \
    --blend_overlap=false \
    --path_output="${DIR_EXPERIMENT}/data/blend_a30"

# advance=20, blend (overlap of 10)
echo "Running eval: advance=20, blend (10 overlap)..."
python "${DIR_ROOT}/scripts/eval/eval_offline_replay.py" \
    --path_policy="${PATH_POLICY}" \
    --id_repo_dataset="${ID_REPO_DATASET}" \
    --indices_episode=0,1,2 \
    --advance=20 \
    --blend_overlap=true \
    --path_output="${DIR_EXPERIMENT}/data/blend_a20"

# advance=15, blend (overlap of 15)
echo "Running eval: advance=15, blend (15 overlap)..."
python "${DIR_ROOT}/scripts/eval/eval_offline_replay.py" \
    --path_policy="${PATH_POLICY}" \
    --id_repo_dataset="${ID_REPO_DATASET}" \
    --indices_episode=0,1,2 \
    --advance=15 \
    --blend_overlap=true \
    --path_output="${DIR_EXPERIMENT}/data/blend_a15"

# advance=10, blend (overlap of 20)
echo "Running eval: advance=10, blend (20 overlap)..."
python "${DIR_ROOT}/scripts/eval/eval_offline_replay.py" \
    --path_policy="${PATH_POLICY}" \
    --id_repo_dataset="${ID_REPO_DATASET}" \
    --indices_episode=0,1,2 \
    --advance=10 \
    --blend_overlap=true \
    --path_output="${DIR_EXPERIMENT}/data/blend_a10"

# Collect results
echo "=== Results ===" > "${DIR_EXPERIMENT}/report.md"
echo "" >> "${DIR_EXPERIMENT}/report.md"
for variant in a30 a20 a15 a10; do
    echo "--- ${variant} ---" >> "${DIR_EXPERIMENT}/report.md"
    echo '```' >> "${DIR_EXPERIMENT}/report.md"
    cat "${DIR_EXPERIMENT}/data/blend_${variant}_results.txt" >> "${DIR_EXPERIMENT}/report.md"
    echo '```' >> "${DIR_EXPERIMENT}/report.md"
    echo "" >> "${DIR_EXPERIMENT}/report.md"
done

cp "${DIR_EXPERIMENT}"/data/*trajectories.png "${DIR_EXPERIMENT}/figures/" 2>/dev/null || true
cp "${DIR_EXPERIMENT}"/data/*accelerations.png "${DIR_EXPERIMENT}/figures/" 2>/dev/null || true

echo "=== 10-overlap-blend: Done ==="
