#!/bin/bash
# 12-overlap-blend-continued: Overlap blending on the 08-continued-training model (60K steps).
# No training -- tests if more training + blending is better than blending alone.
set -e

DIR_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$(dirname "$0")/../env.sh"
DIR_EXPERIMENT="$(cd "$(dirname "$0")" && pwd)"
PATH_POLICY="${DIR_ROOT}/experiments/2026-02-07_22-05-improve-chunk-continuity/08-continued-training/checkpoints/checkpoints/last/pretrained_model"
ID_REPO_DATASET="giacomoran/lerobot_policy_act_smooth_30fps"

echo "=== 12-overlap-blend-continued: Overlap blending on 60K model ==="

# advance=20, blend (overlap of 10) -- best variant from 10-overlap-blend
echo "Running eval: advance=20, blend (10 overlap) on 60K model..."
python "${DIR_ROOT}/scripts/eval/eval_offline_replay.py" \
    --path_policy="${PATH_POLICY}" \
    --id_repo_dataset="${ID_REPO_DATASET}" \
    --indices_episode=0,1,2 \
    --advance=20 \
    --blend_overlap=true \
    --path_output="${DIR_EXPERIMENT}/data/blend_continued_a20"

# advance=15, blend (overlap of 15)
echo "Running eval: advance=15, blend (15 overlap) on 60K model..."
python "${DIR_ROOT}/scripts/eval/eval_offline_replay.py" \
    --path_policy="${PATH_POLICY}" \
    --id_repo_dataset="${ID_REPO_DATASET}" \
    --indices_episode=0,1,2 \
    --advance=15 \
    --blend_overlap=true \
    --path_output="${DIR_EXPERIMENT}/data/blend_continued_a15"

# Collect results
echo "=== Results ===" > "${DIR_EXPERIMENT}/report.md"
echo "" >> "${DIR_EXPERIMENT}/report.md"
for variant in a20 a15; do
    echo "--- ${variant} ---" >> "${DIR_EXPERIMENT}/report.md"
    echo '```' >> "${DIR_EXPERIMENT}/report.md"
    cat "${DIR_EXPERIMENT}/data/blend_continued_${variant}_results.txt" >> "${DIR_EXPERIMENT}/report.md"
    echo '```' >> "${DIR_EXPERIMENT}/report.md"
    echo "" >> "${DIR_EXPERIMENT}/report.md"
done

cp "${DIR_EXPERIMENT}"/data/*trajectories.png "${DIR_EXPERIMENT}/figures/" 2>/dev/null || true
cp "${DIR_EXPERIMENT}"/data/*accelerations.png "${DIR_EXPERIMENT}/figures/" 2>/dev/null || true

echo "=== 12-overlap-blend-continued: Done ==="
