#!/bin/bash
#
# Run policy evaluation with the SO101 follower arm.
#
# Usage:
#   ./eval.sh --variant_eval=sync --fps_policy=10 --variant_policy=smooth
#   ./eval.sh --variant_eval=sync_discard --fps_policy=30 --variant_policy=smooth
#   ./eval.sh --variant_eval=async_smooth --fps_policy=30 --variant_policy=smooth
#   ./eval.sh --variant_eval=async_discard --fps_policy=30 --variant_policy=smooth
#
# Policy paths are derived from fps and variant:
#   - smooth + 10fps -> outputs/lerobot_policy_act_smooth_10fps_smooth_p1f2/...
#   - smooth + 30fps -> outputs/lerobot_policy_act_smooth_30fps_smooth_p4f2/...
#
# NOTE: The configuration below is specific to Giacomo's setup.
# You will need to substitute your own values:
#   - robot.port: USB port for your arm (find with `lerobot-find-port`)
#   - robot.id: ID for your arm
#   - robot.cameras: Camera indices depend on your system (find with `lerobot-find-cameras`)

VARIANT_EVAL=""
VARIANT_POLICY=""
FPS_POLICY=""
CHECKPOINT="030003"
FPS_INTERPOLATION=30
FPS_OBSERVATION=60
EPISODE_TIME_S=30
DISPLAY_DATA=true

for arg in "$@"; do
  case $arg in
    --variant_eval=*)
      VARIANT_EVAL="${arg#*=}"
      ;;
    --variant_policy=*)
      VARIANT_POLICY="${arg#*=}"
      ;;
    --fps_policy=*)
      FPS_POLICY="${arg#*=}"
      ;;
    --checkpoint=*)
      CHECKPOINT="${arg#*=}"
      ;;
  esac
done

if [[ -z "$VARIANT_EVAL" || -z "$FPS_POLICY" || -z "$VARIANT_POLICY" ]]; then
  echo "Usage: $0 --variant_eval=<sync|sync_discard|async_smooth|async_discard> --fps_policy=<fps> --variant_policy=<smooth|vanilla> [--checkpoint=030003]"
  exit 1
fi

if [[ "$VARIANT_POLICY" != "smooth" && "$VARIANT_POLICY" != "vanilla" ]]; then
  echo "Error: --variant_policy must be 'smooth' or 'vanilla'"
  exit 1
fi

CAMERAS="{
  wrist: { type: opencv, index_or_path: 1, width: 480, height: 640, fps: 30, rotation: -90 },
  top:   { type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30 }
}"

# Build variant suffix based on policy variant and fps
# Trained models use: smooth_p{past}f{future} naming (see train-act-smooth.sh)
if [[ "$VARIANT_POLICY" == "smooth" ]]; then
  case "$FPS_POLICY" in
    10)
      VARIANT_SUFFIX="smooth_p1f2"
      THRESHOLD_REMAINING_ACTIONS=2
      ;;
    30)
      VARIANT_SUFFIX="smooth_p4f2"
      THRESHOLD_REMAINING_ACTIONS=2
      ;;
    *)
      echo "Error: No trained smooth model for fps=$FPS_POLICY. Available: 10, 30"
      exit 1
      ;;
  esac
else
  VARIANT_SUFFIX="vanilla"
  THRESHOLD_REMAINING_ACTIONS=1
fi

PATH_POLICY=outputs/lerobot_policy_act_smooth_${FPS_POLICY}fps_${VARIANT_SUFFIX}/checkpoints/${CHECKPOINT}/pretrained_model

# Compute recording path with auto-incrementing run number
DIR_RECORDING=outputs/recordings
mkdir -p "$DIR_RECORDING"
PREFIX_RECORDING="${VARIANT_EVAL}_${VARIANT_POLICY}_${FPS_POLICY}"
# Find next run number (3-digit, zero-padded)
IDX_RUN=0
while [[ -f "${DIR_RECORDING}/${PREFIX_RECORDING}_$(printf '%03d' $IDX_RUN).rrd" ]]; do
  IDX_RUN=$((IDX_RUN + 1))
done
PATH_RECORDING="${DIR_RECORDING}/${PREFIX_RECORDING}_$(printf '%03d' $IDX_RUN).rrd"
echo "Recording to: $PATH_RECORDING"

case "$VARIANT_EVAL" in
  sync)
    python scripts/eval/eval_sync.py \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem5A460829821 \
        --robot.id=arm_follower_0 \
        --robot.cameras="$CAMERAS" \
        --policy.path="$PATH_POLICY" \
        --fps_policy="$FPS_POLICY" \
        --fps_observation="$FPS_OBSERVATION" \
        --episode_time_s="$EPISODE_TIME_S" \
        --display_data="$DISPLAY_DATA" \
        --path_recording="$PATH_RECORDING"
    ;;

  sync_discard)
    python scripts/eval/eval_sync_discard.py \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem5A460829821 \
        --robot.id=arm_follower_0 \
        --robot.cameras="$CAMERAS" \
        --policy.path="$PATH_POLICY" \
        --fps_policy="$FPS_POLICY" \
        --fps_observation="$FPS_OBSERVATION" \
        --episode_time_s="$EPISODE_TIME_S" \
        --display_data="$DISPLAY_DATA" \
        --path_recording="$PATH_RECORDING"
    ;;

  async_discard)
    python scripts/eval/eval_async_discard.py \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem5A460829821 \
        --robot.id=arm_follower_0 \
        --robot.cameras="$CAMERAS" \
        --policy.path="$PATH_POLICY" \
        --fps_policy="$FPS_POLICY" \
        --fps_observation="$FPS_OBSERVATION" \
        --threshold_remaining_actions="$THRESHOLD_REMAINING_ACTIONS" \
        --episode_time_s="$EPISODE_TIME_S" \
        --display_data="$DISPLAY_DATA" \
        --path_recording="$PATH_RECORDING"
    ;;

  async_smooth)
    python scripts/eval/eval_async_smooth.py \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem5A460829821 \
        --robot.id=arm_follower_0 \
        --robot.cameras="$CAMERAS" \
        --policy.path="$PATH_POLICY" \
        --fps_policy="$FPS_POLICY" \
        --fps_interpolation=30 \
        --fps_observation=60 \
        --episode_time_s="$EPISODE_TIME_S" \
        --display_data="$DISPLAY_DATA" \
        --path_recording="$PATH_RECORDING"
    ;;

  *)
    echo "Error: --variant_eval must be one of: sync, sync_discard, async_smooth, async_discard"
    exit 1
    ;;
esac
