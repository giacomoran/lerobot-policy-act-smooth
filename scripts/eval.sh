#!/bin/bash
#
# Run policy evaluation with the SO101 follower arm.
#
# Usage:
#   ./eval.sh --policy=lerobot_policy_act_smooth_10fps_smooth_p1f2 --eval=async_smooth
#   ./eval.sh --policy=lerobot_policy_act_smooth_30fps_smooth_p4f2 --eval=async_smooth
#   ./eval.sh --policy=lerobot_policy_act_smooth_10fps_vanilla --eval=sync
#   ./eval.sh --policy=lerobot_policy_act_smooth_30fps_vanilla --eval=sync --checkpoint=030003
#   ./eval.sh --policy=lerobot_policy_act_smooth_10fps_smooth_p1f4 --eval=async_smooth --delay_ms_injected=100
#   ./eval.sh --policy=lerobot_policy_act_smooth_30fps_smooth_p4f8 --eval=async_smooth --threshold_remaining_actions=2
#
# The policy fps is extracted from the policy name (e.g. "10fps" -> 10).
#
# NOTE: The configuration below is specific to Giacomo's setup.
# You will need to substitute your own values:
#   - robot.port: USB port for your arm (find with `lerobot-find-port`)
#   - robot.id: ID for your arm
#   - robot.cameras: Camera indices depend on your system (find with `lerobot-find-cameras`)

set -e

# --- Constants ---
FPS_INTERPOLATION=30
FPS_OBSERVATION=60
DISPLAY_DATA=true
EPISODE_TIME_S=30
CHECKPOINT="030003"
DELAY_MS_INJECTED=0
THRESHOLD_REMAINING_ACTIONS=""

CAMERAS="{
  wrist: { type: opencv, index_or_path: 1, width: 480, height: 640, fps: 30, rotation: -90 },
  top:   { type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30 }
}"

# --- Parse CLI args ---
NAME_POLICY=""
VARIANT_EVAL=""

for arg in "$@"; do
  case $arg in
    --policy=*)
      NAME_POLICY="${arg#*=}"
      ;;
    --eval=*)
      VARIANT_EVAL="${arg#*=}"
      ;;
    --checkpoint=*)
      CHECKPOINT="${arg#*=}"
      ;;
    --episode_time_s=*)
      EPISODE_TIME_S="${arg#*=}"
      ;;
    --delay_ms_injected=*)
      DELAY_MS_INJECTED="${arg#*=}"
      ;;
    --threshold_remaining_actions=*)
      THRESHOLD_REMAINING_ACTIONS="${arg#*=}"
      ;;
  esac
done

if [[ -z "$NAME_POLICY" || -z "$VARIANT_EVAL" ]]; then
  echo "Usage: $0 --policy=<policy_name> --eval=<sync|sync_discard|async_smooth|async_discard> [--checkpoint=030003] [--episode_time_s=30] [--delay_ms_injected=0] [--threshold_remaining_actions=N]"
  exit 1
fi

# --- Derive fps from policy name ---
FPS_POLICY=$(echo "$NAME_POLICY" | grep -oE '[0-9]+fps' | grep -oE '[0-9]+')
if [[ -z "$FPS_POLICY" ]]; then
  echo "Error: Could not extract fps from policy name '$NAME_POLICY' (expected e.g. '10fps' or '30fps')"
  exit 1
fi

# --- Build policy path ---
PATH_POLICY="outputs/${NAME_POLICY}/checkpoints/${CHECKPOINT}/pretrained_model"
if [[ ! -d "$PATH_POLICY" ]]; then
  echo "Error: Policy path not found: $PATH_POLICY"
  exit 1
fi

# --- Recording path (auto-incrementing) ---
DIR_RECORDING=outputs/recordings
mkdir -p "$DIR_RECORDING"
PREFIX_RECORDING="${VARIANT_EVAL}_${NAME_POLICY}"
IDX_RUN=0
while [[ -f "${DIR_RECORDING}/${PREFIX_RECORDING}_$(printf '%03d' $IDX_RUN).rrd" ]]; do
  IDX_RUN=$((IDX_RUN + 1))
done
PATH_RECORDING="${DIR_RECORDING}/${PREFIX_RECORDING}_$(printf '%03d' $IDX_RUN).rrd"

echo "Policy:    $NAME_POLICY (${FPS_POLICY}fps, checkpoint ${CHECKPOINT})"
echo "Eval:      $VARIANT_EVAL"
echo "Delay:     ${DELAY_MS_INJECTED}ms injected"
echo "Recording: $PATH_RECORDING"

# --- Dispatch ---
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
        --delay_ms_injected="$DELAY_MS_INJECTED" \
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
        --delay_ms_injected="$DELAY_MS_INJECTED" \
        --episode_time_s="$EPISODE_TIME_S" \
        --display_data="$DISPLAY_DATA" \
        --path_recording="$PATH_RECORDING"
    ;;

  async_discard)
    ARGS_ASYNC_DISCARD=(
        --robot.type=so101_follower
        --robot.port=/dev/tty.usbmodem5A460829821
        --robot.id=arm_follower_0
        --robot.cameras="$CAMERAS"
        --policy.path="$PATH_POLICY"
        --fps_policy="$FPS_POLICY"
        --fps_interpolation="$FPS_INTERPOLATION"
        --fps_observation="$FPS_OBSERVATION"
        --delay_ms_injected="$DELAY_MS_INJECTED"
        --episode_time_s="$EPISODE_TIME_S"
        --display_data="$DISPLAY_DATA"
        --path_recording="$PATH_RECORDING"
    )
    if [[ -n "$THRESHOLD_REMAINING_ACTIONS" ]]; then
        ARGS_ASYNC_DISCARD+=(--threshold_remaining_actions="$THRESHOLD_REMAINING_ACTIONS")
    fi
    python scripts/eval/eval_async_discard.py "${ARGS_ASYNC_DISCARD[@]}"
    ;;

  async_smooth)
    ARGS_ASYNC_SMOOTH=(
        --robot.type=so101_follower
        --robot.port=/dev/tty.usbmodem5A460829821
        --robot.id=arm_follower_0
        --robot.cameras="$CAMERAS"
        --policy.path="$PATH_POLICY"
        --fps_policy="$FPS_POLICY"
        --fps_interpolation="$FPS_INTERPOLATION"
        --fps_observation="$FPS_OBSERVATION"
        --delay_ms_injected="$DELAY_MS_INJECTED"
        --episode_time_s="$EPISODE_TIME_S"
        --display_data="$DISPLAY_DATA"
        --path_recording="$PATH_RECORDING"
    )
    if [[ -n "$THRESHOLD_REMAINING_ACTIONS" ]]; then
        ARGS_ASYNC_SMOOTH+=(--threshold_remaining_actions="$THRESHOLD_REMAINING_ACTIONS")
    fi
    python scripts/eval/eval_async_smooth.py "${ARGS_ASYNC_SMOOTH[@]}"
    ;;

  *)
    echo "Error: --eval must be one of: sync, sync_discard, async_smooth, async_discard"
    exit 1
    ;;
esac
