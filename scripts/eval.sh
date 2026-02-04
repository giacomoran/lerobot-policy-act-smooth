#!/bin/bash
#
# Run policy evaluation with the SO101 follower arm.
#
# Usage:
#   ./eval.sh --variant_eval=sync --fps_policy=10 --variant_policy=vanilla
#   ./eval.sh --variant_eval=sync_discard --fps_policy=30 --variant_policy=smooth
#   ./eval.sh --variant_eval=async_rtc --fps_policy=30 --variant_policy=smooth
#   ./eval.sh --variant_eval=async_discard --fps_policy=30 --variant_policy=smooth
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
MAX_DELAY="2"
FPS_OBSERVATION=30
EPISODE_TIME_S=60
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
    --max_delay=*)
      MAX_DELAY="${arg#*=}"
      ;;
  esac
done

if [[ -z "$VARIANT_EVAL" || -z "$FPS_POLICY" || -z "$VARIANT_POLICY" ]]; then
  echo "Usage: $0 --variant_eval=<sync|sync_discard|async_rtc|async_discard> --fps_policy=<fps> --variant_policy=<smooth|vanilla> [--checkpoint=030003] [--max_delay=2]"
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

# Build variant suffix: vanilla stays as-is, smooth becomes smooth_d{delay}
if [[ "$VARIANT_POLICY" == "smooth" ]]; then
  VARIANT_SUFFIX="smooth_d${MAX_DELAY}"
else
  VARIANT_SUFFIX="vanilla"
fi

PATH_POLICY=outputs/lerobot_policy_act_smooth_${FPS_POLICY}fps_${VARIANT_SUFFIX}/checkpoints/${CHECKPOINT}/pretrained_model
THRESHOLD_REMAINING_ACTIONS=$((MAX_DELAY))

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
        --display_data="$DISPLAY_DATA"
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
        --display_data="$DISPLAY_DATA"
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
        --display_data="$DISPLAY_DATA"
    ;;

  async_rtc)
    python scripts/eval/eval_async_rtc.py \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem5A460829821 \
        --robot.id=arm_follower_0 \
        --robot.cameras="$CAMERAS" \
        --policy.path="$PATH_POLICY" \
        --fps_policy="$FPS_POLICY" \
        --fps_observation="$FPS_OBSERVATION" \
        --threshold_remaining_actions="$THRESHOLD_REMAINING_ACTIONS" \
        --episode_time_s="$EPISODE_TIME_S" \
        --display_data="$DISPLAY_DATA"
    ;;

  *)
    echo "Error: --variant_eval must be one of: sync, sync_discard, async_rtc, async_discard"
    exit 1
    ;;
esac
