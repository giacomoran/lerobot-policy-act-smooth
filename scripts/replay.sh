#!/bin/bash
#
# Replay recorded episodes on the SO101 follower arm.
#
# Usage:
#   ./replay.sh --fps 30 --episode 0
#   ./replay.sh --fps 10 --episode 2
#
# Arguments:
#   --fps N      Dataset FPS (10 or 30)
#   --episode N  Episode number to replay
#
# NOTE: The configuration below is specific to Giacomo's setup.
# You will need to substitute your own values:
#   - robot.port: USB port for your arm (find with `lerobot-find-port`)
#   - robot.id: ID for your arm

# Parse arguments
FPS=""
EPISODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --fps)
            FPS="$2"
            shift 2
            ;;
        --episode)
            EPISODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./replay.sh --fps <10|30> --episode <N>"
            exit 1
            ;;
    esac
done

# Validate arguments
if [[ -z "$FPS" ]]; then
    echo "Error: --fps is required"
    echo "Usage: ./replay.sh --fps <10|30> --episode <N>"
    exit 1
fi

if [[ "$FPS" != "10" && "$FPS" != "30" ]]; then
    echo "Error: --fps must be 10 or 30"
    exit 1
fi

if [[ -z "$EPISODE" ]]; then
    echo "Error: --episode is required"
    echo "Usage: ./replay.sh --fps <10|30> --episode <N>"
    exit 1
fi

# Select dataset based on fps
REPO_ID="giacomoran/lerobot_policy_act_smooth_${FPS}fps"

lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A460829821 \
    --robot.id=arm_follower_0 \
    --dataset.repo_id="$REPO_ID" \
    --dataset.episode="$EPISODE"
