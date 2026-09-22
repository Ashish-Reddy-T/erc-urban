#!/usr/bin/env bash
# Local Mini+ testing on `emu-ion-volt`. Free-drive to a GPS goal, no mission.
#
#   ./erc_urban_test/local_run.sh sign                 # verify angular sign FIRST
#   ./erc_urban_test/local_run.sh sensors              # what is alive on this unit
#   ./erc_urban_test/local_run.sh drive <lat> <lon>    # free-drive to a goal
#
# Free-drive (USE_MISSION=0) on purpose: no mission clock, no checkpoint
# scoring, nothing burned if you abort. Ctrl-C is safe — the Mini+ has a command
# watchdog, so it coasts to a stop as soon as commands stop arriving.
#
# BOT_SLUG lives in earth-rovers-sdk/.env, NOT here, because the SDK is what
# talks to FrodoBots. Switching bots means editing that file and restarting the
# SDK; this script only checks that you did.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/miniconda3/envs/rover_env/bin/python}"
SDK_URL="${SDK_URL:-http://localhost:8000}"
WANT_BOT="${WANT_BOT:-emu-ion-volt}"

cd "$REPO"

if ! curl -sf --max-time 5 "$SDK_URL/data" >/dev/null 2>&1; then
  echo "SDK not answering at $SDK_URL/data"
  echo "  start it:  cd $REPO/earth-rovers-sdk && hypercorn main:app --reload"
  echo "  if it IS up, it may need:  curl -X POST $SDK_URL/start-mission"
  exit 1
fi

HAVE_BOT="$(grep -E '^BOT_SLUG=' earth-rovers-sdk/.env 2>/dev/null | cut -d= -f2- || true)"
if [ "$HAVE_BOT" != "$WANT_BOT" ]; then
  echo "WARNING: earth-rovers-sdk/.env has BOT_SLUG=${HAVE_BOT:-<unset>}, expected $WANT_BOT"
  echo "         Edit that file and restart the SDK, or set WANT_BOT= to silence this."
  if [ -t 0 ]; then
    read -r -p "         Continue anyway? [y/N] " ok
    [ "$ok" = "y" ] || exit 1
  else
    echo "         (non-interactive: set ALLOW_BOT_MISMATCH=1 to proceed)"
    [ "${ALLOW_BOT_MISMATCH:-0}" = "1" ] || exit 1
  fi
fi

MODE="${1:-}"
case "$MODE" in
  sign)    exec "$PY" -m erc_urban_test.bench sign ;;
  sensors) exec "$PY" -m erc_urban_test.bench sensors ;;
  map)     exec "$PY" -m erc_urban_test.bench map ;;
  drive)   ;;
  *) echo "usage: $0 {sign|sensors|map|drive <lat> <lon>}"; exit 1 ;;
esac

GOAL_LAT="${2:-}"; GOAL_LON="${3:-}"
if [ -z "$GOAL_LAT" ] || [ -z "$GOAL_LON" ]; then
  echo "drive needs a goal:  $0 drive <lat> <lon>"
  echo "current position:"
  curl -s --max-time 5 "$SDK_URL/data" | "$PY" -c \
    'import json,sys; d=json.load(sys.stdin); print("   ", d.get("latitude"), d.get("longitude"))'
  exit 1
fi

# The sign check gates autonomous driving. It is cheap and it is the difference
# between correcting toward the goal and steering away from it.
if [ ! -f "$REPO/.bench_sign_ok" ]; then
  echo "No .bench_sign_ok in the repo root — the angular sign has not been verified"
  echo "on this unit. Run:  $0 sign"
  echo "Then, if it reported POSITIVE = LEFT:  touch $REPO/.bench_sign_ok"
  if [ -t 0 ]; then
    read -r -p "Skip the check and drive anyway? [y/N] " ok
    [ "$ok" = "y" ] || exit 1
  else
    echo "(non-interactive: refusing to drive unverified; run the sign check)"
    exit 1
  fi
fi

echo "free-drive -> ($GOAL_LAT, $GOAL_LON)   Ctrl-C to stop"
exec env \
  STAGE=1 \
  USE_MISSION=0 \
  GOAL_LAT="$GOAL_LAT" \
  GOAL_LON="$GOAL_LON" \
  USE_VLM="${USE_VLM:-1}" \
  GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-erc-urban-nav-2610}" \
  PYTORCH_ENABLE_MPS_FALLBACK=1 \
  V_NOM="${V_NOM:-0.24}" \
  "$PY" -m erc_urban_test.run
