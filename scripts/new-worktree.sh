#!/usr/bin/env bash
# Create an isolated git worktree for feature work, with its own venv.
#
# Usage:
#   scripts/new-worktree.sh <branch-name> [base-ref]
#
#   base-ref defaults to origin/main.
#
# What it does:
#   1. Creates .worktrees/<branch-name> as a new linked worktree, branching
#      off base-ref (creates the branch if it does not exist yet).
#   2. Copies earth-rovers-sdk/.env into the new worktree if the source repo
#      has one, so the SDK server has credentials without touching the
#      tracked file in this checkout.
#   3. Creates a Python venv at .venv inside the new worktree and installs
#      requirements.txt (SKIP_PIP=1 to skip).
#
# It does not download model checkpoints or pre-cache DINOv2/SegFormer —
# run setup.sh inside the new worktree for that when you actually need it.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <branch-name> [base-ref]" >&2
  exit 1
fi

BRANCH=$1
BASE=${2:-origin/main}

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

WT_DIR=".worktrees/$BRANCH"

if [[ -e $WT_DIR ]]; then
  echo "error: $WT_DIR already exists" >&2
  exit 1
fi

mkdir -p .worktrees

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "==> branch $BRANCH already exists, reusing it"
  git worktree add "$WT_DIR" "$BRANCH"
else
  echo "==> creating branch $BRANCH from $BASE"
  git worktree add "$WT_DIR" -b "$BRANCH" "$BASE"
fi

if [[ -f earth-rovers-sdk/.env && ! -f "$WT_DIR/earth-rovers-sdk/.env" ]]; then
  echo "==> copying earth-rovers-sdk/.env into new worktree"
  mkdir -p "$WT_DIR/earth-rovers-sdk"
  cp earth-rovers-sdk/.env "$WT_DIR/earth-rovers-sdk/.env"
fi

if [[ ${SKIP_PIP:-0} != 1 ]]; then
  PY=${PYTHON:-python3}
  echo "==> creating venv at $WT_DIR/.venv"
  "$PY" -m venv "$WT_DIR/.venv"
  "$WT_DIR/.venv/bin/pip" install -q -U pip
  "$WT_DIR/.venv/bin/pip" install -q -r "$WT_DIR/requirements.txt"
else
  echo "==> SKIP_PIP=1, not creating a venv"
fi

echo
echo "Worktree ready: $WT_DIR"
echo "Next:"
echo "  cd $WT_DIR"
[[ ${SKIP_PIP:-0} != 1 ]] && echo "  source .venv/bin/activate"
echo "  ./setup.sh            # only if you need checkpoints / model caches"
