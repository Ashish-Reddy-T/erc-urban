#!/usr/bin/env bash
# The single entry point for "is this change done": runs both offline
# pytest suites (test/ and erc_urban_test/tests, the same ones documented
# under Tests in the README), then a mirror-drift check, then
# git diff --check. Exits nonzero if anything fails.
#
# Mirror-drift check: erc_urban and erc_urban_test are two forks of the
# same stack and are expected to diverge (erc_urban_test carries sandbox
# tools like local_run.sh and its own tests/, and files like run.py differ
# by design). But when a file exists at the same relative path under both
# trees (e.g. modules/baseline.py, sdk_client.py), a real change to one
# side that has the same code on the other side is supposed to be mirrored
# there too. This check compares the current diff (commits since BASE_REF,
# plus any staged/unstaged changes) against origin/main by default and
# fails if a shared-path file changed on one side but not the other.
# Files that only exist on one side are not mirrors and are skipped.
# Runs outside a git checkout, or against a BASE_REF that doesn't exist
# locally, print "skip" instead of failing (the pytest suites still run).
#
# Usage:
#   scripts/verify.sh                          # checks against origin/main
#   BASE_REF=main scripts/verify.sh             # compare against another ref
#   PYTHON=/path/to/python3 scripts/verify.sh   # use a specific interpreter
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python
BASE_REF="${BASE_REF:-origin/main}"

fail=0

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

echo "== test/"
if ! "$PYTHON" -m pytest -q -p no:cacheprovider test --ignore=test/test_live_sdk.py; then
  fail=1
fi

echo "== erc_urban_test/tests"
if ! "$PYTHON" -m pytest -q -p no:cacheprovider erc_urban_test/tests; then
  fail=1
fi

echo "== mirror-drift (erc_urban vs erc_urban_test)"
if ! command -v git >/dev/null 2>&1 || ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "skip: not a git checkout"
elif ! git rev-parse --verify -q "$BASE_REF" >/dev/null 2>&1; then
  echo "skip: base ref '$BASE_REF' not found (set BASE_REF=<ref>)"
else
  # Files with the same relative path under both erc_urban/ and erc_urban_test/
  # are mirrors: a real code change to one must show up as a change to the
  # other in the same diff, or the two stacks silently diverge. Files that
  # only exist on one side (sandbox-only tools, tests/) are not mirrors and
  # are skipped.
  changed="$(git diff --name-only "$BASE_REF"...HEAD -- erc_urban erc_urban_test; git diff --name-only -- erc_urban erc_urban_test; git diff --cached --name-only -- erc_urban erc_urban_test)"
  changed="$(printf '%s\n' "$changed" | sort -u | sed '/^$/d')"
  drift=0
  while IFS= read -r path; do
    [ -z "$path" ] && continue
    case "$path" in
      erc_urban/*)
        other="erc_urban_test/${path#erc_urban/}"
        [ -f "$other" ] || continue
        if ! printf '%s\n' "$changed" | grep -qxF "$other"; then
          echo "drift: $path changed but $other did not"
          drift=1
        fi
        ;;
      erc_urban_test/*)
        other="erc_urban/${path#erc_urban_test/}"
        [ -f "$other" ] || continue
        if ! printf '%s\n' "$changed" | grep -qxF "$other"; then
          echo "drift: $path changed but $other did not"
          drift=1
        fi
        ;;
    esac
  done <<EOF
$changed
EOF
  if [ "$drift" -eq 1 ]; then
    fail=1
  else
    echo "ok: no one-sided changes to mirrored files"
  fi
fi

echo "== git diff --check (whitespace)"
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if ! git diff --check; then
    fail=1
  fi
  if ! git diff --cached --check; then
    fail=1
  fi
else
  echo "skip: not a git checkout"
fi

if [ "$fail" -ne 0 ]; then
  echo "VERIFY FAILED"
  exit 1
fi
echo "VERIFY OK"
