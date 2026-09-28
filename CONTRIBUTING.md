# Contributing

## Isolated worktrees for feature work

Working directly in the main checkout means every branch switch drags
`erc_indoor/data/route_map`, `earth-rovers-sdk/.venv`, and any uncommitted
experiment along with it. Use a linked git worktree instead so feature work,
bugfixes, and the main checkout never step on each other.

```bash
# macOS / Linux / WSL
scripts/new-worktree.sh my-feature-branch

# Windows PowerShell
scripts\new-worktree.ps1 -Branch my-feature-branch
```

This creates `.worktrees/my-feature-branch` as a new linked worktree
(branching off `origin/main`, or an existing local branch if one with that
name already exists), copies `earth-rovers-sdk/.env` into it so the SDK
server has credentials, and creates a `.venv` there with `requirements.txt`
installed. It does not download model checkpoints or pre-cache DINOv2 /
SegFormer — those are large and usually already warm in your global cache;
run `./setup.sh` inside the new worktree if you need them there too.

Useful flags:

- `scripts/new-worktree.sh <branch> <base-ref>` — branch off something other
  than `origin/main`.
- `SKIP_PIP=1 scripts/new-worktree.sh <branch>` / `-SkipPip` on Windows —
  skip the venv, e.g. if you'll activate `conda activate rover_env` instead.

When you're done with a worktree:

```bash
git worktree remove .worktrees/my-feature-branch
```

`.worktrees/` and `worktrees/` are already in `.gitignore`, so nothing under
them is ever staged by accident.

## Tests

Offline suites (no rover, no SDK server, no network):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --ignore=test/test_live_sdk.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest erc_urban_test/tests
```

`test_live_sdk.py` needs the SDK running with a rover connected; skip it for
routine PR checks. Run both suites above before opening a PR.

## erc_urban / erc_urban_test

`erc_urban_test/` is a sandbox copy of `erc_urban/` used for local
free-drive testing. If a change to one applies to the other's copy of the
same code, mirror it there too.
