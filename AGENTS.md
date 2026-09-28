# AGENTS.md

Guide for coding agents working in this repo. Read README.md first for the
product picture (what each stack does, how to run against a real rover);
this file is about editing code and running tests here without breaking
things.

## Layout

| Path | What |
|---|---|
| `erc_urban/` | Urban track stack: LogoNav planner, Depth Anything obstacles, SegFormer semantics. `erc_urban.run` is the entry point used at a venue. |
| `erc_urban_test/` | Sandbox copy of `erc_urban` for local free-drive testing plus its own unit tests (`erc_urban_test/tests/`). |
| `erc_indoor/` | Indoor track: teach-and-repeat stack. Separate module tree, own tests under top-level `test/`. |
| `erc_offroad/` | Off-road trial tools (teleop, climb/creep). See `erc_offroad/FINDINGS.md`. |
| `earth-rovers-sdk/` | Vendored FrodoBots SDK server (own venv, own numpy pin). Not part of the stacks' test suite. |
| `ERC-3-earthrover-challenge/` | Vendored snapshot of a third-party repo; `erc_urban`/`erc_urban_test` import its LogoNav and Depth Anything code. Don't edit it as if it were ours — changes here are upstream drift, not local fixes. |
| `test/` | Unit tests for `erc_urban` (imports `erc_urban.*`). |
| `erc_urban_test/tests/` | The same tests, importing `erc_urban_test.*` instead. See mirror rule below. |

## The erc_urban / erc_urban_test mirror

`erc_urban_test` is a parallel copy of `erc_urban`, not a re-export or a
subclass. Some files are byte-identical between the two trees right now
(`calib.py`, `camera_front.json`, `geo.py`, `interfaces.py`); others have
already diverged (e.g. `erc_urban/modules/dirt_path.py` and
`erc_urban/modules/upgrades.py` exist only in `erc_urban`,
`erc_urban_test/modules/osm_route.py` exists only in `erc_urban_test`,
`baseline.py`/`traversability.py`/`run.py`/`sdk_client.py` differ between the
two). Divergence is allowed; it happens when one side gets a fix or feature
the other doesn't need yet.

**The rule:** if you change a file in `erc_urban/` and the same logic exists
in `erc_urban_test/` (same filename, same function/class, not already
diverged for a reason), apply the equivalent change there too, and vice
versa. Before editing, diff the two copies of the file you're touching
(`diff erc_urban/modules/X.py erc_urban_test/modules/X.py`) to see whether
they're still in sync. If they're already different, use judgment: mirror
the fix if the diverged code still has the same bug, skip it if the
diverged version doesn't have that code path at all, and say what you
decided in your PR description either way.

The same pairing applies to tests: `test/test_X.py` imports from
`erc_urban.*`, `erc_urban_test/tests/test_X.py` imports the same test body
from `erc_urban_test.*`. Add or fix a test in one and mirror it in the
other with the import swapped.

## Running tests

Both suites use `unittest`-style tests collected by pytest, no live rover
or SDK needed except one file:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test --ignore=test/test_live_sdk.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest erc_urban_test/tests
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is required in this environment: the
global conda env has a broken `deepeval`/`langchain` plugin that raises
`ModuleNotFoundError` during pytest's plugin autoload and kills collection
before any of this repo's tests run (see `pytest.ini`). It can't be set from
`pytest.ini` because autoload happens before ini parsing.

`test/test_live_sdk.py` needs the SDK running with a rover connected; it is
excluded above and will fail/hang otherwise. Everything else in both suites
is offline (synthetic `Observation`s, fixture images in
`erc_urban_test/tests/fixtures/`).

Current baseline on this checkout: `test/` (minus `test_live_sdk.py`) is
286 passed, 1 skipped; `erc_urban_test/tests` is 273 passed, 3 skipped.

## Conventions

- Python 3.10, dataclasses, `from __future__ import annotations`. Pinned
  versions are in `requirements.txt`; this project does not use `uv` or
  Poetry, just `pip install -r requirements.txt` into a conda env (see
  `setup.sh`, `README.md` Setup).
- Module tuning comes from env vars read in `run.py` (`STAGE`, `USE_*`
  flags), not hardcoded config files or CLI flags. When adding a knob,
  follow that pattern and document the default in `.env.example`.
- Core data types (`Candidate`, `Observation`) live in `interfaces.py` and
  are shared across every module in that stack — check there before adding
  a new ad hoc dict/tuple shape.
- `# >>> WIRE:` comments (see `erc_urban/DESIGN.md`) mark seams the authors
  deliberately left unimplemented rather than guess at. Don't silently fill
  these in with invented signatures; if you wire one up, say so explicitly.
- Tests build synthetic `Observation`/`Candidate` objects directly rather
  than mocking the SDK; follow that pattern for new offline tests instead
  of reaching for `unittest.mock` on the SDK client.

## Pitfalls

- **Don't confuse the two numpy pins.** `requirements.txt` (used by the
  stacks, `erc_urban`/`erc_urban_test`/`erc_indoor`) pins numpy 2.x;
  `earth-rovers-sdk/requirements.txt` (used only by the SDK server, its own
  venv) pins numpy 1.26. They intentionally don't share an environment —
  don't "fix" this by unifying the pins.
- **`ERC-3-earthrover-challenge/` is vendored**, pinned to a specific
  upstream commit (see README.md). Treat it as read-only unless you're
  intentionally updating the vendored snapshot; local fixes belong in
  `erc_urban`/`erc_urban_test`, which import from it.
- **`__pycache__` dirs exist under `erc_urban/modules/` and
  `erc_urban_test/modules/`** and will show up in `diff -r` between the two
  trees as noise (different bytecode even for identical/unrelated source).
  Diff the `.py` files directly, not the directories, when checking mirror
  status.
- **Running the full test command without
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`** fails at collection with a
  `ModuleNotFoundError` from a global-env plugin that has nothing to do
  with this repo; it's easy to misread that as a real regression.
- `erc_urban_test/local_run.sh` and any `test_live_sdk.py`-style test talk
  to a real SDK server and, for `drive`, a real rover. Never run those from
  an automated/offline agent session — they're for a human standing next to
  the bot.
