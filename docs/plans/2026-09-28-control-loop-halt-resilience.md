# Control-Loop Halt Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `RoverClient.send_cmd` report a network failure instead of raising, so the control loop's own tick-error handler can always halt the rover instead of occasionally crashing outright while the rover's last, possibly nonzero, command is still standing.

**Architecture:** `send_cmd` wraps its `/control` POST in a `try/except requests.exceptions.RequestException` and also treats any 5xx response as failure, returning `bool` instead of `None` and never raising. `_halt()` in `run.py` checks that return value and prints a visible warning when the stop itself failed to send, instead of silently swallowing it. Both `erc_urban` and `erc_urban_test` carry duplicate copies of `sdk_client.py` and `run.py`, so the same two edits land in both packages.

**Tech Stack:** Python 3, `requests`, `unittest`/`pytest`, run through the Windows `rover_env` interpreter via WSL interop (`test.sh`). No new dependencies.

**Spec:** none — this plan documents a defect found directly in the code, not from a written spec. Cited locations, current worktree HEAD `6bd2577` (verified 2026-09-28):
- `erc_urban/sdk_client.py:65-76` (`send_cmd`, unguarded `requests.post`)
- `erc_urban/run.py:423-428` (`_cmd`/`_halt`) and `erc_urban/run.py:1984-2000` (`main()`'s tick loop: `except Exception as e: _halt(client); print(...)`)
- `erc_urban_test/sdk_client.py:65-76` (identical `send_cmd`)
- `erc_urban_test/run.py:411-416` (`_cmd`/`_halt`) and `erc_urban_test/run.py:1858-1868` (identical tick loop)

The bug: when a tick raises (e.g. the SDK is unreachable), `main()`'s `except Exception` handler calls `_halt(client)` to stop the rover. `_halt` calls `client.send_cmd(...)`, which does a bare `requests.post(...)` with no `try/except`. If the SDK is unreachable for the *same* reason the tick just failed, that second `send_cmd` call raises again — this time with nothing catching it, since it is already inside the loop's own exception handler. The exception escapes `main()` entirely, the process exits, and the rover is left running whatever command it was last given, un-halted, with no further ticks. Confirmed this is unguarded by reading the three most recent related open PRs against this repo (`gh pr diff`, all read-only): PR #30 (`fix/urban-sdk-client-robustness`) shortens the default timeout and hardens payload/telemetry parsing but does not touch `send_cmd`; PR #31 (`fix/urban-control-loop-safety`) adds a `SEND_CONTROL=0` dry-run bypass to `_cmd`/`_halt` but still calls the same unguarded `client.send_cmd(...)` when `SEND_CONTROL=1`; PR #27 (`fix/sdk-rtm-send-and-reset`) is server-side (the `earth-rovers-sdk`/RTM layer), not this client. None of the nine open PRs listed for this repo cover this path.

## Global Constraints

- Git via `git.exe` only (Windows git); never Linux `git`. LF line endings. Conventional commit messages (`fix:`/`feat:`/`test:`/`docs:`/`chore:`/`refactor:`). Never amend, rebase, force-push, or push.
- Any change to `erc_urban` must be mirrored into `erc_urban_test`, and vice versa, when the other copy carries the same code (both `sdk_client.py` and `run.py` do here).
- Tests run only via `bash /mnt/c/Users/007ha/Desktop/self-drive/.agents/sp_lanes/test.sh writing-plans`, which runs `test/` (`--ignore=test/test_live_sdk.py`) and `erc_urban_test/tests`. Both suites must be no worse after the change. Baseline measured 2026-09-28: `test/` 286 passed, 1 skipped; `erc_urban_test/tests` 273 passed, 3 skipped.
- Keep the change small and reviewable (well under 400 changed lines — this is a ~20-line fix duplicated across two packages plus tests).
- `erc_urban` has no `tests/` directory and `test.sh` runs no suite against it; verification there is by equivalence against the already-tested `erc_urban_test` copy plus an import smoke check, not by a second pytest run.
- Never touch `nyu-earthrover`, never edit or print any `.env` file, never load model weights for benchmarks.

## Review Focus

1. **Halt-after-tick-error double fault.** When the SDK is unreachable, `main()`'s `except Exception` handler calls `_halt(client)`, which today calls `client.send_cmd(...)` unguarded — a second network failure there raises out of the handler and kills the whole run with the rover's last command still standing. Task 1's tests pin that `send_cmd` returns `False` rather than raising on `ConnectionError`/`Timeout`.
2. **Silent halt failure.** Simply swallowing the failure (`send_cmd` returns `False` and the caller does nothing) hides from the operator that the rover was *not* actually told to stop. Task 2's tests pin that `_halt` prints a visible warning when the stop itself fails to send, and that the routine per-tick `_cmd` path is unaffected (no warning spam on ordinary transient misses).
3. **Server-side failure without a raised exception.** `/control` can answer with a 5xx (e.g. the 503 added by PR #27's rate-limit path on the SDK side) without `requests.post` raising at all. A bare `try/except RequestException` would miss this silently (report success on a failed command). Task 1's test pins that a mocked 5xx response is also treated as failure.
4. **Backward compatibility of existing callers.** `stop()`, `_cmd`, and the final `client.send_cmd(0.0, 0.0, lamp=0)` in `main()` all currently ignore `send_cmd`'s return value; changing it from `None` to `bool` must not break them. Task 1 re-runs the existing `test_send_cmd_clamps` test unmodified as a regression check, and Task 2 adds an explicit test that `_cmd` still works when `send_cmd` returns `False`.
5. **Known limitation, not solved by this plan.** A `/control` POST that times out client-side may have already landed server-side (the rover moved before the timeout fired). This plan makes that failure visible and non-fatal; it cannot make the network idempotent or retroactively confirm the rover actually stopped. Documented in NOTES, not silently claimed as fixed.

---

### Task 1: `RoverClient.send_cmd` never raises on a network failure (erc_urban_test)

**Files:**
- Modify: `erc_urban_test/sdk_client.py:65-76` (`send_cmd`)
- Test: `erc_urban_test/tests/test_sdk_client.py` (append; also add `import requests` to the top import block)

**Interfaces:**
- Produces: `RoverClient.send_cmd(linear: float, angular: float, lamp: Optional[int] = None) -> bool` (was `-> None`). Returns `True` iff the POST returned a status code below 500; `False` on any `requests.exceptions.RequestException` or a 5xx status. Never raises.
- Consumes: existing `self.base`, `self.timeout`, module-level `requests` import — nothing new.

- [ ] **Step 1: Write the failing tests**

Add `import requests` near the top of `erc_urban_test/tests/test_sdk_client.py` (alongside the existing `import math` / `import unittest` block), then append this class before the `if __name__ == "__main__":` line:

```python
class TestSendCmdResilience(unittest.TestCase):
    """erc_urban/run.py's tick-error handler calls _halt -> send_cmd while
    the SDK may be the thing that is unreachable. send_cmd must report that
    instead of raising, or the handler itself crashes the run."""

    def setUp(self):
        self.client = RoverClient("http://localhost:8000", timeout=1.0)

    @patch("erc_urban_test.sdk_client.requests.post")
    def test_send_cmd_returns_true_on_2xx(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        self.assertIs(self.client.send_cmd(0.1, 0.0), True)

    @patch(
        "erc_urban_test.sdk_client.requests.post",
        side_effect=requests.exceptions.ConnectionError("refused"),
    )
    def test_send_cmd_returns_false_instead_of_raising_on_connection_error(self, _):
        result = self.client.send_cmd(0.0, 0.0)
        self.assertIs(result, False)

    @patch(
        "erc_urban_test.sdk_client.requests.post",
        side_effect=requests.exceptions.Timeout("timed out"),
    )
    def test_send_cmd_returns_false_instead_of_raising_on_timeout(self, _):
        self.assertIs(self.client.send_cmd(0.0, 0.0), False)

    @patch("erc_urban_test.sdk_client.requests.post")
    def test_send_cmd_treats_5xx_as_failure_without_raising(self, mock_post):
        mock_post.return_value = MagicMock(status_code=503)
        self.assertIs(self.client.send_cmd(0.0, 0.0), False)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/mnt/c/Users/007ha/miniforge3/envs/rover_env/python.exe -m pytest -q erc_urban_test/tests/test_sdk_client.py -k TestSendCmdResilience -v`
Expected: FAIL/ERROR on all four — `send_cmd` today always returns `None` (`assertIs(None, True)` and `assertIs(None, False)` both fail), and the connection-error/timeout cases raise straight out of `send_cmd` (reported as ERROR, not FAIL, since nothing catches them yet).

- [ ] **Step 3: Write the minimal implementation**

Replace `erc_urban_test/sdk_client.py:65-76`:

```python
    def send_cmd(self, linear: float, angular: float, lamp: Optional[int] = None) -> bool:
        """linear/angular clamped to [-1, 1]. lamp is 0/1 if provided.

        Returns True once the POST lands with a non-5xx status, False on any
        network failure (timeout, connection refused, 5xx) rather than
        raising. main()'s tick-error handler calls this (via _halt) to stop
        the rover after a tick has already failed; a raise here would escape
        that handler uncaught and end the run with the rover's last
        (possibly nonzero) command still standing.
        """
        linear = float(np.clip(linear, -1.0, 1.0))
        angular = float(np.clip(angular, -1.0, 1.0))
        command: dict[str, Any] = {"linear": linear, "angular": angular}
        if lamp is not None:
            command["lamp"] = int(lamp)
        try:
            r = requests.post(
                f"{self.base}/control",
                json={"command": command},
                timeout=self.timeout,
            )
            return r.status_code < 500
        except requests.exceptions.RequestException:
            return False
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/mnt/c/Users/007ha/miniforge3/envs/rover_env/python.exe -m pytest -q erc_urban_test/tests/test_sdk_client.py -v`
Expected: PASS — all four new tests green, plus the pre-existing `test_send_cmd_clamps` still green (it never inspects `send_cmd`'s return value, so the `None -> bool` change is invisible to it).

- [ ] **Step 5: Commit**

```bash
git.exe add erc_urban_test/sdk_client.py erc_urban_test/tests/test_sdk_client.py
git.exe commit -m "fix(urban-test): send_cmd reports network failure instead of raising"
```

### Task 2: `_halt()` surfaces a failed stop command instead of swallowing it (erc_urban_test)

**Files:**
- Modify: `erc_urban_test/run.py:415-416` (`_halt`)
- Create: `erc_urban_test/tests/test_control_resilience.py`

**Interfaces:**
- Consumes: `RoverClient.send_cmd(...) -> bool` from Task 1.
- Produces: `_halt(client: RoverClient) -> None` — signature unchanged; now prints `"  halt command failed to send — rover may still be moving"` (flushed) when `send_cmd` returns `False`. `_cmd(client, lin, ang) -> None` is unchanged — it stays a fire-and-forget per-tick call, so a transient miss on an ordinary (non-halt) command does not spam the log; only the safety-critical stop gets the visible warning.

- [ ] **Step 1: Write the failing test**

Create `erc_urban_test/tests/test_control_resilience.py`:

```python
"""Halt visibility when the SDK is unreachable.

erc_urban_test/run.py:1858-1868 is the double-fault path this guards: a
tick error triggers `except Exception: _halt(client)`, and `_halt`'s own
send can fail too (same outage). `_halt` must not raise (Task 1 makes
send_cmd itself safe) and must say so when it happens (this task)."""
from __future__ import annotations
import contextlib
import io
import unittest
from unittest.mock import ANY, MagicMock

from erc_urban_test.run import _halt, _cmd


class TestHaltVisibility(unittest.TestCase):
    def test_halt_prints_warning_when_send_fails(self):
        client = MagicMock()
        client.send_cmd.return_value = False
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _halt(client)
        self.assertIn("halt command failed to send", buf.getvalue())
        client.send_cmd.assert_called_once_with(0.0, 0.0, lamp=ANY)

    def test_halt_does_not_print_when_send_succeeds(self):
        client = MagicMock()
        client.send_cmd.return_value = True
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _halt(client)
        self.assertEqual(buf.getvalue(), "")

    def test_cmd_ignores_send_cmd_return_value(self):
        """_cmd is the routine per-tick path, not the safety path — it must
        keep working whether send_cmd reports True or False."""
        client = MagicMock()
        client.send_cmd.return_value = False
        _cmd(client, 0.1, 0.0)  # must not raise
        client.send_cmd.assert_called_once()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/mnt/c/Users/007ha/miniforge3/envs/rover_env/python.exe -m pytest -q erc_urban_test/tests/test_control_resilience.py -v`
Expected: `test_halt_prints_warning_when_send_fails` FAILS (`buf.getvalue()` is `""` today — `_halt` ignores `send_cmd`'s return value and never prints). The other two tests already pass against today's `_halt`/`_cmd` (they don't raise given a mock), which is fine — they exist to lock the behavior in place once Task 2 lands, not to prove a second bug.

- [ ] **Step 3: Write the minimal implementation**

Replace `erc_urban_test/run.py:415-416`:

```python
def _halt(client: RoverClient) -> None:
    if not client.send_cmd(0.0, 0.0, lamp=LAMP):
        print("  halt command failed to send — rover may still be moving", flush=True)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/mnt/c/Users/007ha/miniforge3/envs/rover_env/python.exe -m pytest -q erc_urban_test/tests/test_control_resilience.py -v`
Expected: PASS — 3 passed.

- [ ] **Step 5: Commit**

```bash
git.exe add erc_urban_test/run.py erc_urban_test/tests/test_control_resilience.py
git.exe commit -m "fix(urban-test): warn when a halt command fails to send"
```

### Task 3: mirror both fixes into `erc_urban` (no test suite there — verified by equivalence + import)

**Files:**
- Modify: `erc_urban/sdk_client.py:65-76` (`send_cmd`) — same body as Task 1, under the `erc_urban` package.
- Modify: `erc_urban/run.py:427-428` (`_halt`) — same body as Task 2, under the `erc_urban` package.

**Interfaces:**
- Consumes: nothing new.
- Produces: same as Task 1/2 but importable as `erc_urban.sdk_client.RoverClient.send_cmd` / `erc_urban.run._halt`.

`erc_urban` has no `tests/` directory and `test.sh` does not run a suite against it (only `test/` and `erc_urban_test/tests`). The "test" for this task is that the two packages' relevant code blocks become identical (since they already carry duplicate logic) and that the module still imports cleanly — that is the closest available substitute for a failing/passing test cycle here.

- [ ] **Step 1: Run the equivalence check to verify it currently fails (blocks differ)**

Run: `diff <(sed -n '65,80p' erc_urban/sdk_client.py) <(sed -n '65,84p' erc_urban_test/sdk_client.py)`
Expected: non-empty diff — `erc_urban/sdk_client.py:65-76` is still the pre-Task-1 unguarded `send_cmd` (raises, returns `None`), while `erc_urban_test/sdk_client.py` already has Task 1's fixed version.

- [ ] **Step 2: Confirm the `_halt` blocks also differ**

Run: `diff <(sed -n '423,428p' erc_urban/run.py) <(sed -n '411,417p' erc_urban_test/run.py)`
Expected: non-empty diff — `erc_urban/run.py`'s `_halt` still silently swallows a failed send; `erc_urban_test/run.py`'s already warns (Task 2).

- [ ] **Step 3: Apply the identical edits**

Replace `erc_urban/sdk_client.py:65-76` with the exact `send_cmd` body from Task 1, Step 3 (byte-identical — the method has no package-qualified references).

Replace `erc_urban/run.py:427-428` with the exact `_halt` body from Task 2, Step 3.

- [ ] **Step 4: Run the equivalence check and an import smoke test to verify it passes**

Run:
```bash
diff <(sed -n '65,80p' erc_urban/sdk_client.py) <(sed -n '65,80p' erc_urban_test/sdk_client.py)
diff <(sed -n '427,432p' erc_urban/run.py) <(sed -n '415,420p' erc_urban_test/run.py)
PYTHONPATH=. /mnt/c/Users/007ha/miniforge3/envs/rover_env/python.exe -c "import erc_urban.sdk_client, erc_urban.run; print('import ok')"
```
Expected: both `diff`s produce no output (blocks now identical modulo the line-number offset already accounted for in the ranges above), and the import line prints `import ok` with no traceback.

Then re-run the full acceptance check for the whole plan:

Run: `bash /mnt/c/Users/007ha/Desktop/self-drive/.agents/sp_lanes/test.sh writing-plans`
Expected: `test/` still 286 passed, 1 skipped; `erc_urban_test/tests` now 273 + 7 = 280 passed (4 new in Task 1, 3 new in Task 2), 3 skipped — no regressions, only additions.

- [ ] **Step 5: Commit**

```bash
git.exe add erc_urban/sdk_client.py erc_urban/run.py
git.exe commit -m "fix(urban): send_cmd reports network failure instead of raising"
```

---

## Self-Review

**Spec coverage:** There is no external spec; the "spec" is the defect described above. Every element of it is covered: `send_cmd` raising (Task 1), `send_cmd` returning success on a 5xx (Task 1's `test_send_cmd_treats_5xx_as_failure_without_raising`), the halt call silently swallowing failure (Task 2), and the duplicate `erc_urban` copy (Task 3).

**Placeholder scan:** No `TBD`/`implement later`/`add error handling` phrasing; every step has real code and a real command. Task 3 references Task 1/Task 2 code rather than repeating it verbatim in the plan text, which the skill flags as a smell ("Similar to Task N") — but it is intentional here: Task 3's whole point is that the two blocks must end up byte-identical, so pointing at the earlier block *is* the spec for Task 3, not a shortcut around writing it. The blocks themselves are written out in full in Tasks 1 and 2.

**Type consistency:** `send_cmd` returns `bool` everywhere it's touched (Task 1's implementation, Task 2's `_halt` consuming it, Task 3's mirror). `_halt`'s and `_cmd`'s signatures (`(client: RoverClient, ...) -> None`) are unchanged from their current form in both packages, so no caller elsewhere in `run.py` needs updating — confirmed by grep: every call site (`_halt(client)` / `_cmd(client, lin, ang)`) ignores the return value already.

**Review Focus:** all five items above have an owning task and test: #1 and #3 in Task 1, #2 and #4 in Task 2 (plus Task 1's regression re-run), #5 is a documented limitation with no task (correctly — it's out of scope, not a gap).

---

Plan complete and saved to `docs/plans/2026-09-28-control-loop-halt-resilience.md`. Please review the plan. Which execution approach would you prefer?

- **Subagent-driven** — a fresh subagent implements each task and a fresh reviewer checks it before the next one starts, then a whole-branch review at the end. Most thorough; costs a fresh context per task and per review.
- **Native** — implement every task in this session directly, then one fresh reviewer on the most capable model checks the whole branch. Cheapest and fastest; no independent review until the end.

**For this plan I recommend Native**, because the three tasks are small, sequentially dependent on the exact same 12-line pattern (Task 2 depends on Task 1's return type, Task 3 mirrors both verbatim), there are only three of them, and a shipped mistake here is a `bool`-vs-`None` typo caught immediately by the existing test suite — not the kind of risk that needs a fresh reviewer gate between every task.
