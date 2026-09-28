# Control-Loop Halt Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `RoverClient.send_cmd` report a network failure instead of raising, so the control loop's own tick-error handler can always halt the rover and keep the run alive instead of occasionally crashing the whole process on a transient SDK outage. After such a crash nothing sends a stop at all; the rover only comes to rest when its firmware watchdog notices commands have stopped arriving.

**Architecture:** `send_cmd` wraps its `/control` POST in a `try/except requests.exceptions.RequestException` and also treats any 5xx response as failure, returning `bool` instead of `None` and never raising. `_halt()` in `run.py` checks that return value and prints a visible warning when the stop itself failed to send, instead of silently swallowing it. Both `erc_urban` and `erc_urban_test` carry duplicate copies of `sdk_client.py` and `run.py`, so the same two edits land in both packages.

**Tech Stack:** Python 3, `requests`, `unittest`/`pytest`. No new dependencies.

**Spec:** none — this plan documents a defect found directly in the code, not from a written spec. Cited locations, current worktree HEAD `6bd2577` (verified 2026-09-28):
- `erc_urban/sdk_client.py:65-76` (`send_cmd`, unguarded `requests.post`)
- `erc_urban/run.py:423-428` (`_cmd`/`_halt`) and `erc_urban/run.py:1984-2000` (`main()`'s tick loop: `except Exception as e: _halt(client); print(...)`)
- `erc_urban_test/sdk_client.py:65-76` (identical `send_cmd`)
- `erc_urban_test/run.py:411-416` (`_cmd`/`_halt`) and `erc_urban_test/run.py:1858-1868` (identical tick loop)

The bug: when a tick raises (e.g. the SDK is unreachable), `main()`'s `except Exception` handler calls `_halt(client)` to stop the rover. `_halt` calls `client.send_cmd(...)`, which does a bare `requests.post(...)` with no `try/except`. If the SDK is unreachable for the *same* reason the tick just failed, that second `send_cmd` call raises again — this time with nothing catching it, since it is already inside the loop's own exception handler. The exception escapes `main()` entirely, the process exits, and the run is over with no explicit halt ever sent. The rover does not keep driving indefinitely (its firmware watchdog coasts it to a stop once commands stop arriving, see `erc_indoor/doc/HOW_IT_WORKS.md` section 9 and the "streamed every tick (watchdog)" note in `erc_urban/eval/record.py`), but it coasts on its last command until that watchdog fires, and a single transient outage ends the mission instead of being retried on the next tick. Confirmed this is unguarded by reading the three most recent related open PRs against this repo (`gh pr diff`, all read-only): PR #30 (`fix/urban-sdk-client-robustness`) shortens the default timeout and hardens payload/telemetry parsing but does not touch `send_cmd`; PR #31 (`fix/urban-control-loop-safety`) adds a `SEND_CONTROL=0` dry-run bypass to `_cmd`/`_halt` but still calls the same unguarded `client.send_cmd(...)` when `SEND_CONTROL=1`; PR #27 (`fix/sdk-rtm-send-and-reset`) is server-side (the `earth-rovers-sdk`/RTM layer), not this client. None of the nine open PRs listed for this repo cover this path.

## Global Constraints

- LF line endings. Conventional commit messages (`fix:`/`feat:`/`test:`/`docs:`/`chore:`/`refactor:`).
- Any change to `erc_urban` must be mirrored into `erc_urban_test`, and vice versa, when the other copy carries the same code (both `sdk_client.py` and `run.py` do here). Tests are mirrored the same way: `test/` is the `erc_urban` suite, `erc_urban_test/tests/` is the `erc_urban_test` suite, same test bodies with the import swapped.
- Run both offline suites before and after: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test --ignore=test/test_live_sdk.py` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest erc_urban_test/tests`. Both must be no worse after the change. Baseline at `6bd2577`: `test/` 286 passed, 1 skipped; `erc_urban_test/tests` 273 passed, 3 skipped.
- Keep the change small and reviewable (well under 400 changed lines; this is a ~20-line fix duplicated across two packages plus tests).
- Never edit or print any `.env` file.

## Review Focus

1. **Halt-after-tick-error double fault.** When the SDK is unreachable, `main()`'s `except Exception` handler calls `_halt(client)`, which today calls `client.send_cmd(...)` unguarded — a second network failure there raises out of the handler and kills the whole run without an explicit stop ever being sent. Task 1's tests pin that `send_cmd` returns `False` rather than raising on `ConnectionError`/`Timeout`.
2. **Silent halt failure.** Simply swallowing the failure (`send_cmd` returns `False` and the caller does nothing) hides from the operator that the rover was *not* actually told to stop. Task 2's tests pin that `_halt` prints a visible warning when the stop itself fails to send, and that the routine per-tick `_cmd` path is unaffected (no warning spam on ordinary transient misses).
3. **Server-side failure without a raised exception.** `/control` can answer with a 5xx (e.g. the 503 added by PR #27's rate-limit path on the SDK side) without `requests.post` raising at all. A bare `try/except RequestException` would miss this silently (report success on a failed command). Task 1's test pins that a mocked 5xx response is also treated as failure.
4. **Backward compatibility of existing callers.** `stop()`, `_cmd`, the final `client.send_cmd(0.0, 0.0, lamp=0)` in `main()`, `eval/record.py` (lines 114 and 161, both packages) and `erc_urban_test/bench.py` (lines 84, 86 and 226) all currently ignore `send_cmd`'s return value; changing it from `None` to `bool` must not break them. They do rely on the raise, though: today a `/control` `ConnectionError` inside `_cmd` aborts the tick and prints `tick error:`, and in `record.py` it ends the recording. After Task 1 those failures are silent unless the caller checks the return value, so decide per caller whether that is acceptable. Task 1 re-runs the existing `test_send_cmd_clamps` test unmodified as a regression check, and Task 2 adds an explicit test that `_cmd` still works when `send_cmd` returns `False`.
5. **Known limitation, not solved by this plan.** A `/control` POST that times out client-side may have already landed server-side (the rover moved before the timeout fired). This plan makes that failure visible and non-fatal; it cannot make the network idempotent or retroactively confirm the rover actually stopped. Recorded here as a known limitation, not silently claimed as fixed.

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

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q erc_urban_test/tests/test_sdk_client.py -k TestSendCmdResilience -v`
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
        that handler uncaught and end the run with no stop ever sent,
        leaving the rover to coast until its firmware watchdog fires.
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

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q erc_urban_test/tests/test_sdk_client.py -v`
Expected: PASS — all four new tests green, plus the pre-existing `test_send_cmd_clamps` still green (it never inspects `send_cmd`'s return value, so the `None -> bool` change is invisible to it).

- [ ] **Step 5: Commit**

```bash
git add erc_urban_test/sdk_client.py erc_urban_test/tests/test_sdk_client.py
git commit -m "fix(urban-test): send_cmd reports network failure instead of raising"
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

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q erc_urban_test/tests/test_control_resilience.py -v`
Expected: `test_halt_prints_warning_when_send_fails` FAILS (`buf.getvalue()` is `""` today — `_halt` ignores `send_cmd`'s return value and never prints). The other two tests already pass against today's `_halt`/`_cmd` (they don't raise given a mock), which is fine — they exist to lock the behavior in place once Task 2 lands, not to prove a second bug.

- [ ] **Step 3: Write the minimal implementation**

Replace `erc_urban_test/run.py:415-416`:

```python
def _halt(client: RoverClient) -> None:
    if not client.send_cmd(0.0, 0.0, lamp=LAMP):
        print("  halt command failed to send — rover may still be moving", flush=True)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q erc_urban_test/tests/test_control_resilience.py -v`
Expected: PASS — 3 passed.

- [ ] **Step 5: Commit**

```bash
git add erc_urban_test/run.py erc_urban_test/tests/test_control_resilience.py
git commit -m "fix(urban-test): warn when a halt command fails to send"
```

### Task 3: mirror both fixes and their tests into `erc_urban`

**Files:**
- Modify: `erc_urban/sdk_client.py:65-76` (`send_cmd`), same body as Task 1, Step 3.
- Modify: `erc_urban/run.py:427-428` (`_halt`), same body as Task 2, Step 3.
- Test: `test/test_sdk_client.py` (append the Task 1 class; add `import requests`)
- Create: `test/test_control_resilience.py` (the Task 2 file)

**Interfaces:**
- Consumes: nothing new.
- Produces: same as Task 1/2 but importable as `erc_urban.sdk_client.RoverClient.send_cmd` / `erc_urban.run._halt`.

`test/` is the `erc_urban` suite (its files import `erc_urban.*`), so this task gets the same red/green cycle as Tasks 1 and 2.

- [ ] **Step 1: Mirror the tests with the import swapped**

Append `TestSendCmdResilience` from Task 1, Step 1 to `test/test_sdk_client.py`, with every `erc_urban_test.sdk_client.requests.post` patch target changed to `erc_urban.sdk_client.requests.post`, and add `import requests` to its import block.

Create `test/test_control_resilience.py` as a copy of the Task 2 file with `from erc_urban_test.run import _halt, _cmd` changed to `from erc_urban.run import _halt, _cmd`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q test/test_sdk_client.py test/test_control_resilience.py -v`
Expected: the four `TestSendCmdResilience` tests FAIL/ERROR and `test_halt_prints_warning_when_send_fails` FAILS, for the same reasons as in Tasks 1 and 2.

- [ ] **Step 3: Apply the identical edits**

Replace `erc_urban/sdk_client.py:65-76` with the exact `send_cmd` body from Task 1, Step 3 (the method has no package-qualified references).

Replace `erc_urban/run.py:427-428` with the exact `_halt` body from Task 2, Step 3.

- [ ] **Step 4: Run both suites to verify they pass**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test --ignore=test/test_live_sdk.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest erc_urban_test/tests
```
Expected: `test/` 286 + 7 = 293 passed, 1 skipped; `erc_urban_test/tests` 273 + 7 = 280 passed, 3 skipped. No regressions, only additions.

- [ ] **Step 5: Commit**

```bash
git add erc_urban/sdk_client.py erc_urban/run.py test/test_sdk_client.py test/test_control_resilience.py
git commit -m "fix(urban): send_cmd reports network failure instead of raising"
```

---

## Self-Review

**Spec coverage:** There is no external spec; the "spec" is the defect described above. Every element of it is covered: `send_cmd` raising (Task 1), `send_cmd` returning success on a 5xx (Task 1's `test_send_cmd_treats_5xx_as_failure_without_raising`), the halt call silently swallowing failure (Task 2), and the duplicate `erc_urban` copy (Task 3).

**Placeholder scan:** No `TBD`/`implement later`/`add error handling` phrasing; every step has real code and a real command. Task 3 points at the Task 1/Task 2 code blocks instead of repeating them, because the two packages must end up with byte-identical bodies; the blocks themselves are written out in full in Tasks 1 and 2.

**Type consistency:** `send_cmd` returns `bool` everywhere it's touched (Task 1's implementation, Task 2's `_halt` consuming it, Task 3's mirror). `_halt`'s and `_cmd`'s signatures (`(client: RoverClient, ...) -> None`) are unchanged from their current form in both packages, so no caller elsewhere in `run.py` needs updating — confirmed by grep: every call site (`_halt(client)` / `_cmd(client, lin, ang)`) ignores the return value already.

**Review Focus:** all five items above have an owning task and test: #1 and #3 in Task 1, #2 and #4 in Task 2 (plus Task 1's regression re-run), #5 is a documented limitation with no task (correctly — it's out of scope, not a gap).
