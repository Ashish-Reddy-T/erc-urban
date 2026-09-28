"""
Issue #9 (offroad: review the nose-up speed boost in speed_policy) asks for
measurement before any behavior change: log pitch_deg, speed and outcome on
the next offroad run, then decide from that data whether the unconditional
1.25x climbing boost in terrain.speed_policy is redundant with climb.py's
ROLLBACK/UNSAFE escalation gate.

auto.py's main loop already logs pitch, tilt, and the commanded speed at
decision time, but the eventual outcome (guard-aborted, wedged, or clean) is
only ever printed to the console, never written to the structured
auto_log.jsonl. That means a completed run cannot be joined on "was this a
climbing tick, and what happened" without re-parsing console output.

This tests the pure classifier that produces the outcome record, so a run's
JSONL can be filtered/joined on tick without touching the network loop or a
live rover.
"""
from erc_offroad.auto import outcome_record


class TestOffroadOutcomeRecord:
    def test_guard_abort_outcome(self):
        rec = outcome_record(tick=3, ok=False, wedged=False, reason="TILT 29.1",
                              d_yaw=0.0, corr=float("nan"), pitch=9.5, lin=0.40,
                              action="forward")
        assert rec["outcome"] == "abort"
        assert rec["reason"] == "TILT 29.1"
        assert rec["tick"] == 3
        assert rec["pitch"] == 9.5
        assert rec["lin"] == 0.40
        assert rec["action"] == "forward"

    def test_wedged_outcome(self):
        rec = outcome_record(tick=7, ok=True, wedged=True, reason=None,
                              d_yaw=1.0, corr=0.9, pitch=10.2, lin=0.45,
                              action="forward")
        assert rec["outcome"] == "wedged"
        assert rec["reason"] is None
        assert rec["d_yaw"] == 1.0
        assert rec["corr"] == 0.9

    def test_clean_outcome(self):
        rec = outcome_record(tick=1, ok=True, wedged=False, reason=None,
                              d_yaw=4.0, corr=0.1, pitch=2.0, lin=0.30,
                              action="turn")
        assert rec["outcome"] == "ok"
        assert rec["reason"] is None

    def test_record_is_json_serializable(self):
        import json
        rec = outcome_record(tick=1, ok=True, wedged=False, reason=None,
                              d_yaw=4.0, corr=0.1, pitch=2.0, lin=0.30,
                              action="turn")
        json.dumps(rec)  # must not raise (NaN corr is the one thing to watch)
