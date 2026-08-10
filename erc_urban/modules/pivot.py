"""
Dead-time tolerant in-place pivot.

Measured against the live SDK on this rover:
    * /data answers at ~17 Hz but the payload only REFRESHES at 2.0 Hz
    * every reading is 0.80-1.44 s old (median 1.09 s)

So a proportional yaw loop is closing around ~1.1 s of dead time. Any gain
large enough to break stiction rotates the rover 60-100 deg before the first
evidence of motion arrives, which is why the naive controller limit-cycled
(+73 -> -84 -> -115 -> ...) and never drove anywhere.

The fix is to stop closing the loop at tick rate. One burst, open loop, sized
from a measured yaw rate; then halt and stay halted until the telemetry
pipeline has certainly flushed; then measure what actually happened, update
the rate estimate, and repeat. Converges in 2-3 bursts and cannot oscillate,
because no correction is ever issued against stale data.
"""
from __future__ import annotations

from typing import Optional


def wrap180(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


class PulsePivot:
    def __init__(
        self,
        ang: float = 0.45,
        rate_prior: float = 30.0,
        settle_s: float = 1.8,
        min_burst: float = 0.35,
        max_burst: float = 1.60,
        aim: float = 0.75,
        tol_deg: float = 22.0,
        ang_floor: float = 0.25,
    ):
        self.ang = ang
        # Stiction floor: below this the tracks do not break loose at all.
        self.ang_floor = min(ang_floor, ang)
        # deg/s produced by |ang|. Prior only — replaced by measurement.
        self.rate = rate_prior
        self.settle_s = settle_s
        self.min_burst = min_burst
        self.max_burst = max_burst
        # Deliberately undershoot: overshoot costs a full settle to undo.
        self.aim = aim
        self.tol_deg = tol_deg
        self.calibrated = False

        self._phase = "idle"
        self._burst_end = 0.0
        self._settle_end = 0.0
        self._cmd_ang = 0.0
        self._yaw0: Optional[float] = None
        self._ts0: Optional[float] = None
        self._dur = 0.0
        self._mag = ang
        self._burst_start = 0.0
        self._actual = 0.0
        self.bursts = 0
        # Completed burst+settle cycles. The caller needs this because the
        # pivot transitions settle->idle->burst inside a single command() call,
        # so busy() is never observed False between cycles — a one-shot turn
        # cannot be detected any other way.
        self.cycles = 0

    def busy(self) -> bool:
        """
        True while a burst/settle cycle is in flight.

        The caller must not preempt this. The bearing it sees is ~1.1 s old, so
        mid-burst it still reflects the pre-burst heading; releasing control on
        that reading aborts the pivot before it can measure anything, and the
        rover oscillates forever without ever calibrating.
        """
        return self._phase in ("burst", "settle")

    def reset(self) -> None:
        self._phase = "idle"
        self._yaw0 = None
        self._ts0 = None

    def command(self, err_deg: float, tel, now: float) -> tuple[float, float, str]:
        """
        err_deg: bearing error, left-positive, degrees.
        Returns (linear, angular, note). Linear is always 0 — this pivots only.
        """
        if self._phase == "burst":
            if now < self._burst_end:
                return 0.0, self._cmd_ang, f"burst {self._burst_end - now:.1f}s"
            # Real length, not requested: the loop tick quantises it upward and
            # learning from the requested value biases the rate estimate.
            self._actual = max(now - self._burst_start, 1e-3)
            self._phase = "settle"
            self._settle_end = now + self.settle_s
            return 0.0, 0.0, "burst done"

        if self._phase == "settle":
            if now < self._settle_end:
                return 0.0, 0.0, f"settle {self._settle_end - now:.1f}s"
            # Require a genuinely newer hardware sample, not just wall time.
            if (
                self._ts0 is not None
                and tel.hw_ts is not None
                and tel.hw_ts <= self._ts0
            ):
                return 0.0, 0.0, "settle (awaiting fresh telemetry)"
            self._learn(tel)
            self._phase = "idle"
            self.cycles += 1

        # idle: plan the next burst
        if abs(err_deg) <= self.effective_tol():
            return 0.0, 0.0, "aligned"

        need = abs(err_deg) * self.aim
        # A burst cannot be shorter than one control tick, so the smallest turn
        # available at full command is rate*min_burst. To turn less than that,
        # weaken the command instead of shortening an unshortenable burst.
        min_turn_full = max(self.rate, 1.0) * self.min_burst
        if need >= min_turn_full:
            dur = min(self.max_burst, need / max(self.rate, 1.0))
            mag = self.ang
        else:
            dur = self.min_burst
            mag = max(self.ang_floor, self.ang * need / min_turn_full)
        direction = 1.0 if err_deg >= 0.0 else -1.0
        self._cmd_ang = direction * mag
        self._mag = mag
        self._dur = dur
        self._burst_start = now
        self._burst_end = now + dur
        self._yaw0 = tel.yaw
        self._ts0 = tel.hw_ts
        self._phase = "burst"
        self.bursts += 1
        return (
            0.0,
            self._cmd_ang,
            f"burst {dur:.2f}s @{mag:.2f} for {err_deg:+.0f}deg "
            f"({self.rate:.0f}deg/s, tol {self.effective_tol():.0f}deg)",
        )

    def effective_tol(self) -> float:
        """
        Finest heading the pivot can actually resolve.

        One tick at the stiction floor is the smallest turn it can produce;
        asking for better than that guarantees an endless hunt. Anything
        finer is left to the forward controller, which steers continuously
        and corrects heading far more precisely than stop-start pivoting.
        """
        finest = (self.rate / max(self.ang, 1e-6)) * self.ang_floor * self.min_burst
        return max(self.tol_deg, 0.8 * finest)

    def _learn(self, tel) -> None:
        """Turn one completed burst into a yaw-rate estimate (deg/s at |ang|)."""
        if self._yaw0 is None or tel.yaw is None or self._actual < 0.25:
            return
        import numpy as np

        moved = abs(wrap180(np.degrees(tel.yaw) - np.degrees(self._yaw0)))
        if moved < 2.0:
            # No response: keep the prior rather than learn a rate of zero,
            # which would make every future burst hit max_burst.
            return
        # Normalise back to the rate at full command, since weak bursts use a
        # reduced magnitude.
        measured = (moved / self._actual) * (self.ang / max(self._mag, 1e-6))
        if not (2.0 <= measured <= 200.0):
            return
        self.rate = measured if not self.calibrated else (
            0.6 * self.rate + 0.4 * measured
        )
        self.calibrated = True
