"""
When to claim a checkpoint.

The SDK accepts a claim within ~15 m, and the rover used to claim the moment
it got there. But checkpoints sit at convenient corners of the route, and
cutting them 15 m short makes the straight line to the next one run into
whatever the corner went round. Wuhan mission-3: CP8 claimed from 15 m out,
and the leg to CP9 went straight at a staircase wall.

So drive in to near_m before claiming, except for the first checkpoint (the
rover starts there; nothing to set up) and the last one (no next leg): those
claim anywhere inside the SDK radius (user, 2026-09-25; middle ones at 5 m). GPS here is good to a few metres, so do not circle chasing
it: claim anyway once, inside the SDK radius, the rover has DRIVEN stall_m
without getting stall_gain_m closer, has spent stall_s there without getting
closer, after max_s in the zone, or if GPS goes stale there. Distance, not
time, is the main test: turning round to face the checkpoint takes ~20 s of
pivots that bring it no closer (Wuhan, the first run of this rule claimed at
5 m that way).
"""
from __future__ import annotations

from typing import Optional


class ApproachClaim:
    def __init__(self, near_m: float = 2.0, claim_m: float = 15.0,
                 stall_s: float = 45.0, stall_gain_m: float = 0.5, max_s: float = 90.0,
                 stall_m: float = 4.0):
        self.stall_m = stall_m
        self.near_m = near_m
        self.claim_m = claim_m
        self.stall_s = stall_s
        self.stall_gain_m = stall_gain_m
        self.max_s = max_s
        self.is_last = False
        self.is_first = False
        self.reset()

    def start_goal(self, is_last: bool, is_first: bool = False) -> None:
        self.is_last = is_last
        self.is_first = is_first
        self.reset()

    def reset(self) -> None:
        self._entered: Optional[float] = None
        self._best = float("inf")
        self._improved_at: Optional[float] = None
        self._improved_travel = 0.0

    def update(self, rng_m: float, now: float, last: Optional[bool] = None,
               gps_stale: bool = False, travelled_m: float = 0.0) -> tuple[bool, str]:
        """(claim now?, why). last defaults to what start_goal() was told."""
        last = self.is_last if last is None else last
        if not rng_m <= self.claim_m:                   # also False for NaN
            self.reset()
            return False, ""
        if last or self.is_first:
            return True, (f"{'last' if last else 'first'} checkpoint, "
                          f"within {self.claim_m:.0f} m")
        if rng_m <= self.near_m:
            return True, f"within {self.near_m:.1f} m"
        if gps_stale:
            return True, "GPS stale inside the claim radius"
        if self._entered is None:
            self._entered, self._best, self._improved_at = now, rng_m, now
            self._improved_travel = travelled_m
        elif rng_m <= self._best - self.stall_gain_m:
            self._best, self._improved_at = rng_m, now
            self._improved_travel = travelled_m
        if travelled_m - self._improved_travel >= self.stall_m:
            return True, (f"drove {self.stall_m:.0f} m without getting closer than "
                          f"{self._best:.1f} m (GPS noise or blocked)")
        if now - self._improved_at >= self.stall_s:
            return True, (f"no closer than {self._best:.1f} m for {self.stall_s:.0f} s "
                          f"(GPS noise or blocked)")
        if now - self._entered >= self.max_s:
            return True, f"{self.max_s:.0f} s inside the claim radius"
        return False, ""

    def describe(self, rng_m: float, now: float) -> str:
        if self._entered is None or not rng_m <= self.claim_m:
            return ""
        return (f"approach {rng_m:.1f}->{self.near_m:.0f}m "
                f"best={self._best:.1f} stall={now - self._improved_at:.0f}s")
