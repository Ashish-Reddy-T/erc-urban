
"""
Module 2 — route/GPS conditioning by visual alignment.

plan.md: take UrbanVLA's route-visual alignment (arXiv 2510.23576) — align the
noisy GPS/route waypoints against what the camera actually sees instead of
treating GPS as a hard target. Feed the checkpoint as a soft directional prior
fused with vision; never servo to a raw fix.

WHY THIS MATTERS MORE HERE THAN THE PAPER SUGGESTS
--------------------------------------------------
The urban track's 15 m tolerance exists because consumer GPS in a street canyon
is bad. On this rover it is worse than "noisy": measured on the live Mini+,
`latitude`/`longitude` came back BIT-IDENTICAL across 24 stationary samples and
again across 30 s of commanded driving, while the IMU updated normally. Whatever
the cause, a controller that servos to that number is servoing to a constant.
Vision is the signal that is actually live.

So the fusion here is not a refinement, it is the difference between a rover
that follows the visible path and one that grinds toward a stale fix.

WHAT THIS REPLACES
------------------
run.py currently blends GPS against vision at three separate sites with three
different rules:

  * `TRACK_BLEND` (run.py:971)      - lerp bearing toward the track centroid
  * `CORRIDOR_GOAL_BIAS` (run.py:980) - corridor overrides, goal biases within it
  * `_steer_nudge` (run.py:231)     - proportional yaw-rate correction

Each was tuned against a different failure and they interact in ways nobody can
hold in their head. This consolidates the *policy* into one tested object; the
callers keep their own gains.

THE POLICY
----------
Trust vision in proportion to how much of it there is:

  * A wide, confident corridor is the constraint, not a hint. On a single-track
    path the line IS the way through, and it may point away from the goal for a
    while. Follow it, and let the goal bias the aim point WITHIN it.
  * A weak or absent corridor means we have no visual opinion, so fall back to
    the GPS bearing.
  * The goal's influence is a BOUNDED offset, never a fraction of the bearing.
    Blending 20% of a -134 deg goal parks the aim 27 deg off the corridor centre
    -- i.e. outside the corridor, which is how the rover loses the path. This is
    the bug run.py:984 already documents; here it is enforced by construction.
  * Turn around for the goal only when it is genuinely behind us.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class RouteAim:
    """The fused steering target for this tick."""

    bearing: float          # radians, robot frame, left-positive
    confidence: float       # 0 = pure GPS, 1 = pure vision
    source: str             # "corridor" | "track" | "gps" | "turnaround"
    speed_scale: float = 1.0
    corridor_width: float = 0.0

    def describe(self) -> str:
        return (f"route[{self.source} conf={self.confidence:.2f} "
                f"aim={np.degrees(self.bearing):+.0f}deg]")


class RouteAligner:
    """
    M2: fuse the GPS goal bearing with the visually-observed corridor.

    Stateless per tick and free of any rover I/O, so it is testable without
    hardware and cannot surprise the control loop.
    """

    def __init__(
        self,
        corridor_min_width: float = 0.12,
        corridor_full_width: float = 0.45,
        goal_bias_max_deg: float = 8.0,
        goal_bias_gain: float = 0.20,
        track_min_frac: float = 0.04,
        turnaround_deg: float = 120.0,
        narrow_speed_scale: float = 0.7,
    ):
        """
        corridor_min_width / corridor_full_width: corridor width, as a fraction
            of frame width, at which vision starts to be trusted and at which it
            is trusted fully. Ramping between them avoids a hard mode flip that
            would make the rover twitch as a corridor comes and goes.
        goal_bias_max_deg: hard cap on how far the goal may pull the aim off the
            corridor centre. This is the load-bearing bound described above.
        turnaround_deg: only above this goal error do we abandon the corridor and
            pivot. Below it, the path is still the way to the goal even if it
            currently points elsewhere.
        """
        self.corridor_min_width = corridor_min_width
        self.corridor_full_width = max(corridor_full_width, corridor_min_width + 1e-6)
        self.goal_bias_max = np.radians(goal_bias_max_deg)
        self.goal_bias_gain = goal_bias_gain
        self.track_min_frac = track_min_frac
        self.turnaround_deg = turnaround_deg
        self.narrow_speed_scale = narrow_speed_scale

    def confidence(self, corridor_width: float) -> float:
        """How much to trust vision, ramped over the corridor width."""
        if corridor_width <= self.corridor_min_width:
            return 0.0
        span = self.corridor_full_width - self.corridor_min_width
        return float(np.clip((corridor_width - self.corridor_min_width) / span, 0.0, 1.0))

    def align(
        self,
        goal_bearing: float,
        corridor_bearing: Optional[float] = None,
        corridor_width: float = 0.0,
        track_bearing: Optional[float] = None,
        track_frac: Optional[float] = None,
        goal_trusted: bool = True,
    ) -> RouteAim:
        """
        goal_bearing: from GPS, robot frame, left-positive (geo.py convention).
        corridor_bearing / corridor_width: from semantics.corridor().
        track_bearing / track_frac: from semantics.track(), the weaker fallback
            when there is traversable surface but no clean corridor.
        goal_trusted: False when the GPS fix is stale or the heading is
            unreliable. Then the goal bearing is not a noisy measurement of the
            right answer, it is not a measurement at all, and must not be
            allowed to steer -- let alone to trigger a turnaround.
        """
        goal_err_deg = abs(np.degrees(goal_bearing))

        # The goal is genuinely behind us: no corridor is worth following
        # backwards. Pivot, and let the caller's align logic take over.
        #
        # Gated on `goal_trusted` because a turnaround driven by a bad bearing
        # is the worst possible failure: observed live with frozen GPS and a
        # drifting compass, the bearing sat at +-177 deg and the SIGN flipped
        # between ticks, so the rover reversed its pivot direction every burst
        # and span in place indefinitely without ever driving.
        if goal_trusted and goal_err_deg > self.turnaround_deg:
            return RouteAim(bearing=_wrap(goal_bearing), confidence=0.0,
                            source="turnaround", speed_scale=0.0,
                            corridor_width=corridor_width)

        conf = self.confidence(corridor_width) if corridor_bearing is not None else 0.0

        # With no usable goal, vision is the ONLY input. Follow the corridor
        # exactly -- there is no goal to bias toward.
        if not goal_trusted and conf > 0.0 and corridor_bearing is not None:
            speed = (self.narrow_speed_scale
                     if corridor_width < 2.0 * self.corridor_min_width else 1.0)
            return RouteAim(bearing=_wrap(corridor_bearing), confidence=conf,
                            source="corridor(no-gps)", speed_scale=speed,
                            corridor_width=corridor_width)
        if not goal_trusted and track_bearing is not None and (track_frac or 0) >= self.track_min_frac:
            return RouteAim(bearing=_wrap(track_bearing), confidence=0.5,
                            source="track(no-gps)", corridor_width=corridor_width)

        if conf > 0.0 and corridor_bearing is not None:
            # Corridor is the constraint; the goal only biases WITHIN it, by a
            # bounded offset. Never a fraction of the raw bearing.
            bias = float(np.clip(self.goal_bias_gain * goal_bearing,
                                 -self.goal_bias_max, self.goal_bias_max))
            aim = _wrap(corridor_bearing + bias)
            # Blend toward the GPS bearing when the corridor is only partly
            # trusted, so confidence ramps the behaviour rather than switching it.
            aim = _wrap((1.0 - conf) * goal_bearing + conf * aim)
            speed = (self.narrow_speed_scale
                     if corridor_width < 2.0 * self.corridor_min_width else 1.0)
            return RouteAim(bearing=aim, confidence=conf, source="corridor",
                            speed_scale=speed, corridor_width=corridor_width)

        # No corridor, but traversable surface is visible: use its centroid as a
        # weak visual prior. Weaker than a corridor because a centroid cannot
        # tell "gap in the middle" from "grass in the middle" -- the reason
        # semantics.corridor() exists at all.
        if (
            track_bearing is not None
            and track_frac is not None
            and track_frac >= self.track_min_frac
        ):
            w = 0.5
            return RouteAim(bearing=_wrap((1.0 - w) * goal_bearing + w * track_bearing),
                            confidence=w, source="track",
                            corridor_width=corridor_width)

        # Nothing visual to say. GPS it is -- as a direction, not a target.
        return RouteAim(bearing=_wrap(goal_bearing), confidence=0.0, source="gps",
                        corridor_width=corridor_width)
