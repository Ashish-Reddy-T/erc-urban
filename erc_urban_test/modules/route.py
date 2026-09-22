"""
Route/GPS conditioning by visual alignment: fuse the GPS goal bearing with the
corridor the camera can see, in one place instead of three ad-hoc blends.

This matters more here than it should. Measured on the live Mini+, lat/lon came
back bit-identical across 24 stationary samples AND across 30 s of commanded
driving. A controller that servos to that is servoing to a constant, so vision
is the only live signal.

The policy — trust vision in proportion to how much of it there is:

  * A wide, confident corridor is the constraint, not a hint. On a single-track
    path the line IS the way through and may point away from the goal for a
    while; the goal only biases the aim point WITHIN it.
  * A weak or absent corridor means no visual opinion: fall back to GPS.
  * The goal's influence is a BOUNDED offset, never a fraction of the bearing.
    Blending 20% of a -134 deg goal parks the aim 27 deg off the corridor
    centre, i.e. outside it, which is how the rover loses the path.
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
    Fuse the GPS goal bearing with the visually-observed corridor.

    Stateless per tick and free of rover I/O, so it is testable without
    hardware and cannot surprise the control loop.
    """

    def __init__(
        self,
        corridor_min_width: float = 0.12,
        corridor_full_width: float = 0.45,
        corridor_open_width: float = 0.97,
        goal_bias_max_deg: float = 8.0,
        goal_bias_gain: float = 0.20,
        track_min_frac: float = 0.04,
        track_sat_frac: float = 0.97,
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
        track_sat_frac: at or above this near-band traversable fraction the
            track centroid carries NO directional information and must not
            steer. `semantics.track` returns the horizontal centroid of
            traversable pixels; when the whole band is traversable that
            centroid is the middle of the frame by construction, i.e. "straight
            ahead", whatever is actually out there. Measured live on grass:
            near=1.00 with the reported bearing pinned at exactly +0 degrees on
            every tick while the goal sat 90 degrees off the bow. Treating that
            as a visual opinion suppressed the align pivot and the rover drove
            straight past the checkpoint.
        turnaround_deg: only above this goal error do we abandon the corridor and
            pivot. Below it, the path is still the way to the goal even if it
            currently points elsewhere.
        """
        self.corridor_min_width = corridor_min_width
        self.corridor_full_width = max(corridor_full_width, corridor_min_width + 1e-6)
        self.corridor_open_width = max(corridor_open_width,
                                       self.corridor_full_width + 1e-6)
        self.goal_bias_max = np.radians(goal_bias_max_deg)
        self.goal_bias_gain = goal_bias_gain
        self.track_min_frac = track_min_frac
        self.track_sat_frac = track_sat_frac
        self.turnaround_deg = turnaround_deg
        self.narrow_speed_scale = narrow_speed_scale

    def confidence(self, corridor_width: float) -> float:
        """
        How much to trust vision, as a function of corridor width.

        Rises over [corridor_min_width, corridor_full_width] — a corridor has to
        be wide enough to be real before it constrains anything — and drops to
        ZERO at `corridor_open_width`, because a corridor that spans essentially
        the whole frame is not a corridor at all.

        That last step is deliberate rather than a ramp, because the property it
        stands for is binary: a corridor is defined by its EDGES. `corridor()`
        returns the longest run of drivable columns over the frame width, so a
        value at 1.0 means no column anywhere was closed — there is no edge, and
        therefore nothing to follow. A width of 0.85 has 15% of columns closed;
        that is a real, wide corridor and it keeps full confidence.

        Why it matters, measured live on grass: every column read open, width
        came back 1.00, this returned 1.00, and the corridor policy then let the
        goal move the aim by at most `goal_bias_max_deg`. Logged as
        src=corridor(1.00@+8deg) with the goal at +90deg and cmd ang=0.02 —
        about 2 deg/s against a 90 degree error. The rover drove a straight line
        past the checkpoint, range 24.4 m -> 30.5 m, every gate reporting clear.

        At confidence 0 the caller falls back to `_aligning()`, which pivots to
        face the goal and then drives: the right move on an open field.
        """
        if corridor_width <= self.corridor_min_width:
            return 0.0
        if corridor_width >= self.corridor_open_width:
            return 0.0
        span = self.corridor_full_width - self.corridor_min_width
        return float(np.clip(
            (corridor_width - self.corridor_min_width) / span, 0.0, 1.0))

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

        # Goal genuinely behind us: no corridor is worth following backwards.
        # Gated on `goal_trusted` — with frozen GPS and a drifting compass the
        # bearing sat at +-177 deg with the SIGN flipping between ticks, so the
        # rover reversed its pivot every burst and span in place forever.
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
        track_usable = (
            track_bearing is not None
            and (track_frac or 0.0) >= self.track_min_frac
            and (track_frac or 0.0) < self.track_sat_frac
        )
        if not goal_trusted and track_usable:
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

        # No corridor, but traversable surface is visible: its centroid is a
        # weak prior. Weaker because a centroid cannot tell "gap in the middle"
        # from "grass in the middle" -- why semantics.corridor() exists.
        if track_usable:
            w = 0.5
            return RouteAim(bearing=_wrap((1.0 - w) * goal_bearing + w * track_bearing),
                            confidence=w, source="track",
                            corridor_width=corridor_width)

        # Nothing visual to say. GPS it is -- as a direction, not a target.
        return RouteAim(bearing=_wrap(goal_bearing), confidence=0.0, source="gps",
                        corridor_width=corridor_width)
