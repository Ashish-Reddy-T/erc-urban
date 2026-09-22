"""
Mission-specific route shaping.

The SDK gives sparse GPS checkpoints, but an urban rover must usually follow the
walkable surface beside them rather than the straight chord to the next fix. So
keep the SCORING checkpoint separate from the STEERING point: the server still
validates the real checkpoint while the controller sees a short lookahead target
on a sidewalk rail.

Only the known Rware Ward mission is enabled by default; other missions fall
through unchanged unless ROUTE_PROFILE selects one.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

_R_EARTH = 6_371_000.0


@dataclass(frozen=True)
class RouteTarget:
    lat: float
    lon: float
    source: str
    lookahead_m: float
    lateral_offset_m: float
    cross_track_m: float

    def describe(self) -> str:
        return (
            f"route_profile={self.source} look={self.lookahead_m:.1f}m "
            f"offset={self.lateral_offset_m:+.1f}m "
            f"xt={self.cross_track_m:+.1f}m"
        )


class SidewalkRoute:
    """
    Shift checkpoint geometry onto a sidewalk rail and pick a local lookahead.

    Works in a local tangent plane, accurate to centimetres over this mission's
    sub-kilometre span.
    """

    def __init__(
        self,
        goals: Sequence[tuple[float, float]],
        *,
        origin: Optional[tuple[float, float]] = None,
        lateral_normal_en: tuple[float, float],
        offset_m: float = 3.0,
        lookahead_m: float = 10.0,
        source: str = "sidewalk",
    ):
        if not goals:
            raise ValueError("SidewalkRoute requires at least one goal")
        self.goals = [(float(a), float(b)) for a, b in goals]
        self.origin = origin or self.goals[0]
        normal = np.asarray(lateral_normal_en, dtype=np.float64)
        n = float(np.linalg.norm(normal))
        if n <= 1e-9:
            raise ValueError("lateral normal must be non-zero")
        self.normal = normal / n
        self.offset_m = float(offset_m)
        self.lookahead_m = float(lookahead_m)
        self.source = source
        self.raw_xy = np.asarray([self._to_xy(p) for p in self.goals], dtype=np.float64)
        self.route_xy = self.raw_xy + self.normal[None, :] * self.offset_m

    def describe(self) -> str:
        return (
            f"{self.source}(goals={len(self.goals)} "
            f"offset={self.offset_m:+.1f}m look={self.lookahead_m:.1f}m)"
        )

    def target(self, goal_index: int, lat: float, lon: float) -> Optional[RouteTarget]:
        if goal_index < 0 or goal_index >= len(self.goals):
            return None
        cur = np.asarray(self._to_xy((float(lat), float(lon))), dtype=np.float64)
        if goal_index == 0:
            xy = self.route_xy[0]
            cross = float(np.dot(cur - self.raw_xy[0], self.normal))
            lat_t, lon_t = self._to_latlon(xy)
            return RouteTarget(
                lat_t, lon_t, self.source, 0.0, self.offset_m, cross
            )

        a = self.route_xy[goal_index - 1]
        b = self.route_xy[goal_index]
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= 1e-6:
            xy = b
            cross = 0.0
            look = 0.0
        else:
            unit = seg / seg_len
            along = float(np.clip(np.dot(cur - a, unit), 0.0, seg_len))
            look = min(self.lookahead_m, seg_len - along)
            xy = a + unit * (along + look)
            rel = cur - a
            cross = float(unit[0] * rel[1] - unit[1] * rel[0])
        lat_t, lon_t = self._to_latlon(xy)
        return RouteTarget(
            lat_t, lon_t, self.source, look, self.offset_m, cross
        )

    def _to_xy(self, p: tuple[float, float]) -> tuple[float, float]:
        lat, lon = p
        lat0, lon0 = self.origin
        x = math.radians(lon - lon0) * _R_EARTH * math.cos(math.radians(lat0))
        y = math.radians(lat - lat0) * _R_EARTH
        return x, y

    def _to_latlon(self, xy: np.ndarray) -> tuple[float, float]:
        lat0, lon0 = self.origin
        lat = lat0 + math.degrees(float(xy[1]) / _R_EARTH)
        lon = lon0 + math.degrees(
            float(xy[0]) / (_R_EARTH * math.cos(math.radians(lat0)))
        )
        return lat, lon


def _rware_sidewalk_normal() -> tuple[float, float]:
    """
    Right side of the outbound Rware Ward road axis, in local east/north metres.

    `build_route_profile` applies a signed offset along this normal; live
    verification put the sidewalk on the negative side. Return legs reuse the
    same absolute rail.
    """
    cp1 = (-0.4144954681, 36.95953369)
    cp2 = (-0.4138350785, 36.95928192)
    origin = cp1

    def xy(p):
        lat, lon = p
        x = math.radians(lon - origin[1]) * _R_EARTH * math.cos(math.radians(origin[0]))
        y = math.radians(lat - origin[0]) * _R_EARTH
        return np.array([x, y], dtype=np.float64)

    v = xy(cp2) - xy(cp1)
    right = np.array([v[1], -v[0]], dtype=np.float64)
    return tuple(right / float(np.linalg.norm(right)))


def _looks_like_rware(goals: Sequence[tuple[float, float]]) -> bool:
    if not goals:
        return False
    lats = [float(p[0]) for p in goals]
    lons = [float(p[1]) for p in goals]
    return (
        min(lats) > -0.416
        and max(lats) < -0.408
        and min(lons) > 36.956
        and max(lons) < 36.961
    )


def build_route_profile(
    goals: Sequence[tuple[float, float]],
    profile: Optional[str] = None,
) -> Optional[SidewalkRoute]:
    """
    Build a route profile for the current mission, or None for raw checkpoints.

    ROUTE_PROFILE values:
      * auto / unset: enable only for the known Rware Ward mission
      * off / none / 0: raw checkpoint navigation
      * rware_sidewalk: force the Rware sidewalk rail
    """
    prof = (profile or os.environ.get("ROUTE_PROFILE", "auto")).strip().lower()
    if prof in {"0", "off", "none", "raw"}:
        return None
    if prof not in {"auto", "rware", "rware_sidewalk"}:
        raise ValueError(f"unknown ROUTE_PROFILE={prof!r}")
    if prof == "auto" and not _looks_like_rware(goals):
        return None

    # Measured: the rover's sidewalk fix sat west of the checkpoint chord.
    # SIDEWALK_SIDE overrides if a mission starts on the opposite side.
    side = os.environ.get("SIDEWALK_SIDE", "left").strip().lower()
    sign = -1.0 if side in {"left", "west"} else 1.0
    offset = sign * float(os.environ.get("SIDEWALK_OFFSET_M", "3.0"))
    lookahead = float(os.environ.get("ROUTE_LOOKAHEAD_M", "10.0"))
    return SidewalkRoute(
        goals,
        lateral_normal_en=_rware_sidewalk_normal(),
        offset_m=offset,
        lookahead_m=lookahead,
        source="rware_sidewalk",
    )
