"""
GPS -> robot-frame goal geometry (Module 2, geometric part).

Given the rover's (lat, lon, yaw) and a checkpoint (lat, lon), compute the
goal's bearing and range in the ROBOT frame. This is the "soft directional
prior" that UrbanVLA-style route-visual alignment later refines — the
baseline uses it directly; the upgrade fuses it with vision instead of
trusting it.

NOTE the 15 m tolerance: a checkpoint counts as reached within ~15 m, and
consumer GPS in urban canyons is noisy, so treat range as approximate and
NEVER hard-servo to a raw fix.
"""
from __future__ import annotations
import math

_R_EARTH = 6_371_000.0  # m


def geodesic_bearing_range(lat0, lon0, yaw, lat1, lon1):
    """Return (bearing_rad_robot_frame, range_m). yaw may be None."""
    p0, p1 = math.radians(lat0), math.radians(lat1)
    dlon = math.radians(lon1 - lon0)
    # great-circle initial bearing (0 = North, CW)
    y = math.sin(dlon) * math.cos(p1)
    x = math.cos(p0) * math.sin(p1) - math.sin(p0) * math.cos(p1) * math.cos(dlon)
    bearing_north = math.atan2(y, x)
    # haversine range
    dlat = p1 - p0
    a = math.sin(dlat / 2) ** 2 + math.cos(p0) * math.cos(p1) * math.sin(dlon / 2) ** 2
    rng = 2 * _R_EARTH * math.asin(min(1.0, math.sqrt(a)))
    # into robot frame, LEFT-POSITIVE (CCW from forward), matching
    # waypoints (x_forward, y_left) and atan2(y, x) in the baseline planner.
    # bearing_north and yaw are both CW-from-North; relative CW is
    # (bearing_north - yaw); negate for left-positive.
    if yaw is None:
        return _wrap(-bearing_north), rng
    return _wrap(-(bearing_north - yaw)), rng


def reached(rng_m: float, tol_m: float = 15.0) -> bool:
    return rng_m <= tol_m


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi