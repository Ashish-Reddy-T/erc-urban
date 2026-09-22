"""
The rolling local map — the stack's memory of SPACE.

Without it the rover re-derived the world from one image every tick and
immediately forgot it, so an obstacle that left the frame stopped existing. That
is exactly the information needed to say "the thing is on my left, the goal is
past it, so I go right and KEEP going right"; its absence produced eight
consecutive escapes at +116, +113, +134, +130, -160, +112, -165, +101 degrees
while the rover never left one spot.

A log-odds occupancy grid in a world frame anchored at the first update, with a
rolling window that recentres on the rover. Deliberately NOT SLAM: no loop
closure, no global consistency claim, and honest about drifting.

FRAME
-----
World X = North, Y = West, theta CCW-positive — the right-handed frame implied
by `geo.geodesic_bearing_range`, whose yaw is CW-from-North. So `theta = -yaw`.
Robot frame stays (x forward, y left).

POSE: TRANSLATION FROM VO, HEADING FROM THE COMPASS
---------------------------------------------------
Not a preference. Accumulated VO yaw came out ~3x LOW over a 360 deg spin, so
integrating it would rotate the map against the world at a third of true rate
and every obstacle laid down while turning would land at the wrong bearing — the
exact defect this module exists to remove. The compass drifts (-141 deg then
back to -7 deg over one run) but does not ACCUMULATE: bounded error beats
unbounded. VO dyaw is the fallback for ticks with no compass reading.

LOG-ODDS, NOT BOOLEAN
---------------------
Depth Anything hallucinates, and a boolean map makes one bad frame permanent.
L_OCC and OCC_THRESH are set so a cell needs two consistent observations before
it can veto anything, and free-space evidence can erase it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# One observation of an obstacle / of free space, in log-odds.
L_OCC = 0.85
L_FREE = -0.40
L_CLAMP = 4.0
# Believe "occupied" only at two consistent hits (2 * 0.85 = 1.7).
OCC_THRESH = 1.2
# Anything this close to zero has never been meaningfully observed.
UNKNOWN_BAND = 0.20


@dataclass
class PolarScan:
    """What the accumulated map says in each direction from the rover."""

    bin_centers: np.ndarray      # (n,) radians, robot frame, left-positive
    clearance_m: np.ndarray      # (n,) distance to first occupied cell
    unknown_frac: np.ndarray     # (n,) fraction of the ray that is unobserved
    max_range_m: float

    def describe(self) -> str:
        i = int(np.argmax(self.clearance_m))
        return (f"scan[best {np.degrees(self.bin_centers[i]):+.0f}deg "
                f"{self.clearance_m[i]:.1f}m]")


class LocalMap:
    """
    Odometry-anchored rolling occupancy map.

    Cheap on purpose — a 160x160 float32 grid and a vectorised ray-cast, no
    network and no extra frame — because it shares a ~1.5 Hz loop with LogoNav,
    SegFormer and Depth Anything.
    """

    def __init__(
        self,
        cell_m: float = 0.15,
        extent_m: float = 24.0,
        recentre_margin_m: float = 4.0,
        rover_radius_m: float = 0.35,
    ):
        self.cell_m = float(cell_m)
        self.n = max(8, int(round(extent_m / cell_m)))
        self.recentre_margin_cells = max(1, int(recentre_margin_m / cell_m))
        self.rover_radius_m = rover_radius_m
        # grid[iy, ix]: iy indexes world Y, ix indexes world X.
        self.grid = np.zeros((self.n, self.n), np.float32)
        self.driven = np.zeros((self.n, self.n), bool)
        # World coordinate of the (0, 0) cell's lower corner.
        self.origin_x = -0.5 * self.n * self.cell_m
        self.origin_y = -0.5 * self.n * self.cell_m
        # Pose in the world frame.
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self._have_yaw = False
        self.travelled_m = 0.0
        self.updates = 0

    # ---- frames -----------------------------------------------------------
    def robot_to_world(self, xy: np.ndarray) -> np.ndarray:
        """(N,2) robot-frame (x_fwd, y_left) -> (N,2) world."""
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        c, s = math.cos(self.theta), math.sin(self.theta)
        return np.stack(
            [self.x + c * xy[:, 0] - s * xy[:, 1],
             self.y + s * xy[:, 0] + c * xy[:, 1]],
            axis=1,
        )

    def world_to_cell(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = np.asarray(w, np.float64).reshape(-1, 2)
        ix = np.floor((w[:, 0] - self.origin_x) / self.cell_m).astype(np.int64)
        iy = np.floor((w[:, 1] - self.origin_y) / self.cell_m).astype(np.int64)
        return ix, iy

    def _in_bounds(self, ix, iy):
        return (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)

    # ---- pose -------------------------------------------------------------
    def update_pose(self, odom=None, yaw: Optional[float] = None) -> None:
        """
        Advance the pose by one tick.

        Order matters: the odometry delta was measured in the PREVIOUS robot
        frame, so rotate it by the OLD heading before replacing the heading.
        The other way applies this tick's rotation to last tick's translation
        and smears every observation along a turn.
        """
        if odom is not None and getattr(odom, "confidence", 0.0) > 0.0:
            c, s = math.cos(self.theta), math.sin(self.theta)
            dx, dy = float(odom.dx), float(odom.dy)
            self.x += c * dx - s * dy
            self.y += s * dx + c * dy
            self.travelled_m += float(np.hypot(dx, dy))

        if yaw is not None and np.isfinite(yaw):
            # Compass: absolute, bounded error. See the module docstring.
            self.theta = _wrap(-float(yaw))
            self._have_yaw = True
        elif odom is not None and getattr(odom, "confidence", 0.0) > 0.0:
            # Stopgap for a compass dropout only: VO rotation is under-scaled.
            self.theta = _wrap(self.theta + float(odom.dyaw))

        self._recentre()

    def _recentre(self) -> None:
        """Roll the window so the rover stays near the middle of the grid."""
        ix, iy = self.world_to_cell(np.array([[self.x, self.y]]))
        cx = cy = self.n // 2
        kx, ky = int(ix[0]) - cx, int(iy[0]) - cy
        if abs(kx) < self.recentre_margin_cells and abs(ky) < self.recentre_margin_cells:
            return
        for arr in (self.grid, self.driven):
            arr[...] = np.roll(arr, (-ky, -kx), axis=(0, 1))
        # Zero the strips rolled in from the far side: wrapped garbage.
        if kx > 0:
            self.grid[:, -kx:] = 0.0
            self.driven[:, -kx:] = False
        elif kx < 0:
            self.grid[:, :-kx] = 0.0
            self.driven[:, :-kx] = False
        if ky > 0:
            self.grid[-ky:, :] = 0.0
            self.driven[-ky:, :] = False
        elif ky < 0:
            self.grid[:-ky, :] = 0.0
            self.driven[:-ky, :] = False
        self.origin_x += kx * self.cell_m
        self.origin_y += ky * self.cell_m

    # ---- ingest -----------------------------------------------------------
    def integrate(
        self,
        obstacles_robot: Optional[np.ndarray] = None,
        clearance_by_bin: Optional[np.ndarray] = None,
        bin_centers: Optional[np.ndarray] = None,
        max_free_m: float = 6.0,
    ) -> None:
        """
        Fold one tick of perception into the map.

        obstacles_robot: (N,2) robot-frame points from `CostmapResult.occupied`,
            which has had the ground plane removed by RANSAC. Do NOT pass raw
            depth — a 148 mm camera sees a great deal of ground, and the ground
            is not an obstacle.
        clearance_by_bin / bin_centers: the costmap's polar free-space estimate.
            Everything nearer than the clearance is evidence of FREE space,
            which is what lets a hallucinated obstacle decay.
        """
        self.updates += 1

        if clearance_by_bin is not None and bin_centers is not None:
            self._mark_free(
                np.asarray(bin_centers, np.float64),
                np.asarray(clearance_by_bin, np.float64),
                max_free_m,
            )

        if obstacles_robot is not None and len(obstacles_robot):
            w = self.robot_to_world(obstacles_robot)
            ix, iy = self.world_to_cell(w)
            ok = self._in_bounds(ix, iy)
            if ok.any():
                np.add.at(self.grid, (iy[ok], ix[ok]), L_OCC)

        # The rover is standing here: the one observation that cannot be wrong.
        self._mark_driven()
        np.clip(self.grid, -L_CLAMP, L_CLAMP, out=self.grid)

    def _mark_free(self, angles, clearance, max_free_m) -> None:
        n_steps = max(2, int(max_free_m / self.cell_m))
        t = np.linspace(0.0, 1.0, n_steps)[None, :]
        # Stop one cell short: the far end of a ray IS the obstacle, and
        # clearing it would fight the hit written above.
        reach = np.clip(
            np.minimum(np.asarray(clearance, np.float64), max_free_m)
            - self.cell_m,
            0.0, None,
        )[:, None]
        r = t * reach
        a = np.asarray(angles, np.float64)[:, None]
        xs = (r * np.cos(a)).ravel()
        ys = (r * np.sin(a)).ravel()
        w = self.robot_to_world(np.stack([xs, ys], axis=1))
        ix, iy = self.world_to_cell(w)
        ok = self._in_bounds(ix, iy)
        if ok.any():
            np.add.at(self.grid, (iy[ok], ix[ok]), L_FREE)

    def _mark_driven(self) -> None:
        r = int(math.ceil(self.rover_radius_m / self.cell_m))
        ix, iy = self.world_to_cell(np.array([[self.x, self.y]]))
        x0, y0 = int(ix[0]), int(iy[0])
        xa, xb = max(0, x0 - r), min(self.n, x0 + r + 1)
        ya, yb = max(0, y0 - r), min(self.n, y0 + r + 1)
        if xa < xb and ya < yb:
            self.driven[ya:yb, xa:xb] = True
            # Also erases any phantom written here earlier.
            self.grid[ya:yb, xa:xb] = np.minimum(
                self.grid[ya:yb, xa:xb], -L_CLAMP * 0.5
            )

    # ---- query ------------------------------------------------------------
    @property
    def occupied(self) -> np.ndarray:
        return self.grid > OCC_THRESH

    @property
    def unknown(self) -> np.ndarray:
        return np.abs(self.grid) < UNKNOWN_BAND

    def clearance_polar(
        self,
        n_bins: int = 36,
        max_range_m: float = 8.0,
        fov_deg: float = 360.0,
        inflate_m: Optional[float] = None,
    ) -> PolarScan:
        """
        Distance to the first occupied cell in each direction, from the current
        pose, over the ACCUMULATED map.

        The query the whole module is for. `DepthCostmap` can only answer it for
        what is in frame right now; because this map persists, an obstacle the
        rover has turned away from still blocks its bearing — which is what lets
        the explorer commit to a side and go around.

        inflate_m: grow obstacles by this first, so a "gap" is one the rover
            actually fits through. Defaults to the rover radius.
        """
        inflate = self.rover_radius_m if inflate_m is None else inflate_m
        half = math.radians(fov_deg) / 2.0
        if fov_deg >= 359.9:
            edges = np.linspace(-math.pi, math.pi, n_bins + 1)
        else:
            edges = np.linspace(-half, half, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        occ = self._inflated(inflate)
        n_steps = max(2, int(max_range_m / self.cell_m))
        radii = np.linspace(self.cell_m, max_range_m, n_steps)      # (S,)
        # (B, S) sample grid in the robot frame, then to world in one shot.
        rr = radii[None, :]
        aa = centers[:, None]
        xs = (rr * np.cos(aa)).ravel()
        ys = (rr * np.sin(aa)).ravel()
        ix, iy = self.world_to_cell(self.robot_to_world(np.stack([xs, ys], 1)))
        inb = self._in_bounds(ix, iy)
        ixc = np.clip(ix, 0, self.n - 1)
        iyc = np.clip(iy, 0, self.n - 1)

        hit = np.where(inb, occ[iyc, ixc], False).reshape(n_bins, n_steps)
        unk = np.where(inb, self.unknown[iyc, ixc], True).reshape(n_bins, n_steps)

        # First True along each ray; rays that never hit get max_range.
        any_hit = hit.any(axis=1)
        first = np.where(any_hit, hit.argmax(axis=1), n_steps - 1)
        clearance = np.where(any_hit, radii[first], float(max_range_m))

        # Only over the VISIBLE part of the ray: counting cells behind an
        # obstacle as frontier sends the rover at walls with unmapped space
        # behind them.
        steps = np.arange(n_steps)[None, :]
        before = steps <= first[:, None]
        unknown_frac = (unk & before).sum(axis=1) / np.maximum(
            before.sum(axis=1), 1
        )

        return PolarScan(
            bin_centers=centers,
            clearance_m=clearance.astype(np.float64),
            unknown_frac=unknown_frac.astype(np.float64),
            max_range_m=float(max_range_m),
        )

    def _inflated(self, inflate_m: float) -> np.ndarray:
        occ = self.occupied
        r = int(round(inflate_m / self.cell_m))
        if r <= 0 or not occ.any():
            return occ
        try:
            import cv2

            k = 2 * r + 1
            return cv2.dilate(
                occ.astype(np.uint8), np.ones((k, k), np.uint8)
            ).astype(bool)
        except Exception:
            out = occ.copy()
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    out |= np.roll(np.roll(occ, dy, axis=0), dx, axis=1)
            return out

    def describe(self) -> str:
        return (f"map(occ={int(self.occupied.sum())} "
                f"driven={int(self.driven.sum())} "
                f"pose={self.x:+.1f},{self.y:+.1f}@{np.degrees(self.theta):+.0f}°)")


def _wrap(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)
