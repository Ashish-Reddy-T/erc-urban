"""
Ground-relative obstacle costmap (the geometric veto).

Rather than guessing which pixels are ground — a fixed image band is
structurally blind to anything low and close, which is most of what can stop
this rover — compute it:

    depth -> 3D points (camera frame)
    -> RANSAC ground plane
    -> per-point height above that plane
    -> anything standing above it is an obstacle, whatever image row it is in
    -> rasterise into a robot-frame occupancy grid
    -> veto any candidate that drives through occupied cells

Geometry needs no weights and answers "is there a thing in my way" directly.
Semantics (dirt vs grass vs person) is a separate layer vetoing on top.

Camera frame: X right, Y down, Z forward. Robot frame: x forward, y left.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CostmapConfig:
    # MEASURED, not assumed: closing a 360 deg in-place spin gives fx 330-350 px
    # on a 1024-wide frame, i.e. ~113 deg, not the 90 deg previously assumed.
    # Lateral extent scales as 1/fx, so 90 deg compressed the world by 1.5x.
    # Barrel distortion degrades the pinhole model near the edges; the
    # near-field centre used for the veto is fine.
    fov_horizontal_deg: float = 112.8
    # A point this far above the fitted ground plane is an obstacle.
    obstacle_height_m: float = 0.18
    # Above this it is overhead (branches, awnings) and the rover passes under.
    max_height_m: float = 2.0
    # Depth range we trust from a monocular metric model.
    min_range_m: float = 0.30
    max_range_m: float = 10.0
    # Occupancy grid.
    cell_m: float = 0.15
    grid_forward_m: float = 8.0
    grid_half_width_m: float = 4.0
    # Only the near field of a long candidate can hit this grid honestly.
    # 10 m arcs otherwise die on smeared memory past the bumper.
    veto_range_m: float = 2.5
    # Rover half-width plus margin; obstacles inflate by this for the veto.
    rover_radius_m: float = 0.35
    # RANSAC.
    ransac_iters: int = 60
    ransac_thresh_m: float = 0.10
    # Ground plane normal must be within this of "up" or the fit is rejected.
    min_up_dot: float = 0.75
    # Persist obstacles briefly so a rock does not vanish when it leaves view.
    # The history is robot-frame and is NOT transported between ticks, so it is
    # only valid while the rover is roughly still: yaw more than
    # `memory_max_yaw_deg` and OR-ing the stored grids smears every past
    # obstacle across the fan at the wrong bearing. Measured — an in-place pivot
    # on empty ground vetoed 12/12 candidates with pts=0 in the same log line.
    # Forgetting a real rock is recoverable; a rock remembered in the wrong
    # direction is not, because it steers while looking like evidence.
    memory_frames: int = 3
    memory_max_yaw_deg: float = 4.0


@dataclass
class CostmapResult:
    occupied: np.ndarray          # (ny, nx) bool, robot frame — hard veto
    offtrack: np.ndarray          # (ny, nx) bool — drivable but off-path
    ground_found: bool
    forward_clearance_m: float    # nearest obstacle in the forward corridor
    clearance_by_bin: np.ndarray  # (num_bins,) metres
    bin_centers: np.ndarray       # (num_bins,) radians, left-positive
    n_obstacle_points: int
    reason: str = ""
    debug: dict = field(default_factory=dict)


class DepthCostmap:
    def __init__(self, config: Optional[CostmapConfig] = None, num_bins: int = 16):
        self.cfg = config or CostmapConfig()
        self.num_bins = num_bins
        self._history: list[np.ndarray] = []
        self._rng = np.random.default_rng(0)

    # ---- geometry ---------------------------------------------------------
    def _backproject(self, depth: np.ndarray):
        """
        (H,W) metric depth -> ((N,3) camera-frame points, (H,W) valid mask).

        The mask is returned so a per-pixel semantic label map can be indexed
        with it and stay aligned with the point list.
        """
        h, w = depth.shape
        fov = math.radians(self.cfg.fov_horizontal_deg)
        fx = w / (2.0 * math.tan(fov / 2.0))
        fy = fx  # square pixels; Depth Anything gives no intrinsics
        cx, cy = w / 2.0, h / 2.0

        u = np.arange(w, dtype=np.float32)[None, :]
        v = np.arange(h, dtype=np.float32)[:, None]
        z = depth.astype(np.float32)
        valid = (
            np.isfinite(z)
            & (z > self.cfg.min_range_m)
            & (z < self.cfg.max_range_m)
        )
        if not valid.any():
            return np.empty((0, 3), np.float32), valid
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.stack([x[valid], y[valid], z[valid]], axis=1), valid

    def _fit_ground(self, pts: np.ndarray) -> tuple[Optional[np.ndarray], float]:
        """
        RANSAC a ground plane. Returns (unit normal oriented UP, offset d)
        such that signed height above ground = n.p - d.
        """
        if len(pts) < 50:
            return None, 0.0
        # Ground candidates: the lower half of the camera's vertical span.
        y_med = float(np.median(pts[:, 1]))
        cand = pts[pts[:, 1] >= y_med]
        if len(cand) < 30:
            cand = pts
        up = np.array([0.0, -1.0, 0.0], np.float32)  # Y is down

        best_n, best_d, best_inl = None, 0.0, 0
        n_cand = len(cand)
        for _ in range(self.cfg.ransac_iters):
            idx = self._rng.choice(n_cand, size=3, replace=False)
            p0, p1, p2 = cand[idx]
            nrm = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(nrm)
            if norm < 1e-6:
                continue
            nrm = nrm / norm
            if float(nrm @ up) < 0:
                nrm = -nrm
            if float(nrm @ up) < self.cfg.min_up_dot:
                continue  # not a ground-like plane
            d = float(nrm @ p0)
            inl = int(
                (np.abs(cand @ nrm - d) < self.cfg.ransac_thresh_m).sum()
            )
            if inl > best_inl:
                best_n, best_d, best_inl = nrm, d, inl

        if best_n is None or best_inl < max(30, int(0.15 * n_cand)):
            return None, 0.0

        # Least-squares refine on the inlier set.
        resid = np.abs(cand @ best_n - best_d)
        inliers = cand[resid < self.cfg.ransac_thresh_m]
        if len(inliers) >= 3:
            centroid = inliers.mean(axis=0)
            _, _, vt = np.linalg.svd(inliers - centroid, full_matrices=False)
            n2 = vt[-1]
            if float(n2 @ up) < 0:
                n2 = -n2
            if float(n2 @ up) >= self.cfg.min_up_dot:
                best_n = n2 / np.linalg.norm(n2)
                best_d = float(best_n @ centroid)
        return best_n, best_d

    # ---- main -------------------------------------------------------------
    def forget_memory(self) -> None:
        """Drop the robot-frame occupancy history (see `memory_max_yaw_deg`)."""
        self._history.clear()

    def compute(
        self,
        depth: np.ndarray,
        hard_mask: Optional[np.ndarray] = None,
        soft_mask: Optional[np.ndarray] = None,
        odom=None,
    ) -> CostmapResult:
        """
        hard_mask / soft_mask: (H,W) bool, aligned with `depth`.
          hard — occupied regardless of height (people, cars, water: a puddle
                 is a perfectly flat ground plane and geometry cannot see it).
          soft — drivable but off-track (grass); costed, never vetoed.
        odom — this tick's `OdomDelta`, used only to decide whether the stored
          history is still valid in the current robot frame. Optional.
        """
        cfg = self.cfg
        if odom is not None and (
            getattr(odom, "confidence", 0.0) <= 0.0
            or abs(math.degrees(getattr(odom, "dyaw", 0.0)))
            > cfg.memory_max_yaw_deg
        ):
            # We turned (or we do not know whether we turned). Either way the
            # stored grids no longer describe the frame we are in.
            self.forget_memory()
        nx = max(1, int(cfg.grid_forward_m / cfg.cell_m))
        ny = max(1, int(2 * cfg.grid_half_width_m / cfg.cell_m))
        occ = np.zeros((ny, nx), bool)
        bin_edges = np.linspace(
            -math.radians(cfg.fov_horizontal_deg) / 2.0,
            math.radians(cfg.fov_horizontal_deg) / 2.0,
            self.num_bins + 1,
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        far = cfg.max_range_m

        pts, valid = self._backproject(depth)
        if len(pts) == 0:
            return CostmapResult(
                occupied=occ,
                offtrack=np.zeros_like(occ),
                ground_found=False,
                forward_clearance_m=far,
                clearance_by_bin=np.full(self.num_bins, far, np.float32),
                bin_centers=bin_centers,
                n_obstacle_points=0,
                reason="no_valid_depth",
            )
        hard_pt = (
            hard_mask[valid] if hard_mask is not None and hard_mask.shape == valid.shape
            else np.zeros(len(pts), bool)
        )
        soft_pt = (
            soft_mask[valid] if soft_mask is not None and soft_mask.shape == valid.shape
            else np.zeros(len(pts), bool)
        )

        n, d = self._fit_ground(pts)
        if n is None:
            # No plane: do NOT invent a fake ground from camera-horizontal.
            # That treated every near-field dirt pixel as an obstacle (10,681
            # pts / 0.92 m stop on a floor the rover can roll). Blind geometry
            # is unknown, not occupied; semantics `hard` still adds hazards.
            ground_found = False
            geo_obs = np.zeros(len(pts), bool)
            d = 0.0
        else:
            ground_found = True
            height = pts @ n - d
            geo_obs = (height > cfg.obstacle_height_m) & (height < cfg.max_height_m)
            # The camera is 15 cm up, so roots 5-25 cm tall at 0.3-0.7 m sit
            # above a valid plane and read as a wall (9k pts / 0.68 m stop on
            # rollable dirt). Real obstacles in that band are taller.
            z = pts[:, 2]
            geo_obs = geo_obs & ~((z < 0.70) & (height < 0.32))
        # Semantics can only ADD hazards, never clear a geometric one: if the
        # segmenter calls a boulder "earth", the boulder still stops us.
        is_obs = geo_obs | hard_pt
        obs = pts[is_obs]
        soft_pts = pts[soft_pt & ~is_obs]

        # Camera -> robot frame: x forward = Z, y left = -X.
        fx_ = obs[:, 2]
        fy_ = -obs[:, 0]

        clearance = np.full(self.num_bins, far, np.float32)
        if len(obs):
            ang = np.arctan2(fy_, fx_)          # left-positive
            rad = np.hypot(fx_, fy_)
            b = np.digitize(ang, bin_edges) - 1
            ok = (b >= 0) & (b < self.num_bins)
            for bi in np.unique(b[ok]):
                sel = rad[ok][b[ok] == bi]
                if len(sel):
                    clearance[bi] = float(np.percentile(sel, 5))

            gx = (fx_ / cfg.cell_m).astype(int)
            gy = ((fy_ + cfg.grid_half_width_m) / cfg.cell_m).astype(int)
            good = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
            occ[gy[good], gx[good]] = True

        soft = np.zeros((ny, nx), bool)
        if len(soft_pts):
            sx = (soft_pts[:, 2] / cfg.cell_m).astype(int)
            sy = ((-soft_pts[:, 0] + cfg.grid_half_width_m) / cfg.cell_m).astype(int)
            g2 = (sx >= 0) & (sx < nx) & (sy >= 0) & (sy < ny)
            soft[sy[g2], sx[g2]] = True

        # Obstacle memory: a rock must not vanish the moment it leaves view.
        self._history.append(occ.copy())
        if len(self._history) > cfg.memory_frames:
            self._history.pop(0)
        occ_mem = np.any(np.stack(self._history), axis=0)

        half = max(1, self.num_bins // 8)
        c = self.num_bins // 2
        fwd = float(np.min(clearance[max(0, c - half): c + half + 1]))

        return CostmapResult(
            occupied=occ_mem,
            offtrack=soft,
            ground_found=ground_found,
            forward_clearance_m=fwd,
            clearance_by_bin=clearance,
            bin_centers=bin_centers,
            n_obstacle_points=int(len(obs)),
            reason="ok" if ground_found else "no_ground_plane",
            debug={"n_points": int(len(pts)), "ground_d": float(d)},
        )

    # ---- the veto ---------------------------------------------------------
    def veto(self, result: CostmapResult, candidates) -> np.ndarray:
        """Drop any candidate crossing occupied space. True = safe to keep."""
        cfg = self.cfg
        occ = result.occupied
        ny, nx = occ.shape
        mask = np.ones(len(candidates), bool)
        if not occ.any():
            return mask
        r_cells = max(1, int(math.ceil(cfg.rover_radius_m / cfg.cell_m)))
        oy, ox = np.nonzero(occ)
        for i, cand in enumerate(candidates):
            wp = np.asarray(cand.waypoints, np.float32)
            gx = (wp[:, 0] / cfg.cell_m).astype(int)
            gy = ((wp[:, 1] + cfg.grid_half_width_m) / cfg.cell_m).astype(int)
            near = wp[:, 0] <= cfg.veto_range_m
            inb = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny) & near
            if not inb.any():
                continue
            # Inflate by the rover footprint: any obstacle cell within
            # r_cells of any waypoint kills the candidate.
            for cxi, cyi in zip(gx[inb], gy[inb]):
                if np.any(
                    (np.abs(ox - cxi) <= r_cells) & (np.abs(oy - cyi) <= r_cells)
                ):
                    mask[i] = False
                    break
        return mask

    def offtrack_cost(self, result: CostmapResult, candidates) -> np.ndarray:
        """
        Fraction of each candidate's waypoints on off-track surface.

        A preference, not a rule: prefer the track, but stay able to cross the
        verge to get around an obstacle.
        """
        cfg = self.cfg
        soft = result.offtrack
        out = np.zeros(len(candidates), np.float32)
        if soft is None or not soft.any():
            return out
        ny, nx = soft.shape
        for i, cand in enumerate(candidates):
            wp = np.asarray(cand.waypoints, np.float32)
            gx = (wp[:, 0] / cfg.cell_m).astype(int)
            gy = ((wp[:, 1] + cfg.grid_half_width_m) / cfg.cell_m).astype(int)
            inb = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
            if not inb.any():
                continue
            out[i] = float(soft[gy[inb], gx[inb]].mean())
        return out
