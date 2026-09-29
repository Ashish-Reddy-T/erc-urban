"""
World-frame memory of places the rover found it could not get through.

The fan planner only sees the camera's +-50 deg, and forgets everything the
moment it turns. Wuhan mission-3, CP9: a low wall at the foot of a staircase
lay between the rover and the goal. Every escape turned it away, the goal pull
turned it straight back into the same wall, and it swung left and right along
it for minutes. With the wall remembered, "towards the goal" stops meaning
"into the wall": the rover backs off, turns, goes along it and round its end.

Two kinds of point, in an odometry frame (x east, y north, metres; yaw
clockwise from north, as the compass reports it):
  * confirmed: what stopped an escape ("can't get through here"). Kept for
    keep_m of travel, forgotten after seen_free_ticks ticks of open ground.
  * seen: obstacles seen within seen_min_m..seen_max_m, every tick. Short-lived
    (seen_keep_m / seen_keep_s) and forgotten the first tick the camera shows
    open ground there. Without them a wall the rover drives alongside leaves
    the +-50 deg view and the goal pull turns it straight back in.
Anything the camera later shows as open ground is dropped, so a pedestrian who
walks off does not leave a phantom wall.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from erc_indoor.freespace import path_clear


class ObstacleMemory:
    def __init__(self, keep_m: float = 40.0, keep_s: float = 600.0,
                 cell_m: float = 0.10, max_pts: int = 4000,
                 add_range_m: float = 3.0, seen_free_ticks: int = 3,
                 seen_keep_m: float = 8.0, seen_keep_s: float = 90.0,
                 seen_min_m: float = 0.5, seen_max_m: float = 2.5):
        self.keep_m = keep_m
        self.keep_s = keep_s
        self.seen_keep_m = seen_keep_m
        self.seen_keep_s = seen_keep_s
        self.seen_min_m = seen_min_m
        self.seen_max_m = seen_max_m
        self.cell_m = cell_m
        self.max_pts = max_pts
        self.add_range_m = add_range_m
        self.seen_free_ticks = seen_free_ticks
        self.x = 0.0            # east, m
        self.y = 0.0            # north, m
        self.travel = 0.0       # |distance| driven, m
        self.yaw = 0.0          # rad, clockwise; the frame's own heading (gyro)
        self._pts = np.zeros((0, 2))
        self._travel0 = np.zeros(0)
        self._t0 = np.zeros(0)
        self._batch = np.zeros(0, int)
        self._free = np.zeros(0, int)
        self._conf = np.zeros(0, bool)
        self._hits = np.zeros(0, int)
        # Felt, not seen (wheels stalled): the camera shows open ground there,
        # so clear_seen must not erase them. Machakos leg-1: a root/mound under
        # grass stalled the wheels, vision cleared the strip within 3 ticks and
        # the rover drove back into it 8 times.
        self._sticky = np.zeros(0, bool)
        self._next_batch = 1

    def __len__(self) -> int:
        return len(self._pts)

    # --- pose -------------------------------------------------------------
    def step_odom(self, dx: float, dy: float, dyaw_left: float) -> None:
        """
        Advance by one wheel+gyro odometry delta (robot frame of the previous
        pose; dyaw left-positive). The memory then lives in its own gyro
        frame, so a compass error does not rotate remembered walls. Panama:
        tilted on a slope, the compass read 40 deg for a 62 deg gyro turn.
        """
        self.update_pose(dx, dy, self.yaw)
        self.yaw -= dyaw_left

    def update_pose(self, dx: float, dy: float, yaw_rad: float) -> None:
        """dx forward / dy left (signed, robot frame) since the last call."""
        s, c = math.sin(yaw_rad), math.cos(yaw_rad)
        self.x += dx * s - dy * c
        self.y += dx * c + dy * s
        self.travel += math.hypot(dx, dy)

    def _to_world(self, pts: np.ndarray, yaw_rad: float) -> np.ndarray:
        s, c = math.sin(yaw_rad), math.cos(yaw_rad)
        e = self.x + pts[:, 0] * s - pts[:, 1] * c
        n = self.y + pts[:, 0] * c + pts[:, 1] * s
        return np.stack([e, n], 1)

    def to_robot(self, yaw_rad: float, max_range_m: float = 6.0,
                 min_hits: int = 1) -> np.ndarray:
        """
        Remembered points in the robot frame (x forward, y left). min_hits:
        seen points only if observed in the same cell that many times; a
        crowd walking past leaves one hit per cell, a wall or a parked
        scooter piles them up (Wuhan: 700 ghost points of students boxed the
        plan in every direction). Confirmed points always count.
        """
        if not len(self._pts):
            return np.zeros((0, 2))
        keep = self._conf | (self._hits >= min_hits)
        s, c = math.sin(yaw_rad), math.cos(yaw_rad)
        de, dn = self._pts[keep, 0] - self.x, self._pts[keep, 1] - self.y
        fx = de * s + dn * c
        ly = -de * c + dn * s
        out = np.stack([fx, ly], 1)
        return out[np.hypot(fx, ly) < max_range_m]

    # --- contents ---------------------------------------------------------
    def add(self, robot_pts: np.ndarray, yaw_rad: float, now: float, sticky: bool = False) -> int:
        """Confirmed: the obstacle points within add_range_m. Returns a batch id.
        sticky: felt rather than seen; open ground in view does not clear it."""
        return self._add(robot_pts, yaw_rad, now, True, 0.0, self.add_range_m, sticky)

    def observe(self, robot_pts: np.ndarray, yaw_rad: float, now: float) -> None:
        """Seen this tick: short-lived points within seen_min_m..seen_max_m."""
        self._add(robot_pts, yaw_rad, now, False, self.seen_min_m, self.seen_max_m)

    def _add(self, robot_pts, yaw_rad, now, confirmed, r_min, r_max, sticky=False) -> int:
        if robot_pts is None or not len(robot_pts):
            return 0
        r = np.hypot(robot_pts[:, 0], robot_pts[:, 1])
        p = robot_pts[(r >= r_min) & (r < r_max)]
        if not len(p):
            return 0
        w = self._to_world(p, yaw_rad)
        # One point per cell; a cell already remembered is refreshed instead.
        cells = np.round(w / self.cell_m).astype(np.int64)
        _, idx = np.unique(cells, axis=0, return_index=True)
        w, cells = w[idx], cells[idx]
        b = self._next_batch if confirmed else 0
        if confirmed:
            self._next_batch += 1
        have = {tuple(c): i for i, c in
                enumerate(np.round(self._pts / self.cell_m).astype(np.int64))}
        new = np.ones(len(w), bool)
        for k, c in enumerate(cells):
            j = have.get(tuple(c))
            if j is None:
                continue
            new[k] = False
            self._t0[j], self._travel0[j], self._free[j] = now, self.travel, 0
            self._hits[j] += 1
            if confirmed:
                self._conf[j], self._batch[j] = True, b
            if sticky:
                self._sticky[j] = True
        w = w[new]
        m = len(w)
        if m:
            self._pts = np.vstack([self._pts, w])
            self._travel0 = np.concatenate([self._travel0, np.full(m, self.travel)])
            self._t0 = np.concatenate([self._t0, np.full(m, now)])
            self._batch = np.concatenate([self._batch, np.full(m, b)])
            self._free = np.concatenate([self._free, np.zeros(m, int)])
            self._conf = np.concatenate([self._conf, np.full(m, confirmed)])
            self._hits = np.concatenate([self._hits, np.ones(m, int)])
            self._sticky = np.concatenate([self._sticky, np.full(m, sticky)])
        if len(self._pts) > self.max_pts:
            self._keep(np.arange(len(self._pts)) >= len(self._pts) - self.max_pts)
        return b

    def n_confirmed(self) -> int:
        return int(self._conf.sum())

    def drop_batch(self, batch: int) -> None:
        if batch:
            self._keep(self._batch != batch)

    def prune(self, now: float) -> None:
        if len(self._pts):
            dm, ds = self.travel - self._travel0, now - self._t0
            keep = np.where(self._conf, (dm < self.keep_m) & (ds < self.keep_s),
                            (dm < self.seen_keep_m) & (ds < self.seen_keep_s))
            self._keep(keep)

    def clear_seen(self, view_pts: np.ndarray, yaw_rad: float, half_fov_deg: float = 45.0,
                   min_r: float = 0.35, max_r: float = 4.5, margin_m: float = 0.3,
                   az_tol_deg: float = 3.0, cap_m: float = 5.0) -> int:
        """
        Forget points the camera now shows as open ground: inside the view,
        nearer than whatever currently ends the ground in that direction, for
        seen_free_ticks ticks running. Returns how many were dropped.
        """
        if not len(self._pts):
            return 0
        s, c = math.sin(yaw_rad), math.cos(yaw_rad)
        de, dn = self._pts[:, 0] - self.x, self._pts[:, 1] - self.y
        rp = np.stack([de * s + dn * c, -de * c + dn * s], 1)
        r = np.hypot(rp[:, 0], rp[:, 1])
        az = np.degrees(np.arctan2(rp[:, 1], rp[:, 0]))
        inview = (np.abs(az) <= half_fov_deg) & (r >= min_r) & (r <= max_r)
        free = np.zeros(len(r), bool)
        if inview.any():
            if view_pts is not None and len(view_pts):
                vr = np.hypot(view_pts[:, 0], view_pts[:, 1])
                vaz = np.degrees(np.arctan2(view_pts[:, 1], view_pts[:, 0]))
            else:
                vr = vaz = np.zeros(0)
            for i in np.nonzero(inview)[0]:
                near = vr[np.abs(vaz - az[i]) <= az_tol_deg]
                seen_to = float(near.min()) if near.size else cap_m
                free[i] = r[i] < seen_to - margin_m
        self._free = np.where(free, self._free + 1, 0)
        drop = (self._free >= np.where(self._conf, self.seen_free_ticks, 1)) & ~self._sticky
        if drop.any():
            self._keep(~drop)
        return int(drop.sum())

    def _keep(self, m: np.ndarray) -> None:
        self._pts, self._travel0, self._t0 = self._pts[m], self._travel0[m], self._t0[m]
        self._batch, self._free, self._conf = self._batch[m], self._free[m], self._conf[m]
        self._hits = self._hits[m]
        self._sticky = self._sticky[m]


def _circ_margin(ok: np.ndarray) -> np.ndarray:
    """Clear only if both circular neighbours are clear too."""
    return ok & np.roll(ok, 1) & np.roll(ok, -1)


def choose_full(fs, want_deg: float, need_m: float, min_m: Optional[float],
                mem_pts: Optional[np.ndarray], forbid_fn=None,
                half_width_m: float = 0.17, half_fov_deg: float = 50.0,
                step_deg: float = 5.0, need_out_m: float = 1.5, cap_m: float = 5.0,
                turn_k: float = 0.5, grass_pen: float = 180.0, short_pen: float = 360.0,
                comf_m: float = 2.5, clear_k: float = 30.0, unknown_pen: float = 15.0,
                side_x_m: float = 0.3, side_y_m: float = 0.35, side_turn_deg: float = 15.0
                ) -> tuple[Optional[float], str, bool]:
    """
    Heading over the full circle: the camera's bins judged by what it sees
    AND what is remembered, the rest by memory alone. Cost = angle to the
    goal + turn_k x turn + clear_k per metre of clearance short of comf_m + a
    tier penalty (grass, short clearance). The clearance term is what makes it
    run along a wall rather than graze into it: "nearest clear heading to the
    goal" always picked the one that met the wall at exactly need_m. Returns
    (heading, which, in_view); which in {"track", "grass", "short", "mem", "none"}.

    forbid_fn(bins_deg) -> bool mask of recently failed headings, or None.

    Side check: anything remembered right beside the body (|x| < side_x_m,
    |y| < side_y_m) forbids turning more than side_turn_deg towards it. The
    swept strip only looks ahead; rounding a wall end, the corner sat 0.18 m
    off the left front and the turn swung the body into it.
    """
    mem = mem_pts if mem_pts is not None and len(mem_pts) else None

    def mem_clear(b: float) -> float:
        return cap_m if mem is None else min(cap_m, path_clear(mem, b, half_width_m))

    vb = fs.drive.bins_deg
    mc = np.array([mem_clear(float(b)) for b in vb])
    tr = np.minimum(fs.track.clear_m, mc)
    dr = np.minimum(fs.drive.clear_m, mc)
    ob = np.array([b for b in np.arange(-180.0, 180.0, step_deg) if abs(b) > half_fov_deg + 1e-6])
    oc = np.array([mem_clear(float(b)) for b in ob])
    bins = np.concatenate([vb, ob])
    order = np.argsort(bins)
    bins_s = bins[order]
    # Tiers on the whole circle (out-of-view bins only have memory).
    track_ok = np.concatenate([tr >= need_m, oc >= need_out_m])[order]
    drive_ok = np.concatenate([dr >= need_m, oc >= need_out_m])[order]
    short_ok = np.concatenate([dr >= (min_m if min_m is not None else 1e9),
                               np.zeros(len(ob), bool)])[order]
    in_view = np.concatenate([np.ones(len(vb), bool), np.zeros(len(ob), bool)])[order]
    clr_t = np.concatenate([tr, oc])[order]
    clr_d = np.concatenate([dr, oc])[order]
    track_ok, drive_ok = _circ_margin(track_ok), _circ_margin(drive_ok)
    allowed = np.ones(len(bins_s), bool)
    if mem is not None:
        side = mem[(np.abs(mem[:, 0]) < side_x_m) & (np.abs(mem[:, 1]) < side_y_m)
                   & (np.abs(mem[:, 1]) > 0.05)]
        if (side[:, 1] > 0).any():
            allowed &= ~(bins_s > side_turn_deg)
        if (side[:, 1] < 0).any():
            allowed &= ~(bins_s < -side_turn_deg)
    if forbid_fn is not None:
        f = forbid_fn(bins_s)
        if f is not None:
            allowed = ~f
    ang = np.abs((bins_s - want_deg + 180.0) % 360.0 - 180.0)
    # Out of view is only "not known to be shut": prefer what the camera
    # sees clear, or a crowd has it turning to look sideways all the time.
    cost = ang + turn_k * np.abs(bins_s) + np.where(in_view, 0.0, unknown_pen)
    clr_pen_t = clear_k * np.maximum(0.0, comf_m - clr_t)
    clr_pen_d = clear_k * np.maximum(0.0, comf_m - clr_d)
    # In view: track, else grass (avoid grass, allow to pass), else short.
    # The tier penalties rank IN-VIEW choices only. Against turning to look
    # out of view they do not apply: on an open grass field (Panama) the
    # 180 deg grass penalty made "turn 60 and look for pavement" win every
    # tick, and the rover pivoted instead of crossing the grass to the goal.
    best_in, which_in, cmp_in, rank_in = None, "none", np.inf, np.inf
    for ok, pen, name, cp in ((track_ok & allowed & in_view, 0.0, "track", clr_pen_t),
                              (drive_ok & allowed & in_view, grass_pen, "grass", clr_pen_d),
                              (short_ok & allowed, short_pen, "short", clr_pen_d)):
        if not ok.any():
            continue
        c = np.where(ok, cost + cp, np.inf)
        i = int(np.argmin(c))
        if c[i] + pen < rank_in:
            best_in, which_in, rank_in = i, name, float(c[i] + pen)
            cmp_in = float(c[i]) + (short_pen if name == "short" else 0.0)
    out_ok = track_ok & allowed & ~in_view
    best_out, cmp_out = None, np.inf
    if out_ok.any():
        c = np.where(out_ok, cost + clr_pen_t, np.inf)       # unknown_pen is in cost
        best_out = int(np.argmin(c))
        cmp_out = float(c[best_out])
    if best_in is None and best_out is None:
        return None, "none", False
    if best_in is not None and cmp_in <= cmp_out:
        return float(bins_s[best_in]), which_in, True
    return float(bins_s[best_out]), "mem", False


def plan_heading(mem_pts: Optional[np.ndarray], want_deg: float, rng_m: float,
                 cell_m: float = 0.15, half_m: float = 7.5, inflate_m: float = 0.25,
                 soft_m: float = 0.6, soft_cost: float = 3.0, look_m: float = 3.0
                 ) -> Optional[float]:
    """
    Heading (deg, left-positive) of the shortest path to the goal around the
    remembered obstacles, or None if nothing is remembered / no path.

    Robot-frame grid (x forward, y left) of half_m either side. Obstacles are
    inflated by inflate_m and cost extra within soft_m (keep off walls).
    Unknown space is free, and each cell on the window edge may leave the grid
    at the straight-line cost from there to the goal, so a wall wider than the
    window is followed towards its nearer end rather than read as a dead end.
    This is what turns "left and right both shut" into "back out, go round":
    from inside a pocket the only way to the goal is out through its mouth.
    """
    if mem_pts is None or not len(mem_pts):
        return None
    import cv2
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    n = int(round(2 * half_m / cell_m))
    ii = np.floor((mem_pts[:, 0] + half_m) / cell_m).astype(int)
    jj = np.floor((mem_pts[:, 1] + half_m) / cell_m).astype(int)
    ok = (ii >= 0) & (ii < n) & (jj >= 0) & (jj < n)
    if not ok.any():
        return None
    occ = np.zeros((n, n), np.uint8)
    occ[ii[ok], jj[ok]] = 1
    dist = cv2.distanceTransform((1 - occ).astype(np.uint8), cv2.DIST_L2, 5) * cell_m
    blocked = dist < inflate_m
    rc = n // 2
    blocked[rc - 1: rc + 2, rc - 1: rc + 2] = False          # never start "inside" a wall
    cost = 1.0 + soft_cost * np.clip((soft_m - dist) / max(soft_m - inflate_m, 1e-6), 0.0, 1.0)

    idx = np.arange(n * n).reshape(n, n)
    rows, cols, wts = [], [], []
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = idx[max(0, -di): n - max(0, di), max(0, -dj): n - max(0, dj)]
        b = idx[max(0, di): n - max(0, -di) if di <= 0 else n, max(0, dj): n - max(0, -dj) if dj <= 0 else n]
        a, b = a.ravel(), b.ravel()
        ai, aj = np.divmod(a, n)
        bi, bj = np.divmod(b, n)
        m = ~blocked[ai, aj] & ~blocked[bi, bj]
        step = cell_m * (math.sqrt(2.0) if di and dj else 1.0)
        w = step * 0.5 * (cost[ai, aj] + cost[bi, bj])
        rows += [a[m], b[m]]
        cols += [b[m], a[m]]
        wts += [w[m], w[m]]
    # Super source -> goal cells (inside the window) / edge cells (leave it).
    sup = n * n
    gx, gy = rng_m * math.cos(math.radians(want_deg)), rng_m * math.sin(math.radians(want_deg))
    ci = (np.arange(n) + 0.5) * cell_m - half_m
    X, Y = np.meshgrid(ci, ci, indexing="ij")
    edge = np.zeros((n, n), bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    tgt = edge & ~blocked
    tw = np.hypot(X - gx, Y - gy)[tgt]
    tidx = idx[tgt]
    if abs(gx) < half_m - cell_m and abs(gy) < half_m - cell_m:
        near = (np.hypot(X - gx, Y - gy) < 0.5) & ~blocked
        tidx = np.concatenate([tidx, idx[near]])
        tw = np.concatenate([tw, np.zeros(int(near.sum()))])
    if not len(tidx):
        return None
    rows.append(np.full(len(tidx), sup))
    cols.append(tidx)
    wts.append(tw + 1e-6)
    g = csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(sup + 1, sup + 1))
    d, pred = dijkstra(g, directed=True, indices=sup, return_predecessors=True)
    start = idx[rc, rc]
    if not np.isfinite(d[start]):
        return None
    # Walk from the rover towards the goal (predecessors point back to sup),
    # then aim at the farthest path cell within look_m that is in straight
    # line of sight: raw 8-connected paths zig-zag at 45 deg.
    c, walked, path = start, 0.0, []
    px, py = X[rc, rc], Y[rc, rc]
    while walked < 2.5 * half_m:
        p = pred[c]
        if p < 0 or p == sup:
            break
        pi, pj = divmod(int(p), n)
        walked += math.hypot(X[pi, pj] - px, Y[pi, pj] - py)
        px, py, c = X[pi, pj], Y[pi, pj], p
        path.append((pi, pj))
    if not path:
        return float(want_deg)
    # Line of sight to the farthest path cell, but at least look_m of it:
    # a corner closer than that is still the way to go.
    for pi, pj in reversed(path):
        k = max(abs(pi - rc), abs(pj - rc)) * 2 + 1
        si = np.round(np.linspace(rc, pi, k)).astype(int)
        sj = np.round(np.linspace(rc, pj, k)).astype(int)
        if not blocked[si, sj].any():
            break
    return float(math.degrees(math.atan2(Y[pi, pj] - Y[rc, rc], X[pi, pj] - X[rc, rc])))
