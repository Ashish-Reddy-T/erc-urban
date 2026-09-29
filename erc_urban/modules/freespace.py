"""
Free space per heading for the urban planner, the erc_indoor method outdoors.

Where does drivable ground end in each image column? Scan up from the bottom to
the first non-drivable pixel; through the calibrated camera that boundary row
is the distance at which something stands on the ground in that direction.
Walls, kerbs, hedges, people, parked robots and poles all end the ground. Each
heading is judged by the rover-wide strip swept along it, so something beside
the front corner counts even when it is not dead ahead.

Why this replaced the K-candidate fan as the urban driver (Wuhan mission-3,
2026-09-23): the rover sat angled 20 deg into a planter wall with open plaza to
its right. Every 10 m arc from the current heading crossed the wall, three
vetoes agreed "no way forward", and the blind escape scan turned it into the
kerb. On that frame this module says: turn 20 deg right, 5 m clear.

Two masks:
  * DRIVABLE is an explicit ground allow-list, unlike semantics.categorize's
    "traversable by exclusion". Exclusion makes a pole or a hedge ("plant")
    drivable, and a boundary that runs through a hedge drives into it.
  * TRACK is DRIVABLE minus grass/field: "avoid grass, allow to pass". The
    planner steers on TRACK first and falls back to DRIVABLE only when no
    track heading is clear.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import math

import numpy as np

from erc_indoor.freespace import FreeSpace, path_clear

GROUND_NAMES = {
    "road", "sidewalk", "pavement", "floor", "earth", "ground", "path",
    "dirt track", "land", "sand", "runway", "rug",
}
GRASS_NAMES = {"grass", "field"}
# Classes that cannot be a LOW object with open ground visible just beyond it.
# Wuhan: pink-brown granite paving read "building" (conf 0.7-0.9) right at the
# bumper, "ground ends 0.2 m", escape after escape in open plaza. The flat-patch
# test cannot reject a patch that big (it could be a 12 cm box), but a building
# 0.2 m ahead with plaza visible over it is not a thing. "wall" is NOT here: a
# kerb labelled wall must stay an obstacle.
NEVER_LOW_NAMES = {"building", "house", "skyscraper", "sky", "ceiling"}
# FAN_TERRAIN=offroad (Panama park): drivable = everything EXCEPT these. Only
# real obstacles stop the rover; grass, earth, leaf litter (SegFormer calls it
# "plant", "bag", even "animal"), roots and the sky over a hill crest do not.
OFFROAD_OBSTACLE_NAMES = {
    "tree", "palm", "rock", "stone", "wall", "building", "house", "fence", "railing",
    "pole", "column", "pillar", "person", "car", "truck", "van", "bus", "bicycle",
    "minibike", "motorbike", "bench", "signboard", "sign", "streetlight", "street lamp",
    "ashcan", "trash can", "water", "sea", "river", "lake", "pool", "swimming pool",
    "fountain", "stairs", "steps", "stairway", "booth", "kiosk", "tower",
    # Bushes and flower beds. Costa Rica mission-1 drove into a planting bed
    # and wedged at 33 deg: "plant" counted as ground. Flat leaf litter
    # labelled plant still passes the depth flat-test (Panama: 3/260 frames
    # newly blocked, all of them standing vegetation).
    "plant", "flower", "flowerpot", "pot",
    # Garden furniture: Costa Rica mission-1 drove under a round stone table
    # (off-road had "bench" but not "table") and tipped up against its top.
    "table", "chair", "seat", "desk", "stool", "ottoman", "sofa", "armchair",
}


def unlabel_impossible(ground: np.ndarray, never_low: np.ndarray, horizon: float) -> np.ndarray:
    """ground plus never_low pixels that have ground above them (below the
    horizon) in the same column: misread paving, not a building."""
    h = ground.shape[0]
    top = int(max(0, min(h - 1, np.ceil(horizon) + 1)))
    above = np.zeros_like(ground)
    # any ground strictly above this row, from the horizon down
    above[top + 1:] = np.maximum.accumulate(ground[top:-1], axis=0)
    return ground | (never_low & above)


def class_ids(id2label: dict, names: set[str]) -> set[int]:
    """ADE20K ids whose comma-separated synonym list contains any of `names`."""
    out = set()
    for i, lbl in id2label.items():
        parts = {p.strip().lower() for p in str(lbl).split(",")}
        if parts & names:
            out.add(int(i))
    return out


@dataclass
class UrbanFreeSpace:
    track: FreeSpace        # clearance on paved ground only
    drive: FreeSpace        # clearance on paved ground + grass
    track_boundary: np.ndarray   # boundary row per column (for the overlay)
    drive_boundary: np.ndarray


class FreeSpaceUrban:
    def __init__(self, calib, seg_hw: tuple[int, int] = (288, 512),
                 half_fov_deg: float = 50.0, step_deg: float = 5.0,
                 cap_m: float = 5.0, bottom_skip: int = 4,
                 min_obstacle_h_m: float = 0.07, half_width_m: float = 0.17,
                 overhang_m: float = 1.2, overhang_ratio: float = 0.5,
                 overhang_rows: int = 12, contact_reach_m: float = 0.9,
                 overhang_pad_m: float = 0.10, max_flat_h_m: float = 0.15,
                 flat_frac: float = 0.85, occlusion_ratio: float = 0.0,
                 range_relief_ratio: float = 0.0,
                 range_relief_min_m: float = 0.0,
                 hard_min_h_m: float = 0.07):
        """
        min_obstacle_h_m: a non-ground patch with ground visible again beyond
            it is only an obstacle if it could be at least this tall. Seen from
            a camera at height h, a patch spanning ground distances d_near..d_far
            is either flat, or a vertical face of height h*(1 - d_near/d_far) at
            d_near; below this it is a leaf, a seam or a painted line. Wuhan: a
            ginkgo leaf under the bumper (0.19-0.3 m, i.e. <= 5.4 cm) read as
            "ground ends 0.2 m" in every direction and fired escape after
            escape. A 10 cm kerb 2 m out implies 10 cm and stays an obstacle;
            walls, hedges and bodies have no ground above them at all.
        overhang_m / overhang_ratio / overhang_rows: the boundary assumes
            whatever ends the ground is STANDING there. An overhang breaks
            that: under a parked e-scooter (Wuhan mission-3) the camera saw
            ground almost to the horizon, the boundary sat on the scooter's
            underside and read "5 m clear" with the floorboard 0.2 m away. So,
            per column, compare depth in the `overhang_rows` just above the
            visible ground with depth of the ground just below: if it is under
            `overhang_m` and under `overhang_ratio` x that ground depth, it is
            an overhang, and becomes an obstacle at the depth distance.
        contact_reach_m: the depth test above only fires within ~1 m (the
            depth fit squashes mid-range), so from further out a parked
            scooter still read as ground to its underside and the planner aimed
            into the parking rows (Wuhan mission-3, twice). But its wheels touch
            the ground and the boundary finds them. A column whose ground runs
            under something takes the range of the nearest wheel-like contact
            within this lateral distance, if depth says the underside is at
            that contact's distance rather than the ground's. Relative depth
            only, so the squash does not matter.
        overhang_pad_m: extra lateral clearance kept from overhang/contact
            points. Mirrors, footboards and number plates stick out past the
            wheel line.
        """
        self.contact_reach_m = contact_reach_m
        # Off-road: a run mostly labelled a real obstacle class (the caller's
        # `hard` mask: wall, rock, fence ...) is judged at this height and never
        # waved through as flat litter. Machakos leg-1: a ~15 cm concrete kerb
        # labelled wall implied 0.115 m (< 0.12 off-road) and depth called it
        # flat; the rover drove into it 10+ times. Leaves are plant/bag/animal.
        self.hard_min_h_m = hard_min_h_m
        self._hard: Optional[np.ndarray] = None
        self.max_flat_h_m = max_flat_h_m
        # Off-road only (0 = off): an obstacle whose base is this many times
        # deeper than the ground just below it is far BEHIND something (a big
        # leaf at the lens, a crest), not standing here. Panama: a leaf 0.3 m
        # out made the tree line beyond read "ground ends 0.36 m". A drop-off
        # looks the same, which is why urban leaves this off.
        self.occlusion_ratio = occlusion_ratio
        # Off-road only (0 = off): an uphill grass/tree boundary violates the
        # flat-ground projection.  On the Panama slope a tree line that raw
        # depth put 1-3 m away intersected the assumed flat ground at 0.5 m,
        # creating a false wall.  `range_relief_min_m` is a conservative floor
        # for that compressed far field after the ratio gate passes.  A
        # caller-supplied `range_relief` mask marks the classes eligible for
        # correction.  Near objects are unchanged because their depth agrees
        # with the projected ground range.
        self.range_relief_ratio = range_relief_ratio
        self.range_relief_min_m = range_relief_min_m
        self.flat_frac = flat_frac
        self.overhang_pad_m = overhang_pad_m
        self.overhang_m = overhang_m
        self.overhang_ratio = overhang_ratio
        self.overhang_rows = overhang_rows
        self.h, self.w = seg_hw
        self.cal = calib.scaled_to(self.w, self.h)
        self.horizon = self.cal.horizon_v()
        self.cap_m = cap_m
        self.bottom_skip = bottom_skip
        self.min_obstacle_h_m = min_obstacle_h_m
        self.half_width_m = half_width_m
        u = np.arange(self.w) + 0.5
        v = np.arange(self.h) + 0.5
        uu, vv = np.meshgrid(u, v)
        xy, ok = self.cal.image_to_ground(np.stack([uu.ravel(), vv.ravel()], 1))
        self._gx = xy[:, 0].reshape(self.h, self.w)
        self._gy = xy[:, 1].reshape(self.h, self.w)
        self._gok = ok.reshape(self.h, self.w)
        # Azimuth of each column, left-positive (for overhang points).
        self._az = np.arctan2(-(u - self.cal.cx), self.cal.fx)
        self.bins = np.arange(-half_fov_deg, half_fov_deg + 1e-6, step_deg)

    def _flat_by_depth(self, depth: np.ndarray, row_ref: np.ndarray, u: int,
                       r0: int, r1: int) -> bool:
        """
        A low non-ground run (rows r1+1..r0) lies flat if, in its upper part,
        it is as deep as the GROUND is on the same image rows elsewhere. A
        standing face is nearer than the ground behind it on those rows. Both
        sides share the depth model's (strong) near-field squash, so it
        cancels. Panama: dry leaf drifts on grass.
        """
        rows = np.arange(r1 + 1, r0 + 1)
        if rows.size < 4:
            return False
        upper = rows[: max(2, rows.size // 3)]
        ref = row_ref[upper]
        d = depth[upper, u]
        ok = np.isfinite(ref) & np.isfinite(d) & (ref > 0) & (d > 0)
        if ok.sum() < 2:
            return False
        return float(np.median(d[ok] / ref[ok])) >= self.flat_frac

    def _vertical_by_depth(self, depth: Optional[np.ndarray], u: int, r0: int, r1: int) -> bool:
        """
        Does depth stand still up this run (a face) instead of receding like
        ground? Monocular depth squashes range roughly to its square root, so
        flat ground from row r0 up to r1 recedes by ~sqrt(geometric ratio)
        (saved frames: 1.7-2.4x from the bumper rows to row 202). Machakos
        kerb: 1.26x. Panama leaf drifts labelled wall recede like ground.
        No depth: trust the label.
        """
        if depth is None:
            return True
        top = depth[r1 + 1: min(r1 + 4, r0 + 1), u]
        bot = depth[max(r1 + 1, r0 - 2): r0 + 1, u]
        top, bot = top[np.isfinite(top) & (top > 0)], bot[np.isfinite(bot) & (bot > 0)]
        gb, gt = self._gx[r0, u], self._gx[min(r1 + 1, r0), u]
        if top.size == 0 or bot.size == 0 or gb <= 0 or gt <= gb:
            return True
        expect = math.sqrt(gt / gb)
        return float(np.median(top) / np.median(bot)) < 1.0 + 0.5 * (expect - 1.0)

    def _behind_occluder(self, depth: Optional[np.ndarray], u: int, r0: int, H: int) -> bool:
        if depth is None or self.occlusion_ratio <= 0.0:
            return False
        obj = depth[max(0, r0 - 3): r0 + 1, u]
        gnd = depth[r0 + 1: min(H, r0 + 5), u]
        obj, gnd = obj[np.isfinite(obj) & (obj > 0)], gnd[np.isfinite(gnd) & (gnd > 0)]
        if obj.size == 0 or gnd.size == 0:
            return False
        return float(np.median(obj)) > self.occlusion_ratio * float(np.median(gnd))

    def _boundary(self, g: np.ndarray, depth: Optional[np.ndarray] = None,
                  row_ref: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Per column, the row where something that could stand in the rover's
        way begins. Scans up from the bottom; a non-ground run with ground
        above it is skipped when it is too low to be an obstacle (see
        min_obstacle_h_m), or, below max_flat_h_m, when depth says it lies
        flat (Panama: dry leaf drifts on grass read as 8 cm boxes). Row 0 =
        ground all the way (contributes nothing).
        """
        H = self.h - self.bottom_skip
        hcam = self.cal.height_m
        boundary = np.zeros(self.w, int)
        for u in range(self.w):
            col = g[:H, u]
            if col.all():
                continue
            v = H - 1
            while v >= 0:
                while v >= 0 and col[v]:
                    v -= 1
                if v < 0:
                    break
                r0 = v                              # lowest row of this run
                while v >= 0 and not col[v]:
                    v -= 1
                r1 = v                              # ground again above it
                if (r1 < 0 or r1 <= self.horizon + 1
                        or not self._gok[r0, u] or not self._gok[r1, u]):
                    if not self._behind_occluder(depth, u, r0, H):
                        boundary[u] = r0
                    break
                dn, df = self._gx[r0, u], self._gx[r1, u]
                h_imp = hcam * (1.0 - dn / df) if df > dn else float("inf")
                hard = (self._hard is not None
                        and self._hard[r1 + 1: r0 + 1, u].mean() > 0.5
                        and self._vertical_by_depth(depth, u, r0, r1))
                if h_imp >= (min(self.min_obstacle_h_m, self.hard_min_h_m) if hard
                             else self.min_obstacle_h_m):
                    if (not hard and depth is not None and row_ref is not None
                            and h_imp < self.max_flat_h_m
                            and self._flat_by_depth(depth, row_ref, u, r0, r1)):
                        continue                    # flat: keep scanning up
                    boundary[u] = r0
                    break
        return boundary

    def _overhang_pts(self, g: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Obstacle points for columns whose ground runs under something close."""
        H = self.h - self.bottom_skip
        nong = ~g[:H]
        has = nong.any(axis=0)
        # Lowest non-ground row per column: where the visible ground stops.
        top = np.where(has, (H - 1) - nong[::-1].argmax(axis=0), -1)
        out = []
        for u in np.nonzero(has)[0]:
            t = int(top[u])
            if t <= self.horizon + 1:
                continue                            # above the horizon: canopy, sky
            band = depth[max(0, t - self.overhang_rows): t + 1, u]
            band = band[np.isfinite(band) & (band > 0.05)]
            below = depth[t + 1: min(H, t + 1 + self.overhang_rows), u]
            below = below[g[t + 1: min(H, t + 1 + self.overhang_rows), u]]
            below = below[np.isfinite(below) & (below > 0.05)]
            if band.size < 3 or below.size < 3:
                continue
            d = float(np.median(band))
            # Relative, not absolute: the thing must be nearer than the ground
            # it hides. The depth response fit squashes the far field (a
            # building 30 m out read 0.7-0.9 m in open plaza, and an absolute
            # test called that an overhang); both sides of this comparison are
            # squashed alike, so it survives.
            if d < self.overhang_m and d < self.overhang_ratio * float(np.median(below)):
                a = self._az[u]
                out.append((d * np.cos(a), d * np.sin(a)))
        return np.asarray(out, float).reshape(-1, 2)

    def _contact_pts(self, g: np.ndarray, boundary: np.ndarray,
                     depth: np.ndarray) -> np.ndarray:
        """Obstacle points for columns whose ground runs under something that
        stands on a nearby contact (a scooter body between its wheels)."""
        H = self.h - self.bottom_skip
        cols = np.arange(self.w)
        valid = (boundary > self.horizon + 1) & self._gok[boundary, cols]
        # A contact is a foot standing ON visible ground: ground just below it.
        # The lens vignette in the bottom corners is non-ground down to the
        # frame edge and otherwise reads as a "contact" 0.6 m out that the
        # reach then spreads over half the image.
        below = np.zeros(self.w, bool)
        for c in np.nonzero(valid)[0]:
            r = g[boundary[c] + 1: min(H, boundary[c] + 5), c]
            below[c] = r.size >= 3 and r.mean() > 0.75
        valid &= below
        rng = np.where(valid, np.hypot(self._gx[boundary, cols], self._gy[boundary, cols]),
                       np.inf)
        reach = np.where(valid, self.cal.fx * self.contact_reach_m / np.maximum(rng, 0.2), 0.0)
        nong = ~g[:H]
        has = nong.any(axis=0)
        top = np.where(has, (H - 1) - nong[::-1].argmax(axis=0), -1)
        ld = np.log(np.maximum(depth, 1e-3))
        k = self.overhang_rows
        out = []
        for u in np.nonzero(has & (top > self.horizon + 1))[0]:
            t = int(top[u])
            r_u = float(np.hypot(self._gx[t, u], self._gy[t, u]))
            cand = np.nonzero(valid & (np.abs(cols - u) <= reach) & (rng < 0.8 * r_u))[0]
            if cand.size == 0:
                continue
            band = ld[max(0, t - k): t + 1, u]
            gb = ld[t + 1: min(H, t + 1 + k), u][g[t + 1: min(H, t + 1 + k), u]]
            if band.size < 3 or gb.size < 3:
                continue
            du, dg = float(np.median(band)), float(np.median(gb))
            if du >= dg - np.log(1.15):            # not nearer than its ground
                continue
            for c in cand[np.argsort(rng[cand])][:4]:
                dc = float(np.median(ld[max(0, boundary[c] - k): boundary[c] + 1, c]))
                if abs(du - dc) < abs(du - dg):     # same thing as that contact
                    a, r = self._az[u], rng[c]
                    out.append((r * np.cos(a), r * np.sin(a)))
                    break
        return np.asarray(out, float).reshape(-1, 2)

    def _one(self, ground: np.ndarray, depth: Optional[np.ndarray] = None,
             sky: Optional[np.ndarray] = None,
             range_relief: Optional[np.ndarray] = None) -> tuple[FreeSpace, np.ndarray]:
        """sky: counts as ground for the boundary scan only (a hill crest is not
        a wall), never for the overhang/contact tests (canopy over sky is not
        an overhang: Panama, OVERHANG x20 on open grass under trees)."""
        g = cv2.morphologyEx(ground.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((5, 3), np.uint8)).astype(bool)
        gb = g | sky if sky is not None else g
        row_ref = None
        if depth is not None:
            # Ground depth per image row (median over this row's ground pixels).
            dg = np.where(g, depth, np.nan)
            cnt = g.sum(axis=1)
            with np.errstate(all="ignore"):
                row_ref = np.nanmedian(dg, axis=1)
            row_ref[cnt < 20] = np.nan
            # Rows with too little visible ground (a band of leaf litter right
            # at the bumper): ground depth is smooth in the row, so fill them
            # from the rows that have it, below the horizon.
            ok = np.isfinite(row_ref)
            ok[: int(self.horizon) + 2] = False
            if ok.sum() >= 5:
                rows = np.arange(self.h)
                fill = np.interp(rows, rows[ok], row_ref[ok])
                below = rows > self.horizon + 1
                row_ref = np.where(np.isfinite(row_ref) | ~below, row_ref, fill)
        boundary = self._boundary(gb, depth, row_ref)
        cols = np.arange(self.w)
        # A boundary at/above the horizon means ground runs to the far end.
        valid = (boundary > self.horizon + 1) & self._gok[boundary, cols]
        pts = np.stack([self._gx[boundary, cols], self._gy[boundary, cols]], 1)[valid]
        # The pinhole/ground intersection is only a range estimate while the
        # terrain is locally flat.  On a rising slope it can turn a distant
        # tree line into a wall at the bumper.  Raw monocular depth retains the
        # useful near-vs-far distinction even when its absolute scale is not
        # trustworthy, so use it as a lower bound only when it strongly
        # disagrees and semantics says this is vegetation in the distance.
        if (depth is not None and range_relief is not None
                and self.range_relief_ratio > 0.0 and len(pts)):
            vc = cols[valid]
            vr = boundary[valid]
            marked = range_relief[vr, vc]
            d = np.array([
                np.nanmedian(depth[max(0, int(r) - 3): int(r) + 1, int(u)])
                for r, u in zip(vr, vc)
            ], dtype=float)
            geom = np.hypot(pts[:, 0], pts[:, 1])
            use = (marked & np.isfinite(d) & (d > 0.05)
                   & (d >= self.range_relief_ratio * np.maximum(geom, 0.05)))
            if use.any():
                a = self._az[vc[use]]
                relieved = np.maximum(d[use], self.range_relief_min_m)
                pts[use, 0] = relieved * np.cos(a)
                pts[use, 1] = relieved * np.sin(a)
        soft = np.zeros((0, 2))                  # overhang/contact: padded
        if depth is not None:
            oh = self._overhang_pts(g, depth)
            self.last_overhang_cols = len(oh)
            ct = self._contact_pts(g, boundary, depth)
            self.last_contact_cols = len(ct)
            soft = np.vstack([oh, ct])
        pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < self.cap_m]
        soft = soft[np.hypot(soft[:, 0], soft[:, 1]) < self.cap_m]
        hw, hwp = self.half_width_m, self.half_width_m + self.overhang_pad_m

        def clr(b: float) -> float:
            c = path_clear(pts, b, hw)
            if len(soft):
                c = min(c, path_clear(soft, b, hwp))
            return min(self.cap_m, c)

        clear = np.array([clr(float(b)) for b in self.bins])
        allp = np.vstack([pts, soft]) if len(soft) else pts
        return FreeSpace(self.bins.copy(), clear, clr(0.0), allp), boundary

    def estimate(self, ground: np.ndarray, grass: Optional[np.ndarray] = None,
                 depth: Optional[np.ndarray] = None,
                 sky: Optional[np.ndarray] = None,
                 range_relief: Optional[np.ndarray] = None,
                 hard: Optional[np.ndarray] = None) -> UrbanFreeSpace:
        """
        ground: paved/track pixels; grass: grass pixels (drivable, not
        preferred); depth: metric depth at seg_hw (enables the overhang check).
        """
        if ground.shape != (self.h, self.w):
            raise ValueError(f"mask {ground.shape} != seg_hw {(self.h, self.w)}")
        if depth is not None and depth.shape != (self.h, self.w):
            depth = cv2.resize(depth.astype(np.float32), (self.w, self.h),
                               interpolation=cv2.INTER_LINEAR)
        if range_relief is not None and range_relief.shape != (self.h, self.w):
            range_relief = cv2.resize(range_relief.astype(np.uint8), (self.w, self.h),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
        if hard is not None and hard.shape != (self.h, self.w):
            hard = cv2.resize(hard.astype(np.uint8), (self.w, self.h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        self._hard = hard
        self.last_overhang_cols = 0
        self.last_contact_cols = 0
        track, tb = self._one(ground, depth, sky, range_relief)
        if grass is None:
            return UrbanFreeSpace(track, track, tb, tb)
        drive, db = self._one(ground | grass, depth, sky, range_relief)
        return UrbanFreeSpace(track, drive, tb, db)


def _with_margin(ok: np.ndarray) -> np.ndarray:
    """
    Clear only if both neighbouring bins are clear too (edges of the fan count
    as clear). "Nearest clear heading to the goal" otherwise skims whatever
    stands between the rover and the goal: mission-3 ran along a low wall
    with CP3 behind it, drifted back into the wall's strip every few metres
    and escaped at 19, 20, 53 deg.
    """
    padded = np.concatenate([[True], ok, [True]])
    return ok & padded[:-2] & padded[2:]


def choose_heading(fs: UrbanFreeSpace, want_deg: float, need_m: float,
                   min_m: Optional[float] = None,
                   forbid: Optional[np.ndarray] = None) -> tuple[Optional[float], str]:
    """
    Heading nearest `want_deg` with a clear swept strip. Track first, then
    grass allowed, then a shorter clearance. Returns (heading, which) with
    which in {"track", "grass", "short", "none"}.

    forbid: bool per bin, headings that just failed (see run.py). Wuhan: the
    nearest-to-goal heading threaded a 0.5 m gap between a kerb and a parked
    rover that read clear from 1.5 m and blocked from 0.3 m, and escape ->
    reverse -> "clear again" looped until stopped. Open plaza to the right was
    never tried.
    """
    allowed = np.ones(len(fs.drive.bins_deg), bool) if forbid is None else ~forbid
    for fsx, need, which in ((fs.track, need_m, "track"), (fs.drive, need_m, "grass")):
        ok = _with_margin(fsx.clear_m >= need) & allowed
        if ok.any():
            cand = fsx.bins_deg[ok]
            return float(cand[np.argmin(np.abs(cand - want_deg))]), which
    if min_m is not None:
        ok = (fs.drive.clear_m >= min_m) & allowed
        if ok.any():
            cand = fs.drive.bins_deg[ok]
            return float(cand[np.argmin(np.abs(cand - want_deg))]), "short"
    return None, "none"


def open_side(fs: UrbanFreeSpace) -> float:
    """+1 if the left half of the fan is more open than the right, else -1."""
    c, b = fs.drive.clear_m, fs.drive.bins_deg
    return 1.0 if c[b > 0].sum() >= c[b < 0].sum() else -1.0
