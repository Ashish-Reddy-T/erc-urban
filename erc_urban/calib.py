"""
Camera calibration — the shared geometry for anything that turns robot-frame
metres into pixels and back (the overlay renderer, the vetoes, odometry).

FrodoBots publishes no intrinsics, so this was scattered across three files as a
magic 90 deg FOV. Here it is explicit and replaceable, and every instance
carries `source` saying where its numbers came from: an assumed FOV, a
checkerboard, or a ground-plane self-calibration. Anything that projects
geometry should be willing to say how much it trusts the projection.

FRAMES
------
Robot:  x forward, y left, z up. Matches Candidate.waypoints.
Camera: X right, Y down, Z forward (optical axis).

The camera sits `height_m` above the ground and is pitched down by `pitch_rad`.
Level camera first, then a rotation about the camera's X axis:

    X =  Xl
    Y =  Yl*cos(t) - Zl*sin(t)
    Z =  Yl*sin(t) + Zl*cos(t)

A downward pitch moves distant ground points UP the image — the sign convention
that trips people up, so it is pinned by a test.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np

# Fallbacks only. camera_front.json carries the measured values (fx 340,
# ~113 deg, height 0.148 m); these apply when it is missing.
DEFAULT_FOV_DEG = 90.0
DEFAULT_HEIGHT_M = 0.14


@dataclass
class CameraCalibration:
    """Intrinsics + ground-relative extrinsics for one camera."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    height_m: float = DEFAULT_HEIGHT_M
    pitch_rad: float = 0.0
    # Provenance: "assumed_fov" means nobody measured this.
    source: str = "assumed_fov"
    notes: str = ""

    # ---- construction ---------------------------------------------------
    @classmethod
    def from_fov(
        cls,
        width: int,
        height: int,
        fov_horizontal_deg: float = DEFAULT_FOV_DEG,
        height_m: float = DEFAULT_HEIGHT_M,
        pitch_rad: float = 0.0,
        source: str = "assumed_fov",
    ) -> "CameraCalibration":
        """Fallback: square pixels, centred principal point, FOV-implied fx."""
        fx = width / (2.0 * math.tan(math.radians(fov_horizontal_deg) / 2.0))
        return cls(
            width=width, height=height,
            fx=fx, fy=fx,                      # square pixels
            cx=width / 2.0, cy=height / 2.0,
            height_m=height_m, pitch_rad=pitch_rad,
            source=source,
            notes=f"assumed {fov_horizontal_deg} deg horizontal FOV, unverified",
        )

    @property
    def measured(self) -> bool:
        """True only if the intrinsics came from real measurement."""
        return self.source not in ("assumed_fov", "")

    @property
    def fov_horizontal_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.fx)))

    def scaled_to(self, width: int, height: int) -> "CameraCalibration":
        """
        Rescale for a resized frame — depth, semantics and the raw frame all run
        at different sizes, and a projection computed at one is wrong at another.

        Assumes a plain resize with no crop, which is what the segmenter does.
        """
        sx, sy = width / self.width, height / self.height
        return CameraCalibration(
            width=width, height=height,
            fx=self.fx * sx, fy=self.fy * sy,
            cx=self.cx * sx, cy=self.cy * sy,
            height_m=self.height_m, pitch_rad=self.pitch_rad,
            source=self.source, notes=self.notes,
        )

    # ---- matrices -------------------------------------------------------
    def K(self) -> np.ndarray:
        """3x3 intrinsics, in the form OpenCV and SAM-TP expect."""
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def T_robot_to_cam(self) -> np.ndarray:
        """
        4x4 robot->camera transform: the axis relabel (robot x-fwd/y-left/z-up
        to camera X-right/Y-down/Z-fwd) composed with the downward pitch.
        """
        t = self.pitch_rad
        c, s = math.cos(t), math.sin(t)
        # Level camera: X=-y, Y=-z, Z=x.
        R_level = np.array(
            [[0.0, -1.0, 0.0],
             [0.0, 0.0, -1.0],
             [1.0, 0.0, 0.0]], dtype=np.float64
        )
        R_pitch = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, c, -s],
             [0.0, s, c]], dtype=np.float64
        )
        T = np.eye(4)
        T[:3, :3] = R_pitch @ R_level
        # Camera sits height_m up the robot's z axis.
        T[:3, 3] = T[:3, :3] @ np.array([0.0, 0.0, -self.height_m])
        return T

    # ---- projection -----------------------------------------------------
    def project(self, pts_robot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        (N,3) robot-frame points -> ((N,2) pixels, (N,) bool in-front mask).

        The mask matters: points behind the camera project to plausible pixels
        with a sign flip, so drawing without it puts a backwards-going candidate
        cleanly in the middle of the image.
        """
        p = np.atleast_2d(np.asarray(pts_robot, dtype=np.float64))
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        t = self.pitch_rad
        c, s = math.cos(t), math.sin(t)
        Xl, Yl, Zl = -y, -(z - self.height_m), x
        X = Xl
        Y = Yl * c - Zl * s
        Z = Yl * s + Zl * c
        valid = Z > 1e-6
        Zs = np.where(valid, Z, 1.0)
        u = self.cx + self.fx * X / Zs
        v = self.cy + self.fy * Y / Zs
        return np.stack([u, v], axis=1), valid

    def project_ground(self, xy_robot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,2) robot-frame ground points -> pixels. What the overlay draws."""
        xy = np.atleast_2d(np.asarray(xy_robot, dtype=np.float64))
        pts = np.concatenate([xy, np.zeros((len(xy), 1))], axis=1)
        return self.project(pts)

    def image_to_ground(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Pixels -> robot-frame ground points, intersecting the pixel ray with z=0.

        Returns (xy, valid). Rays at or above the horizon never meet the ground
        and come back INVALID rather than as a huge bogus distance — which would
        put a phantom drivable surface at 400 m and drive the rover at a wall.
        """
        p = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        X = (p[:, 0] - self.cx) / self.fx
        Y = (p[:, 1] - self.cy) / self.fy
        Z = np.ones_like(X)
        t = self.pitch_rad
        c, s = math.cos(t), math.sin(t)
        # Un-pitch back to the level camera frame.
        Yl = Y * c + Z * s
        Zl = -Y * s + Z * c
        # Level camera -> robot direction.
        dx, dy, dz = Zl, -X, -Yl
        # Ray from (0, 0, height_m) hits z=0 at t = height_m / -dz.
        valid = dz < -1e-9
        tt = np.where(valid, self.height_m / np.where(valid, -dz, 1.0), 0.0)
        return np.stack([tt * dx, tt * dy], axis=1), valid

    def horizon_v(self) -> float:
        """
        Image row of the horizon; above it is never ground.

        As a ground point recedes, Y -> -D*sin(t) and Z -> D*cos(t), so
        v -> cy - fy*tan(t). Note the MINUS: pitching down raises the horizon.
        """
        return self.cy - self.fy * math.tan(self.pitch_rad)

    # ---- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CameraCalibration":
        return cls(**json.loads(Path(path).read_text()))

    @classmethod
    def load_or_default(
        cls, path: str | Path, width: int, height: int
    ) -> "CameraCalibration":
        """Real calibration if one exists, else the documented guess."""
        p = Path(path)
        if p.is_file():
            try:
                return cls.load(p).scaled_to(width, height)
            except Exception as e:
                print(f"[calib] {p} unreadable ({e}) — falling back to assumed FOV")
        return cls.from_fov(width, height)


def extrinsics_from_ground_plane(
    normal_cam: np.ndarray, offset_d: float
) -> tuple[float, float]:
    """
    Recover (height_m, pitch_rad) from a fitted ground plane — a free extrinsic
    calibration, since `DepthCostmap._fit_ground` RANSACs one every tick.

    Same convention `_fit_ground` returns: `normal_cam` a unit normal oriented
    up (camera frame, Y down), `offset_d` such that height above ground is
    `n . p - d`. For a level camera the up-normal is (0, -1, 0); pitching down
    by t rotates it to (0, -cos t, -sin t). The camera is at the origin, so its
    own height above the plane is -d.
    """
    n = np.asarray(normal_cam, dtype=np.float64).ravel()
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    pitch = math.atan2(-float(n[2]), -float(n[1]))
    return float(-offset_d), float(pitch)
