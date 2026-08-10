"""
Module 4 (+ M5 slow half) — VLM candidate selection with latency-resilient fusion.

Source: "Slow Brain, Fast Planner: Latency-Resilient VLM-Augmented Urban
Navigation" (arXiv 2606.20458). Verified against the paper rather than
paraphrased; every default below traces to one of their ablations.

WHY IT EXISTS
-------------
The planner's own scoring picks badly in exactly the situations that drive the
difficulty rubric -- junctions, pedestrian encounters, sidewalk/grass edges.
The paper measures a 30% ADE reduction from VLM selection over the planner's
best pick, and our own oracle-gap probe reproduces the shape: on this stack,
top-1 error is 16-22 deg where the oracle-best candidate is 0.1-2.1 deg, so
there is 15-41 deg of headroom for a selector to exploit -- and exactly 0 deg on
an unobstructed straight, matching their finding that the planner is already
competitive in routine situations.

WHAT WE TOOK, AND WHY IT IS NOT OPTIONAL
----------------------------------------
* Numbered overlays, VLM returns an index. Training-free.
* Goal and planner scores are HIDDEN from the prompt. Their ablation: shown the
  scores, the VLM defers to the planner's ranking instead of judging.
* Probability Fusion, not Score Fusion:
      p_fused = (1-a)*p_planner + a*p_vlm,   a = (lam/(lam+1)) * w(dt)
  bounded by a <= 1. The paper prefers it because Score Fusion's similarity term
  is unbounded and it is "markedly less sensitive to hyperparameter choice".
* Exponential staleness decay w(dt) = exp(-dt/tau), tau = 5 s deployed.
* Similarity with progress alignment and horizon truncation:
      Sim = -(1/d_scale) * (1/N_use) * sum ||tau_i[n] - tau_vlm_hat[n]||
  where tau_vlm_hat is the REMAINING segment of the stale trajectory after the
  rover's progress is accounted for. The old `_traj_dist` in upgrades.py did
  neither, so a stale pick was compared against paths starting from a position
  the rover had already left.

ASYNCHRONY IS A HARD REQUIREMENT HERE
-------------------------------------
Measured on this rover: it has a command watchdog. A single /control POST yields
+0.7 deg of yaw; streaming the same command at 2 Hz yields +12.2 deg and at 5 Hz
+18.6 deg. Command RATE is effective speed. A VLM call takes 1-3 s, so running
it inside the control tick would drop the command rate by an order of magnitude
and physically slow the rover to a crawl. The query therefore runs on its own
thread and the tick consumes whatever answer is already available.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np

from ..interfaces import Candidate, Observation

# Distinct, high-contrast, and colour-blind-tolerable. Deliberately not a
# rainbow ramp: adjacent candidates must be told apart at a glance, and the VLM
# is being asked to name one.
_PALETTE = [
    (255, 64, 64), (64, 200, 255), (255, 200, 0), (120, 255, 120),
    (255, 128, 220), (180, 160, 255), (255, 150, 60), (0, 220, 190),
    (240, 240, 240), (200, 120, 60), (150, 255, 220), (255, 90, 150),
    (120, 200, 60), (90, 150, 255), (230, 180, 120), (170, 170, 170),
    (255, 255, 140), (110, 230, 255),
]


def render_overlay(
    frame: np.ndarray,
    candidates: list[Candidate],
    calib,
    line_px: int = 3,
    label_px: float = 0.62,
    label_range_m: float = 1.1,
) -> np.ndarray:
    """
    Draw the candidate set onto the frame as numbered coloured polylines.

    This is the entire interface between our geometry and the VLM's judgement,
    so a mis-projected overlay is worse than no overlay: the model answers
    confidently about a path that is not where we will actually drive. It is
    also why calib.py exists -- the previous `_render_overlay` in upgrades.py
    returned the raw frame unchanged, giving the VLM nothing to select over.

    Follows the paper's rendering: coloured polyline per candidate, a dot at the
    endpoint, and an integer index on a colour-matched background.

    `label_range_m` is a deviation forced by this camera. The paper labels
    endpoints, but at 148 mm height the ground compresses brutally with range:
    measured, image row 350 is 0.74 m and row 300 is 1.99 m, with the horizon at
    270. A 2.5 m endpoint therefore lands within ~30 px of the horizon, so
    labelling endpoints stacks every number into a thin band against the
    buildings, where they are hard to associate with any ground position. The
    fan is visibly separated around 1 m, so the label goes there instead --
    same information, placed where a reader can actually use it.
    """
    try:
        import cv2
    except Exception:
        return frame

    img = np.ascontiguousarray(frame.copy())
    h, w = img.shape[:2]
    cal = calib.scaled_to(w, h)

    for i, cand in enumerate(candidates):
        colour = _PALETTE[i % len(_PALETTE)]
        wp = np.asarray(cand.waypoints, dtype=np.float64)
        uv, ok = cal.project_ground(wp)
        pts = [
            (int(round(u)), int(round(v)))
            for (u, v), good in zip(uv, ok)
            if good and -w < u < 2 * w and -h < v < 2 * h
        ]
        if len(pts) < 2:
            continue
        cv2.polylines(img, [np.array(pts, np.int32)], False, colour, line_px,
                      lineType=cv2.LINE_AA)
        end = pts[-1]
        cv2.circle(img, end, line_px + 3, colour, -1, lineType=cv2.LINE_AA)

        # Label where the fan is actually separated, not at the horizon.
        rng = np.linalg.norm(wp, axis=1)
        j = int(np.argmin(np.abs(rng - label_range_m)))
        anchor = end
        if ok[j] and -w < uv[j, 0] < 2 * w and -h < uv[j, 1] < 2 * h:
            anchor = (int(round(uv[j, 0])), int(round(uv[j, 1])))

        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_px, 2)
        # Stagger across three rows. The ground is so foreshortened at this
        # camera height that a whole fan of labels lands within ~40 px
        # vertically and they overprint each other; a VLM cannot pick an index
        # it cannot read. Staggering costs nothing and keeps them legible.
        row = (i % 3) * (th + 7)
        lx = int(np.clip(anchor[0] - tw // 2, 2, w - tw - 4))
        ly = int(np.clip(anchor[1] + th + 6 + row, th + 4, h - 4))
        cv2.rectangle(img, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 4), colour, -1)
        # Dark text on bright fills, light on dark: the index has to be legible
        # or the VLM is guessing at which number it is looking at.
        lum = 0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2]
        cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, label_px,
                    (0, 0, 0) if lum > 140 else (255, 255, 255), 2, cv2.LINE_AA)
    return img


PROMPT = (
    "You are helping a small ground robot drive on streets and sidewalks.\n"
    "The image shows several possible short paths the robot could take, each "
    "drawn as a coloured line with a number at its end.\n\n"
    "Choose the ONE numbered path that is safest and most sensible to drive "
    "now. Prefer paths that:\n"
    "- stay on drivable surface (pavement, sidewalk, road) and avoid grass, "
    "kerbs, steps and drops\n"
    "- keep clear of people, animals, vehicles and obstacles\n"
    "- follow the direction the walkway or road actually goes\n\n"
    "Reply with ONLY the number. No explanation."
)


class GeminiVertexVLM:
    """
    Gemini on Vertex AI, so the calls bill against existing GCP credits rather
    than a separate AI Studio key.

    Model default is `gemini-3.5-flash-lite`: the paper's ablation picks the
    Flash-Lite tier for the latency/quality sweet spot (1.7 s median), and at
    $0.30/1M input this is roughly $1.60 per driving hour at 1 Hz.

    Requires `aiplatform.googleapis.com` enabled on the project:
        gcloud services enable aiplatform.googleapis.com --project <PROJECT>
    """

    def __init__(
        self,
        project: Optional[str] = None,
        location: str = "global",
        model: str = "gemini-3.5-flash-lite",
        jpeg_quality: int = 80,
        timeout_s: float = 8.0,
    ):
        import os

        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.model = model
        self.jpeg_quality = jpeg_quality
        self.timeout_s = timeout_s
        self._client = None
        self.last_error: Optional[str] = None
        self.n_calls = 0
        self.n_errors = 0
        self.last_latency_s: Optional[float] = None

    def _client_or_none(self):
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(
                    vertexai=True, project=self.project, location=self.location
                )
            except Exception as e:
                self.last_error = str(e)
                return None
        return self._client

    def __call__(self, image: np.ndarray, k: int) -> Optional[int]:
        """Overlay image -> chosen index, or None on any failure."""
        client = self._client_or_none()
        if client is None:
            return None
        try:
            import cv2
            from google.genai import types

            ok, buf = cv2.imencode(
                ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                return None
            t0 = time.monotonic()
            resp = client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"),
                    PROMPT,
                ],
            )
            self.last_latency_s = time.monotonic() - t0
            self.n_calls += 1
            digits = "".join(ch for ch in (resp.text or "") if ch.isdigit())
            if not digits:
                return None
            idx = int(digits[:3])
            return idx if 0 <= idx < k else None
        except Exception as e:
            self.n_errors += 1
            self.last_error = str(e)
            return None


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 0 else np.full_like(e, 1.0 / len(e))


def _similarity(cand: np.ndarray, stale: np.ndarray, progress_m: float,
                d_scale: float) -> float:
    """
    Paper's Sim, with the two pieces the old implementation omitted.

    progress alignment: the rover has driven `progress_m` along the stale
    trajectory since it was chosen, so compare against the REMAINING segment.
    Without it, every fresh candidate is scored against a path that starts
    behind the rover, and the stale pick is penalised for the robot having moved.

    horizon truncation: only compare over the overlapping horizon N_use.
    """
    stale = np.asarray(stale, dtype=np.float64)
    if len(stale) < 2:
        return 0.0
    seg = np.linalg.norm(np.diff(stale, axis=0), axis=1)
    travelled = np.concatenate([[0.0], np.cumsum(seg)])
    start = int(np.searchsorted(travelled, max(progress_m, 0.0)))
    remaining = stale[start:]
    if len(remaining) < 2:
        return 0.0
    n = min(len(cand), len(remaining))
    d = float(np.mean(np.linalg.norm(cand[:n] - remaining[:n], axis=1)))
    return -d / max(d_scale, 1e-9)


class VLMSelector:
    """
    M4 + M5(slow). Honors the `Selector` protocol: select(obs, cands) -> index.

    Supersedes `upgrades.VLMSelectorFusion`, whose fusion algebra was already
    right but whose overlay was a no-op and whose similarity had neither
    progress alignment nor horizon truncation.
    """

    def __init__(
        self,
        calib,
        vlm_call: Optional[Callable[[np.ndarray, int], Optional[int]]] = None,
        tau_decay_s: float = 5.0,
        lam: float = 1.0,
        query_hz: float = 1.0,
        d_scale_m: float = 1.0,
        max_stale_s: float = 15.0,
        gate_queries: bool = True,
        contest_ratio: float = 0.25,
        hazard_frac: float = 0.01,
        ambiguous_corridor: float = 0.25,
    ):
        self.calib = calib
        self.vlm_call = vlm_call
        self.tau = tau_decay_s
        self.lam = lam
        self.query_dt = 1.0 / max(query_hz, 1e-6)
        self.d_scale = d_scale_m
        self.max_stale_s = max_stale_s

        self._lock = threading.Lock()
        self._pick: Optional[dict] = None      # {waypoints, stamp, idx}
        self._last_query = -1e9
        self._inflight = False
        self.gate_queries = gate_queries
        self.contest_ratio = contest_ratio
        self.hazard_frac = hazard_frac
        self.ambiguous_corridor = ambiguous_corridor
        self.n_queries = 0
        self.n_answers = 0
        self.n_skipped = 0
        self.last_gate = 'init'
        self.last_alpha = 0.0
        self.last_note = "no_pick"

    # ---- when is a query worth making ------------------------------------
    def is_decision_point(
        self,
        cands: list[Candidate],
        hazard_frac: float = 0.0,
        corridor_width: float | None = None,
    ) -> tuple[bool, str]:
        """
        Should the VLM be consulted at all on this tick?

        The VLM is a HIGH-LEVEL planner, not a per-tick controller. Querying it
        everywhere is wasteful on three axes -- API cost, 1.6-2.1 s of staleness,
        and (because the rover has a command watchdog) tick rate, which is
        literally speed. It is also unnecessary: Slow-Brain's own result is that
        the planner is already competitive in routine situations and the VLM
        earns its keep at junctions, pedestrian encounters and surface
        boundaries. Our oracle-gap probe reproduced exactly that shape -- 15-41
        deg of headroom at off-axis goals, and 0 deg on an unobstructed straight.

        So: query where the decision is genuinely contested, and stay quiet on
        an empty path. Three independent triggers, any of which fires:

          1. the candidate set disagrees -- top scores are close, so the
             planner itself has no strong opinion
          2. a hazard is visible (person, vehicle, water) in the near field
          3. the corridor is ambiguous -- narrow, or absent entirely
        """
        if len(cands) < 2:
            return False, "single_candidate"
        scores = np.sort(np.array([c.score for c in cands], dtype=np.float64))[::-1]
        margin = float(scores[0] - scores[1])
        spread = float(scores[0] - scores[-1])
        contested = spread > 1e-9 and (margin / spread) < self.contest_ratio

        if hazard_frac >= self.hazard_frac:
            return True, f"hazard({hazard_frac:.2f})"
        if corridor_width is not None and corridor_width < self.ambiguous_corridor:
            return True, f"ambiguous_corridor({corridor_width:.2f})"
        if contested:
            return True, f"contested(margin={margin/max(spread,1e-9):.2f})"
        return False, "routine"

    # ---- slow loop -------------------------------------------------------
    def maybe_query(self, obs: Observation, cands: list[Candidate],
                    hazard_frac: float = 0.0,
                    corridor_width: float | None = None,
                    force: bool = False) -> None:
        """Fire a query if due, warranted, and none is in flight. Never blocks."""
        now = time.monotonic()
        if not force and self.gate_queries:
            want, why = self.is_decision_point(cands, hazard_frac, corridor_width)
            self.last_gate = why
            if not want:
                self.n_skipped += 1
                return
        with self._lock:
            if (
                self.vlm_call is None
                or self._inflight
                or now - self._last_query < self.query_dt
                or not cands
            ):
                return
            self._last_query = now
            self._inflight = True

        overlay = render_overlay(obs.front, cands, self.calib)
        snapshot = [c.waypoints.copy() for c in cands]
        stamp = obs.stamp
        self.n_queries += 1

        def _run():
            idx = None
            try:
                idx = self.vlm_call(overlay, len(snapshot))
            except Exception:
                idx = None
            with self._lock:
                self._inflight = False
                if idx is not None and 0 <= idx < len(snapshot):
                    self._pick = {"waypoints": snapshot[idx], "stamp": stamp,
                                  "idx": int(idx)}
                    self.n_answers += 1

        threading.Thread(target=_run, daemon=True).start()

    # ---- fast loop -------------------------------------------------------
    def select(self, obs: Observation, cands: list[Candidate],
               progress_m: float = 0.0, hazard_frac: float = 0.0,
               corridor_width: float | None = None) -> int:
        if not cands:
            return 0
        self.maybe_query(obs, cands, hazard_frac, corridor_width)
        with self._lock:
            pick = dict(self._pick) if self._pick else None

        planner = np.array([c.score for c in cands], dtype=np.float64)
        if pick is None:
            self.last_alpha = 0.0
            self.last_note = "planner_only"
            return int(np.argmax(planner))

        dt = float(obs.stamp - pick["stamp"])
        if dt > self.max_stale_s or dt < 0.0:
            self.last_alpha = 0.0
            self.last_note = f"too_stale({dt:.1f}s)"
            return int(np.argmax(planner))

        w = float(np.exp(-dt / self.tau))
        alpha = (self.lam / (self.lam + 1.0)) * w
        sims = np.array(
            [_similarity(np.asarray(c.waypoints, float), pick["waypoints"],
                         progress_m, self.d_scale) for c in cands]
        )
        fused = (1.0 - alpha) * _softmax(planner) + alpha * _softmax(sims)
        self.last_alpha = alpha
        self.last_note = f"fused(dt={dt:.1f}s a={alpha:.2f})"
        return int(np.argmax(fused))

    def describe(self) -> str:
        return (f"vlm[{self.last_note} gate={self.last_gate} q={self.n_queries} "
                f"a={self.n_answers} skip={self.n_skipped}"
                f"{' inflight' if self._inflight else ''}]")
