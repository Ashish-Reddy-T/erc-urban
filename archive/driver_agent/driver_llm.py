"""
The API driver's brain: one model call per decision, one action back.

The model sees what a remote driver sees (front camera with numbered heading
candidates, a north-up map, the rear camera) plus a status line and a journal of
its last few actions and what they did. It returns ONE action as JSON:

  auto  distance_m      let the local planner drive toward the checkpoint
  drive candidate, d_m  hold candidate heading 1..7 for d_m (planner keeps it safe)
  turn  turn_deg        face a new direction (+ = left), then creep 0.5 m
  back  distance_m      reverse out of a dead end
  push  candidate, d_m  drive candidate heading 0.5..3 m IGNORING the camera obstacle
                        stop (leaf litter / grass it wrongly calls blocked); tilt and
                        stall guards still stop it

Backends, picked by whichever key is set (override with DRIVER_BACKEND):
  anthropic   ANTHROPIC_API_KEY  -> official SDK, DRIVER_MODEL (default claude-sonnet-5)
  openrouter  OPENROUTER_API_KEY -> OpenAI-compatible, DRIVER_MODEL e.g. google/gemini-2.5-flash
  gemini      GEMINI_API_KEY     -> Gemini's OpenAI-compatible endpoint,
                                    DRIVER_MODEL default gemini-3.8-flash (or gemini-robotics-er-2-preview)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

ACTIONS = ("auto", "drive", "turn", "back", "push")

SYSTEM = """You are the remote driver of a small wheeled robot (35 cm long, 20 cm wide, camera 15 cm above the ground) in a navigation competition. You score one point per GPS checkpoint reached, in order. The robot FLIPPING scores zero for the whole run, so never drive onto steep banks, kerbs, steps, rocks, stairs, or anything it would have to climb.

A local planner executes your action safely: it stops before obstacles it can see, backs off if the robot starts to tilt, and claims each checkpoint automatically as soon as it is within 15 m, so you only need to get within 15 m. Your job is the judgement it lacks: which way actually leads to the checkpoint along ground a small robot can drive.

Each step you get:
- FRONT camera with numbered heading candidates 1 (far left, +60 deg) to 7 (far right, -60 deg), 4 = straight. Arrow colour = measured clearance along it (green open, yellow short, red blocked). A magenta line marks the checkpoint direction when it is in view.
- MAP (north up): grey = mapped roads/paths, green dots = checkpoints (big = current), orange arrow = robot and its heading, blue = where it has been.
- Optionally the REAR camera.
- STATUS: checkpoint distance and bearing, tilt, and a journal of your last actions and what happened.

Actions (answer with exactly one):
- "auto": the local planner drives toward the checkpoint by itself for distance_m (2-10). Use it whenever the way toward the checkpoint is open, flat and obvious (a clear road, path, lawn or plaza). It is the fastest option.
- "drive": hold candidate heading (1-7) for distance_m (1-6). Use it to follow a path or sidewalk that bends, go through a gap, or go around something the direct line would hit.
- "turn": rotate in place toward turn_side ("left" or "right") by turn_deg (30-180), then creep 0.5 m. Use it to look for a way when nothing good is in view, or when the checkpoint is behind.
- "back": reverse distance_m (0.5-2) when pressed against something or in a dead end.
- "push": drive candidate heading (1-7) for distance_m (0.5-3) IGNORING the planner's camera obstacle stop. The planner's obstacle detector is often wrong about dry leaves, leaf litter, short or patchy grass, twigs, shadows, sun patches and painted lines: it marks them red/yellow and refuses to drive, or keeps escaping. When you can clearly see that the ground along a candidate is flat and drivable and it is only that kind of harmless cover, push through it. NEVER push toward tree trunks, roots taller than a finger, rocks, bushes, walls, kerbs, steps, fences, people, water, or any slope or drop. The tilt and stall guards still stop a push.

Rules of thumb: prefer roads, sidewalks, paths and flat open ground; stay off flower beds, bushes, grass verges next to drops, and anything soft or steep; go around, not over. If the journal shows the same action failing twice (escape, no progress), choose something different; if auto/drive keep ending in escapes on ground you can see is harmless leaves or grass, use push. Keep "reason" short (under 20 words); it is shown to spectators.

Reply with JSON only: {"action": "auto|drive|turn|back|push", "candidate": 1-7 or 0, "turn_side": "left|right", "turn_deg": 0-180, "distance_m": number, "reason": "..."}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "candidate": {"type": "integer"},
        "turn_side": {"type": "string", "enum": ["left", "right"]},
        "turn_deg": {"type": "integer"},
        "distance_m": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["action", "candidate", "turn_side", "turn_deg", "distance_m", "reason"],
    "additionalProperties": False,
}


@dataclass
class Action:
    action: str
    candidate: int = 4
    turn_deg: int = 0
    distance_m: float = 3.0
    reason: str = ""
    latency_s: float = 0.0
    fallback: bool = False


def parse_action(obj) -> Action:
    """Validate and clamp a model reply; anything unusable becomes 'auto'."""
    if isinstance(obj, str):
        t = obj[obj.find("{"): obj.rfind("}") + 1] if "{" in obj else ""
        obj = json.loads(t) if t else {}
    if not isinstance(obj, dict) or "action" not in obj:
        return Action("auto", reason="(no action in reply -> auto)", fallback=True)
    act = str(obj.get("action", "auto")).lower().strip()
    if act not in ACTIONS:
        return Action("auto", reason="(unparseable reply -> auto)", fallback=True)
    try:
        cand = int(obj.get("candidate", 4) or 4)
        turn = abs(int(obj.get("turn_deg", 0) or 0))
        side = str(obj.get("turn_side", "")).lower()
        if side == "right":
            turn = -turn
        elif side != "left":            # no side given: trust the sign as sent
            turn = int(obj.get("turn_deg", 0) or 0)
        dist = float(obj.get("distance_m", 3.0) or 3.0)
    except (TypeError, ValueError):
        return Action("auto", reason="(bad numbers -> auto)", fallback=True)
    limits = {"auto": (2.0, 10.0), "drive": (1.0, 6.0), "turn": (0.5, 0.5), "back": (0.5, 2.0),
              "push": (0.5, 3.0)}
    lo, hi = limits[act]
    if act in ("drive", "push") and not 1 <= cand <= 7:
        cand = 4
    if act == "turn" and turn == 0:
        turn = 90
    return Action(act, cand, max(-180, min(180, turn)), max(lo, min(hi, dist)),
                  str(obj.get("reason", ""))[:200])


class _Anthropic:
    def __init__(self, model: str):
        import anthropic
        self.client = anthropic.Anthropic(timeout=25.0, max_retries=1)
        self.model = model

    def ask(self, system: str, images: list[tuple[str, str]], text: str) -> dict:
        content = []
        for label, b64 in images:
            content.append({"type": "text", "text": label})
            content.append({"type": "image", "source": {"type": "base64",
                                                        "media_type": "image/jpeg", "data": b64}})
        content.append({"type": "text", "text": text})
        r = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": content}],
        )
        if r.stop_reason == "refusal":
            raise RuntimeError("model refused")
        return json.loads(next(b.text for b in r.content if b.type == "text"))


class _OpenAICompat:
    """OpenRouter (Qwen / Kimi / Gemini ...) or Gemini's OpenAI-compatible endpoint."""

    def __init__(self, model: str, base_url: str, api_key: str):
        import openai
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=25.0, max_retries=1)
        self.model = model

    def ask(self, system: str, images: list[tuple[str, str]], text: str):
        content = []
        for label, b64 in images:
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": text})
        # Gemini 3.x thinks before answering and the thinking counts against
        # max_tokens: at 400 half the replies came back cut off (empty "auto").
        kw = {"reasoning_effort": "low"} if "gemini" in self.model else {}
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=2048,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            **kw,
        )
        return r.choices[0].message.content or ""


class _Fake:
    """Scripted replies for tests (DRIVER_BACKEND=fake). DRIVER_FAKE_SCRIPT is a
    comma list cycled forever: auto:6 | drive:2:3 | push:4:2 | turn:right:60 | back:1 | error | junk"""

    def __init__(self, script: str):
        self.script = [x for x in script.split(",") if x] or ["auto:6"]
        self.i = 0

    def ask(self, system, images, text):
        s = self.script[self.i % len(self.script)].split(":")
        self.i += 1
        if s[0] == "error":
            raise RuntimeError("scripted API error")
        if s[0] == "junk":
            return "I think you should go left"
        if s[0] in ("drive", "push"):
            return {"action": s[0], "candidate": int(s[1]), "turn_side": "left", "turn_deg": 0,
                    "distance_m": float(s[2]), "reason": f"scripted {s[0]}"}
        if s[0] == "turn":
            return {"action": "turn", "candidate": 0, "turn_side": s[1], "turn_deg": int(s[2]),
                    "distance_m": 0.5, "reason": "scripted turn"}
        if s[0] == "back":
            return {"action": "back", "candidate": 0, "turn_side": "left", "turn_deg": 0,
                    "distance_m": float(s[1]), "reason": "scripted back"}
        return {"action": "auto", "candidate": 0, "turn_side": "left", "turn_deg": 0,
                "distance_m": float(s[1]) if len(s) > 1 else 6.0, "reason": "scripted auto"}


def make_backend():
    """(backend, name) from DRIVER_BACKEND / DRIVER_MODEL and whichever key is set."""
    want = os.environ.get("DRIVER_BACKEND", "").strip().lower()
    model = os.environ.get("DRIVER_MODEL", "").strip()
    if want == "fake":
        return _Fake(os.environ.get("DRIVER_FAKE_SCRIPT", "auto:6")), "fake"
    if want == "anthropic" or (not want and os.environ.get("ANTHROPIC_API_KEY")):
        m = model or "claude-sonnet-5"
        return _Anthropic(m), f"anthropic:{m}"
    if want == "openrouter" or (not want and os.environ.get("OPENROUTER_API_KEY")):
        m = model or "google/gemini-2.5-flash"
        return (_OpenAICompat(m, "https://openrouter.ai/api/v1", os.environ["OPENROUTER_API_KEY"]),
                f"openrouter:{m}")
    if want == "gemini" or (not want and os.environ.get("GEMINI_API_KEY")):
        m = model or "gemini-3.8-flash"
        return (_OpenAICompat(m, "https://generativelanguage.googleapis.com/v1beta/openai/",
                              os.environ["GEMINI_API_KEY"]), f"gemini:{m}")
    raise RuntimeError("no driver model key: set ANTHROPIC_API_KEY, OPENROUTER_API_KEY or GEMINI_API_KEY")


class Driver:
    def __init__(self, site_notes: str = ""):
        self.backend, self.name = make_backend()
        self.system = SYSTEM + (f"\n\nSITE NOTES (from the team):\n{site_notes}" if site_notes else "")
        self.errors = 0

    def decide(self, images: list[tuple[str, str]], text: str) -> Action:
        t0 = time.monotonic()
        try:
            a = parse_action(self.backend.ask(self.system, images, text))
            self.errors = 0
        except Exception as e:
            self.errors += 1
            a = Action("auto", distance_m=4.0, fallback=True,
                       reason=f"(model error {type(e).__name__}: {str(e)[:80]} -> auto)")
        a.latency_s = time.monotonic() - t0
        return a
