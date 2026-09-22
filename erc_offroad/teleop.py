"""
WASD teleop that RECORDS. Stdlib only -- no `keyboard` package, no sudo.

    python -m erc_offroad.teleop

  w / s ...... burst forward / backward
  a / d ...... burst left / right   (positive angular = LEFT, verified)
  + / - ...... adjust burst power
  [ / ] ...... adjust burst duration
  space ...... STOP now
  f .......... grab a full-resolution frame right now (use this at each flag)
  l .......... toggle lamp
  q .......... quit (sends stop)

BURSTS, NOT HOLD-TO-DRIVE
-------------------------
A terminal in raw mode gets key-DOWN only; there is no key-up event, and
auto-repeat does not start for ~0.5 s. Hold-to-drive built on that either
runs away when a key-up is missed or stutters on the repeat delay. Neither is
acceptable on a rover that reached 44 deg of transient tilt tonight.

So each keypress issues one bounded burst that stops itself. Holding a key
repeats bursts. Space stops immediately. This is the same discipline as
`creep.py`, just driven by a human instead of a planner.

The command still STREAMS at 5 Hz for the burst duration, because the Mini+
stops when commands stop arriving.

WHAT IT RECORDS
---------------
Into runs/offroad/teleop_<stamp>/ :
  * telemetry.jsonl -- one line per tick: full raw /data, the command in force,
    and derived roll/pitch/tilt. This is what makes the log trainable: commands
    aligned to outcomes on the same clock.
  * frames/NNNNNN.png + frames.jsonl -- front frames on a background thread at
    ~1 Hz so capture latency never stalls the control loop, each stamped with
    the same monotonic clock as the telemetry.

SAFETY
------
Absolute tilt is displayed every tick and an auto-stop fires above
AUTOSTOP_TILT_DEG. The guard cannot save you from a slope you chose to climb --
it can only stop adding power once you are already going over.
"""
from __future__ import annotations

import json
import os
import select
import sys
import termios
import threading
import time
import tty

from .creep import _get, _post, attitude, HZ

AUTOSTOP_TILT_DEG = 35.0
FRAME_HZ = 1.0


class Recorder:
    def __init__(self, out: str):
        self.dir = out
        self.fdir = os.path.join(out, "frames")
        os.makedirs(self.fdir, exist_ok=True)
        self.tel = open(os.path.join(out, "telemetry.jsonl"), "a")
        self.fidx = open(os.path.join(out, "frames.jsonl"), "a")
        self.n = 0
        self.stop = False
        self.t0 = time.monotonic()

    def clock(self) -> float:
        return time.monotonic() - self.t0

    def log_tick(self, raw: dict, lin: float, ang: float, att) -> None:
        rec = {"t": self.clock(), "wall": time.time(),
               "cmd": {"linear": lin, "angular": ang}, "data": raw}
        if att is not None:
            rec["att"] = {"roll": att.roll, "pitch": att.pitch,
                          "tilt": att.tilt, "gz": att.gz,
                          "moving": att.moving, "age_s": att.age_s}
        self.tel.write(json.dumps(rec) + "\n")
        self.tel.flush()

    def grab(self, tag: str = "auto") -> None:
        """Called from the frame thread AND from the 'f' key."""
        import base64, io
        from PIL import Image
        try:
            d = _get("/v2/screenshot", timeout=90)
        except Exception:
            return
        b = d.get("front_frame")
        if not b:
            return
        if "," in b:
            b = b.split(",", 1)[1]
        t = self.clock()
        name = f"{self.n:06d}.png"
        Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB").save(
            os.path.join(self.fdir, name))
        self.fidx.write(json.dumps({"t": t, "file": name, "tag": tag}) + "\n")
        self.fidx.flush()
        self.n += 1

    def frame_loop(self) -> None:
        while not self.stop:
            self.grab("auto")
            time.sleep(1.0 / FRAME_HZ)

    def close(self) -> None:
        self.stop = True
        self.tel.close()
        self.fidx.close()


class Streamer(threading.Thread):
    """
    Streams the CURRENT command continuously at HZ, with no gaps.

    The previous design issued each keypress as a self-terminating burst, which
    meant every burst ENDED by sending a zero command. Holding a key therefore
    produced drive-stop-drive-stop, and on rock every restart has to break
    static friction from scratch -- the single worst pattern for this surface,
    and a large part of why turning felt dead.

    Now a key only sets a target command and pushes out a deadline. This thread
    keeps sending that command, unchanged and ungapped, until the deadline
    passes; then it sends zero. Terminal auto-repeat (which idles ~0.5 s before
    repeating) refreshes the deadline well within HOLD_S, so holding a key gives
    genuinely continuous drive, while releasing it still stops the rover
    promptly. The command watchdog is satisfied either way.
    """

    def __init__(self, rec: "Recorder"):
        super().__init__(daemon=True)
        self.rec = rec
        self.lock = threading.Lock()
        self.lin = 0.0
        self.ang = 0.0
        self.lamp = 0
        self.deadline = 0.0
        self.stop_flag = False
        self.last_att = None
        self.autostopped = False

    def set(self, lin: float, ang: float, hold_s: float) -> None:
        """
        Apply a command immediately AND latch it for `hold_s`.

        The POST happens here, synchronously, rather than waiting for this
        thread's next tick. Measured: waiting cost up to 0.22 s between the
        keypress and the rover hearing about it, which on top of the ~0.8 s
        telemetry lag is exactly what makes teleop feel dead.
        """
        with self.lock:
            self.lin, self.ang = lin, ang
            self.deadline = time.monotonic() + hold_s
            self.autostopped = False
            lamp = self.lamp
        try:
            _post("/control", {"command": {"linear": lin, "angular": ang,
                                           "lamp": lamp}})
        except Exception:
            pass

    def halt(self) -> None:
        with self.lock:
            self.lin = self.ang = 0.0
            self.deadline = 0.0
            lamp = self.lamp
        try:
            _post("/control", {"command": {"linear": 0.0, "angular": 0.0,
                                           "lamp": lamp}})
        except Exception:
            pass

    def set_lamp(self, v: int) -> None:
        with self.lock:
            self.lamp = v

    def run(self) -> None:
        dt = 1.0 / HZ
        while not self.stop_flag:
            with self.lock:
                live = time.monotonic() < self.deadline
                lin = self.lin if live else 0.0
                ang = self.ang if live else 0.0
                lamp = self.lamp
            try:
                _post("/control", {"command": {"linear": lin, "angular": ang,
                                               "lamp": lamp}})
                raw = _get("/data", timeout=8)
                att = attitude(raw)
                self.last_att = att
                self.rec.log_tick(raw, lin, ang, att)
                if att is not None and att.tilt > AUTOSTOP_TILT_DEG:
                    with self.lock:
                        self.deadline = 0.0
                        self.lin = self.ang = 0.0
                        self.autostopped = True
                    _post("/control", {"command": {"linear": 0.0,
                                                   "angular": 0.0,
                                                   "lamp": lamp}})
            except Exception:
                pass
            time.sleep(dt)


def main() -> int:
    stamp = time.strftime("%H%M%S")
    out = os.path.join("runs", "offroad", f"teleop_{stamp}")
    os.makedirs(out, exist_ok=True)
    rec = Recorder(out)
    threading.Thread(target=rec.frame_loop, daemon=True).start()
    st = Streamer(rec)
    st.start()

    power = 0.30
    # `hold` is how long motion CONTINUES AFTER THE LAST KEYPRESS -- it is not
    # a delay, and not a gap. Tap = a short nudge; hold the key = continuous
    # motion, because macOS auto-repeat (~0.5 s to first repeat) refreshes the
    # deadline inside this window. Release and the rover stops within `hold`.
    hold_lin = 0.7
    hold_turn = 0.7
    lamp = 0
    print(f"recording -> {out}")
    print("w/s drive   a/d turn   space STOP   +/- power")
    print("[ / ] turn-hold   , / . drive-hold   f frame   l lamp   q quit")
    print("TAP = short nudge.  HOLD the key = continuous motion.")
    print("'hold' = how long it keeps moving after you let go, not a delay.\n")

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            if not select.select([sys.stdin], [], [], 0.2)[0]:
                a = st.last_att
                if a:
                    warn = ("  <-- AUTO-STOP" if st.autostopped
                            else ("  <-- HIGH" if a.tilt > 25 else ""))
                    print(f"\r  tilt {a.tilt:5.1f}  roll{a.roll:+6.1f} "
                          f"pitch{a.pitch:+6.1f}  pwr {power:.2f}  "
                          f"turn-hold {hold_turn:.1f}s{warn}      ",
                          end="", flush=True)
                continue

            c = sys.stdin.read(1).lower()
            if c == "q":
                break
            elif c == " ":
                st.halt(); print("\n  STOP"); continue
            elif c == "f":
                rec.grab("manual"); print(f"\n  frame {rec.n}"); continue
            elif c == "l":
                lamp = 0 if lamp else 1
                st.set_lamp(lamp); print(f"\n  lamp {lamp}"); continue
            elif c in "+=":
                power = min(0.60, round(power + 0.02, 3))
                print(f"\n  power {power:.2f}"); continue
            elif c in "-_":
                power = max(0.10, round(power - 0.02, 3))
                print(f"\n  power {power:.2f}"); continue
            elif c == "]":
                hold_turn = min(6.0, round(hold_turn + 0.5, 2))
                print(f"\n  turn-hold {hold_turn:.1f}s"); continue
            elif c == "[":
                hold_turn = max(0.5, round(hold_turn - 0.5, 2))
                print(f"\n  turn-hold {hold_turn:.1f}s"); continue
            elif c == ".":
                hold_lin = min(3.0, round(hold_lin + 0.2, 2))
                print(f"\n  drive-hold {hold_lin:.1f}s"); continue
            elif c == ",":
                hold_lin = max(0.2, round(hold_lin - 0.2, 2))
                print(f"\n  drive-hold {hold_lin:.1f}s"); continue
            elif c == "w":
                st.set(power, 0.0, hold_lin)
            elif c == "s":
                st.set(-power, 0.0, hold_lin)
            elif c == "a":
                st.set(0.0, power, hold_turn)     # + = LEFT (verified)
            elif c == "d":
                st.set(0.0, -power, hold_turn)
            else:
                continue
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        st.halt()
        st.stop_flag = True
        time.sleep(0.4)
        for _ in range(3):
            try:
                _post("/control", {"command": {"linear": 0.0, "angular": 0.0,
                                               "lamp": 0}})
            except Exception:
                pass
            time.sleep(0.1)
        rec.close()
        print(f"\nstopped. recorded {rec.n} frames -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
