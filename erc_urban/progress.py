"""
Build-progress checker.

    python -m erc_urban.progress

Deliberately not a hand-maintained checklist. Each step declares the artifacts
that would exist if it were really done — modules, tests, config — and this
verifies them on disk. A hand-ticked list drifts from reality the moment
something is half-wired; this cannot, because the tick IS the check.

"Wired" is tracked separately from "built" on purpose. Step 2's modules existed
and passed their tests for a while before run.py imported any of them, and a
live log showed the rover still using the old dead-depth path throughout. Built
and wired are different claims.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _exists(rel: str) -> bool:
    return os.path.exists(os.path.join(ROOT, rel))


def _contains(rel: str, needle: str) -> bool:
    p = os.path.join(ROOT, rel)
    if not os.path.isfile(p):
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            return needle in f.read()
    except OSError:
        return False


@dataclass
class Step:
    n: str
    title: str
    module: str                       # plan.md module id
    files: list = field(default_factory=list)      # must exist
    tests: list = field(default_factory=list)      # must exist
    wired: list = field(default_factory=list)      # (file, needle) in run.py etc
    note: str = ""

    def status(self) -> tuple[str, str]:
        missing = [f for f in self.files if not _exists(f)]
        if self.files and missing:
            return "todo", f"missing {', '.join(missing[:2])}"
        if not self.files:
            return "todo", self.note
        untested = [t for t in self.tests if not _exists(t)]
        unwired = [n for f, n in self.wired if not _contains(f, n)]
        if unwired:
            return "built", f"not wired: {', '.join(unwired[:2])}"
        if untested:
            return "built", f"no tests: {', '.join(untested[:2])}"
        return "done", self.note


STEPS = [
    Step("0", "Repo hygiene + reference clones", "-",
         files=[".gitignore", "pytest.ini", "repos/clone.sh", "repos/DIVERGENCE.md"],
         note="10 upstream repos cloned; MBRA divergence audited"),
    Step("1", "K-candidate generator", "M1",
         files=["erc_urban/modules/candidates.py"],
         tests=["test/test_candidates.py"],
         wired=[("erc_urban/run.py", "MultiHypothesisPlanner")],
         note="anchors + LogoNav as scorer"),
    Step("2", "Camera calibration + traversability veto", "M6",
         files=["erc_urban/calib.py", "erc_urban/camera_front.json",
                "erc_urban/modules/traversability.py",
                "erc_urban/modules/depth_scale.py"],
         tests=["test/test_calib.py", "test/test_traversability.py"],
         wired=[("erc_urban/run.py", "TraversabilityVeto"),
                ("erc_urban/run.py", "DepthResponseFit"),
                ("erc_urban/modules/depth_gate.py", 'checkpoint_domain: str = "indoor"')],
         note="fx measured 340px/113deg; depth response fit"),
    Step("3", "Route-visual alignment", "M2",
         files=["erc_urban/modules/route.py"],
         tests=["test/test_route.py"],
         wired=[("erc_urban/run.py", "RouteAligner")],
         note="UrbanVLA-style; GPS as soft prior"),
    Step("4", "Visual odometry, trail, latency", "M5",
         files=["erc_urban/modules/odometry.py", "erc_urban/modules/commit.py"],
         tests=["test/test_odometry.py", "test/test_latency.py"],
         wired=[("erc_urban/run.py", "VisualOdometry"),
                ("erc_urban/run.py", "CommitWindow"),
                ("erc_urban/run.py", "GpsHealth"),
                ("erc_urban/modules/upgrades.py", "def integrate")],
         note="VO + breadcrumbs + turn-around escape; tau=5s"),
    Step("5", "VLM candidate selector", "M4",
         files=["erc_urban/modules/selector.py"],
         tests=["test/test_selector.py"],
         wired=[("erc_urban/run.py", "VLMSelector")],
         note="overlay + Vertex Gemini 3.5 Flash-Lite + Probability Fusion"),
    Step("6", "RL reactivity residual", "M3",
         files=[], note="S2E RAM residual in URBAN-SIM"),
    Step("7", "Train traversability head", "M8",
         files=[], note="FrodoBots-Mini-4K, adverse conditions ($250 GCP)"),
    Step("8", "Eval harness + benchmarks", "M9",
         files=["erc_urban/eval/tick_budget.py", "erc_urban/eval/record.py",
                "erc_urban/eval/oracle_gap.py", "erc_urban/eval/selector_bench.py",
                "erc_urban/eval/frodobots.py"],
         tests=["test/test_eval.py"],
         wired=[("erc_urban/eval/oracle_gap.py", "def ade"),
                ("erc_urban/eval/frodobots.py", "_video_")],
         note="tick budget, FrodoBots loader, oracle gap, selector vs random"),
    Step("9", "Documentation", "-",
         files=[], note="docs/documentation.md + PROVENANCE.md"),
]

MARK = {"done": "[x]", "built": "[~]", "todo": "[ ]"}


def main() -> int:
    rows = [(s, *s.status()) for s in STEPS]
    done = sum(1 for _, st, _ in rows if st == "done")
    built = sum(1 for _, st, _ in rows if st == "built")

    print("\n  ERC Urban — build progress\n")
    print(f"  {'':4}{'step':<40} {'module':<7} note")
    print("  " + "-" * 76)
    for s, st, note in rows:
        print(f"  {MARK[st]} {s.n}. {s.title:<36} {s.module:<7} {note}")
    print("  " + "-" * 76)
    print(f"  {done}/{len(STEPS)} done" + (f", {built} built-but-incomplete" if built else ""))
    print("\n  [x] done   [~] built, not wired/tested   [ ] not started\n")
    print("  Verify:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test -q")
    print("           python -m unittest discover -s test -t .   (rover_env)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
