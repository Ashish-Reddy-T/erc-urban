#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

REQUIRED_DIRS = [
    "docs",
    "earth-rovers-sdk",
    "mbra_repo",
    "src",
    "third_party",
    "models",
]

REQUIRED_FILES = [
    "earth-rovers-sdk/.env.sample",
    "earth-rovers-sdk/main.py",
    "mbra_repo/deployment/LogoNav_frodobot.py",
    "mbra_repo/deployment/utils_logonav.py",
    "src/earthrover_interface.py",
]

ENV_REQUIRED_KEYS = [
    "SDK_API_TOKEN",
    "BOT_SLUG",
    "CHROME_EXECUTABLE_PATH",
    "MAP_ZOOM_LEVEL",
]

ENV_OPTIONAL_KEYS = [
    "MISSION_SLUG",
]

TEXT_SCAN_EXTS = {
    ".md",
    ".py",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".sh",
}

PARENT_PATH_PATTERNS = [
    str(ROOT.parent),
]


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def is_placeholder(value: str) -> bool:
    lowered = value.lower()
    markers = [
        "example",
        "ex4mpl3",
        "/path/to/",
        "weld-arm-ultron",
        "mission-ex4mpl3",
        "3x4mpl3",
    ]
    return any(marker in lowered for marker in markers)


def find_checkpoints(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix in {".pth", ".pt", ".ckpt"})


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    ok: list[str] = []

    for rel in REQUIRED_DIRS:
        path = ROOT / rel
        if path.is_dir():
            ok.append(f"found directory: {rel}")
        else:
            errors.append(f"missing directory: {rel}")

    for rel in REQUIRED_FILES:
        path = ROOT / rel
        if path.is_file():
            ok.append(f"found file: {rel}")
        else:
            errors.append(f"missing file: {rel}")

    env_path = ROOT / "earth-rovers-sdk/.env"
    if not env_path.exists():
        errors.append(
            "missing local config: earth-rovers-sdk/.env "
            "(create it with `cp earth-rovers-sdk/.env.sample earth-rovers-sdk/.env`)"
        )
    else:
        env_values = parse_env(env_path)
        for key in ENV_REQUIRED_KEYS:
            value = env_values.get(key)
            if not value:
                errors.append(f"missing .env key: {key}")
            elif is_placeholder(value):
                errors.append(f".env key still looks like a placeholder: {key}")

        for key in ENV_OPTIONAL_KEYS:
            value = env_values.get(key)
            if not value:
                warnings.append(
                    f"optional .env key not set: {key} "
                    "(SDK can still run without it if you are not using mission mode)"
                )
            elif is_placeholder(value):
                warnings.append(f"optional .env key still looks like a placeholder: {key}")
        ok.append("validated local SDK .env")

    mbra_weights = find_checkpoints(ROOT / "mbra_repo/deployment/model_weights")
    if mbra_weights:
        ok.append(f"found MBRA/LogoNav weights: {len(mbra_weights)} file(s)")
    else:
        warnings.append(
            "no MBRA/LogoNav checkpoint found under mbra_repo/deployment/model_weights/ "
            "(required only for --controller mbra)"
        )

    project_weights = find_checkpoints(ROOT / "models")
    if project_weights:
        ok.append(f"found project checkpoints: {len(project_weights)} file(s)")
    else:
        warnings.append("no project-level checkpoints found under models/")

    depth_weights = find_checkpoints(ROOT / "third_party/Depth-Anything-V2/checkpoints")
    if depth_weights:
        ok.append(f"found depth checkpoints: {len(depth_weights)} file(s)")
    else:
        warnings.append("no Depth Anything checkpoints found under third_party/Depth-Anything-V2/checkpoints/")

    hardcoded_hits: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in TEXT_SCAN_EXTS:
            continue
        rel = path.relative_to(ROOT)
        if any(part.startswith(".git") for part in rel.parts):
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for pattern in PARENT_PATH_PATTERNS:
            if pattern in text:
                hardcoded_hits.append(f"{rel}: contains hardcoded parent path `{pattern}`")

    if hardcoded_hits:
        errors.extend(hardcoded_hits)
    else:
        ok.append("no hardcoded parent-repo paths detected")

    env_yml = ROOT / "mbra_repo/train/environment_mbra.yml"
    if env_yml.exists():
        text = env_yml.read_text()
        if re.search(r"^prefix:\s*/", text, flags=re.MULTILINE):
            errors.append("mbra_repo/train/environment_mbra.yml still contains a machine-specific conda prefix")
        else:
            ok.append("MBRA conda environment file has no machine-specific prefix")

    print("ERC-3 workspace verification")
    print(f"Root: {ROOT}")
    print()

    if ok:
        print("OK:")
        for item in ok:
            print(f"  - {item}")
        print()

    if warnings:
        print("WARNINGS:")
        for item in warnings:
            print(f"  - {item}")
        print()

    if errors:
        print("ERRORS:")
        for item in errors:
            print(f"  - {item}")
        print()
        print("Result: NOT READY")
        return 1

    print("Result: READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
