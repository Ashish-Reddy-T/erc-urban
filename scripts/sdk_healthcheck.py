#!/usr/bin/env python3
"""Exit 0 if the earth-rovers-sdk answers GET /data with a JSON object.

    python scripts/sdk_healthcheck.py                    # $SDK_URL or localhost:8000
    python scripts/sdk_healthcheck.py --url http://host:8000 --timeout 3

Run it after starting the SDK and before launching a stack, so a dead
server or browser session fails fast instead of mid-run. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://localhost:8000"


def check(url: str, timeout: float) -> str | None:
    """Return None if healthy, else a one-line reason."""
    endpoint = url.rstrip("/") + "/data"
    try:
        with urllib.request.urlopen(endpoint, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        return f"{endpoint}: HTTP {e.code}"
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", e)
        return f"{endpoint}: {reason}"
    try:
        data = json.loads(body)
    except ValueError:
        return f"{endpoint}: response is not JSON"
    if not isinstance(data, dict):
        return f"{endpoint}: expected a JSON object, got {type(data).__name__}"
    return None


def main(argv: list[str] | None = None, environ=os.environ) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=environ.get("SDK_URL", DEFAULT_URL))
    ap.add_argument("--timeout", type=float, default=5.0,
                    help="seconds before giving up (default 5)")
    args = ap.parse_args(argv)
    err = check(args.url, args.timeout)
    if err:
        print(f"SDK unhealthy: {err}", file=sys.stderr)
        return 1
    print(f"SDK ok: {args.url.rstrip('/')}/data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
