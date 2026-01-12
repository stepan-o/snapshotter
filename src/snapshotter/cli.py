from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from . import __version__
from . import main as main_module


def _load_job_payload(args: argparse.Namespace) -> Dict[str, Any]:
    # Prefer explicit file when provided (local/dev), otherwise use env var (AWS)
    if args.job_file:
        path = Path(args.job_file)
        if not path.exists():
            raise FileNotFoundError(f"Job file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    env_value = os.environ.get("SNAPSHOTTER_JOB_JSON")
    if not env_value:
        raise RuntimeError(
            "No job payload provided. Set SNAPSHOTTER_JOB_JSON or use --job-file path.json"
        )
    return json.loads(env_value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="snapshotter",
        description="Repo Snapshotter v0.1 (scaffold)",
    )
    parser.add_argument(
        "--job-file",
        dest="job_file",
        metavar="PATH",
        help="Path to a JSON file containing the job payload (local/dev)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no side effects)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"snapshotter {__version__}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        job = _load_job_payload(args)
        # Call the (currently unimplemented) core logic. We intentionally
        # allow NotImplementedError to be handled the same as any other error
        # to provide structured error JSON on stdout and a non-zero exit code.
        result = main_module.run(job_payload=job, dry_run=bool(args.dry_run))
        print(json.dumps({"ok": True, "result": result}, separators=(",", ":")))
        return 0
    except Exception as e:  # noqa: BLE001 - top-level CLI error handler
        error_obj = {
            "ok": False,
            "error": type(e).__name__,
            "message": str(e),
        }
        print(json.dumps(error_obj, separators=(",", ":")), file=sys.stdout)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
