# src/snapshotter/cli.py
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from . import __version__
from . import main as main_module

STAGE_PARSE_JOB = "parse_job"


def _read_job_payload_required(_: argparse.Namespace) -> tuple[dict[str, Any], str]:
    """
    Contract (hard):
      - The ONLY canonical job input is SNAPSHOTTER_JOB_JSON (env JSON string).
      - No file paths, no stdin, no dotenv, no fallback keys.
      - Missing/empty/invalid JSON -> hard fail at stage=parse_job.

    Returns: (payload_dict, payload_src_string)
    """
    raw = os.environ.get("SNAPSHOTTER_JOB_JSON")
    if not raw or not raw.strip():
        raise RuntimeError(
            "Missing required job payload: set SNAPSHOTTER_JOB_JSON to a JSON object string."
        )

    payload = json.loads(raw)

    if not isinstance(payload, dict):
        raise TypeError("SNAPSHOTTER_JOB_JSON must decode to a JSON object (dict).")

    return payload, "env:SNAPSHOTTER_JOB_JSON"


def _print_success(obj: dict[str, Any]) -> None:
    # Keep output schema stable: graph result is already structured.
    print(json.dumps(obj, separators=(",", ":")), file=sys.stdout)


def _print_failure(stage: str, err: Exception) -> None:
    out = {
        "ok": False,
        "stage": stage,
        "error_code": f"SNAPSHOTTER_FAILED_{stage.upper()}",
        "error_message": str(err),
    }
    print(json.dumps(out, separators=(",", ":")), file=sys.stdout)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="snapshotter",
        description="Repo Snapshotter (graph-only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no side effects).",
    )
    parser.add_argument(
        "--aws-region",
        dest="aws_region",
        metavar="REGION",
        help="Optional AWS region override (otherwise AWS_REGION/AWS_DEFAULT_REGION are used).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"snapshotter {__version__}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Contract: dry_run is CLI-controlled only (no env fallback).
    dry_run = bool(args.dry_run)

    try:
        payload, payload_src = _read_job_payload_required(args)

        result = main_module.run(
            job_payload=payload,
            dry_run=dry_run,
            payload_src=payload_src,
            aws_region=args.aws_region,
        )
        _print_success(result)
        return 0

    except Exception as e:  # noqa: BLE001 - top-level CLI error handler
        # If graph surfaced a stage-aware error, render it.
        try:
            from snapshotter.graph import SnapshotterStageError  # type: ignore
        except Exception:
            SnapshotterStageError = None  # type: ignore

        if SnapshotterStageError is not None and isinstance(e, SnapshotterStageError):
            _print_failure(e.stage, e)
            return 1

        # Payload / argument issues are parse_job
        if isinstance(e, (json.JSONDecodeError, RuntimeError, TypeError)):
            _print_failure(STAGE_PARSE_JOB, e)
            return 1

        # Everything else: unknown stage
        _print_failure("unknown", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
