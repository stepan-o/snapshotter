# src/snapshotter/cli.py
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from . import __version__
from . import main as main_module


STAGE_PARSE_JOB = "parse_job"


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "y", "on"}


def _maybe_load_dotenv() -> None:
    """
    Local convenience only.

    - If python-dotenv isn't installed, this is a no-op.
    - If SNAPSHOTTER_DISABLE_DOTENV is truthy, skip.
    - We load once, early, before reading env-based payloads/flags.
    """
    if _env_bool("SNAPSHOTTER_DISABLE_DOTENV", default=False):
        return
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


def _read_job_payload_required(args: argparse.Namespace) -> Tuple[Dict[str, Any], str]:
    """
    Canonical job input (priority order):
      1) --job-file (CLI arg)
      2) SNAPSHOTTER_JOB_JSON env (JSON string)
      3) SNAPSHOTTER_JOB_FILE env (path to JSON file)
      4) stdin (if piped)

    Returns: (payload_dict, payload_src_string)
    """
    # 1) explicit CLI file
    if args.job_file:
        p = Path(args.job_file)
        if not p.exists():
            raise FileNotFoundError(f"Job file not found: {p}")
        return json.loads(p.read_text(encoding="utf-8")), f"file:{p}"

    # 2) env JSON string
    raw = os.environ.get("SNAPSHOTTER_JOB_JSON")
    if raw and raw.strip():
        return json.loads(raw), "env:SNAPSHOTTER_JOB_JSON"

    # 3) env file path
    job_file = os.environ.get("SNAPSHOTTER_JOB_FILE")
    if job_file and job_file.strip():
        p = Path(job_file)
        if not p.exists():
            raise FileNotFoundError(f"Job file not found: {p}")
        return json.loads(p.read_text(encoding="utf-8")), f"file:{p}"

    # 4) stdin
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data and data.strip():
            return json.loads(data), "stdin"

    raise RuntimeError(
        "Missing required job payload. Provide one of: "
        "--job-file PATH, SNAPSHOTTER_JOB_JSON (env), SNAPSHOTTER_JOB_FILE (env path), or pipe JSON to stdin."
    )


def _print_success(obj: Dict[str, Any]) -> None:
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
        "--job-file",
        dest="job_file",
        metavar="PATH",
        help="Path to a JSON file containing the job payload (local/dev)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no side effects). Also honors SNAPSHOTTER_DRY_RUN=1/true.",
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
    # Local .env support (optional import, safe in AWS).
    _maybe_load_dotenv()

    args = parse_args(argv)

    # dry_run: CLI flag OR env flag (matches old behavior expectations)
    dry_run = bool(args.dry_run) or _env_bool("SNAPSHOTTER_DRY_RUN", default=False)

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
        if isinstance(e, (json.JSONDecodeError, FileNotFoundError, RuntimeError)):
            # RuntimeError is too broad, but for missing payload we want parse_job;
            # other runtime errors should keep stage=unknown. We detect missing payload message.
            msg = str(e)
            if isinstance(e, RuntimeError) and not msg.startswith("Missing required job payload"):
                _print_failure("unknown", e)
            else:
                _print_failure(STAGE_PARSE_JOB, e)
            return 1

        # Everything else: unknown stage
        _print_failure("unknown", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
