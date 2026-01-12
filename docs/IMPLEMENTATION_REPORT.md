Implementation Report — Repo Snapshotter v0.1 Scaffold

Date: 2026-01-11 19:00 (local)

Overview
This document summarizes the current implementation of the Repo Snapshotter v0.1 scaffold. It is intended for engineers and operators who need to understand the repository’s structure, chosen tooling, operational contracts, and next steps. No business logic for snapshotting is implemented yet; this repository provides the foundation to run locally and on AWS (ECS Fargate or AWS Batch).

Scope
- Provide a uv-first Python project with modern packaging and src/ layout
- Stub CLI that enforces payload-only configuration
- Containerization suitable for AWS
- Makefile-driven local developer workflow
- Minimal CI (lint + tests)
- Basic documentation and hygiene

Architecture and Layout
- Packaging: Modern PEP 621 metadata in pyproject.toml; build backend: hatchling.
- Python version: 3.11+
- Package name: snapshotter
- Source layout: src/snapshotter/
  - __init__.py: version and placeholder
  - cli.py: user-facing entrypoint
  - main.py: stub run() raising NotImplementedError

Dependencies
- Runtime:
  - pydantic (v2) — configuration/data validation (future use)
  - boto3 — AWS interactions (future use)
  - httpx — HTTP client (future use)
  - typing-extensions (for py<3.12)
- Dev (optional group): ruff, mypy, pytest

CLI Contract (src/snapshotter/cli.py)
- Invocation: python -m snapshotter.cli (also installed as console script: snapshotter)
- Configuration sources (payload-only):
  1) --job-file PATH (local/dev)
  2) SNAPSHOTTER_JOB_JSON environment variable (AWS-friendly)
- Flags:
  - --dry-run: future behavior hook; currently passed through to main.run()
  - --version: prints version and exits
- Behavior:
  - Parses and loads job payload from the chosen source
  - Calls snapshotter.main.run(job_payload, dry_run)
  - Success path: prints {"ok": true, "result": ...} as minified JSON and exits 0
  - Error path: catches any Exception (including NotImplementedError), prints {"ok": false, "error": "...", "message": "..."} and exits 1

Core Stub (src/snapshotter/main.py)
- run(job_payload: dict, dry_run: bool) -> dict
- Currently raises NotImplementedError to clearly indicate missing business logic
- This will be replaced by real implementation after pasting existing Snapshotter code from the separate repository

Developer Workflow (Makefile)
- make init: uv sync --all-groups (creates venv and installs deps)
- make fmt: ruff format
- make lint: ruff check
- make type: mypy (package mode)
- make test: pytest
- make run: runs CLI locally; supports JOB_FILE=job.json or relies on SNAPSHOTTER_JOB_JSON
- make docker-build: uv lock and docker build
- make docker-run: runs container, supports env var or mounts a job file

Containerization (Dockerfile)
- Base image: python:3.11-slim
- Installs uv in the image
- Uses layering best practices: copy pyproject.toml + uv.lock, then uv sync --frozen --no-dev, then copy src/
- Default CMD: ["python", "-m", "snapshotter.cli"]

Continuous Integration
- GitHub Actions workflow (.github/workflows/ci.yml)
- Steps:
  - Checkout + setup Python 3.11
  - Install uv
  - uv lock --all-groups (ensures lock is generated in CI)
  - uv sync --frozen --all-groups
  - ruff check
  - pytest -q

Testing
- tests/test_cli.py covers two behaviors:
  - Missing payload leads to JSON error with RuntimeError and exit code 1
  - NotImplementedError from main.run is caught and returned as structured JSON error

Documentation
- README.md: local setup, dev workflow, Docker usage, AWS notes, where to paste real logic
- docs/AWS_RUNBOOK.md: practical guide for ECS Fargate and AWS Batch runs, configuration via env var, IAM role usage, Secrets Manager for secrets, CloudWatch logging, and operational notes
- This report (docs/IMPLEMENTATION_REPORT.md) describes the current scaffold implementation

Security and Operational Considerations
- No hardcoded secrets; recommend AWS Secrets Manager for sensitive values
- Use IAM roles for AWS resource access; avoid static keys
- Logging via stdout/stderr suitable for CloudWatch
- S3 SSE recommended for any data at rest (policy-level enforcement, not code)
- Env var size limits exist; for large payloads consider storing in S3 and passing a reference (future enhancement)

AWS Readiness
- Container designed to run on ECS Fargate or AWS Batch as a single container task
- Contract relies on SNAPSHOTTER_JOB_JSON (string); CLI is resilient with structured error output
- No service bindings or sidecars required at this stage

Limitations and Next Steps
- Core logic is absent: snapshotter.main.run() raises NotImplementedError
- No job schema validation yet (pydantic can be used when implementing)
- No retries/backoff implemented at app level (leverage ECS/Batch policies for now)
- Large payload handling via S3 reference not yet implemented
- Observability limited to stdout; no metrics/tracing integrated yet

How to Extend
1) Paste the existing Snapshotter logic into src/snapshotter/ (e.g., git_ops.py and related modules)
2) Implement snapshotter.main.run() to orchestrate the job
3) Add pydantic models for job payload schema and validation
4) Add targeted tests for new logic and edge cases
5) Expand CI to include type check (already present) and potentially build artifacts

Reproducibility
- Dependency management uses uv with a committed lockfile (uv.lock)
- Docker image build installs with --frozen to ensure reproducible dependency resolution

Acknowledgements
- Scaffold designed to be compatible with later code that uses git CLI via subprocess
