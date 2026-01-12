Repo Snapshotter v0.1 — Scaffold

What is this?
- An initial, AWS-ready scaffold for a "Repo Snapshotter" service. The actual snapshotting logic is intentionally NOT implemented yet. This repo provides the project structure, CLI, containerization, and developer workflow.

Key design points
- uv-first dependency management
- Makefile is the primary developer interface
- Runs locally and on AWS (ECS Fargate or AWS Batch)
- Payload-only configuration via SNAPSHOTTER_JOB_JSON or --job-file
- No Replit dependencies; no hardcoded secrets

Requirements
- Python 3.11+
- uv (https://docs.astral.sh/uv/)
- Docker (optional, for container builds)

Getting started (local)
1) Sync dependencies (including dev tools):
   make init

2) Lint, format, type-check, and run tests:
   make fmt
   make lint
   make type
   make test

3) Run the CLI locally using a job file:
   echo '{"repo_url": "https://github.com/example/repo", "ref": "main"}' > job.json
   JOB_FILE=job.json make run

   Or via environment variable:
   export SNAPSHOTTER_JOB_JSON='{"repo_url": "https://github.com/example/repo", "ref": "main"}'
   make run

Container build and run
- Build:
  make docker-build

- Run with an inline payload:
  docker run --rm -e SNAPSHOTTER_JOB_JSON='{"repo_url":"https://github.com/example/repo","ref":"main"}' snapshotter:0.1.0

- Or mount a local job file:
  docker run --rm -v "$(pwd)":/app -w /app snapshotter:0.1.0 python -m snapshotter.cli --job-file job.json

AWS runbook
See docs/AWS_RUNBOOK.md for guidance on running as an ECS Task or Batch job, passing configuration, using IAM roles and Secrets Manager, and operational notes.

Where to paste the real logic later
- Put real implementation modules under src/snapshotter and wire them into snapshotter.main.run(). The current snapshotter.main.run() raises NotImplementedError as a clear signal until the code is pasted.

License
MIT
