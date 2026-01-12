import json
import os
import subprocess
import sys


def run_cli(env: dict[str, str] | None = None, args: list[str] | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "snapshotter.cli"]
    if args:
        cmd.extend(args)
    env_vars = os.environ.copy()
    if env:
        env_vars.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, env=env_vars)


def test_cli_errors_when_no_payload():
    # Ensure neither env var nor file is provided
    env = os.environ.copy()
    env.pop("SNAPSHOTTER_JOB_JSON", None)
    proc = run_cli(env=env)
    assert proc.returncode == 1
    payload = json.loads(proc.stdout.strip())
    assert payload["ok"] is False
    assert payload["error"] in {"RuntimeError"}


def test_cli_catches_not_implemented_and_returns_json_error():
    env = {"SNAPSHOTTER_JOB_JSON": json.dumps({"repo_url": "https://example.com/repo", "ref": "main"})}
    proc = run_cli(env=env)
    assert proc.returncode == 1
    payload = json.loads(proc.stdout.strip())
    assert payload["ok"] is False
    assert payload["error"] in {"NotImplementedError"}
    assert "not implemented" in payload["message"].lower()
