import os
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: str | None = None) -> str:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )
    return p.stdout.strip()


def clone_and_checkout(repo_url: str, ref: str, workdir: str) -> str:
    """
    Clone repo into <workdir>/repo and checkout <ref> if possible.

    IMPORTANT: Avoid double-nesting by NEVER using:
      cwd=workdir + dest=workdir/repo
    """
    workdir_path = Path(workdir).resolve()
    workdir_path.mkdir(parents=True, exist_ok=True)

    repo_dir = workdir_path / "repo"
    if repo_dir.exists():
        run(["rm", "-rf", str(repo_dir)])

    # Clone using absolute destination; no cwd tricks -> no double nesting.
    run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], cwd=None)

    # Best-effort checkout of ref (branch/tag). If it fails, stay on default.
    try:
        run(["git", "fetch", "--depth", "1", "origin", ref], cwd=str(repo_dir))
        run(["git", "checkout", ref], cwd=str(repo_dir))
    except Exception:
        pass

    resolved_commit = run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir))
    return resolved_commit