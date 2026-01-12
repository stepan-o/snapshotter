from __future__ import annotations

from typing import Any, Dict


def run(job_payload: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """Stub entrypoint for core Snapshotter logic.

    This function is intentionally not implemented yet. It returns a minimal
    JSON-shaped structure to keep the CLI and integrations working, but raises
    NotImplementedError to signal that real logic must be pasted later.
    """
    # Return a minimal structure to aid observability during integration tests
    # and container runs, while still making it clear the core logic is missing.
    raise NotImplementedError(
        "Snapshotter main.run is not implemented yet. Paste the real logic later."
    )
