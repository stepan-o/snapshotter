# src/snapshotter/validate_basic.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# -------------------------------------------------------------------
# Core validation - hard contracts
# -------------------------------------------------------------------
def _must_exist(path: str | Path, label: str) -> None:
    """Hard contract: file must exist and be non-empty."""
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Missing artifact: {label} at {p}")
    if not p.is_file():
        raise RuntimeError(f"Artifact path is not a file: {label} at {p}")
    if p.stat().st_size == 0:
        raise RuntimeError(f"Artifact is empty: {label} at {p}")


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    """Load and validate JSON is a dict."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Invalid JSON for {label} at {p}: {e}") from e

    if not isinstance(data, dict):
        raise RuntimeError(f"{label} must be a JSON object (dict), got {type(data).__name__}")

    return data


# -------------------------------------------------------------------
# Artifact-specific validators
# -------------------------------------------------------------------
def _validate_repo_index(obj: dict[str, Any]) -> None:
    """Validate PASS1_REPO_INDEX.json structure."""
    required = ["schema_version", "generated_at", "job", "files", "read_plan"]
    for key in required:
        if key not in obj:
            raise RuntimeError(f"repo_index missing required key: {key}")

    if not isinstance(obj["files"], list):
        raise RuntimeError("repo_index.files must be a list")

    if not obj["files"]:
        raise RuntimeError("repo_index.files must not be empty")

    # Each file must have path
    for i, f in enumerate(obj["files"]):
        if not isinstance(f, dict):
            raise RuntimeError(f"repo_index.files[{i}] must be a dict")
        if not isinstance(f.get("path"), str):
            raise RuntimeError(f"repo_index.files[{i}] missing 'path' string")


def _validate_architecture_snapshot(obj: dict[str, Any]) -> None:
    """Validate ARCHITECTURE_SUMMARY_SNAPSHOT.json structure."""
    required = ["generated_at", "repo", "coverage", "files_read", "files_not_read"]
    for key in required:
        if key not in obj:
            raise RuntimeError(f"architecture_snapshot missing required key: {key}")

    # Validate coverage stats
    coverage = obj["coverage"]
    if not isinstance(coverage, dict):
        raise RuntimeError("architecture_snapshot.coverage must be an object")

    for key in ["files_scanned", "files_read", "files_not_read", "files_included_from_pass1"]:
        if key not in coverage:
            raise RuntimeError(f"architecture_snapshot.coverage missing key: {key}")
        if not isinstance(coverage[key], int):
            raise RuntimeError(f"architecture_snapshot.coverage.{key} must be an integer")


def _validate_gaps(obj: dict[str, Any]) -> None:
    """Validate GAPS_AND_INCONSISTENCIES.json structure."""
    if "source" not in obj:
        raise RuntimeError("gaps missing 'source' key")

    risks = obj.get("risks_or_gaps")
    if risks is not None and not isinstance(risks, list):
        raise RuntimeError("gaps.risks_or_gaps must be a list if present")


def _validate_pass2_semantic(obj: dict[str, Any]) -> None:
    """Validate PASS2_SEMANTIC.json structure - matches pass2_semantic.py output."""
    # Top-level keys from pass2_semantic.py
    required = ["schema_version", "generated_at", "repo", "caps", "inputs", "llm_output", "fingerprint_sha256"]
    for key in required:
        if key not in obj:
            raise RuntimeError(f"pass2_semantic missing required key: {key}")

    # Validate llm_output has summary with expected structure
    llm_output = obj["llm_output"]
    if not isinstance(llm_output, dict):
        raise RuntimeError("pass2_semantic.llm_output must be an object")

    summary = llm_output.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("pass2_semantic.llm_output.summary must be an object")

    # Summary must have these keys (allow empty/null values)
    summary_required = ["primary_stack", "architecture_overview", "key_components",
                        "data_flows", "auth_and_routing_notes", "risks_or_gaps"]
    for key in summary_required:
        if key not in summary:
            raise RuntimeError(f"pass2_semantic.llm_output.summary missing key: {key}")

    # Validate types within summary
    if not isinstance(summary.get("primary_stack"), (str, type(None))):
        raise RuntimeError("pass2_semantic.llm_output.summary.primary_stack must be string or null")

    if not isinstance(summary.get("architecture_overview"), str):
        raise RuntimeError("pass2_semantic.llm_output.summary.architecture_overview must be string")

    if not isinstance(summary.get("key_components"), list):
        raise RuntimeError("pass2_semantic.llm_output.summary.key_components must be list")

    if not isinstance(summary.get("data_flows"), list):
        raise RuntimeError("pass2_semantic.llm_output.summary.data_flows must be list")

    if not isinstance(summary.get("auth_and_routing_notes"), list):
        raise RuntimeError("pass2_semantic.llm_output.summary.auth_and_routing_notes must be list")

    if not isinstance(summary.get("risks_or_gaps"), list):
        raise RuntimeError("pass2_semantic.llm_output.summary.risks_or_gaps must be list")

    # Validate caps structure
    caps = obj["caps"]
    if not isinstance(caps, dict):
        raise RuntimeError("pass2_semantic.caps must be an object")

    # Validate inputs structure
    inputs = obj["inputs"]
    if not isinstance(inputs, dict):
        raise RuntimeError("pass2_semantic.inputs must be an object")

    # Validate fingerprint
    fingerprint = obj["fingerprint_sha256"]
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise RuntimeError("pass2_semantic.fingerprint_sha256 must be a valid 64-char hex string")


def _validate_onboarding(path: str | Path) -> None:
    """Validate ONBOARDING.md exists and has content."""
    p = Path(path)
    _must_exist(p, "onboarding")

    content = p.read_text(encoding="utf-8")
    if len(content.strip()) < 50:
        raise RuntimeError("onboarding.md appears too short or empty")


def _validate_artifact_manifest(obj: dict[str, Any]) -> None:
    """Validate artifact_manifest.json structure."""
    required = ["generated_at", "items", "stable_fingerprints", "run_fingerprint_sha256"]
    for key in required:
        if key not in obj:
            raise RuntimeError(f"artifact_manifest missing required key: {key}")

    if not isinstance(obj["items"], list):
        raise RuntimeError("artifact_manifest.items must be a list")

    # Each item must have name, filename, bytes, sha256
    for i, item in enumerate(obj["items"]):
        if not isinstance(item, dict):
            raise RuntimeError(f"artifact_manifest.items[{i}] must be a dict")

        for key in ["name", "filename", "bytes", "sha256"]:
            if key not in item:
                raise RuntimeError(f"artifact_manifest.items[{i}] missing key: {key}")


# -------------------------------------------------------------------
# Cross-artifact validation
# -------------------------------------------------------------------
def _validate_cross_artifact_consistency(
        repo_index: dict[str, Any],
        architecture_snapshot: dict[str, Any],
        pass2_semantic: dict[str, Any],
) -> None:
    """Validate consistency between different artifacts."""

    # 1. Verify repo URLs match
    repo_urls = [
        repo_index.get("job", {}).get("repo_url"),
        architecture_snapshot.get("repo", {}).get("repo_url"),
        pass2_semantic.get("repo", {}).get("repo_url"),
    ]

    # Filter out None values
    repo_urls = [url for url in repo_urls if url is not None]

    # All non-None URLs must match
    if repo_urls:
        first_url = repo_urls[0]
        for i, url in enumerate(repo_urls):
            if url != first_url:
                raise RuntimeError(f"Repo URL mismatch: {url} != {first_url}")

    # 2. Verify commit hashes match
    commits = [
        repo_index.get("job", {}).get("resolved_commit"),
        architecture_snapshot.get("repo", {}).get("resolved_commit"),
        pass2_semantic.get("repo", {}).get("resolved_commit"),
    ]

    # Filter out None and "unknown"
    commits = [commit for commit in commits if commit is not None and commit != "unknown"]

    # All non-None commits must match
    if commits:
        first_commit = commits[0]
        for i, commit in enumerate(commits):
            if commit != first_commit:
                raise RuntimeError(f"Commit hash mismatch: {commit} != {first_commit}")

    # 3. Verify architecture_snapshot has at least some files read
    files_read = architecture_snapshot.get("files_read", [])
    if not isinstance(files_read, list):
        raise RuntimeError("architecture_snapshot.files_read must be a list")

    if not files_read:
        raise RuntimeError("architecture_snapshot.files_read is empty - no files were analyzed")

    # 4. Verify pass2_semantic summary has content
    llm_output = pass2_semantic.get("llm_output", {})
    summary = llm_output.get("summary", {})

    if not isinstance(summary, dict):
        raise RuntimeError("pass2_semantic.llm_output.summary must be a dict")

    # Summary must have non-empty architecture_overview
    overview = summary.get("architecture_overview", "")
    if not isinstance(overview, str) or not overview.strip():
        raise RuntimeError("pass2_semantic.llm_output.summary.architecture_overview must be non-empty")


# -------------------------------------------------------------------
# Main validation entry point
# -------------------------------------------------------------------
def validate_basic_artifacts(local_paths: dict[str, str | None]) -> None:
    """
    Hard contract validation of all required artifacts.

    Args:
        local_paths: Must contain at minimum these keys (from graph.py's get_validation_paths):
            - repo_index
            - artifact_manifest
            - architecture_snapshot
            - gaps
            - onboarding
            - pass2_semantic
    """
    # 1. Validate we have all required paths
    required_keys = [
        "repo_index",
        "artifact_manifest",
        "architecture_snapshot",
        "gaps",
        "onboarding",
        "pass2_semantic",
    ]

    for key in required_keys:
        if key not in local_paths:
            raise RuntimeError(f"Missing required artifact key in local_paths: {key}")

        path = local_paths[key]
        if not path:
            raise RuntimeError(f"Path for artifact '{key}' is None or empty")

        # Basic file existence
        _must_exist(path, key)

    # 2. Load and validate each artifact's structure
    repo_index = _load_json(local_paths["repo_index"], "repo_index")
    _validate_repo_index(repo_index)

    architecture_snapshot = _load_json(local_paths["architecture_snapshot"], "architecture_snapshot")
    _validate_architecture_snapshot(architecture_snapshot)

    gaps = _load_json(local_paths["gaps"], "gaps")
    _validate_gaps(gaps)

    pass2_semantic = _load_json(local_paths["pass2_semantic"], "pass2_semantic")
    _validate_pass2_semantic(pass2_semantic)

    artifact_manifest = _load_json(local_paths["artifact_manifest"], "artifact_manifest")
    _validate_artifact_manifest(artifact_manifest)

    # 3. Validate onboarding (markdown file)
    _validate_onboarding(local_paths["onboarding"])

    # 4. Validate cross-artifact consistency
    _validate_cross_artifact_consistency(repo_index, architecture_snapshot, pass2_semantic)

    # 5. Verify artifact_manifest includes the artifacts we validated
    manifest_items = {item.get("name") for item in artifact_manifest.get("items", [])
                      if isinstance(item, dict)}

    expected_in_manifest = {"repo_index", "architecture_snapshot", "gaps", "pass2_semantic"}
    missing = expected_in_manifest - manifest_items
    if missing:
        raise RuntimeError(f"artifact_manifest missing items: {missing}")