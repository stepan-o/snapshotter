# snapshotter/validate_basic.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

JSON_FILENAMES = {
    "repo_index.json",
    "artifact_manifest.json",
    "ARCHITECTURE_SUMMARY_SNAPSHOT.json",
    "GAPS_AND_INCONSISTENCIES.json",
}


def _must_exist(path: str | Path, label: str) -> None:
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Missing artifact: {label} at {p}")
    if not p.is_file():
        raise RuntimeError(f"Artifact path is not a file: {label} at {p}")


def _load_json(path: str | Path, label: str) -> Any:
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Invalid JSON for {label} at {p}: {e}") from e


def _must_parse_json(path: str | Path, label: str) -> None:
    _load_json(path, label)


def _extract_pass1_import_edges(repo_index: dict[str, Any]) -> dict[str, Set[str]]:
    """
    Best-effort extraction of Pass 1 literal imports. Supports a few possible shapes
    so validation doesn't hard-couple to one repo_index schema.
    Returns: {file_path: {import_string, ...}}
    """
    edges: dict[str, Set[str]] = {}
    files = repo_index.get("files", []) or []
    if not isinstance(files, list):
        return edges

    for f in files:
        if not isinstance(f, dict):
            continue
        path = f.get("path")
        if not isinstance(path, str) or not path:
            continue

        imports: Set[str] = set()

        # 1) f["imports"] as list[str]
        imp = f.get("imports")
        if isinstance(imp, list):
            for x in imp:
                if isinstance(x, str) and x.strip():
                    imports.add(x.strip())

        # 2) f["imports"] as dict with lists (literal/raw/all)
        imp2 = f.get("imports")
        if isinstance(imp2, dict):
            for key in ("literal", "raw", "all"):
                v = imp2.get(key)
                if isinstance(v, list):
                    for x in v:
                        if isinstance(x, str) and x.strip():
                            imports.add(x.strip())

        # 3) f["literal_imports"] as list[str]
        imp3 = f.get("literal_imports")
        if isinstance(imp3, list):
            for x in imp3:
                if isinstance(x, str) and x.strip():
                    imports.add(x.strip())

        if imports:
            edges[path] = imports

    return edges


def _validate_semantic_artifacts(
        *,
        repo_index_obj: Any,
        architecture_obj: Any,
        gaps_obj: Any,
        repo_index_label: str,
        architecture_label: str,
        gaps_label: str,
) -> None:
    if not isinstance(repo_index_obj, dict):
        raise RuntimeError(f"{repo_index_label} must be a JSON object")
    if not isinstance(architecture_obj, dict):
        raise RuntimeError(f"{architecture_label} must be a JSON object")
    if not isinstance(gaps_obj, dict):
        raise RuntimeError(f"{gaps_label} must be a JSON object")

    # ---- Gaps basic shape ----
    for k in ("generated_at", "job_id", "items"):
        if k not in gaps_obj:
            raise RuntimeError(f"{gaps_label} missing required key: {k}")
    if not isinstance(gaps_obj.get("items"), list):
        raise RuntimeError(f"{gaps_label}.items must be a list")

    gap_items = gaps_obj.get("items") or []
    # normalize gap items we can search through
    gap_items_dicts = [x for x in gap_items if isinstance(x, dict)]

    # ---- Architecture snapshot required keys (v0.1: self-auditing) ----
    required_arch_keys = (
        "generated_at",
        "repo",
        "read_plan",
        "coverage",
        "modules",
        "uncertainties",
        "files_read",
        "files_not_read",
    )
    for k in required_arch_keys:
        if k not in architecture_obj:
            raise RuntimeError(f"{architecture_label} missing required key: {k}")

    files_read = architecture_obj.get("files_read")
    files_not_read = architecture_obj.get("files_not_read")
    modules = architecture_obj.get("modules")

    if not isinstance(files_read, list):
        raise RuntimeError(f"{architecture_label}.files_read must be a list")
    if not isinstance(files_not_read, list):
        raise RuntimeError(f"{architecture_label}.files_not_read must be a list")
    if not isinstance(modules, list):
        raise RuntimeError(f"{architecture_label}.modules must be a list")

    # Build read_set
    read_set: Set[str] = set()
    for it in files_read:
        if not isinstance(it, dict):
            raise RuntimeError(f"{architecture_label}.files_read must contain objects")
        p = it.get("path")
        if not isinstance(p, str) or not p:
            raise RuntimeError(f"{architecture_label}.files_read items must include non-empty 'path'")
        read_set.add(p)

    # files_not_read: soft validation (path required, reason optional but recommended)
    for it in files_not_read:
        if not isinstance(it, dict):
            raise RuntimeError(f"{architecture_label}.files_not_read must contain objects")
        p = it.get("path")
        if not isinstance(p, str) or not p:
            raise RuntimeError(f"{architecture_label}.files_not_read items must include non-empty 'path'")

    # ---- Modules: evidence_paths constraint ----
    for idx, m in enumerate(modules):
        if not isinstance(m, dict):
            raise RuntimeError(f"{architecture_label}.modules[{idx}] must be an object")

        # evidence_paths is required by 1.4 acceptance criteria
        if "evidence_paths" not in m:
            raise RuntimeError(f"{architecture_label}.modules[{idx}] missing required key: evidence_paths")

        ev = m.get("evidence_paths")
        if not isinstance(ev, list):
            raise RuntimeError(f"{architecture_label}.modules[{idx}].evidence_paths must be a list")

        bad = [p for p in ev if not isinstance(p, str) or p not in read_set]
        if bad:
            raise RuntimeError(
                f"{architecture_label}.modules[{idx}].evidence_paths must be subset of files_read. Bad: {bad[:10]}"
            )

    # ---- Dependencies must reconcile with Pass 1 imports, OR be flagged in gaps ----
    imports_by_file = _extract_pass1_import_edges(repo_index_obj)

    # If repo_index doesn't expose imports, we can't strictly validate reconciliation.
    # But if it does, enforce the rule.
    if imports_by_file:
        def _is_flagged_dependency_mismatch(module_name: str, dependency: str) -> bool:
            for gi in gap_items_dicts:
                t = gi.get("type")
                if t not in ("dependency_mismatch", "dependency_mismatch_pass1"):
                    continue
                dep = gi.get("dependency")
                mod = gi.get("module")
                if isinstance(dep, str) and dep == dependency:
                    if isinstance(mod, str):
                        if mod == module_name:
                            return True
                    else:
                        # module not present; still accept as a flag
                        return True
            return False

        for idx, m in enumerate(modules):
            deps = m.get("dependencies")
            if deps is None:
                continue
            if not isinstance(deps, list):
                raise RuntimeError(f"{architecture_label}.modules[{idx}].dependencies must be a list (or omitted)")

            deps_clean = [d.strip() for d in deps if isinstance(d, str) and d.strip()]
            if not deps_clean:
                continue

            ev = m.get("evidence_paths") or []
            if not isinstance(ev, list) or not ev:
                continue

            # Merge literal imports from evidence files
            evidence_imports: Set[str] = set()
            for p in ev:
                if isinstance(p, str):
                    evidence_imports |= imports_by_file.get(p, set())

            # If we have no import evidence for these files, we can't validate these deps.
            if not evidence_imports:
                continue

            module_name = m.get("name") if isinstance(m.get("name"), str) else f"modules[{idx}]"

            for d in deps_clean:
                supported = any(d in imp for imp in evidence_imports)
                if supported:
                    continue
                if _is_flagged_dependency_mismatch(module_name=str(module_name), dependency=d):
                    continue
                raise RuntimeError(
                    f"Dependency reconciliation failed: module={module_name} dependency={d} "
                    f"not supported by Pass 1 imports from evidence_paths, and not flagged in {gaps_label}."
                )


def validate_basic_artifacts(local_paths: Dict[str, Optional[str]]) -> None:
    """
    Minimal sanity validation:
    - required artifacts exist
    - JSON artifacts parse as JSON
    - (v0.1+) semantic validation:
        * ARCHITECTURE_SUMMARY_SNAPSHOT has files_read/files_not_read + modules with evidence_paths ⊆ files_read
        * dependencies reconcile with Pass 1 imports OR are explicitly flagged in gaps
        * GAPS_AND_INCONSISTENCIES has required shape
    """
    required_keys = [
        "repo_index",
        "artifact_manifest",
        "architecture_snapshot",
        "gaps",
        "onboarding",
    ]

    # Existence + parse
    for key in required_keys:
        p = local_paths.get(key)
        if not p:
            raise RuntimeError(f"Missing local path for required artifact key: {key}")
        _must_exist(p, key)

        name = Path(p).name
        if name in JSON_FILENAMES:
            _must_parse_json(p, key)

    # Semantic validations (uses parsed JSON)
    repo_index_path = local_paths.get("repo_index")
    arch_path = local_paths.get("architecture_snapshot")
    gaps_path = local_paths.get("gaps")
    if not repo_index_path or not arch_path or not gaps_path:
        return

    repo_index_obj = _load_json(repo_index_path, "repo_index")
    architecture_obj = _load_json(arch_path, "architecture_snapshot")
    gaps_obj = _load_json(gaps_path, "gaps")

    _validate_semantic_artifacts(
        repo_index_obj=repo_index_obj,
        architecture_obj=architecture_obj,
        gaps_obj=gaps_obj,
        repo_index_label="repo_index",
        architecture_label="architecture_snapshot",
        gaps_label="gaps",
    )