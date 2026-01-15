# snapshotter/validate_basic.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, List


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


def _repo_paths_set(repo_index: dict[str, Any]) -> Set[str]:
    s: Set[str] = set()
    files = repo_index.get("files", []) or []
    if not isinstance(files, list):
        return s
    for f in files:
        if not isinstance(f, dict):
            continue
        p = f.get("path")
        if isinstance(p, str) and p:
            s.add(p)
    return s


def _extract_pass1_deps_by_file(repo_index: dict[str, Any]) -> dict[str, Tuple[Set[str], Set[str]]]:
    """
    Extract Pass1 deps in the CURRENT schema.

    Returns:
      { file_path: (internal_repo_paths_set, external_specs_set) }

    Source order:
      - file["deps"]["internal"] / file["deps"]["external"]   (preferred)
      - file["imports_resolved_internal"] / file["imports_external"] (fallback)
    """
    out: dict[str, Tuple[Set[str], Set[str]]] = {}
    files = repo_index.get("files", []) or []
    if not isinstance(files, list):
        return out

    for f in files:
        if not isinstance(f, dict):
            continue
        path = f.get("path")
        if not isinstance(path, str) or not path:
            continue

        internal: Set[str] = set()
        external: Set[str] = set()

        deps = f.get("deps")
        if isinstance(deps, dict):
            v_int = deps.get("internal")
            v_ext = deps.get("external")
            if isinstance(v_int, list):
                for x in v_int:
                    if isinstance(x, str) and x.strip():
                        internal.add(x.strip())
            if isinstance(v_ext, list):
                for x in v_ext:
                    if isinstance(x, str) and x.strip():
                        external.add(x.strip())

        # fallbacks
        if not internal:
            v = f.get("imports_resolved_internal")
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, str) and x.strip():
                        internal.add(x.strip())
        if not external:
            v = f.get("imports_external")
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, str) and x.strip():
                        external.add(x.strip())

        out[path] = (internal, external)

    return out


def _extract_arch_files_read_set(architecture_obj: dict[str, Any]) -> Set[str]:
    """
    Pass2 semantic generator uses `allowed_paths = set(file_contents_map.keys())`,
    which is typically the same set written to architecture_snapshot.files_read[*].path
    (but we support older / alternate shapes too).
    """
    read_set: Set[str] = set()

    files_read = architecture_obj.get("files_read")
    if isinstance(files_read, list):
        for it in files_read:
            if not isinstance(it, dict):
                continue
            p = it.get("path")
            if isinstance(p, str) and p:
                read_set.add(p)

    # Some historical shapes may embed `pass2.files` (list of {path, content}) or similar.
    # We treat those as "read" too if present.
    pass2 = architecture_obj.get("pass2")
    if isinstance(pass2, dict):
        files = pass2.get("files")
        if isinstance(files, list):
            for it in files:
                if not isinstance(it, dict):
                    continue
                p = it.get("path")
                if isinstance(p, str) and p:
                    read_set.add(p)

    return read_set


def _normalize_modules_list(architecture_obj: dict[str, Any]) -> List[dict[str, Any]]:
    """
    Support both:
      - New semantic shape: {"modules":[...], "uncertainties":[...]} (what pass2_semantic returns)
      - Legacy wrapper shape containing "modules" at top-level as well
    """
    modules = architecture_obj.get("modules")
    if not isinstance(modules, list):
        return []
    return [m for m in modules if isinstance(m, dict)]


def _normalize_uncertainties_list(architecture_obj: dict[str, Any]) -> List[dict[str, Any]]:
    u = architecture_obj.get("uncertainties")
    if not isinstance(u, list):
        return []
    return [x for x in u if isinstance(x, dict)]


def _validate_gaps_shape(gaps_obj: dict[str, Any], gaps_label: str) -> List[dict[str, Any]]:
    for k in ("generated_at", "job_id", "items"):
        if k not in gaps_obj:
            raise RuntimeError(f"{gaps_label} missing required key: {k}")
    if not isinstance(gaps_obj.get("items"), list):
        raise RuntimeError(f"{gaps_label}.items must be a list")

    items = gaps_obj.get("items") or []
    return [x for x in items if isinstance(x, dict)]


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

    gap_items_dicts = _validate_gaps_shape(gaps_obj, gaps_label)

    # ---- Architecture: current pass2_semantic output contract ----
    modules = _normalize_modules_list(architecture_obj)
    _ = _normalize_uncertainties_list(architecture_obj)  # keep: ensure it parses as list[dict] if present

    if not modules:
        raise RuntimeError(f"{architecture_label}.modules must be a non-empty list")

    repo_paths = _repo_paths_set(repo_index_obj)
    if not repo_paths:
        raise RuntimeError(f"{repo_index_label}.files must contain at least one file with a 'path'")

    # Prefer validating evidence_paths ⊆ files_read if present (stronger),
    # else fall back to evidence_paths ⊆ repo_index paths (still useful).
    read_set = _extract_arch_files_read_set(architecture_obj)

    deps_by_file = _extract_pass1_deps_by_file(repo_index_obj)

    # helper: recognize gap items that explicitly waive determinism checks (kept for flexibility)
    def _is_flagged_dependency_issue(module_name: str, dependency: str) -> bool:
        for gi in gap_items_dicts:
            t = gi.get("type")
            if t not in ("dependency_mismatch", "dependency_mismatch_pass1"):
                continue
            dep = gi.get("dependency")
            mod = gi.get("module")
            if isinstance(dep, str) and dep == dependency:
                if isinstance(mod, str):
                    return mod == module_name
                return True
        return False

    for idx, m in enumerate(modules):
        module_name = m.get("name") if isinstance(m.get("name"), str) else f"modules[{idx}]"

        # Required-ish semantic fields
        if not isinstance(m.get("type"), str) or not m.get("type"):
            raise RuntimeError(f"{architecture_label}.modules[{idx}].type must be a non-empty string")
        if not isinstance(m.get("summary"), str) or not m.get("summary"):
            raise RuntimeError(f"{architecture_label}.modules[{idx}].summary must be a non-empty string")

        # evidence_paths is the downstream compatibility field. In current pass2_semantic,
        # evidence_paths is set to anchor_paths.
        ev = m.get("evidence_paths")
        if not isinstance(ev, list):
            raise RuntimeError(f"{architecture_label}.modules[{idx}].evidence_paths must be a list")
        ev_paths = [p for p in ev if isinstance(p, str) and p.strip()]

        # anchor_paths (new) should exist; but be tolerant if only evidence_paths exists.
        ap = m.get("anchor_paths")
        if ap is not None and not isinstance(ap, list):
            raise RuntimeError(f"{architecture_label}.modules[{idx}].anchor_paths must be a list (or omitted)")
        anchor_paths = [p for p in ap if isinstance(p, str) and p.strip()] if isinstance(ap, list) else []

        # If both exist, they should match (pass2_semantic sets evidence_paths := anchor_paths).
        if anchor_paths and ev_paths and anchor_paths != ev_paths:
            raise RuntimeError(
                f"{architecture_label}.modules[{idx}] anchor_paths and evidence_paths diverge "
                f"(expected identical order/contents). module={module_name}"
            )

        # Validate evidence paths refer to real repo files.
        bad_not_in_repo = [p for p in ev_paths if p not in repo_paths]
        if bad_not_in_repo:
            raise RuntimeError(
                f"{architecture_label}.modules[{idx}].evidence_paths contains paths not present in {repo_index_label}.files. "
                f"module={module_name} bad={bad_not_in_repo[:10]}"
            )

        # Stronger constraint when files_read is present in architecture snapshot.
        if read_set:
            bad_not_in_read = [p for p in ev_paths if p not in read_set]
            if bad_not_in_read:
                raise RuntimeError(
                    f"{architecture_label}.modules[{idx}].evidence_paths must be subset of files_read when files_read is present. "
                    f"module={module_name} bad={bad_not_in_read[:10]}"
                )

        # Dependencies: CURRENT rule
        # pass2_semantic derives module.dependencies deterministically from pass1 deps_by_file over anchor/evidence paths.
        deps = m.get("dependencies")
        if deps is None:
            # keep permissive: allow omitted, but in our pipeline it should exist.
            continue
        if not isinstance(deps, list):
            raise RuntimeError(f"{architecture_label}.modules[{idx}].dependencies must be a list (or omitted)")

        deps_clean = [d.strip() for d in deps if isinstance(d, str) and d.strip()]
        deps_set = set(deps_clean)

        # Compute expected deps from pass1 for this module based on evidence paths.
        expected_internal: Set[str] = set()
        expected_external: Set[str] = set()
        for p in ev_paths:
            ints, exts = deps_by_file.get(p, (set(), set()))
            expected_internal |= set(ints or set())
            expected_external |= set(exts or set())

        expected_set = set(expected_internal) | set(expected_external)

        # If we have any evidence paths, we should be able to compute deterministic deps
        # and they should match exactly (since pass2_semantic sets them that way).
        # However, allow explicit waiver via gaps items for edge cases.
        if ev_paths:
            # unexpected deps: not derivable from pass1 deps for those files
            unexpected = [d for d in deps_clean if d not in expected_set and not _is_flagged_dependency_issue(module_name, d)]
            if unexpected:
                raise RuntimeError(
                    f"Dependency reconciliation failed (deterministic rule): module={module_name} "
                    f"unexpected_dependencies={unexpected[:10]} (not derivable from Pass1 deps for evidence_paths)"
                )

            # missing deps: present in pass1 derivation but absent from module.dependencies
            missing = [d for d in sorted(expected_set) if d not in deps_set]
            if missing:
                raise RuntimeError(
                    f"Dependency reconciliation failed (deterministic rule): module={module_name} "
                    f"missing_dependencies={missing[:10]} (Pass1 derivation not reflected in module.dependencies)"
                )

    # Validate files_not_read shape if present (keep legacy-ish check; harmless)
    files_not_read = architecture_obj.get("files_not_read")
    if files_not_read is not None:
        if not isinstance(files_not_read, list):
            raise RuntimeError(f"{architecture_label}.files_not_read must be a list (or omitted)")
        for it in files_not_read:
            if not isinstance(it, dict):
                raise RuntimeError(f"{architecture_label}.files_not_read must contain objects")
            p = it.get("path")
            if not isinstance(p, str) or not p:
                raise RuntimeError(f"{architecture_label}.files_not_read items must include non-empty 'path'")


def validate_basic_artifacts(local_paths: Dict[str, Optional[str]]) -> None:
    """
    Minimal sanity validation:
    - required artifacts exist
    - JSON artifacts parse as JSON
    - semantic validation (aligned with current Pass1 + Pass2):
        * GAPS_AND_INCONSISTENCIES has required shape (generated_at, job_id, items[])
        * ARCHITECTURE_SUMMARY_SNAPSHOT has modules[] with evidence_paths[]
        * evidence_paths point to real repo files, and (if files_read present) are subset of files_read
        * module.dependencies are deterministically derivable from Pass1 deps over evidence_paths
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
