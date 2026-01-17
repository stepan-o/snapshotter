# snapshotter/graph.py
from __future__ import annotations

import codecs
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypedDict

from snapshotter.git_ops import clone_and_checkout
from snapshotter.job import Job
from snapshotter.pass1 import write_json
from snapshotter.pass2_semantic import Pass2SemanticError, generate_pass2_semantic_artifacts
from snapshotter.s3_uploader import S3Uploader
from snapshotter.utils import sha256_bytes, stable_json_fingerprint_sha256, utc_ts
from snapshotter.validate_basic import validate_basic_artifacts

try:
    from langgraph.graph import END, StateGraph
except Exception as e:  # pragma: no cover
    raise RuntimeError("LangGraph is required. Install 'langgraph'.") from e


# -----------------------------
# Stages (canonical)
# -----------------------------
STAGE_INIT = "init"
STAGE_PARSE_JOB = "parse_job"
STAGE_CLONE = "clone"
STAGE_PASS1_REPO_INDEX = "pass1_repo_index"
STAGE_PASS2_MAKE_READ_PLAN = "pass2_make_read_plan"
STAGE_PASS2_FETCH_FILES = "pass2_fetch_files"
STAGE_PASS2_GENERATE_OUTPUTS = "pass2_generate_outputs"
STAGE_PASS1_MANIFEST = "pass1_manifest"
STAGE_VALIDATE_BASIC = "validate_basic"
STAGE_UPLOAD = "upload"
STAGE_EMIT_RESULT = "emit_result"
STAGE_DONE = "done"
STAGE_DONE_DRY_RUN = "done_dry_run"


class SnapshotterStageError(RuntimeError):
    def __init__(self, stage: str, inner: Exception):
        super().__init__(str(inner))
        self.stage = stage
        self.inner = inner


@dataclass(frozen=True)
class RuntimeConfig:
    dry_run: bool
    aws_region: str | None


class SnapshotterState(TypedDict, total=False):
    payload: dict[str, Any]
    payload_src: str
    config: RuntimeConfig
    stage: str

    job: Job
    resolved_commit: str

    workdir: str
    repo_dir: str
    out_dir: str

    local_paths: dict[str, str]

    repo_index: dict[str, Any]

    # pass2 planning/fetching
    read_plan: list[str]
    pass2_caps: dict[str, int]
    read_plan_missing: list[str]
    file_contents_map: dict[str, str]

    # pass2 fetch reporting
    pass2_total_chars: int
    pass2_files_read: list[dict[str, Any]]  # [{path, chars, truncated}]
    pass2_not_read_reasons: dict[str, str]  # {path: reason}

    # pass2 planning debug (optional)
    pass2_read_plan_debug: dict[str, Any]

    # upload/output
    s3_paths: dict[str, Optional[str]]
    result: dict[str, Any]


# -----------------------------
# Helpers
# -----------------------------
def _file_sha256(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _stable_fingerprint_for_artifact(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix.lower() == ".json":
        try:
            obj = json.loads(raw.decode("utf-8"))
            return stable_json_fingerprint_sha256(obj)
        except Exception:
            return sha256_bytes(raw)
    return sha256_bytes(raw)


def _read_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    obj = json.loads(raw.decode("utf-8"))
    return obj if isinstance(obj, dict) else {}


def _require_file(path: Path, *, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"required artifact missing: {label} ({path})")
    try:
        if path.stat().st_size <= 0:
            raise RuntimeError(f"required artifact empty: {label} ({path})")
    except FileNotFoundError:
        raise RuntimeError(f"required artifact missing: {label} ({path})")


def build_artifact_manifest(local_paths: dict[str, Optional[str]]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    stable_fingerprints: dict[str, str] = {}

    for name, p in local_paths.items():
        if not p:
            continue
        path = Path(p)
        if not path.exists():
            continue

        raw = path.read_bytes()
        items.append(
            {
                "name": name,
                "filename": path.name,
                "bytes": len(raw),
                "sha256": sha256_bytes(raw),
            }
        )
        stable_fingerprints[name] = _stable_fingerprint_for_artifact(path)

    items.sort(key=lambda x: x["name"])

    canonical = json.dumps(
        stable_fingerprints,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    run_fingerprint_sha256 = sha256_bytes(canonical)

    return {
        "generated_at": utc_ts(),
        "items": items,
        "stable_fingerprints": stable_fingerprints,
        "run_fingerprint_sha256": run_fingerprint_sha256,
    }


def _repo_index_included_paths(repo_index: dict[str, Any]) -> list[str]:
    files = repo_index.get("files", []) or []
    out: list[str] = []
    for f in files:
        if not isinstance(f, dict):
            continue
        p = f.get("path")
        if isinstance(p, str) and p:
            out.append(p)
    return sorted(out)


def _pass2_defaults_from_env() -> dict[str, int]:
    """
    Pass 2 fetch caps.

    Notes:
    - We accept both *_CHARS and legacy-ish *_CHAR variants to avoid config typos breaking runs.
    - max_chars_per_file is optional; 0/absent disables per-file truncation.
    """
    max_files = int(os.environ.get("SNAPSHOTTER_PASS2_MAX_FILES", "120"))

    max_total_chars_raw = os.environ.get("SNAPSHOTTER_PASS2_MAX_TOTAL_CHARS", "").strip()
    if not max_total_chars_raw:
        max_total_chars_raw = os.environ.get("SNAPSHOTTER_PASS2_MAX_TOTAL_CHAR", "").strip()
    max_total_chars = int(max_total_chars_raw) if max_total_chars_raw else 250000

    max_chars_per_file_raw = os.environ.get("SNAPSHOTTER_PASS2_MAX_CHARS_PER_FILE", "").strip()
    max_chars_per_file = int(max_chars_per_file_raw) if max_chars_per_file_raw else 0

    caps: dict[str, int] = {"max_files": max_files, "max_total_chars": max_total_chars}
    if max_chars_per_file > 0:
        caps["max_chars_per_file"] = max_chars_per_file
    return caps


def _extract_candidate_paths(repo_index: dict[str, Any]) -> list[str]:
    """
    Extract Pass 1 read-plan suggestions.

    Canonical shape:
      {"candidates":[{"path":"..."}, ...]}

    We keep list/dict tolerance because it costs nothing and doesn't introduce split-brain.
    Returns: candidate paths in original order (may include duplicates).
    """
    v = repo_index.get("read_plan_suggestions")
    out: list[str] = []

    if isinstance(v, dict):
        cands = v.get("candidates", []) or []
        if isinstance(cands, list):
            for c in cands:
                if isinstance(c, str) and c:
                    out.append(c)
                elif isinstance(c, dict):
                    p = c.get("path")
                    if isinstance(p, str) and p:
                        out.append(p)
        return out

    if isinstance(v, list):
        for it in v:
            if isinstance(it, str) and it:
                out.append(it)
            elif isinstance(it, dict):
                p = it.get("path")
                if isinstance(p, str) and p:
                    out.append(p)
        return out

    return out


def _extract_closure_seed_paths(repo_index: dict[str, Any]) -> list[str]:
    """
    Pass1 emits read_plan_closure_seeds. We treat it as deterministic priority.
    """
    v = repo_index.get("read_plan_closure_seeds")
    out: list[str] = []

    if isinstance(v, list):
        for it in v:
            if isinstance(it, str) and it:
                out.append(it)
            elif isinstance(it, dict):
                p = it.get("path")
                if isinstance(p, str) and p:
                    out.append(p)

    return out


def _deterministic_read_plan(repo_index: dict[str, Any], *, max_files: int) -> tuple[list[str], dict[str, Any]]:
    """
    Deterministic plan selection (LOCKED):
      1) read_plan_closure_seeds (order preserved)
      2) read_plan_suggestions.candidates (order preserved)
      3) top-off with remaining included paths (lexicographic)

    Then:
      - de-dupe while preserving order
      - gate to included paths
      - cap at max_files
    """
    seed_paths = _extract_closure_seed_paths(repo_index)
    cand_paths = _extract_candidate_paths(repo_index)

    raw_combined = seed_paths + cand_paths
    combined_dedup = list(dict.fromkeys(raw_combined))

    included = _repo_index_included_paths(repo_index)
    included_set = set(included)

    before_filter = len(combined_dedup)
    plan = [p for p in combined_dedup if p in included_set]
    filtered_out = before_filter - len(plan)

    used_topoff = False
    if len(plan) < max_files:
        used_topoff = True
        seen = set(plan)
        for p in included:
            if p in seen:
                continue
            plan.append(p)
            seen.add(p)
            if len(plan) >= max_files:
                break

    final = plan[:max_files]

    debug: dict[str, Any] = {
        "max_files": max_files,
        "closure_seeds_len_raw": len(seed_paths),
        "candidates_len_raw": len(cand_paths),
        "combined_len_raw": len(raw_combined),
        "combined_len_deduped": len(combined_dedup),
        "combined_filtered_out_not_in_repo": filtered_out,
        "included_len": len(included),
        "used_topoff": used_topoff,
        "final_plan_len": len(final),
    }
    return final, debug


def _llm_read_plan_stub(repo_index: dict[str, Any], *, max_files: int) -> tuple[list[str], list[str], dict[str, Any]]:
    """
    v0.1 uses deterministic plan selection; semantic generation is in pass2_semantic.py.
    """
    plan, debug = _deterministic_read_plan(repo_index, max_files=max_files)
    return plan, [], debug


def _stream_read_utf8_with_replacement(path: Path, *, max_chars: int) -> tuple[str, bool]:
    """
    Stream read file as UTF-8 with replacement, up to max_chars characters.
    Returns (text, hit_limit) where hit_limit means we read >= max_chars chars.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    chunks: list[str] = []
    total = 0
    hit_limit = False

    with path.open("rb") as f:
        while True:
            b = f.read(8192)
            if not b:
                break
            s = decoder.decode(b)
            if not s:
                continue
            chunks.append(s)
            total += len(s)

            if total >= max_chars:
                overflow = total - max_chars
                if overflow > 0:
                    chunks[-1] = chunks[-1][:-overflow]
                hit_limit = True
                break

        if not hit_limit:
            tail = decoder.decode(b"", final=True)
            if tail:
                chunks.append(tail)

    return "".join(chunks), hit_limit


def _compute_files_not_read(
        *,
        repo_index: dict[str, Any],
        files_read: list[dict[str, Any]],
        read_plan_selected: list[str],
        not_read_reasons_for_selected: dict[str, str],
) -> list[dict[str, Any]]:
    """
    Build files_not_read across *all included* paths, with explicit reasons:
    - if file is read => omitted
    - if selected but not read => use recorded reason or "unknown_not_read"
    - if included but not selected => "not_in_read_plan"
    """
    included_paths = _repo_index_included_paths(repo_index)

    read_paths = [
        it.get("path")
        for it in files_read
        if isinstance(it, dict) and isinstance(it.get("path"), str) and it.get("path")
    ]
    read_set = set(read_paths)

    plan_set = {p for p in read_plan_selected if isinstance(p, str) and p}

    out: list[dict[str, Any]] = []
    for p in included_paths:
        if p in read_set:
            continue

        reason = not_read_reasons_for_selected.get(p)
        if not reason:
            reason = "not_in_read_plan" if p not in plan_set else "unknown_not_read"
        out.append({"path": p, "reason": reason})

    return out


def _normalize_architecture_snapshot(
        *,
        arch: dict[str, Any],
        repo_url: str,
        resolved_commit: str,
        job_id: str,
        repo_index: dict[str, Any],
        read_plan_selected: list[str],
        read_plan_missing: list[str],
        pass2_caps: dict[str, int],
        read_plan_source: str,
        read_plan_debug: dict[str, Any] | None,
        pass2_total_chars: int,
        files_read: list[dict[str, Any]],
        files_not_read: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Force self-auditing + required top-level keys, regardless of LLM drift.
    Also ensures each module has evidence_paths and responsibilities (unknown if missing).
    """
    out = dict(arch or {})
    out["generated_at"] = utc_ts()
    out["repo"] = {"repo_url": repo_url, "resolved_commit": resolved_commit or "unknown", "job_id": job_id}

    out["read_plan"] = {
        "selected_paths": read_plan_selected,
        "missing_paths": read_plan_missing,
        "caps": pass2_caps,
        "source": read_plan_source,
    }
    if read_plan_debug is not None:
        out["read_plan_debug"] = read_plan_debug

    files_scanned = int(repo_index.get("counts", {}).get("files_scanned", 0))
    files_included = int(repo_index.get("counts", {}).get("files_included", 0))

    out["coverage"] = {
        "files_scanned": files_scanned,
        "files_read": len(files_read),
        "files_not_read": len(files_not_read),
        "files_included_from_pass1": files_included,
    }
    out["pass2_total_chars"] = int(pass2_total_chars)

    out["files_read"] = files_read
    out["files_not_read"] = files_not_read

    modules = out.get("modules")
    if not isinstance(modules, list):
        modules = []
        out["modules"] = modules

    uncertainties = out.get("uncertainties")
    if not isinstance(uncertainties, list):
        uncertainties = []
        out["uncertainties"] = uncertainties

    for i, m in enumerate(modules):
        if not isinstance(m, dict):
            continue

        name = m.get("name")
        if not isinstance(name, str) or not name.strip():
            m["name"] = f"unknown_module_{i}"

        mtype = m.get("type")
        if not isinstance(mtype, str) or not mtype.strip():
            m["type"] = "unknown"

        ev = m.get("evidence_paths")
        if not isinstance(ev, list):
            ev = []
        ev = [p for p in ev if isinstance(p, str) and p]
        m["evidence_paths"] = ev

        resp = m.get("responsibilities")
        if not isinstance(resp, list) or not resp:
            resp = []
        resp = [r for r in resp if isinstance(r, str) and r.strip()]
        if not ev:
            m["responsibilities"] = ["unknown"]
            uncertainties.append(
                {
                    "type": "ungrounded_module",
                    "description": f"Module '{m['name']}' lacks evidence_paths in files_read; responsibilities set to unknown.",
                    "files_involved": [],
                    "suggested_questions": ["Which files define this module's responsibilities? Add them to read_plan."],
                }
            )
        else:
            if not resp:
                m["responsibilities"] = ["unknown"]
                uncertainties.append(
                    {
                        "type": "empty_responsibilities",
                        "description": f"Module '{m['name']}' had no responsibilities listed; set to unknown.",
                        "files_involved": ev,
                        "suggested_questions": ["What does this module do? Add explicit responsibilities."],
                    }
                )
            else:
                m["responsibilities"] = resp

        deps = m.get("dependencies")
        if not isinstance(deps, list):
            deps = []
        deps = [d for d in deps if isinstance(d, str) and d.strip()]
        m["dependencies"] = deps

    return out


# -----------------------------
# Nodes
# -----------------------------
def node_load_job(state: SnapshotterState) -> SnapshotterState:
    stage = STAGE_PARSE_JOB
    try:
        job = Job.model_validate(state["payload"]).finalize()

        workdir = ".snapshotter_tmp"
        repo_dir = f"{workdir}/repo"
        out_dir = f"out/{job.repo_slug or 'repo'}/{job.timestamp_utc or 'ts'}/{job.job_id or 'job'}"
        Path(workdir).mkdir(parents=True, exist_ok=True)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # Canonical filenames come from pass1 when available.
        # We still keep repo_index.json as a *convenience alias* for existing tooling,
        # but PASS1_REPO_INDEX.json is the source of truth.
        try:
            from snapshotter.pass1 import PASS1_REPO_INDEX_FILENAME  # type: ignore
        except Exception:
            PASS1_REPO_INDEX_FILENAME = "PASS1_REPO_INDEX.json"  # type: ignore

        try:
            from snapshotter.pass1 import DEPENDENCY_GRAPH_FILENAME  # type: ignore
        except Exception:
            DEPENDENCY_GRAPH_FILENAME = "DEPENDENCY_GRAPH.json"  # type: ignore

        local_paths = {
            # Canonical Pass1 artifacts (required by contract)
            "pass1_repo_index": str(Path(out_dir) / str(PASS1_REPO_INDEX_FILENAME)),
            "dependency_graph": str(Path(out_dir) / str(DEPENDENCY_GRAPH_FILENAME)),
            # Convenience alias (non-canonical)
            "repo_index": str(Path(out_dir) / "repo_index.json"),
            # Pass2 + misc
            "artifact_manifest": str(Path(out_dir) / "artifact_manifest.json"),
            "architecture_snapshot": str(Path(out_dir) / "ARCHITECTURE_SUMMARY_SNAPSHOT.json"),
            "gaps": str(Path(out_dir) / "GAPS_AND_INCONSISTENCIES.json"),
            "onboarding": str(Path(out_dir) / "ONBOARDING.md"),
            # Debug-only LLM raw/repaired (not uploaded by default)
            "pass2_llm_raw_output": str(Path(out_dir) / "PASS2_LLM_RAW_OUTPUT.txt"),
            "pass2_llm_repaired_output": str(Path(out_dir) / "PASS2_LLM_REPAIRED_OUTPUT.txt"),
        }

        state["stage"] = stage
        state["job"] = job
        state["workdir"] = workdir
        state["repo_dir"] = repo_dir
        state["out_dir"] = out_dir
        state["local_paths"] = local_paths
        return state
    except Exception as e:
        raise SnapshotterStageError(stage, e) from e


def node_clone_repo(state: SnapshotterState) -> SnapshotterState:
    stage = STAGE_CLONE
    try:
        job = state["job"]
        resolved_commit = clone_and_checkout(job.repo_url, job.ref, state["workdir"])
        state["stage"] = stage
        state["resolved_commit"] = resolved_commit
        return state
    except Exception as e:
        raise SnapshotterStageError(stage, e) from e


def node_pass1_build_index(state: SnapshotterState) -> SnapshotterState:
    """
    LOCKED CONTRACT:
      - Pass1 stage MUST produce BOTH:
          * PASS1_REPO_INDEX.json (canonical)
          * DEPENDENCY_GRAPH.json (canonical)
      - graph.py MUST call pass1.generate_pass1_artifacts(...) to do so.
      - graph.py MUST fail fast if either artifact is missing/empty.
      - resolved_commit correctness is enforced (branch/ref runs).
      - repo_index.json is written as a convenience alias only (non-canonical).
    """
    stage = STAGE_PASS1_REPO_INDEX
    try:
        job = state["job"]
        lp = state["local_paths"]
        out_dir = Path(state["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        resolved_commit = state.get("resolved_commit", "unknown")

        try:
            from snapshotter.pass1 import generate_pass1_artifacts  # type: ignore
        except Exception as e:
            raise RuntimeError("pass1.generate_pass1_artifacts is required by the locked contract") from e

        # Run Pass1 (canonical artifact writer).
        generate_pass1_artifacts(  # type: ignore[misc]
            repo_dir=state["repo_dir"],
            job=job,
            out_dir=str(out_dir),
            resolved_commit=resolved_commit,
        )

        pass1_repo_index_path = Path(lp["pass1_repo_index"])
        dependency_graph_path = Path(lp["dependency_graph"])

        # Enforce contract: both artifacts must exist and be non-empty.
        _require_file(pass1_repo_index_path, label="PASS1_REPO_INDEX")
        _require_file(dependency_graph_path, label="DEPENDENCY_GRAPH")

        # Load canonical repo_index.
        repo_index = _read_json(pass1_repo_index_path)

        # Enforce resolved_commit injection correctness.
        job_obj = repo_index.get("job")
        if not isinstance(job_obj, dict):
            raise RuntimeError("PASS1_REPO_INDEX.job must be an object (dict)")
        rc = job_obj.get("resolved_commit")
        if not isinstance(rc, str) or not rc:
            raise RuntimeError("PASS1_REPO_INDEX.job.resolved_commit is required (missing/empty)")
        if rc != resolved_commit:
            raise RuntimeError(
                f"resolved_commit mismatch: pass1_repo_index has '{rc}' but orchestrator resolved '{resolved_commit}'"
            )

        # Convenience alias for tools that still look for repo_index.json.
        write_json(lp["repo_index"], repo_index)

        state["stage"] = stage
        state["repo_index"] = repo_index
        return state
    except Exception as e:
        raise SnapshotterStageError(stage, e) from e


def node_pass2_make_read_plan(state: SnapshotterState) -> SnapshotterState:
    stage = STAGE_PASS2_MAKE_READ_PLAN
    try:
        caps = _pass2_defaults_from_env()
        repo_index = state["repo_index"]

        included_paths = _repo_index_included_paths(repo_index)
        included_set = set(included_paths)

        selected, requested_missing, plan_debug = _llm_read_plan_stub(repo_index, max_files=caps["max_files"])

        # Gate selection to repo-included paths
        selected_in_repo = [p for p in selected if isinstance(p, str) and p in included_set]

        # Missing: anything explicitly requested_missing + anything selected but not in repo
        missing_set: set[str] = set()
        for p in requested_missing:
            if isinstance(p, str) and p and p not in included_set:
                missing_set.add(p)
        for p in selected:
            if isinstance(p, str) and p and p not in included_set:
                missing_set.add(p)
        missing = sorted(missing_set)

        # Cap then de-dupe preserving order
        selected_in_repo = selected_in_repo[: caps["max_files"]]
        seen: set[str] = set()
        final_plan: list[str] = []
        for p in selected_in_repo:
            if p in seen:
                continue
            seen.add(p)
            final_plan.append(p)

        plan_debug = dict(plan_debug or {})
        plan_debug["final_plan_len_after_repo_gate"] = len(final_plan)
        plan_debug["caps_max_files"] = int(caps.get("max_files", 0))

        state["stage"] = stage
        state["read_plan"] = final_plan
        state["pass2_caps"] = caps
        state["read_plan_missing"] = missing
        state["pass2_read_plan_debug"] = plan_debug
        return state
    except Exception as e:
        raise SnapshotterStageError(stage, e) from e


def node_pass2_fetch_files(state: SnapshotterState) -> SnapshotterState:
    stage = STAGE_PASS2_FETCH_FILES
    try:
        repo_dir = Path(state["repo_dir"])
        caps = state.get("pass2_caps", _pass2_defaults_from_env())
        max_total = int(caps.get("max_total_chars", 250000))
        max_per_file = int(caps.get("max_chars_per_file", 0))  # optional; 0 => disabled

        total_chars = 0
        contents: dict[str, str] = {}
        files_read: list[dict[str, Any]] = []
        not_read_reasons: dict[str, str] = {}

        plan = state.get("read_plan", [])

        for rel in plan:
            if not isinstance(rel, str) or not rel:
                continue

            remaining = max_total - total_chars
            if remaining <= 0:
                not_read_reasons[rel] = "exceeds_total_char_cap"
                continue

            fp = repo_dir / rel
            if not fp.exists():
                not_read_reasons[rel] = "missing_on_disk"
                continue

            try:
                if max_per_file > 0 and max_per_file <= remaining:
                    text, hit = _stream_read_utf8_with_replacement(fp, max_chars=max_per_file + 1)
                    longer_than_cap = hit or (len(text) > max_per_file)
                    if longer_than_cap:
                        text = text[:max_per_file]
                    n = len(text)

                    if n > remaining:
                        not_read_reasons[rel] = "exceeds_total_char_cap"
                        continue

                    contents[rel] = text
                    total_chars += n
                    files_read.append({"path": rel, "chars": n, "truncated": bool(longer_than_cap)})
                    continue

                text, hit = _stream_read_utf8_with_replacement(fp, max_chars=remaining + 1)
                too_big = hit or (len(text) > remaining)
                if too_big:
                    not_read_reasons[rel] = "exceeds_total_char_cap"
                    continue

                n = len(text)
                contents[rel] = text
                total_chars += n
                files_read.append({"path": rel, "chars": n, "truncated": False})

            except Exception:
                not_read_reasons[rel] = "decode_issues"
                continue

        state["stage"] = stage
        state["file_contents_map"] = contents
        state["pass2_total_chars"] = total_chars
        state["pass2_files_read"] = files_read
        state["pass2_not_read_reasons"] = not_read_reasons
        return state
    except Exception as e:
        raise SnapshotterStageError(stage, e) from e


def node_pass2_generate_outputs(state: SnapshotterState) -> SnapshotterState:
    stage = STAGE_PASS2_GENERATE_OUTPUTS
    try:
        job = state["job"]
        lp = state["local_paths"]
        repo_index = state["repo_index"]
        resolved_commit = state.get("resolved_commit", "unknown")

        files_read = state.get("pass2_files_read", [])
        not_read_reasons = state.get("pass2_not_read_reasons", {})
        files_not_read = _compute_files_not_read(
            repo_index=repo_index,
            files_read=files_read,
            read_plan_selected=state.get("read_plan", []),
            not_read_reasons_for_selected=not_read_reasons,
        )

        try:
            arch_raw, gaps_raw, onboarding_md = generate_pass2_semantic_artifacts(
                repo_url=job.repo_url,
                resolved_commit=resolved_commit,
                job_id=job.job_id or "unknown",
                repo_index=repo_index,
                files_read=files_read,
                files_not_read=files_not_read,
                file_contents_map=state.get("file_contents_map", {}),
            )
        except Pass2SemanticError as e:
            raw_text = getattr(e, "raw_text", None)
            repaired_text = getattr(e, "repaired_text", None)

            raw_path = lp.get("pass2_llm_raw_output") or str(Path(state["out_dir"]) / "PASS2_LLM_RAW_OUTPUT.txt")
            repaired_path = lp.get("pass2_llm_repaired_output") or str(
                Path(state["out_dir"]) / "PASS2_LLM_REPAIRED_OUTPUT.txt"
            )

            try:
                if isinstance(raw_text, str) and raw_text:
                    Path(raw_path).write_text(raw_text, encoding="utf-8")
                if isinstance(repaired_text, str) and repaired_text:
                    Path(repaired_path).write_text(repaired_text, encoding="utf-8")
            except Exception:
                pass

            msg = f"pass2_semantic failed: {e}"
            if isinstance(raw_text, str) and raw_text:
                msg += f"\nRaw LLM output saved at: {raw_path}"
            if isinstance(repaired_text, str) and repaired_text:
                msg += f"\nRepaired LLM output saved at: {repaired_path}"
            raise RuntimeError(msg) from e

        arch = _normalize_architecture_snapshot(
            arch=arch_raw,
            repo_url=job.repo_url,
            resolved_commit=resolved_commit,
            job_id=job.job_id or "unknown",
            repo_index=repo_index,
            read_plan_selected=state.get("read_plan", []),
            read_plan_missing=state.get("read_plan_missing", []),
            pass2_caps=state.get("pass2_caps", _pass2_defaults_from_env()),
            read_plan_source="pass2_semantic",
            read_plan_debug=state.get("pass2_read_plan_debug"),
            pass2_total_chars=int(state.get("pass2_total_chars", 0)),
            files_read=files_read,
            files_not_read=files_not_read,
        )

        write_json(lp["architecture_snapshot"], arch)
        write_json(lp["gaps"], gaps_raw)
        Path(lp["onboarding"]).write_text(onboarding_md or "", encoding="utf-8")

        state["stage"] = stage
        return state
    except Exception as e:
        raise SnapshotterStageError(stage, e) from e


def node_pass1_manifest(state: SnapshotterState) -> SnapshotterState:
    stage = STAGE_PASS1_MANIFEST
    try:
        lp = state["local_paths"]

        # Enforce contract again before fingerprinting/upload selection.
        _require_file(Path(lp["pass1_repo_index"]), label="PASS1_REPO_INDEX")
        _require_file(Path(lp["dependency_graph"]), label="DEPENDENCY_GRAPH")

        manifest = build_artifact_manifest(
            {
                "pass1_repo_index": lp["pass1_repo_index"],
                "dependency_graph": lp["dependency_graph"],
                "repo_index": lp["repo_index"],  # convenience alias (may be useful for humans/tools)
                "architecture_snapshot": lp["architecture_snapshot"],
                "gaps": lp["gaps"],
                "onboarding": lp["onboarding"],
            }
        )
        write_json(lp["artifact_manifest"], manifest)
        state["stage"] = stage
        return state
    except Exception as e:
        raise SnapshotterStageError(stage, e) from e


def node_validate_basic(state: SnapshotterState) -> SnapshotterState:
    stage = STAGE_VALIDATE_BASIC
    try:
        lp = state["local_paths"]

        # LOCKED CONTRACT: dependency_graph is required.
        validate_basic_artifacts(
            {
                "pass1_repo_index": lp["pass1_repo_index"],
                "dependency_graph": lp["dependency_graph"],
                "repo_index": lp["repo_index"],  # alias, allowed
                "artifact_manifest": lp["artifact_manifest"],
                "architecture_snapshot": lp["architecture_snapshot"],
                "gaps": lp["gaps"],
                "onboarding": lp["onboarding"],
            }
        )
        state["stage"] = stage
        return state
    except Exception as e:
        raise SnapshotterStageError(stage, e) from e


def node_upload_artifacts(state: SnapshotterState) -> SnapshotterState:
    stage = STAGE_UPLOAD
    try:
        job = state["job"]
        cfg = state["config"]
        lp = state["local_paths"]

        uploader = S3Uploader(bucket=job.output.s3_bucket, prefix=job.s3_job_prefix(), region=cfg.aws_region)

        if cfg.dry_run:
            state["stage"] = stage
            state["s3_paths"] = {}
            return state

        # Upload canonical + useful convenience artifacts.
        s3_paths: dict[str, Optional[str]] = {
            "pass1_repo_index": uploader.upload_file(
                Path(lp["pass1_repo_index"]).name, lp["pass1_repo_index"], content_type="application/json"
            ),
            "dependency_graph": uploader.upload_file(
                Path(lp["dependency_graph"]).name, lp["dependency_graph"], content_type="application/json"
            ),
            "repo_index": uploader.upload_file("repo_index.json", lp["repo_index"], content_type="application/json"),
            "artifact_manifest": uploader.upload_file(
                "artifact_manifest.json", lp["artifact_manifest"], content_type="application/json"
            ),
            "architecture_snapshot": uploader.upload_file(
                "ARCHITECTURE_SUMMARY_SNAPSHOT.json", lp["architecture_snapshot"], content_type="application/json"
            ),
            "gaps": uploader.upload_file("GAPS_AND_INCONSISTENCIES.json", lp["gaps"], content_type="application/json"),
            "onboarding": uploader.upload_file("ONBOARDING.md", lp["onboarding"], content_type="text/markdown"),
        }

        state["stage"] = stage
        state["s3_paths"] = s3_paths
        return state
    except Exception as e:
        raise SnapshotterStageError(stage, e) from e


def node_emit_result(state: SnapshotterState) -> SnapshotterState:
    stage = STAGE_EMIT_RESULT
    try:
        job = state["job"]
        cfg = state["config"]
        resolved_commit = state.get("resolved_commit", "unknown")
        lp = state["local_paths"]

        repo_index_sha = _file_sha256(lp["repo_index"])

        if cfg.dry_run:
            state["result"] = {
                "ok": True,
                "stage": STAGE_DONE_DRY_RUN,
                "job_id": job.job_id,
                "repo_url": job.repo_url,
                "requested_ref": job.ref,
                "resolved_commit": resolved_commit,
                "s3_bucket": job.output.s3_bucket,
                "s3_prefix": job.s3_job_prefix(),
                "job_payload_source": state.get("payload_src", "unknown"),
                "artifacts": {
                    "pass1_repo_index_local": lp["pass1_repo_index"],
                    "dependency_graph_local": lp["dependency_graph"],
                    "repo_index_local": lp["repo_index"],
                    "artifact_manifest_local": lp["artifact_manifest"],
                    "architecture_snapshot_local": lp["architecture_snapshot"],
                    "gaps_local": lp["gaps"],
                    "onboarding_local": lp["onboarding"],
                },
                "hashes": {"repo_index_sha256": repo_index_sha},
            }
            state["stage"] = stage
            return state

        s3p = state.get("s3_paths", {})
        state["result"] = {
            "ok": True,
            "stage": STAGE_DONE,
            "job_id": job.job_id,
            "repo_url": job.repo_url,
            "requested_ref": job.ref,
            "resolved_commit": resolved_commit,
            "s3_bucket": job.output.s3_bucket,
            "s3_prefix": job.s3_job_prefix(),
            "job_payload_source": state.get("payload_src", "unknown"),
            "artifacts": {
                "pass1_repo_index": s3p.get("pass1_repo_index"),
                "dependency_graph": s3p.get("dependency_graph"),
                "repo_index": s3p.get("repo_index"),
                "artifact_manifest": s3p.get("artifact_manifest"),
                "architecture_snapshot": s3p.get("architecture_snapshot"),
                "gaps": s3p.get("gaps"),
                "onboarding": s3p.get("onboarding"),
            },
            "hashes": {"repo_index_sha256": repo_index_sha},
        }
        state["stage"] = stage
        return state
    except Exception as e:
        raise SnapshotterStageError(stage, e) from e


# -----------------------------
# Graph wiring
# -----------------------------
def build_snapshotter_graph():
    g = StateGraph(SnapshotterState)

    g.add_node("load_job", node_load_job)
    g.add_node("clone_repo", node_clone_repo)
    g.add_node("pass1_build_index", node_pass1_build_index)
    g.add_node("pass2_make_read_plan", node_pass2_make_read_plan)
    g.add_node("pass2_fetch_files", node_pass2_fetch_files)
    g.add_node("pass2_generate_outputs", node_pass2_generate_outputs)
    g.add_node("pass1_manifest", node_pass1_manifest)
    g.add_node("validate_basic", node_validate_basic)
    g.add_node("upload_artifacts", node_upload_artifacts)
    g.add_node("emit_result", node_emit_result)

    g.set_entry_point("load_job")
    g.add_edge("load_job", "clone_repo")
    g.add_edge("clone_repo", "pass1_build_index")
    g.add_edge("pass1_build_index", "pass2_make_read_plan")
    g.add_edge("pass2_make_read_plan", "pass2_fetch_files")
    g.add_edge("pass2_fetch_files", "pass2_generate_outputs")
    g.add_edge("pass2_generate_outputs", "pass1_manifest")
    g.add_edge("pass1_manifest", "validate_basic")
    g.add_edge("validate_basic", "upload_artifacts")
    g.add_edge("upload_artifacts", "emit_result")
    g.add_edge("emit_result", END)

    return g.compile()


def run_snapshotter_graph(
        *,
        payload: dict[str, Any],
        payload_src: str,
        dry_run: bool,
        aws_region: str | None,
) -> dict[str, Any]:
    app = build_snapshotter_graph()
    state: SnapshotterState = {
        "payload": payload,
        "payload_src": payload_src,
        "config": RuntimeConfig(dry_run=dry_run, aws_region=aws_region),
        "stage": STAGE_INIT,
    }
    final_state = app.invoke(state)
    return final_state["result"]
