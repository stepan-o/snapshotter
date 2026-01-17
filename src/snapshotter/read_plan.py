# snapshotter/read_plan.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------------------
# Deterministic Read Plan (Pass1-only inputs) — STRICT CONTRACT (LOCKED)
#
# This module is deterministic and assumes Pass1 provides the evidence it needs.
# We DO NOT "guess" imports via legacy heuristics anymore.
#
# REQUIRED Pass1 file-shape invariants for this module:
#   - files: list[dict] where each file has:
#       * path: str
#       * deps.import_edges: list[dict]  (preferred)
#         OR import_edges: list[dict]    (acceptable, but still evidence-based)
#
# REQUIRED edge-shape invariants:
#   - spec: str (may be external)
#   - For internal edges: resolved_path: str and is_external == False
#
# Output contract:
#   - suggest_files_to_read(...) returns {"max_files_to_read_default": int, "candidates": [...]}
#   - "candidates" is a list of dicts: {"path": str, "score": float, "reasons": [str...]}
#   - No v2 list. Downstream closure seeding must operate on read_plan_suggestions["candidates"].
# --------------------------------------------------------------------------------------

# Known "app roots" we want to treat as equivalent for scoring/bucketing.
# Keep this bounded and deterministic (no globbing).
_KNOWN_PREFIXES: tuple[str, ...] = (
    "frontend/",
    "apps/web/",
    "apps/frontend/",
)

# Regex that matches any of the prefixes above, optional.
_PREFIX_RX = r"(?:frontend/|apps/web/|apps/frontend/)"


def _strip_known_prefix(path: str) -> tuple[str, str]:
    """
    Returns (prefix, stripped) where stripped has the known prefix removed.
    Deterministic; first-match wins based on _KNOWN_PREFIXES ordering.
    """
    p = (path or "").replace("\\", "/")
    for pref in _KNOWN_PREFIXES:
        if p.startswith(pref):
            return pref, p[len(pref) :]
    return "", p


# --- Pattern-based priorities (Next.js + backend/security) ---

HIGH_PATTERNS: list[tuple[re.Pattern[str], int, str]] = [
    # Next.js app router key files
    (re.compile(rf"^(?:{_PREFIX_RX})?app/(layout|page)\.(t|j)sx?$"), 80, "next_root_layout_or_page"),
    (re.compile(rf"^(?:{_PREFIX_RX})?app/.+/(layout|page)\.(t|j)sx?$"), 80, "next_layout_or_page"),
    (re.compile(rf"^(?:{_PREFIX_RX})?app/.+/(error|loading|not-found)\.(t|j)sx?$"), 70, "next_special_file"),
    (re.compile(rf"^(?:{_PREFIX_RX})?app/api/.+/route\.(t|j)s$"), 75, "next_app_api_route"),
    # Next.js pages router
    (re.compile(rf"^(?:{_PREFIX_RX})?pages/(_app|_document)\.(t|j)sx?$"), 75, "next_pages_router_core"),
    (re.compile(rf"^(?:{_PREFIX_RX})?pages/api/.+\.(t|j)s$"), 75, "next_pages_api_route"),
    (re.compile(rf"^(?:{_PREFIX_RX})?pages/index\.(t|j)sx?$"), 65, "next_pages_index"),
    # Middleware + config
    (re.compile(rf"^(?:{_PREFIX_RX})?middleware\.(t|j)s$"), 80, "middleware"),
    (re.compile(r"^next\.config\.(js|mjs|cjs)$"), 60, "next_config"),
    (re.compile(rf"^(?:{_PREFIX_RX})next\.config\.(js|mjs|cjs)$"), 55, "next_config_prefixed"),
    (re.compile(rf"^(?:{_PREFIX_RX})?instrumentation\.(t|j)s$"), 55, "instrumentation"),
    # Backend entrypoints (generic)
    (re.compile(r"(^|/)(main|app|server)\.py$"), 60, "backend_entrypoint"),
    (re.compile(r"(^|/)(routes|routers|controllers)/"), 55, "backend_routing"),
    (re.compile(r"(^|/)(api)/"), 55, "backend_api_dir"),
    # DB / schema / migrations
    (re.compile(r"(^|/)(db|database|models|schemas|schema|entities)/"), 65, "db_schema_models"),
    (re.compile(r"(^|/)migrations?/"), 60, "migrations"),
    (re.compile(r"(^|/)alembic/"), 60, "alembic"),
    (re.compile(r"(^|/)schema\.prisma$"), 70, "prisma_schema"),
    # Contracts / clients / SDK
    (re.compile(r"(^|/)(contracts|client|clients|sdk|openapi|swagger)/"), 70, "contract_or_sdk"),
    (re.compile(r"(^|/)(types|interfaces|dto|validators)/"), 55, "types_dto_validators"),
    # Auth / security
    (re.compile(r"(^|/)(auth|security|permissions|rbac)/"), 70, "auth_security_dir"),
    (re.compile(r"(^|/).*(jwt|token|session|oauth|csrf|cors).*\."), 50, "auth_security_keyword"),
]

NEG_PATTERNS: list[tuple[re.Pattern[str], int, str]] = [
    (re.compile(r"(^|/)__tests__(/|$)"), 60, "tests"),
    (re.compile(r"(^|/)(test|tests)(/|$)"), 60, "tests"),
    (re.compile(r"\.(spec|test)\.(t|j)sx?$"), 60, "tests"),
    (re.compile(r"\.stories\.(t|j)sx?$"), 40, "storybook"),
]

DEFAULT_CAPS = {
    # Caps are keyed by "logical roots" (after stripping known prefixes).
    "app/": 35,
    "pages/": 20,
    "src/": 25,
    "lib/": 25,
    "backend/": 25,
    "api/": 25,
    "server/": 25,
    "other": 9999,
}

DEFAULT_MUST_INCLUDE = [
    rf"^(?:{_PREFIX_RX})?middleware\.(t|j)s$",
    r"^next\.config\.(js|mjs|cjs)$",
    rf"^(?:{_PREFIX_RX})?next\.config\.(js|mjs|cjs)$",
    rf"^(?:{_PREFIX_RX})?app/layout\.(t|j)sx?$",
    rf"^(?:{_PREFIX_RX})?app/page\.(t|j)sx?$",
    rf"^(?:{_PREFIX_RX})?pages/_app\.(t|j)sx?$",
    rf"^(?:{_PREFIX_RX})?pages/_document\.(t|j)sx?$",
]


@dataclass(frozen=True)
class Candidate:
    path: str
    score: float
    reasons: list[str]


def _bucket(path: str) -> str:
    """
    Bucketing honors known monorepo prefixes by stripping them first.
    """
    _, stripped = _strip_known_prefix(path)
    for k in ("app/", "pages/", "src/", "lib/", "backend/", "api/", "server/"):
        if stripped.startswith(k):
            return k
    return "other"


def _edges_for_file(f: dict[str, Any]) -> list[dict[str, Any]]:
    """
    STRICT: only evidence-bearing edges are accepted.
    Looks for:
      - f["deps"]["import_edges"] (preferred)
      - f["import_edges"] (acceptable)
    """
    deps = f.get("deps")
    if isinstance(deps, dict):
        edges = deps.get("import_edges")
        if isinstance(edges, list):
            return [e for e in edges if isinstance(e, dict)]
    edges2 = f.get("import_edges")
    if isinstance(edges2, list):
        return [e for e in edges2 if isinstance(e, dict)]
    return []


def _compute_imported_by(files: list[dict[str, Any]]) -> dict[str, int]:
    """
    STRICT fan-in:
      imported_by_count[path] = number of internal edges (is_external==False) whose resolved_path == path
    Requires Pass1 evidence edges.
    """
    if not isinstance(files, list) or not files:
        raise RuntimeError("read_plan: files must be a non-empty list")

    included_paths: list[str] = []
    for f in files:
        p = f.get("path")
        if isinstance(p, str) and p:
            included_paths.append(p)
    if not included_paths:
        raise RuntimeError("read_plan: files list contains no valid 'path' strings")

    included_set = set(included_paths)
    imported_by: dict[str, int] = {p: 0 for p in included_paths}

    saw_any_edges = False

    for f in files:
        edges = _edges_for_file(f)
        if not edges:
            continue

        saw_any_edges = True
        for e in edges:
            # Count only internal edges with a resolved_path.
            if bool(e.get("is_external", False)):
                continue
            rp = e.get("resolved_path")
            if not isinstance(rp, str) or not rp:
                # Internal edge without resolved_path breaks determinism & usefulness.
                raise RuntimeError("read_plan: internal import edge missing resolved_path")
            if rp in included_set:
                imported_by[rp] = imported_by.get(rp, 0) + 1

    if not saw_any_edges:
        raise RuntimeError(
            "read_plan: no import_edges found on any file; Pass1 must provide deps.import_edges evidence"
        )

    return imported_by


def _imports_count_for_file(f: dict[str, Any]) -> int:
    """
    STRICT fan-out:
      imports_count = number of edges with a non-empty 'spec'
    Requires Pass1 evidence edges (same source as _compute_imported_by).
    """
    edges = _edges_for_file(f)
    if not edges:
        raise RuntimeError("read_plan: missing import_edges on file (Pass1 contract violation)")
    return len([e for e in edges if isinstance(e.get("spec"), str) and e.get("spec")])


def _score(path: str, imports_count: int, imported_by_count: int) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    for rx, w, r in HIGH_PATTERNS:
        if rx.search(path):
            score += w
            reasons.append(r)

    for rx, w, r in NEG_PATTERNS:
        if rx.search(path):
            score -= w
            reasons.append(f"deprioritized:{r}")

    # graph-ish signals (deterministic)
    score += 2.0 * float(imports_count)
    score += 10.0 * float(imported_by_count)

    return score, reasons


def suggest_files_to_read(
        files: list[dict[str, Any]],
        max_files: int = 120,
        caps: dict[str, int] | None = None,
) -> dict[str, Any]:
    """
    Returns a stable, deterministic selection of top files.

    Output contract:
      {
        "max_files_to_read_default": int,
        "candidates": [
          {"path": str, "score": float, "reasons": [str... (bounded)]},
          ...
        ]
      }
    """
    caps = caps or dict(DEFAULT_CAPS)

    imported_by = _compute_imported_by(files)

    cands: list[Candidate] = []
    for f in files:
        path = f.get("path")
        if not isinstance(path, str) or not path:
            continue

        imports_count = _imports_count_for_file(f)
        score, reasons = _score(path, imports_count, imported_by.get(path, 0))
        cands.append(Candidate(path=path, score=score, reasons=reasons))

    # stable ordering: score desc, then path asc
    cands.sort(key=lambda c: (-c.score, c.path))

    must_include_rx = [re.compile(p) for p in DEFAULT_MUST_INCLUDE]
    picked: list[Candidate] = []
    picked_set: set[str] = set()
    bucket_counts: dict[str, int] = {k: 0 for k in caps.keys()}

    def try_pick(c: Candidate) -> None:
        b = _bucket(c.path)
        if c.path in picked_set:
            return
        if bucket_counts.get(b, 0) >= caps.get(b, 9999):
            return
        picked.append(c)
        picked_set.add(c.path)
        bucket_counts[b] = bucket_counts.get(b, 0) + 1

    # seed must-include
    for c in cands:
        if any(rx.search(c.path) for rx in must_include_rx):
            try_pick(c)

    # fill remainder under caps
    for c in cands:
        if len(picked) >= max_files:
            break
        try_pick(c)

    return {
        "max_files_to_read_default": max_files,
        "candidates": [
            {"path": c.path, "score": round(c.score, 3), "reasons": c.reasons[:8]}
            for c in picked
        ],
    }
