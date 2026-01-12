# snapshotter/read_plan.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# --- Pattern-based priorities (Next.js + backend/security) ---

# Highest-priority "what defines the app"
HIGH_PATTERNS: list[tuple[re.Pattern[str], int, str]] = [
    # Next.js app router key files
    (re.compile(r"^app/(layout|page)\.(t|j)sx?$"), 80, "next_root_layout_or_page"),
    (re.compile(r"^app/.+/(layout|page)\.(t|j)sx?$"), 80, "next_layout_or_page"),
    (re.compile(r"^app/.+/(error|loading|not-found)\.(t|j)sx?$"), 70, "next_special_file"),
    (re.compile(r"^app/api/.+/route\.(t|j)s$"), 75, "next_app_api_route"),

    # Next.js pages router
    (re.compile(r"^pages/(_app|_document)\.(t|j)sx?$"), 75, "next_pages_router_core"),
    (re.compile(r"^pages/api/.+\.(t|j)s$"), 75, "next_pages_api_route"),
    (re.compile(r"^pages/index\.(t|j)sx?$"), 65, "next_pages_index"),

    # Middleware + config
    (re.compile(r"^middleware\.(t|j)s$"), 80, "middleware"),
    (re.compile(r"^next\.config\.(js|mjs|cjs)$"), 60, "next_config"),
    (re.compile(r"^instrumentation\.(t|j)s$"), 55, "instrumentation"),

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

# Things we generally don't want in the top 120 unless repo is tiny
NEG_PATTERNS: list[tuple[re.Pattern[str], int, str]] = [
    (re.compile(r"(^|/)__tests__(/|$)"), 60, "tests"),
    (re.compile(r"(^|/)(test|tests)(/|$)"), 60, "tests"),
    (re.compile(r"\.(spec|test)\.(t|j)sx?$"), 60, "tests"),
    (re.compile(r"\.stories\.(t|j)sx?$"), 40, "storybook"),
]

DEFAULT_CAPS = {
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
    # If present, always include:
    r"^middleware\.(t|j)s$",
    r"^next\.config\.(js|mjs|cjs)$",
    r"^app/layout\.(t|j)sx?$",
    r"^app/page\.(t|j)sx?$",
    r"^pages/_app\.(t|j)sx?$",
    r"^pages/_document\.(t|j)sx?$",
]

@dataclass(frozen=True)
class Candidate:
    path: str
    score: float
    reasons: list[str]

def _bucket(path: str) -> str:
    for k in ("app/", "pages/", "src/", "lib/", "backend/", "api/", "server/"):
        if path.startswith(k):
            return k
    return "other"

def _module_id_from_path(path: str) -> str:
    """
    Best-effort mapping from a file path to an import-like module id.
    This is used ONLY to estimate "imported_by_count" deterministically.
    """
    p = Path(path)
    stem = str(p.with_suffix(""))  # drop extension
    stem = stem.replace("\\", "/")

    # normalize common roots
    for prefix in ("src/", "lib/"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]

    # app/ and pages/ are not real module roots, but keep them anyway
    # so internal imports like "app.foo" might match if present.
    return stem.replace("/", ".").strip(".")

def _compute_imported_by(files: list[dict[str, Any]]) -> dict[str, int]:
    """
    imported_by_count: how many other files import a module id that maps to this file.
    This is heuristic but deterministic.
    """
    mod_to_path: dict[str, str] = {}
    for f in files:
        mod_to_path[_module_id_from_path(f["path"])] = f["path"]

    imported_by: dict[str, int] = {f["path"]: 0 for f in files}

    for f in files:
        imports: list[str] = f.get("imports") or []
        for imp in imports:
            # direct match
            if imp in mod_to_path:
                imported_by[mod_to_path[imp]] += 1
                continue
            # prefix match (importing package.subthing)
            # find best target by trying progressively shorter prefixes
            parts = imp.split(".")
            for i in range(len(parts), 0, -1):
                pref = ".".join(parts[:i])
                if pref in mod_to_path:
                    imported_by[mod_to_path[pref]] += 1
                    break

    return imported_by

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

    # graph-ish signals
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
    """
    caps = caps or dict(DEFAULT_CAPS)
    imported_by = _compute_imported_by(files)

    # precompute candidates
    cands: list[Candidate] = []
    for f in files:
        path = f["path"]
        imports_count = len(f.get("imports") or [])
        score, reasons = _score(path, imports_count, imported_by.get(path, 0))
        cands.append(Candidate(path=path, score=score, reasons=reasons))

    # stable ordering: score desc, then path asc
    cands.sort(key=lambda c: (-c.score, c.path))

    # seed must-include
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
            {"path": c.path, "score": round(c.score, 3), "reasons": c.reasons[:8]}  # keep it bounded
            for c in picked
        ],
    }