# snapshotter/pass1.py
from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Any

from snapshotter.job import Job

# deterministic read-plan suggestions
from snapshotter.read_plan import suggest_files_to_read
from snapshotter.utils import is_probably_binary, sha256_bytes, utc_ts

LANG_BY_EXT = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
}

# Python-only fallback import regex (kept for non-python text formats)
PY_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([a-zA-Z0-9_\.]+)\s+import|import\s+([a-zA-Z0-9_\.]+))",
    re.M,
)

# --------------------------------------------------------------------------------------
# JS/TS import extraction (regex + comment stripping; deterministic)
# --------------------------------------------------------------------------------------

JS_ANY_IMPORT_EXPORT_FROM_RE = re.compile(
    r"""(?msx)
    ^\s*
    (?P<kw>import|export)\s+
    (?:type\s+)?                 # "import type ..." / "export type ..."
    (?:[\s\S]*?)                 # bindings (may be multiline)
    \sfrom\s*
    ["'](?P<spec>[^"']+)["']\s*;?
    """
)

JS_IMPORT_SIDE_EFFECT_RE = re.compile(
    r"""(?mx)
    ^\s*import\s*["'](?P<spec>[^"']+)["']\s*;?
    """
)

JS_DYNAMIC_IMPORT_RE = re.compile(r"""\bimport\s*\(\s*["'](?P<spec>[^"']+)["']\s*\)""")
JS_REQUIRE_RE = re.compile(r"""\brequire\s*\(\s*["'](?P<spec>[^"']+)["']\s*\)""")

TS_IMPORT_ASSIGN_REQUIRE_RE = re.compile(
    r"""(?mx)
    ^\s*import\s+[A-Za-z_\$][\w\$]*\s*=\s*require\s*\(\s*["'](?P<spec>[^"']+)["']\s*\)\s*;?
    """
)

JS_TOP_DEF_RE = re.compile(
    r"""(?mx)
    ^\s*
    (?:export\s+(?:default\s+)?)?
    (?:declare\s+)?                 # TS
    (?:async\s+)?                   # async function
    (?:
        function\s+([A-Za-z_\$][\w\$]*)
      | class\s+([A-Za-z_\$][\w\$]*)
      | (?:const|let|var)\s+([A-Za-z_\$][\w\$]*)\s*=
      | interface\s+([A-Za-z_\$][\w\$]*)      # TS
      | type\s+([A-Za-z_\$][\w\$]*)\s*=       # TS
      | enum\s+([A-Za-z_\$][\w\$]*)           # TS
    )
    """
)

# ------------------------------------------------------------
# TS/JS path alias + module resolution (Option B)
# ------------------------------------------------------------

# Typical resolver extensions for TS/JS imports (ordered)
JS_RESOLVE_EXTS = (
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".d.ts",
    ".json",
)

# Extremely common config locations (we scan these deterministically)
CONFIG_CANDIDATES = (
    "tsconfig.json",
    "jsconfig.json",
    "frontend/tsconfig.json",
    "frontend/jsconfig.json",
)

# -----------------------------
# Environment variable extraction
# -----------------------------
PY_ENV_PATTERNS = [
    # os.getenv("X") / os.getenv('X')
    re.compile(r"""\bos\.getenv\(\s*["']([A-Z0-9_]+)["']"""),
    # os.environ.get("X") / os.environ.get('X')
    re.compile(r"""\bos\.environ\.get\(\s*["']([A-Z0-9_]+)["']"""),
    # os.environ["X"]
    re.compile(r"""\bos\.environ\[\s*["']([A-Z0-9_]+)["']\s*\]"""),
]

JS_ENV_PATTERNS = [
    # process.env.X
    re.compile(r"""\bprocess\.env\.([A-Z0-9_]+)\b"""),
    # process.env["X"] / process.env['X']
    re.compile(r"""\bprocess\.env\[\s*["']([A-Z0-9_]+)["']\s*\]"""),
]


def infer_language(path: str) -> str:
    ext = Path(path).suffix.lower()
    return LANG_BY_EXT.get(ext, "unknown")


def _lineno_from_index(text: str, idx: int) -> int:
    if idx <= 0:
        return 1
    return text.count("\n", 0, idx) + 1


def parse_python_defs_and_imports(text: str) -> tuple[list[str], list[str]]:
    """Backward-compatible: defs + unique import module tokens (no evidence)."""
    defs: list[str] = []
    imports: list[str] = []
    try:
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defs.append(node.name)
            elif isinstance(node, ast.Import):
                for n in node.names:
                    imports.append(n.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # keep only module token (matches previous behavior)
                    imports.append(node.module)
    except Exception:
        pass
    return defs, sorted(set(imports))


def parse_python_import_edges(text: str) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Evidence-bearing import edges for python.
    Returns:
      - top_defs
      - import_edges: [{kind, spec, lineno}]
    """
    defs: list[str] = []
    edges: list[dict[str, Any]] = []
    try:
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defs.append(node.name)
            elif isinstance(node, ast.Import):
                for n in node.names:
                    spec = n.name
                    if spec:
                        edges.append(
                            {
                                "kind": "import",
                                "spec": spec,
                                "lineno": int(getattr(node, "lineno", 1) or 1),
                            }
                        )
            elif isinstance(node, ast.ImportFrom):
                level = int(getattr(node, "level", 0) or 0)
                mod = getattr(node, "module", None)
                # spec includes leading dots for relative imports
                spec = ("." * level) + (mod or "")
                spec = spec if spec else ("." * level) if level else ""
                if spec:
                    edges.append(
                        {
                            "kind": "from_import",
                            "spec": spec,
                            "lineno": int(getattr(node, "lineno", 1) or 1),
                        }
                    )
    except Exception:
        # caller will set flags
        pass
    # stable ordering
    edges.sort(key=lambda e: (int(e.get("lineno", 1)), str(e.get("kind", "")), str(e.get("spec", ""))))
    return sorted(set(defs)), edges


def parse_best_effort_pythonish_imports(text: str) -> list[str]:
    """Fallback for non-Python, non-JS files (e.g., shell-ish snippets in docs)."""
    found = set()
    for m in PY_IMPORT_RE.finditer(text):
        mod = m.group(1) or m.group(2)
        if mod:
            found.add(mod)
    return sorted(found)


def strip_js_ts_comments(text: str) -> str:
    """
    Remove // and /* */ comments while preserving newlines and string contents.
    Important: we preserve newlines so match.start() -> lineno remains stable.
    """
    out: list[str] = []
    i = 0
    n = len(text)

    in_line = False
    in_block = False
    in_s = False
    in_d = False
    in_t = False
    esc = False

    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line:
            if c == "\n":
                in_line = False
                out.append("\n")
            else:
                out.append(" ")
            i += 1
            continue

        if in_block:
            if c == "*" and nxt == "/":
                in_block = False
                out.append("  ")
                i += 2
            else:
                out.append("\n" if c == "\n" else " ")
                i += 1
            continue

        # strings
        if in_s:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == "'":
                in_s = False
            i += 1
            continue

        if in_d:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_d = False
            i += 1
            continue

        if in_t:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == "`":
                in_t = False
            i += 1
            continue

        # comment starts (only when not in a string)
        if c == "/" and nxt == "/":
            in_line = True
            out.append("  ")
            i += 2
            continue

        if c == "/" and nxt == "*":
            in_block = True
            out.append("  ")
            i += 2
            continue

        # string starts
        if c == "'":
            in_s = True
            out.append(c)
            i += 1
            continue

        if c == '"':
            in_d = True
            out.append(c)
            i += 1
            continue

        if c == "`":
            in_t = True
            out.append(c)
            i += 1
            continue

        out.append(c)
        i += 1

    return "".join(out)


def parse_js_ts_import_edges(text: str) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Evidence-bearing JS/TS import edges.
    Returns:
      - top_defs
      - import_edges: [{kind, spec, lineno}]
    """
    cleaned = strip_js_ts_comments(text)
    edges: list[dict[str, Any]] = []

    # side-effect imports: import "x";
    for m in JS_IMPORT_SIDE_EFFECT_RE.finditer(cleaned):
        spec = m.group("spec")
        if spec:
            edges.append(
                {
                    "kind": "side_effect_import",
                    "spec": spec,
                    "lineno": _lineno_from_index(cleaned, m.start()),
                }
            )

    # import/export ... from "x"
    for m in JS_ANY_IMPORT_EXPORT_FROM_RE.finditer(cleaned):
        kw = (m.group("kw") or "").strip()
        spec = m.group("spec")
        if spec:
            edges.append(
                {
                    "kind": "export_from" if kw == "export" else "static_import",
                    "spec": spec,
                    "lineno": _lineno_from_index(cleaned, m.start()),
                }
            )

    # TS: import Foo = require("x");
    for m in TS_IMPORT_ASSIGN_REQUIRE_RE.finditer(cleaned):
        spec = m.group("spec")
        if spec:
            edges.append(
                {
                    "kind": "import_assign_require",
                    "spec": spec,
                    "lineno": _lineno_from_index(cleaned, m.start()),
                }
            )

    # dynamic import("x")
    for m in JS_DYNAMIC_IMPORT_RE.finditer(cleaned):
        spec = m.group("spec")
        if spec:
            edges.append(
                {
                    "kind": "dynamic_import",
                    "spec": spec,
                    "lineno": _lineno_from_index(cleaned, m.start()),
                }
            )

    # require("x")
    for m in JS_REQUIRE_RE.finditer(cleaned):
        spec = m.group("spec")
        if spec:
            edges.append(
                {
                    "kind": "require",
                    "spec": spec,
                    "lineno": _lineno_from_index(cleaned, m.start()),
                }
            )

    # top defs (best-effort)
    defs: set[str] = set()
    for m in JS_TOP_DEF_RE.finditer(cleaned):
        for g in m.groups():
            if g and isinstance(g, str):
                defs.add(g)

    edges.sort(key=lambda e: (int(e.get("lineno", 1)), str(e.get("kind", "")), str(e.get("spec", ""))))
    return sorted(defs), edges


# -----------------------------
# Path alias parsing (tsconfig/jsconfig)
# -----------------------------
def _strip_jsonc(text: str) -> str:
    return strip_js_ts_comments(text)


def _load_jsonc_file(abs_path: Path) -> dict[str, Any] | None:
    try:
        raw = abs_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    cleaned = _strip_jsonc(raw).strip()
    if not cleaned:
        return None
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _normalize_rel_path(p: str) -> str:
    return p.replace("\\", "/").lstrip("./")


def _resolve_tsconfig_extends_path(current_cfg: Path, extends_value: str) -> Path | None:
    v = (extends_value or "").strip()
    if not v:
        return None

    # Ignore package-based extends (best-effort: only local file resolution here)
    if not v.startswith(("./", "../", "/")):
        return None

    cand = v
    if not Path(cand).suffix:
        cand = cand + ".json"

    if cand.startswith("/"):
        p = Path(cand)
    else:
        p = (current_cfg.parent / cand)

    try:
        p = p.resolve()
    except Exception:
        p = p

    if p.exists() and p.is_file():
        return p
    return None


def _load_tsconfig_with_extends(cfg_path: Path, *, max_depth: int = 6) -> list[Path]:
    chain: list[Path] = []
    visited: set[str] = set()

    cur = cfg_path
    depth = 0
    while cur and depth < max_depth:
        key = str(cur)
        if key in visited:
            break
        visited.add(key)

        obj = _load_jsonc_file(cur)
        if not obj:
            break

        chain.append(cur)

        ext = obj.get("extends")
        if isinstance(ext, str) and ext.strip():
            nxt = _resolve_tsconfig_extends_path(cur, ext)
            if nxt is None:
                break
            cur = nxt
            depth += 1
            continue

        break

    chain = list(reversed(chain))
    return chain


def _merge_tsconfig_compiler_options(chain_paths: list[Path]) -> tuple[str | None, dict[str, list[str]]]:
    merged_base_url: str | None = None
    merged_paths: dict[str, list[str]] = {}

    for p in chain_paths:
        obj = _load_jsonc_file(p)
        if not obj:
            continue
        co = obj.get("compilerOptions")
        if not isinstance(co, dict):
            continue

        bu = co.get("baseUrl")
        if isinstance(bu, str) and bu.strip():
            merged_base_url = bu.strip()

        ps = co.get("paths")
        if isinstance(ps, dict):
            for k, v in ps.items():
                if not isinstance(k, str) or not k.strip():
                    continue
                if isinstance(v, list):
                    vv = [x for x in v if isinstance(x, str) and x.strip()]
                    if vv:
                        merged_paths[k] = vv

    return merged_base_url, merged_paths


def _extract_tsconfig_aliases(repo_dir: str) -> dict[str, Any]:
    repo_root = Path(repo_dir)
    used: list[str] = []
    base_url_repo_rel: str | None = None
    paths_repo_rel: dict[str, list[str]] = {}

    for rel in CONFIG_CANDIDATES:
        cfg = repo_root / rel
        if not cfg.exists() or not cfg.is_file():
            continue

        chain = _load_tsconfig_with_extends(cfg)
        if not chain:
            continue

        for cp in chain:
            try:
                rp = _normalize_rel_path(os.path.relpath(str(cp), repo_dir))
            except Exception:
                rp = _normalize_rel_path(cp.as_posix())
            if rp not in used:
                used.append(rp)

        merged_bu, merged_paths = _merge_tsconfig_compiler_options(chain)

        anchor_dir: Path = cfg.parent
        if isinstance(merged_bu, str) and merged_bu.strip():
            bu_norm = merged_bu.strip()
            if bu_norm == ".":
                anchor_dir = cfg.parent
            elif bu_norm.startswith("/"):
                anchor_dir = repo_root / bu_norm.lstrip("/")
            else:
                anchor_dir = (cfg.parent / bu_norm)

        if base_url_repo_rel is None and isinstance(merged_bu, str) and merged_bu.strip():
            try:
                base_url_repo_rel = _normalize_rel_path(os.path.relpath(str(anchor_dir.resolve()), repo_dir))
            except Exception:
                base_url_repo_rel = _normalize_rel_path(anchor_dir.as_posix())

        for alias_pat, targets in merged_paths.items():
            if not isinstance(alias_pat, str) or not alias_pat.strip():
                continue
            if not isinstance(targets, list):
                continue

            norm_targets: list[str] = []
            for t in targets:
                if not isinstance(t, str) or not t.strip():
                    continue
                t0 = t.strip()

                if t0.startswith("/"):
                    abs_t = (repo_root / t0.lstrip("/"))
                else:
                    abs_t = (anchor_dir / t0)

                try:
                    abs_t = abs_t.resolve()
                except Exception:
                    abs_t = abs_t

                try:
                    rel_t = os.path.relpath(str(abs_t), repo_dir)
                except Exception:
                    rel_t = str(abs_t)

                nt = _normalize_rel_path(rel_t)
                if nt not in norm_targets:
                    norm_targets.append(nt)

            if norm_targets:
                paths_repo_rel.setdefault(alias_pat, [])
                for nt in norm_targets:
                    if nt not in paths_repo_rel[alias_pat]:
                        paths_repo_rel[alias_pat].append(nt)

    alias_prefixes: list[dict[str, Any]] = []
    for alias_pat, targets in paths_repo_rel.items():
        if not isinstance(alias_pat, str):
            continue
        if "*" not in alias_pat:
            continue

        if alias_pat.endswith("/*"):
            alias_prefix = alias_pat[:-1]
        elif alias_pat.endswith("*"):
            alias_prefix = alias_pat[:-1]
        else:
            continue

        t_prefixes: list[str] = []
        for t in targets:
            if not isinstance(t, str) or not t:
                continue
            if t.endswith("/*"):
                tp = t[:-1]
            elif t.endswith("*"):
                tp = t[:-1]
            else:
                tp = t if t.endswith("/") else (t + "/")
            tp = _normalize_rel_path(tp)
            if tp and tp not in t_prefixes:
                t_prefixes.append(tp)

        if alias_prefix and t_prefixes:
            alias_prefixes.append({"alias_prefix": alias_prefix, "targets": t_prefixes})

    alias_prefixes.sort(
        key=lambda x: (
            -len(str(x.get("alias_prefix", ""))),
            str(x.get("alias_prefix", "")),
            json.dumps(x, sort_keys=True),
        )
    )

    used = sorted(used)

    return {
        "configs_used": used,
        "baseUrl": base_url_repo_rel,
        "paths": paths_repo_rel,
        "alias_prefixes": alias_prefixes,
    }


def _is_probably_external_js_spec(spec: str) -> bool:
    s = (spec or "").strip()
    if not s:
        return True
    if s.startswith(("http://", "https://", "data:")):
        return True
    if s.startswith(("node:", "bun:", "deno:")):
        return True
    if s.startswith(("./", "../", "/")):
        return False
    if s.startswith("@/"):
        return False
    return True


def _candidate_paths_for_module_noext(module_noext: str) -> list[str]:
    out: list[str] = []
    p = module_noext.rstrip("/")
    if not p:
        return out

    for ext in JS_RESOLVE_EXTS:
        out.append(p + ext)

    for ext in JS_RESOLVE_EXTS:
        out.append(p + "/index" + ext)

    return out


def _resolve_js_ts_import_to_repo_path(
        *,
        spec: str,
        from_file_repo_path: str,
        repo_dir: str,
        alias_info: dict[str, Any],
) -> str | None:
    s = (spec or "").strip()
    if not s:
        return None

    repo_root = Path(repo_dir)
    from_dir = Path(from_file_repo_path).parent.as_posix().rstrip("/")

    # 1) relative
    if s.startswith(("./", "../")):
        base = f"{from_dir}/{s}" if from_dir else s
        module_noext = Path(base).as_posix()
        module_noext = os.path.normpath(module_noext).replace("\\", "/")
        module_noext = module_noext.lstrip("./")

        if Path(s).suffix:
            cand = module_noext
            return cand if (repo_root / cand).exists() else None

        for cand in _candidate_paths_for_module_noext(module_noext):
            if (repo_root / cand).exists():
                return cand
        return None

    # 2) rooted from repo root
    if s.startswith("/"):
        module_noext = s.lstrip("/")
        if Path(module_noext).suffix:
            return _normalize_rel_path(module_noext) if (repo_root / module_noext).exists() else None
        for cand in _candidate_paths_for_module_noext(module_noext):
            if (repo_root / cand).exists():
                return _normalize_rel_path(cand)
        return None

    # 3) alias prefixes (from tsconfig/jsconfig)
    alias_prefixes = alias_info.get("alias_prefixes", [])
    if isinstance(alias_prefixes, list):
        for rule in alias_prefixes:
            if not isinstance(rule, dict):
                continue
            ap = rule.get("alias_prefix")
            targets = rule.get("targets")
            if not isinstance(ap, str) or not ap:
                continue
            if not isinstance(targets, list) or not targets:
                continue
            if s.startswith(ap):
                tail = s[len(ap) :].lstrip("/")
                for tp in targets:
                    if not isinstance(tp, str) or not tp:
                        continue
                    module_noext = (tp + tail).replace("\\", "/")
                    module_noext = os.path.normpath(module_noext).replace("\\", "/")
                    module_noext = _normalize_rel_path(module_noext)

                    if Path(s).suffix:
                        if (repo_root / module_noext).exists():
                            return module_noext
                        continue

                    for cand in _candidate_paths_for_module_noext(module_noext):
                        if (repo_root / cand).exists():
                            return _normalize_rel_path(cand)

    # 4) common Next alias fallback: "@/..." -> "frontend/..." (only if it exists)
    if s.startswith("@/"):
        tail = s[2:].lstrip("/")
        module_noext = _normalize_rel_path(f"frontend/{tail}")
        if Path(module_noext).suffix:
            return module_noext if (repo_root / module_noext).exists() else None
        for cand in _candidate_paths_for_module_noext(module_noext):
            if (repo_root / cand).exists():
                return _normalize_rel_path(cand)

    # 5) baseUrl fallback for bare-ish internal paths (best-effort)
    base_url = alias_info.get("baseUrl")
    if isinstance(base_url, str) and base_url.strip():
        module_noext = _normalize_rel_path(f"{base_url.rstrip('/')}/{s.lstrip('/')}")
        if Path(module_noext).suffix:
            return module_noext if (repo_root / module_noext).exists() else None
        for cand in _candidate_paths_for_module_noext(module_noext):
            if (repo_root / cand).exists():
                return _normalize_rel_path(cand)

    return None


def _candidate_paths_for_py_module(module_path: str) -> list[str]:
    """
    module_path: e.g. "snapshotter/utils" (no .py)
    """
    out: list[str] = []
    p = module_path.strip("/").replace("\\", "/")
    if not p:
        return out
    out.append(p + ".py")
    out.append(p + "/__init__.py")
    return out


def _resolve_python_import_to_repo_path(
        *,
        spec: str,
        from_file_repo_path: str,
        repo_dir: str,
) -> str | None:
    """
    Best-effort python import -> repo path resolver.
    Supports:
      - absolute module paths: "snapshotter.utils" -> snapshotter/utils.py or snapshotter/utils/__init__.py
      - relative module specs with leading dots: ".utils", "..core", "."
    """
    s = (spec or "").strip()
    if not s:
        return None

    repo_root = Path(repo_dir)
    from_dir = Path(from_file_repo_path).parent.as_posix().rstrip("/")

    # count leading dots for relative
    m = re.match(r"^(\.+)(.*)$", s)
    if m:
        dots = m.group(1) or ""
        rest = (m.group(2) or "").lstrip(".")
        level = len(dots)

        # go up 'level' packages from current directory
        base = from_dir
        for _ in range(max(level, 1)):  # ".": level=1 means same dir's package context
            if "/" in base:
                base = base.rsplit("/", 1)[0]
            else:
                base = ""
                break

        if rest:
            module_path = f"{base}/{rest.replace('.', '/')}" if base else rest.replace(".", "/")
        else:
            module_path = base  # "from . import X" -> base package

        module_path = _normalize_rel_path(os.path.normpath(module_path).replace("\\", "/"))
        for cand in _candidate_paths_for_py_module(module_path):
            if (repo_root / cand).exists():
                return _normalize_rel_path(cand)
        return None

    # absolute
    module_path = s.replace(".", "/")
    for cand in _candidate_paths_for_py_module(module_path):
        if (repo_root / cand).exists():
            return _normalize_rel_path(cand)
    return None


def _extract_env_var_edges(text: str, *, language: str) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    if not text:
        return edges

    if language == "python":
        # ast-based would be cleaner, but regex is deterministic + cheap
        for rx in PY_ENV_PATTERNS:
            for m in rx.finditer(text):
                key = m.group(1)
                if key:
                    edges.append(
                        {
                            "kind": "env_var",
                            "key": key,
                            "lineno": _lineno_from_index(text, m.start()),
                        }
                    )
    elif language in ("typescript", "javascript"):
        cleaned = strip_js_ts_comments(text)
        for rx in JS_ENV_PATTERNS:
            for m in rx.finditer(cleaned):
                key = m.group(1)
                if key:
                    edges.append(
                        {
                            "kind": "env_var",
                            "key": key,
                            "lineno": _lineno_from_index(cleaned, m.start()),
                        }
                    )

    edges.sort(key=lambda e: (int(e.get("lineno", 1)), str(e.get("key", ""))))
    # de-dupe exact triples
    dedup: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for e in edges:
        t = (int(e.get("lineno", 1)), str(e.get("kind", "")), str(e.get("key", "")))
        if t not in seen:
            seen.add(t)
            dedup.append(e)
    return dedup


def _entrypoint_signals_for_file(rel_path: str, language: str, text: str) -> list[dict[str, Any]]:
    """
    Very lightweight heuristics that help architects quickly find "where it starts",
    without turning pass1 into pass2.
    """
    signals: list[dict[str, Any]] = []
    p = rel_path.replace("\\", "/")

    # path-based Next.js conventions
    if p.startswith("frontend/") and ("/app/" in p or "/pages/" in p):
        if p.endswith(("/page.tsx", "/page.jsx", "/page.ts", "/page.js")):
            signals.append({"kind": "entrypoint_hint", "path": p, "why": "next_page"})
        if p.endswith(("/layout.tsx", "/layout.jsx")):
            signals.append({"kind": "entrypoint_hint", "path": p, "why": "next_layout"})
        if p.endswith(("/route.ts", "/route.js")):
            signals.append({"kind": "entrypoint_hint", "path": p, "why": "next_route"})

    # common backend entry filenames
    if p.endswith(("/main.py", "/app.py", "/server.py")) and language == "python":
        signals.append({"kind": "entrypoint_hint", "path": p, "why": "common_backend_filename"})

    # content-based hints
    if language == "python" and "__name__" in text and "if __name__" in text and "__main__" in text:
        signals.append({"kind": "entrypoint_hint", "path": p, "why": "python___main__"})
    if language == "python" and "FastAPI(" in text:
        signals.append({"kind": "entrypoint_hint", "path": p, "why": "fastapi_app"})
    if language == "python" and "uvicorn.run" in text:
        signals.append({"kind": "entrypoint_hint", "path": p, "why": "uvicorn_run"})
    if language in ("typescript", "javascript") and ("createRoot(" in text or "ReactDOM.render" in text):
        signals.append({"kind": "entrypoint_hint", "path": p, "why": "react_dom_mount"})

    return signals


def _package_json_signals(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    scripts = obj.get("scripts")
    sig: dict[str, Any] = {}
    if isinstance(scripts, dict):
        # keep deterministic order by key
        sig["scripts"] = {k: scripts[k] for k in sorted(scripts.keys()) if isinstance(k, str)}
    for k in ("main", "module", "type", "bin", "name", "workspaces"):
        v = obj.get(k)
        if v is not None:
            sig[k] = v
    for dep_key in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
        v = obj.get(dep_key)
        if isinstance(v, dict):
            sig[dep_key] = sorted([str(x) for x in v.keys()])

    return sig or None


def build_repo_index(repo_dir: str, job: Job) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    total_bytes = 0
    files_scanned = 0
    files_included = 0

    deny_dirs = set(job.filters.deny_dirs)
    deny_file_regex = [re.compile(p) for p in job.filters.deny_file_regex]
    allow_all = "*" in job.filters.allow_exts
    allow_exts = set([e.lower().lstrip(".") for e in job.filters.allow_exts if e != "*"])

    # v0.1: binary is skipped unless explicitly allowed
    allow_binary = bool(getattr(job.filters, "allow_binary", False))

    max_files_reached = False

    # Option B: collect alias info once (deterministic)
    path_aliases = _extract_tsconfig_aliases(repo_dir)

    # repo-level signals aggregated during scan
    repo_signals: dict[str, Any] = {
        "entrypoints": [],  # list of {kind,path,why}
        "package_json": {},  # rel_path -> extracted signals
        "env_vars": [],  # list of {key, locations:[{path, lineno}]}
    }
    _env_locations: dict[str, list[dict[str, Any]]] = {}
    _entrypoint_set: set[tuple[str, str]] = set()  # (path, why)

    def record_skip(rel_path: str, reason: str, size: int | None = None):
        skipped.append({"path": rel_path, "reason": reason, "bytes": size or 0})

    for root, dirs, filenames in os.walk(repo_dir):
        dirs[:] = sorted([d for d in dirs if d not in deny_dirs])
        filenames = sorted(filenames)

        if max_files_reached:
            break

        for fn in filenames:
            if files_scanned >= job.limits.max_files:
                if not max_files_reached:
                    record_skip("*", "max_files_reached")
                    max_files_reached = True
                break

            abs_path = os.path.join(root, fn)
            rel_path = os.path.relpath(abs_path, repo_dir).replace("\\", "/")
            files_scanned += 1

            if any(rx.search(rel_path) for rx in deny_file_regex):
                record_skip(rel_path, "deny_file_regex")
                continue

            ext = Path(rel_path).suffix.lower().lstrip(".")
            if not allow_all and ext not in allow_exts:
                record_skip(rel_path, "ext_not_allowed")
                continue

            try:
                size = os.stat(abs_path).st_size
            except OSError:
                record_skip(rel_path, "stat_failed")
                continue

            if size > job.limits.max_file_bytes:
                record_skip(rel_path, "max_file_bytes_exceeded", size)
                continue

            if total_bytes + size > job.limits.max_total_bytes:
                record_skip(rel_path, "max_total_bytes_exceeded", size)
                continue

            try:
                raw = Path(abs_path).read_bytes()
            except Exception:
                record_skip(rel_path, "read_failed", size)
                continue

            if (not allow_binary) and is_probably_binary(raw):
                record_skip(rel_path, "binary_file", size)
                continue

            sha = sha256_bytes(raw)
            language = infer_language(rel_path)

            imports_raw: list[str] = []
            imports_resolved_internal: list[str] = []
            imports_external: list[str] = []
            import_edges: list[dict[str, Any]] = []

            top_defs: list[str] = []
            flags: list[str] = []

            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                record_skip(rel_path, "text_decode_failed", size)
                continue

            # env var edges (small + deterministic)
            resource_edges = _extract_env_var_edges(text, language=language)
            for e in resource_edges:
                key = str(e.get("key", "")).strip()
                if not key:
                    continue
                _env_locations.setdefault(key, [])
                _env_locations[key].append({"path": rel_path, "lineno": int(e.get("lineno", 1) or 1)})

            # entrypoint hints (small + deterministic)
            for s in _entrypoint_signals_for_file(rel_path, language, text):
                p = str(s.get("path", "")).strip()
                why = str(s.get("why", "")).strip()
                if p and why and (p, why) not in _entrypoint_set:
                    _entrypoint_set.add((p, why))

            if language == "python":
                try:
                    top_defs, import_edges = parse_python_import_edges(text)
                    # backward compat
                    imports_raw = sorted({str(e.get("spec")) for e in import_edges if e.get("spec")})
                except Exception:
                    flags.append("python_parse_failed")
                    top_defs, imports_raw = parse_python_defs_and_imports(text)
                    import_edges = []

                internal_set: set[str] = set()
                external_set: set[str] = set()

                for e in import_edges:
                    spec = str(e.get("spec", "")).strip()
                    if not spec:
                        continue
                    resolved = _resolve_python_import_to_repo_path(
                        spec=spec, from_file_repo_path=rel_path, repo_dir=repo_dir
                    )
                    if resolved:
                        e["resolved_path"] = resolved
                        e["is_external"] = False
                        internal_set.add(resolved)
                    else:
                        e["resolved_path"] = None
                        # relative specs are internal-candidate; mark unresolved
                        if spec.startswith("."):
                            flags.append("import_unresolved")
                            e["is_external"] = False
                        else:
                            e["is_external"] = True
                        external_set.add(spec)

                imports_resolved_internal = sorted(internal_set)
                imports_external = sorted(external_set)

            elif language in ("typescript", "javascript"):
                try:
                    top_defs, import_edges = parse_js_ts_import_edges(text)
                    if not top_defs:
                        flags.append("top_level_defs_best_effort_empty")
                    imports_raw = sorted({str(e.get("spec")) for e in import_edges if e.get("spec")})
                except Exception:
                    flags.append("js_ts_parse_failed")
                    imports_raw = []
                    import_edges = []
                    top_defs = []

                internal_set: set[str] = set()
                external_set: set[str] = set()

                for e in import_edges:
                    spec = str(e.get("spec", "")).strip()
                    if not spec:
                        continue
                    resolved = _resolve_js_ts_import_to_repo_path(
                        spec=spec,
                        from_file_repo_path=rel_path,
                        repo_dir=repo_dir,
                        alias_info=path_aliases,
                    )
                    if resolved:
                        e["resolved_path"] = resolved
                        e["is_external"] = False
                        internal_set.add(resolved)
                    else:
                        e["resolved_path"] = None
                        if not _is_probably_external_js_spec(spec):
                            flags.append("import_unresolved")
                            e["is_external"] = False
                        else:
                            e["is_external"] = True
                        external_set.add(spec)

                imports_resolved_internal = sorted(internal_set)
                imports_external = sorted(external_set)

            else:
                imports_raw = parse_best_effort_pythonish_imports(text)
                flags.append("top_level_defs_unknown")
                import_edges = [{"kind": "best_effort_import", "spec": s, "lineno": 1} for s in imports_raw]
                # no reliable resolution here; treat as external tokens
                imports_resolved_internal = []
                imports_external = sorted(set(imports_raw))

            # package.json signals (repo-level)
            if rel_path.endswith("package.json") and language == "json":
                sig = _package_json_signals(text)
                if sig:
                    repo_signals["package_json"][rel_path] = sig

            total_bytes += size
            files_included += 1

            # canonical deps block: THIS is what Pass2 should consume to avoid "dependency meaning drift"
            import_edges_sorted = sorted(
                import_edges,
                key=lambda e: (
                    int(e.get("lineno", 1) or 1),
                    str(e.get("kind", "")),
                    str(e.get("spec", "")),
                    str(e.get("resolved_path") or ""),
                    str(e.get("is_external", "")),
                ),
            )

            deps_block = {
                "import_edges": import_edges_sorted,
                "internal": imports_resolved_internal,
                "external": imports_external,
            }

            files.append(
                {
                    "path": rel_path,
                    "bytes": size,
                    "sha256": sha,
                    "language": language,
                    # Backward compatibility: keep the old fields
                    "imports": imports_raw,
                    "imports_raw": imports_raw,
                    "imports_resolved_internal": imports_resolved_internal,
                    "imports_external": imports_external,
                    "top_level_defs": top_defs,
                    "flags": flags,
                    # New, canonical evidence-bearing facts
                    "import_edges": import_edges_sorted,
                    "deps": deps_block,
                    "resource_edges": resource_edges,  # currently env-var facts only
                }
            )

        if max_files_reached:
            break

    files.sort(key=lambda x: x["path"])
    skipped.sort(key=lambda x: x["path"])

    read_plan_suggestions = suggest_files_to_read(files, max_files=120)

    # finalize repo-level signals deterministically
    repo_signals["entrypoints"] = [
        {"kind": "entrypoint_hint", "path": p, "why": why}
        for (p, why) in sorted(_entrypoint_set, key=lambda t: (t[0], t[1]))
    ]

    env_vars_out: list[dict[str, Any]] = []
    for k in sorted(_env_locations.keys()):
        locs = _env_locations[k]
        # stable sort + de-dupe
        locs_sorted = sorted(locs, key=lambda x: (str(x.get("path", "")), int(x.get("lineno", 1) or 1)))
        dedup: list[dict[str, Any]] = []
        seen = set()
        for l in locs_sorted:
            t = (str(l.get("path", "")), int(l.get("lineno", 1) or 1))
            if t not in seen:
                seen.add(t)
                dedup.append({"path": t[0], "lineno": t[1]})
        env_vars_out.append({"key": k, "locations": dedup})
    repo_signals["env_vars"] = env_vars_out

    # Job contract: payload + derived fields live under repo_index.job
    job_block = job.model_dump()
    job_block["resolved_commit"] = "unknown"

    return {
        "generated_at": utc_ts(),
        "job": job_block,
        "counts": {
            "files_scanned": files_scanned,
            "files_included": files_included,
            "files_skipped": len(skipped),
            "bytes_included": total_bytes,
        },
        "path_aliases": path_aliases,  # unchanged
        "signals": repo_signals,  # NEW: repo-level grounding signals
        "files": files,
        "skipped_files": skipped,
        "read_plan_suggestions": read_plan_suggestions,
    }


def write_json(path: str | Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")