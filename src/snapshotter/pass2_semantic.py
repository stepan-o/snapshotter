# snapshotter/pass2_semantic.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any


class Pass2SemanticError(RuntimeError):
    pass


class Pass2SemanticLLMOutputError(Pass2SemanticError):
    """
    Raised when the model returns text that cannot be parsed into the required JSON object.

    Carries the raw (and optionally repaired) model output so the caller can persist it for inspection.
    """

    def __init__(self, message: str, *, raw_text: str, repaired_text: str | None = None):
        super().__init__(message)
        self.raw_text = raw_text
        self.repaired_text = repaired_text


@dataclass(frozen=True)
class SemanticCaps:
    onboarding_enabled: bool
    model: str
    max_output_tokens: int

    # input caps (prevents accidental huge prompts)
    max_arch_input_chars: int
    max_arch_files: int
    max_arch_chars_per_file: int

    # supporting pack caps (gaps + onboarding)
    max_support_files: int
    max_support_chars: int
    max_support_chars_per_file: int

    # pack graph expansion bounds
    pack_dep_hops: int
    pack_max_dep_edges_per_file: int


def _bool_from_env(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    return v not in ("0", "false", "False", "no", "NO", "off", "OFF")


def _int_from_env(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _semantic_caps_from_env() -> SemanticCaps:
    onboarding_enabled = _bool_from_env("SNAPSHOTTER_PASS2_ONBOARDING", True)
    model = os.environ.get("SNAPSHOTTER_LLM_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
    max_output_tokens = _int_from_env("SNAPSHOTTER_LLM_MAX_OUTPUT_TOKENS", 2000)

    max_arch_input_chars = _int_from_env("SNAPSHOTTER_PASS2_MAX_ARCH_INPUT_CHARS", 240_000)
    max_arch_files = _int_from_env("SNAPSHOTTER_PASS2_MAX_ARCH_FILES", 120)
    max_arch_chars_per_file = _int_from_env("SNAPSHOTTER_PASS2_MAX_ARCH_CHARS_PER_FILE", 9000)

    max_support_files = _int_from_env("SNAPSHOTTER_PASS2_MAX_SUPPORT_FILES", 28)
    max_support_chars = _int_from_env("SNAPSHOTTER_PASS2_MAX_SUPPORT_CHARS", 120_000)
    max_support_chars_per_file = _int_from_env("SNAPSHOTTER_PASS2_MAX_SUPPORT_CHARS_PER_FILE", 9000)

    pack_dep_hops = _int_from_env("SNAPSHOTTER_PASS2_PACK_DEP_HOPS", 1)
    pack_max_dep_edges_per_file = _int_from_env("SNAPSHOTTER_PASS2_PACK_MAX_DEP_EDGES_PER_FILE", 12)

    return SemanticCaps(
        onboarding_enabled=onboarding_enabled,
        model=model,
        max_output_tokens=max_output_tokens,
        max_arch_input_chars=max_arch_input_chars,
        max_arch_files=max_arch_files,
        max_arch_chars_per_file=max_arch_chars_per_file,
        max_support_files=max_support_files,
        max_support_chars=max_support_chars,
        max_support_chars_per_file=max_support_chars_per_file,
        pack_dep_hops=pack_dep_hops,
        pack_max_dep_edges_per_file=pack_max_dep_edges_per_file,
    )


# -------------------------------------------------------------------
# OpenAI Responses API (JSON-only)
# -------------------------------------------------------------------


def _extract_text_from_responses_obj(resp: Any) -> str:
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t

    out = getattr(resp, "output", None)
    if not isinstance(out, list):
        return ""

    chunks: list[str] = []
    for item in out:
        if isinstance(item, dict):
            item_content = item.get("content")
        else:
            item_content = getattr(item, "content", None)

        if item_content and isinstance(item_content, list):
            for c in item_content:
                if isinstance(c, dict):
                    c_text = c.get("text")
                else:
                    c_text = getattr(c, "text", None)

                if isinstance(c_text, str) and c_text:
                    chunks.append(c_text)
                elif isinstance(c_text, dict):
                    chunks.append(json.dumps(c_text, ensure_ascii=False))

    return "".join(chunks)


def _looks_truncated(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    if not s.endswith("}"):
        return True

    in_str = False
    esc = False
    bal = 0
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            bal += 1
        elif ch == "}":
            bal -= 1
    return bal != 0


def _extract_first_json_object_span(text: str) -> str | None:
    s = text or ""
    start = None

    in_str = False
    esc = False
    bal = 0

    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            if start is None:
                start = i
            bal += 1
        elif ch == "}":
            if start is not None:
                bal -= 1
                if bal == 0:
                    return s[start : i + 1]
    return None


def _try_parse_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise Pass2SemanticError("OpenAI response was empty; expected a JSON object.")

    # STRICT: expect a single JSON object. We only allow extracting the first object span if the
    # SDK returns extra wrapping text.
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise Pass2SemanticError("OpenAI response parsed but is not a JSON object.")
        return obj
    except Exception:
        candidate = _extract_first_json_object_span(text)
        if not candidate:
            raise Pass2SemanticError(f"OpenAI response was not valid JSON. First 400 chars:\n{text[:400]}")
        obj = json.loads(candidate)
        if not isinstance(obj, dict):
            raise Pass2SemanticError("Salvaged JSON parsed but is not a JSON object.")
        return obj


def _build_json_repair_prompt(bad_text: str) -> str:
    return (
            "You are a JSON repair tool.\n"
            "You will be given text that is intended to be a single JSON object, but may contain minor JSON syntax errors.\n"
            "Your task: output ONLY a valid JSON object that preserves the SAME structure and content as closely as possible.\n"
            "Rules:\n"
            "- Output JSON only. No markdown, no commentary.\n"
            "- Do not change top-level keys or semantics.\n"
            "- Only fix syntax (missing commas, quotes, escaping, trailing commas, etc.).\n\n"
            "INPUT (verbatim):\n"
            + (bad_text or "")
    )


def _openai_call_json(*, prompt: str, model: str, max_output_tokens: int, system: str) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise Pass2SemanticError("OPENAI_API_KEY is not set; cannot run pass2 semantic generation.")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise Pass2SemanticError(f"openai python SDK not available or too old for Responses API: {e}") from e

    client = OpenAI(api_key=api_key)

    input_payload = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    def _responses_create_text(inp: Any) -> str:
        # Keep a small compatibility shim for SDK arg names; this supports artifact creation.
        last_err: Exception | None = None
        for attempt_kwargs in (
                {
                    "model": model,
                    "input": inp,
                    "max_output_tokens": max_output_tokens,
                    "text": {"format": {"type": "json_object"}},
                    "temperature": 0,
                },
                {
                    "model": model,
                    "input": inp,
                    "max_output_tokens": max_output_tokens,
                    "text": {"format": {"type": "json_object"}},
                },
                {
                    "model": model,
                    "input": inp,
                    "max_tokens": max_output_tokens,
                    "text": {"format": {"type": "json_object"}},
                },
        ):
            try:
                resp = client.responses.create(**attempt_kwargs)
                return _extract_text_from_responses_obj(resp)
            except TypeError as e:
                last_err = e
                continue
        raise Pass2SemanticError(f"OpenAI Responses API call failed due to incompatible SDK args: {last_err}")

    try:
        text = _responses_create_text(input_payload)
    except Exception as e:
        raise Pass2SemanticError(f"OpenAI Responses API call failed: {e}") from e

    try:
        return _try_parse_json(text)
    except Exception as parse_err:
        if _looks_truncated(text):
            raise Pass2SemanticLLMOutputError(
                "OpenAI returned truncated/incomplete JSON (likely hit max output tokens). "
                "Increase SNAPSHOTTER_LLM_MAX_OUTPUT_TOKENS and retry.\n"
                f"Parse error: {parse_err}\n"
                f"First 400 chars:\n{text[:400]}",
                raw_text=text,
            ) from parse_err

        # Repair is allowed because it increases odds we successfully produce the required artifacts.
        repair_prompt = _build_json_repair_prompt(text)
        repair_input = [
            {"role": "system", "content": "You are a JSON repair tool. Output JSON only."},
            {"role": "user", "content": repair_prompt},
        ]

        repaired_text: str | None = None
        try:
            repaired_text = _responses_create_text(repair_input)
        except Exception as e:
            raise Pass2SemanticLLMOutputError(
                "OpenAI JSON repair call failed.\n"
                f"First 400 chars of original:\n{text[:400]}",
                raw_text=text,
            ) from e

        try:
            return _try_parse_json(repaired_text)
        except Exception as e:
            raise Pass2SemanticLLMOutputError(
                "Failed to parse OpenAI JSON response (including repair attempt).\n"
                f"Original first 400 chars:\n{text[:400]}",
                raw_text=text,
                repaired_text=repaired_text,
            ) from e


# -------------------------------------------------------------------
# Pass1 -> deterministic facts (STRICT CONTRACT)
# -------------------------------------------------------------------


def _require_dict(v: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(v, dict):
        raise Pass2SemanticError(f"Pass1 contract violation: expected '{name}' to be a dict.")
    return v


def _require_list(v: Any, *, name: str) -> list[Any]:
    if not isinstance(v, list):
        raise Pass2SemanticError(f"Pass1 contract violation: expected '{name}' to be a list.")
    return v


def _require_list_str(v: Any, *, name: str) -> list[str]:
    lst = _require_list(v, name=name)
    out: list[str] = []
    for it in lst:
        if not isinstance(it, str) or not it.strip():
            raise Pass2SemanticError(f"Pass1 contract violation: '{name}' must contain non-empty strings.")
        out.append(it.strip())
    return out


def _repo_paths_set(repo_index: dict[str, Any]) -> set[str]:
    files = _require_list(repo_index.get("files"), name="repo_index.files")
    s: set[str] = set()
    for f in files:
        if not isinstance(f, dict):
            raise Pass2SemanticError("Pass1 contract violation: repo_index.files must contain dict entries.")
        p = f.get("path")
        if not isinstance(p, str) or not p:
            raise Pass2SemanticError("Pass1 contract violation: each file must have a non-empty 'path'.")
        s.add(p)
    return s


def _language_by_path_from_repo_index(repo_index: dict[str, Any]) -> dict[str, str]:
    files = _require_list(repo_index.get("files"), name="repo_index.files")
    out: dict[str, str] = {}
    for f in files:
        if not isinstance(f, dict):
            raise Pass2SemanticError("Pass1 contract violation: repo_index.files must contain dict entries.")
        p = f.get("path")
        if not isinstance(p, str) or not p:
            raise Pass2SemanticError("Pass1 contract violation: each file must have a non-empty 'path'.")
        lang = f.get("language")
        if lang is None:
            continue
        if not isinstance(lang, str) or not lang.strip():
            raise Pass2SemanticError(f"Pass1 contract violation: language must be a string when present ({p}).")
        out[p] = lang.strip()
    return out


def _signals_from_repo_index(repo_index: dict[str, Any]) -> dict[str, Any]:
    sig = _require_dict(repo_index.get("signals"), name="repo_index.signals")
    # we keep this as a loose dict, but do not accept wrong types where we depend on them.
    return sig


def _read_plan_candidates(repo_index: dict[str, Any]) -> list[str]:
    """
    STRICT contract:
      repo_index["read_plan_suggestions"] is a dict produced by read_plan.suggest_files_to_read
      and contains:
        - candidates: list[{"path": str, ...}]
    """
    rps = _require_dict(repo_index.get("read_plan_suggestions"), name="repo_index.read_plan_suggestions")
    cands = _require_list(rps.get("candidates"), name="repo_index.read_plan_suggestions.candidates")

    out: list[str] = []
    for it in cands:
        if not isinstance(it, dict):
            raise Pass2SemanticError("Pass1 contract violation: candidates must contain dict entries.")
        p = it.get("path")
        if not isinstance(p, str) or not p.strip():
            raise Pass2SemanticError("Pass1 contract violation: candidate.path must be a non-empty string.")
        out.append(p.strip())

    # deterministic unique preserving order
    seen: set[str] = set()
    dedup: list[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return dedup


def _read_plan_closure_seeds(repo_index: dict[str, Any]) -> list[str]:
    """
    STRICT contract:
      repo_index["read_plan_closure_seeds"] is a list[str] emitted by Pass1.
    """
    seeds = _require_list_str(repo_index.get("read_plan_closure_seeds"), name="repo_index.read_plan_closure_seeds")
    seen: set[str] = set()
    out: list[str] = []
    for p in seeds:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _extract_pass1_deps(repo_index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    STRICT extraction based on Pass1 schema. No compat fallbacks.

    Required per file:
      - path: str
      - deps: dict
        - import_edges: list[dict]
          Each edge must include:
            - spec: str
            - is_external: bool (or coercible)
            - if is_external == False: resolved_path: str

    Output per file:
      {
        "resolved_internal": set[str],
        "external_specs": set[str],
        "import_edges": list[dict],
        "flags": set[str],
        "language": str|None,
        "top_level_defs": list[str],
      }
    """
    files = _require_list(repo_index.get("files"), name="repo_index.files")
    out: dict[str, dict[str, Any]] = {}

    for f in files:
        if not isinstance(f, dict):
            raise Pass2SemanticError("Pass1 contract violation: repo_index.files must contain dict entries.")
        path = f.get("path")
        if not isinstance(path, str) or not path:
            raise Pass2SemanticError("Pass1 contract violation: each file must have a non-empty 'path'.")

        deps = _require_dict(f.get("deps"), name=f"file.deps ({path})")
        edges_any = deps.get("import_edges")
        edges_raw = _require_list(edges_any, name=f"file.deps.import_edges ({path})")
        import_edges: list[dict[str, Any]] = []
        for e in edges_raw:
            if not isinstance(e, dict):
                raise Pass2SemanticError(f"Pass1 contract violation: import_edges must contain dict entries ({path}).")
            spec = e.get("spec")
            if not isinstance(spec, str) or not spec.strip():
                raise Pass2SemanticError(f"Pass1 contract violation: import edge missing spec ({path}).")
            import_edges.append(dict(e))

        resolved_internal: set[str] = set()
        external_specs: set[str] = set()

        for e in import_edges:
            is_external = bool(e.get("is_external", False))
            spec = str(e.get("spec", "")).strip()
            if is_external:
                external_specs.add(spec)
                continue
            rp = e.get("resolved_path")
            if not isinstance(rp, str) or not rp.strip():
                raise Pass2SemanticError(
                    f"Pass1 contract violation: internal import edge missing resolved_path ({path})."
                )
            resolved_internal.add(rp.strip())

        flags_set: set[str] = set()
        fl = f.get("flags")
        if fl is not None:
            if not isinstance(fl, list):
                raise Pass2SemanticError(f"Pass1 contract violation: flags must be a list when present ({path}).")
            for x in fl:
                if not isinstance(x, str) or not x.strip():
                    raise Pass2SemanticError(f"Pass1 contract violation: flags must be non-empty strings ({path}).")
                flags_set.add(x.strip())

        lang = f.get("language")
        if lang is not None and (not isinstance(lang, str) or not lang.strip()):
            raise Pass2SemanticError(f"Pass1 contract violation: language must be a string when present ({path}).")
        lang_str = lang.strip() if isinstance(lang, str) else None

        tdefs = f.get("top_level_defs")
        top_defs: list[str] = []
        if tdefs is not None:
            if not isinstance(tdefs, list):
                raise Pass2SemanticError(f"Pass1 contract violation: top_level_defs must be a list when present ({path}).")
            for x in tdefs:
                if not isinstance(x, str) or not x.strip():
                    raise Pass2SemanticError(f"Pass1 contract violation: top_level_defs must contain strings ({path}).")
                top_defs.append(x.strip())

        out[path] = {
            "resolved_internal": resolved_internal,
            "external_specs": external_specs,
            "import_edges": import_edges,
            "flags": flags_set,
            "language": lang_str,
            "top_level_defs": top_defs,
        }

    return out


def _known_present_route_hints(repo_paths: set[str]) -> dict[str, Any]:
    # keep: this is deterministic and prevents false "missing from snapshot" claims
    hints: dict[str, Any] = {}

    login_files = []
    for p in (
            "frontend/app/login/page.tsx",
            "frontend/app/login/LoginPageClient.tsx",
            "frontend/app/login/loginpageclient.tsx",
    ):
        if p in repo_paths:
            login_files.append(p)
    if login_files:
        hints["login_route_present"] = True
        hints["login_route_files"] = login_files

    cs_files = []
    for p in (
            "frontend/app/case-studies/page.tsx",
            "frontend/app/case-studies/layout.tsx",
    ):
        if p in repo_paths:
            cs_files.append(p)
    if cs_files:
        hints["case_studies_route_present"] = True
        hints["case_studies_route_files"] = cs_files

    return hints


# -------------------------------------------------------------------
# Pack selection (deterministic; bounded; driven by pass1 facts)
# -------------------------------------------------------------------


def _truncate_with_tail(text: str, max_chars: int) -> str:
    if not isinstance(text, str):
        return ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = int(max_chars * 0.75)
    tail = max_chars - head
    if tail < 200:
        return text[:max_chars]
    return text[:head] + "\n/* …TRUNCATED… */\n" + text[-tail:]


def _compute_available_dep_graph(
        *,
        available_paths: set[str],
        deps_by_file: dict[str, dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    out_edges: dict[str, set[str]] = {p: set() for p in available_paths}
    in_edges: dict[str, set[str]] = {p: set() for p in available_paths}

    for p in available_paths:
        info = deps_by_file[p]  # STRICT: deps_by_file must cover all pass1 files; available is subset
        for dep in info["resolved_internal"]:
            if dep in available_paths:
                out_edges[p].add(dep)
                in_edges[dep].add(p)

    return out_edges, in_edges


def _expand_seeds_by_deps(
        *,
        seeds: list[str],
        out_edges: dict[str, set[str]],
        hops: int,
        max_added_per_file: int,
) -> list[str]:
    if hops <= 0:
        return seeds

    seen: set[str] = set()
    order: list[str] = []

    def _add(p: str) -> None:
        if p not in seen:
            seen.add(p)
            order.append(p)

    for s in seeds:
        _add(s)

    frontier = list(seeds)
    for _ in range(hops):
        nxt: list[str] = []
        for p in frontier:
            deps = sorted(list(out_edges.get(p, set())))
            if max_added_per_file > 0:
                deps = deps[:max_added_per_file]
            for d in deps:
                if d not in seen:
                    _add(d)
                    nxt.append(d)
        frontier = nxt
        if not frontier:
            break

    return order


def _entrypoints_from_signals(repo_index: dict[str, Any], *, available_paths: set[str]) -> list[str]:
    sig = _signals_from_repo_index(repo_index)
    eps = sig.get("entrypoints")
    if eps is None:
        return []
    if not isinstance(eps, list):
        raise Pass2SemanticError("Pass1 contract violation: signals.entrypoints must be a list when present.")
    out: list[str] = []
    for it in eps:
        if not isinstance(it, dict):
            raise Pass2SemanticError("Pass1 contract violation: signals.entrypoints must contain dict entries.")
        p = it.get("path")
        if not isinstance(p, str) or not p.strip():
            raise Pass2SemanticError("Pass1 contract violation: entrypoint.path must be a non-empty string.")
        p = p.strip()
        if p in available_paths:
            out.append(p)
    return sorted(set(out))


def _candidate_spines_for_known_roots(available_paths: set[str]) -> list[str]:
    """
    Deterministic "spine" file hints across known web roots:
      - frontend/
      - apps/web/
      - apps/frontend/
      - root-level
    """
    prefixes = ("", "frontend/", "apps/web/", "apps/frontend/")
    out: list[str] = []

    def add(p: str) -> None:
        if p in available_paths and p not in out:
            out.append(p)

    # web runtime spines
    for pref in prefixes:
        add(f"{pref}middleware.ts")
        add(f"{pref}middleware.js")
        add(f"{pref}app/layout.tsx")
        add(f"{pref}app/layout.ts")
        add(f"{pref}app/page.tsx")
        add(f"{pref}app/page.ts")
        add(f"{pref}next.config.ts")
        add(f"{pref}next.config.js")
        add(f"{pref}package.json")
        add(f"{pref}tsconfig.json")
        add(f"{pref}jsconfig.json")

    # repo-global spines
    for p in ("pyproject.toml", "uv.lock", "alembic.ini", "package.json", "tsconfig.json", "README.md", "readme.md"):
        add(p)

    # backend-ish spines (no prefix expansion; backend tends to be root-level)
    for p in ("backend/main.py", "backend/app.py", "backend/server.py", "backend/security.py", "backend/config.py"):
        add(p)

    return out


def _select_pack_paths_for_architecture(
        *,
        file_contents_map: dict[str, str],
        repo_index: dict[str, Any],
        deps_by_file: dict[str, dict[str, Any]],
        caps: SemanticCaps,
) -> tuple[list[str], dict[str, Any]]:
    available_paths = set(file_contents_map.keys())
    if not available_paths:
        raise Pass2SemanticError("pass2: file_contents_map is empty; cannot build LLM evidence pack.")

    entrypoints = _entrypoints_from_signals(repo_index, available_paths=available_paths)

    closure_seeds = [p for p in _read_plan_closure_seeds(repo_index) if p in available_paths]
    read_plan = [p for p in _read_plan_candidates(repo_index) if p in available_paths]

    spines = _candidate_spines_for_known_roots(available_paths)

    # STRICT seed order (deterministic, dedup, fixed rationale):
    # 1) closure seeds (highest signal)
    # 2) read plan candidates (rich coverage)
    # 3) entrypoints (if not already present)
    # 4) spines (conventions/config/runtime boundary files)
    seeds: list[str] = []
    for p in closure_seeds:
        if p not in seeds:
            seeds.append(p)
    for p in read_plan:
        if p not in seeds:
            seeds.append(p)
    for p in entrypoints:
        if p not in seeds:
            seeds.append(p)
    for p in spines:
        if p not in seeds:
            seeds.append(p)

    out_edges, in_edges = _compute_available_dep_graph(available_paths=available_paths, deps_by_file=deps_by_file)

    expanded = _expand_seeds_by_deps(
        seeds=seeds,
        out_edges=out_edges,
        hops=max(0, int(caps.pack_dep_hops)),
        max_added_per_file=max(0, int(caps.pack_max_dep_edges_per_file)),
    )

    lang_by_path = _language_by_path_from_repo_index(repo_index)

    def score(p: str) -> int:
        pl = p.lower()
        s = 0

        if p in closure_seeds:
            s += 1200
        if p in read_plan:
            s += 900
        if p in entrypoints:
            s += 800
        if p in spines:
            s += 650

        if pl.endswith(("main.py", "app.py", "server.py")):
            s += 240
        if pl.endswith(("/route.ts", "/route.js", "/page.tsx", "/layout.tsx")):
            s += 220
        if pl.endswith(("middleware.ts", "middleware.js")):
            s += 240
        if "security" in pl or "auth" in pl:
            s += 220

        s += min(80, 10 * len(in_edges.get(p, set())))
        s += min(40, 5 * len(out_edges.get(p, set())))

        if pl.startswith("backend/routers/"):
            s += 220
        if pl.startswith("backend/"):
            s += 60
        if "/app/api/" in pl and pl.endswith(("/route.ts", "/route.js")):
            s += 180

        if pl.startswith(("frontend/lib/", "apps/web/lib/", "apps/frontend/lib/")):
            s += 140
        if pl.startswith(("frontend/components/", "apps/web/components/", "apps/frontend/components/")):
            s += 120

        if pl.endswith("readme.md") or pl == "readme.md":
            s += 200
        if pl.startswith("docs/"):
            s += 120
        if pl.endswith(("pyproject.toml", "alembic.ini", "package.json", "next.config.ts", "next.config.js")):
            s += 160

        lang = lang_by_path.get(p, "")
        if lang in ("python", "typescript", "javascript"):
            s += 10

        return s

    ranked_all = sorted(list(available_paths), key=lambda p: (-score(p), p))

    ordered: list[str] = []
    seen: set[str] = set()

    def push(p: str) -> None:
        if p not in seen:
            seen.add(p)
            ordered.append(p)

    for p in expanded:
        push(p)
    for p in ranked_all:
        push(p)

    selection_debug = {
        "available_files": len(available_paths),
        "closure_seeds_count": len(closure_seeds),
        "read_plan_count": len(read_plan),
        "entrypoints_count": len(entrypoints),
        "spines_count": len(spines),
        "dep_hops": caps.pack_dep_hops,
        "dep_edges_per_file": caps.pack_max_dep_edges_per_file,
        "expanded_count": len(expanded),
    }
    return ordered, selection_debug


def _build_arch_files_pack(
        *,
        ordered_paths: list[str],
        file_contents_map: dict[str, str],
        caps: SemanticCaps,
) -> dict[str, str]:
    out: dict[str, str] = {}
    total = 0

    for p in ordered_paths:
        if len(out) >= caps.max_arch_files:
            break

        c = file_contents_map[p]
        if not isinstance(c, str) or not c:
            continue

        remaining = caps.max_arch_input_chars - total
        if remaining <= 0:
            break

        c2 = _truncate_with_tail(c, caps.max_arch_chars_per_file)
        if len(c2) > remaining:
            c2 = _truncate_with_tail(c2, remaining)
        if not c2:
            continue

        out[p] = c2
        total += len(c2)

    # ensure minimum breadth deterministically (still strict, just “fill”)
    floor = min(12, caps.max_arch_files)
    if len(out) < floor:
        for p in ordered_paths:
            if len(out) >= min(24, caps.max_arch_files):
                break
            if p in out:
                continue
            c = file_contents_map[p]
            if not isinstance(c, str) or not c:
                continue
            remaining = caps.max_arch_input_chars - total
            if remaining <= 0:
                break
            c2 = _truncate_with_tail(c, min(caps.max_arch_chars_per_file, remaining))
            if not c2:
                continue
            out[p] = c2
            total += len(c2)

    return out


def _select_supporting_files_for_gaps_and_onboarding(
        file_contents_map: dict[str, str],
        repo_index: dict[str, Any],
        *,
        max_files: int,
        max_total_chars: int,
        max_chars_per_file: int,
) -> dict[str, str]:
    available = set(file_contents_map.keys())
    entrypoints = set(_entrypoints_from_signals(repo_index, available_paths=available))

    closure_seeds = [p for p in _read_plan_closure_seeds(repo_index) if p in available]
    read_plan = [p for p in _read_plan_candidates(repo_index) if p in available]

    spines = _candidate_spines_for_known_roots(available)

    def score(p: str) -> int:
        pl = p.lower()
        s = 0
        if p in closure_seeds:
            s += 1100
        if p in read_plan:
            s += 900
        if p in entrypoints:
            s += 800
        if p in spines:
            s += 650
        if pl.endswith("readme.md") or pl == "readme.md":
            s += 260
        if pl.startswith("docs/") or "/docs/" in pl:
            s += 200
        if pl.endswith(".md"):
            s += 150
        if pl.endswith(("pyproject.toml", "alembic.ini", "uv.lock")):
            s += 140
        if "next.config" in pl or "eslint" in pl:
            s += 110
        if pl.endswith(("package.json", "tsconfig.json", "jsconfig.json")):
            s += 85
        if pl.endswith((".ts", ".tsx", ".py")):
            s += 10
        return s

    ranked = sorted(list(available), key=lambda p: (-score(p), p))

    ordered: list[str] = []
    seen: set[str] = set()

    def push(p: str) -> None:
        if p not in seen:
            seen.add(p)
            ordered.append(p)

    for p in closure_seeds:
        push(p)
    for p in read_plan:
        push(p)
    for p in spines:
        push(p)
    for p in ranked:
        push(p)

    out: dict[str, str] = {}
    total = 0
    for p in ordered:
        if len(out) >= max_files:
            break
        c = file_contents_map[p]
        if not isinstance(c, str) or not c:
            continue
        remaining = max_total_chars - total
        if remaining <= 0:
            break
        c2 = _truncate_with_tail(c, min(max_chars_per_file, remaining))
        if not c2:
            continue
        out[p] = c2
        total += len(c2)

    return out


def _deps_hints_for_pack(
        *,
        pack_paths: list[str],
        deps_by_file: dict[str, dict[str, Any]],
        max_internal: int = 10,
        max_external: int = 8,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in pack_paths:
        info = deps_by_file[p]
        internal = sorted(info["resolved_internal"])
        external = sorted(info["external_specs"])
        flags = sorted(info["flags"])

        out[p] = {
            "internal_deps_sample": internal[:max_internal],
            "external_specs_sample": external[:max_external],
            "internal_dep_count": len(internal),
            "external_spec_count": len(external),
            "flags": flags[:12],
        }
    return out


# -------------------------------------------------------------------
# Prompt payloads (LLM output stays SMALL + grounded)
# -------------------------------------------------------------------


def _build_architecture_payload(
        *,
        repo_url: str,
        resolved_commit: str,
        job_id: str,
        repo_index: dict[str, Any],
        file_contents_map: dict[str, str],
        deps_by_file: dict[str, dict[str, Any]],
        caps: SemanticCaps,
) -> tuple[dict[str, Any], list[str]]:
    ordered_paths, selection_debug = _select_pack_paths_for_architecture(
        file_contents_map=file_contents_map,
        repo_index=repo_index,
        deps_by_file=deps_by_file,
        caps=caps,
    )
    arch_files = _build_arch_files_pack(ordered_paths=ordered_paths, file_contents_map=file_contents_map, caps=caps)
    pack_paths = list(arch_files.keys())
    files_pack: list[dict[str, Any]] = [{"path": p, "content": c} for p, c in arch_files.items()]

    signals = _signals_from_repo_index(repo_index)
    path_aliases = repo_index.get("path_aliases", {})
    if not isinstance(path_aliases, dict):
        raise Pass2SemanticError("Pass1 contract violation: repo_index.path_aliases must be a dict when present.")

    repo_paths = _repo_paths_set(repo_index)
    presence_hints = _known_present_route_hints(repo_paths)

    deps_hints = _deps_hints_for_pack(pack_paths=pack_paths, deps_by_file=deps_by_file)

    # include a sample of the plan (contracted shapes only)
    rps = _require_dict(repo_index.get("read_plan_suggestions"), name="repo_index.read_plan_suggestions")
    rps_sample = _read_plan_candidates(repo_index)[:40]
    closure_sample = _read_plan_closure_seeds(repo_index)[:80]

    return (
        {
            "repo": {"repo_url": repo_url, "resolved_commit": resolved_commit, "job_id": job_id},
            "rules": {
                "grounding": (
                    "Your statements MUST be grounded in the anchor_paths you cite. "
                    "If unsure, set responsibilities=['unknown'] and add an uncertainty with questions."
                ),
                "anchor_paths_constraint": "Every module anchor_paths MUST be a subset of pass2.files[].path.",
                "no_appendices": (
                    "Do NOT output dependency lists, file inventories, read plans, or audit appendices. "
                    "This pass is semantic: boundaries, runtime entrypoints, key flows."
                ),
                "missing_claims_rule": (
                    "You can only claim a feature/file is 'missing from the snapshot' if it is absent from repo_presence_hints. "
                    "If you did not see a file in pass2.files, say 'not included in provided file contents' instead of 'missing from repo'."
                ),
            },
            "pass1": {
                "counts": repo_index.get("counts", {}),
                "path_aliases": path_aliases,
                "signals": {
                    "entrypoints": signals.get("entrypoints", []),
                    "env_vars": signals.get("env_vars", []),
                    "package_json": signals.get("package_json", {}),
                },
                "read_plan": {
                    "closure_seeds_sample": closure_sample,
                    "candidates_sample": rps_sample,
                    "max_files_to_read_default": rps.get("max_files_to_read_default"),
                },
                "pack_selection": selection_debug,
            },
            "repo_presence_hints": presence_hints,
            "pass2": {
                "files": files_pack,
                "deps_hints_by_file": deps_hints,
            },
            "output_contract": {
                "return_json_object_with_keys": ["modules", "uncertainties"],
                "modules_require": ["name", "type", "summary", "anchor_paths"],
                "recommended_module_fields": [
                    "responsibilities",
                    "entrypoints",
                    "public_interfaces",
                    "data_flows",
                    "runtime_notes",
                    "where_to_change",
                    "risk_notes",
                ],
            },
        },
        pack_paths,
    )


def _architecture_prompt_text(payload: dict[str, Any]) -> str:
    return (
            "Generate Pass 2 architecture semantics for this repo snapshot.\n"
            "Return ONLY a single JSON object (no markdown) with keys:\n"
            "  - modules: array of module objects\n"
            "  - uncertainties: array\n\n"
            "Hard requirements:\n"
            "1) Each module MUST include anchor_paths (3–10) and they MUST be a subset of pass2.files[].path.\n"
            "2) summary must be grounded in anchor_paths. If unsure, set responsibilities=['unknown'] and add an uncertainty.\n"
            "3) Do NOT output dependency lists, read plans, file inventories, or other appendices.\n"
            "4) Keep output concise and onboarding-useful: define boundaries, runtime entrypoints, and key flows.\n"
            "5) IMPORTANT: If you did not see a file in pass2.files, do NOT claim it is missing from the repo snapshot.\n"
            "   Instead say it was not included in the provided file contents.\n\n"
            "Style guidance:\n"
            "- Prefer ~6–14 modules total.\n"
            "- Use module.type from: ['frontend','backend','shared_lib','db','jobs','infra','tooling','docs','unknown'].\n"
            "- anchor_paths should point to representative files, not every file.\n\n"
            "INPUT PAYLOAD (JSON):\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def _build_gaps_onboarding_payload(
        *,
        repo_url: str,
        resolved_commit: str,
        job_id: str,
        repo_index: dict[str, Any],
        onboarding_enabled: bool,
        arch_modules: list[dict[str, Any]],
        arch_uncertainties: list[dict[str, Any]],
        deterministic_gap_items: list[dict[str, Any]],
        file_contents_map: dict[str, str],
        caps: SemanticCaps,
) -> dict[str, Any]:
    support_files = _select_supporting_files_for_gaps_and_onboarding(
        file_contents_map,
        repo_index,
        max_files=caps.max_support_files,
        max_total_chars=caps.max_support_chars,
        max_chars_per_file=caps.max_support_chars_per_file,
    )

    modules_summary: list[dict[str, Any]] = []
    for m in arch_modules:
        modules_summary.append(
            {
                "name": m.get("name"),
                "type": m.get("type"),
                "summary": m.get("summary"),
                "anchor_paths": m.get("anchor_paths", []),
                "entrypoints": m.get("entrypoints", []),
                "where_to_change": m.get("where_to_change", []),
                "risk_notes": m.get("risk_notes", []),
            }
        )

    signals = _signals_from_repo_index(repo_index)
    path_aliases = repo_index.get("path_aliases", {})
    if not isinstance(path_aliases, dict):
        raise Pass2SemanticError("Pass1 contract violation: repo_index.path_aliases must be a dict when present.")

    repo_paths = _repo_paths_set(repo_index)
    presence_hints = _known_present_route_hints(repo_paths)

    return {
        "repo": {"repo_url": repo_url, "resolved_commit": resolved_commit, "job_id": job_id},
        "rules": {
            "onboarding_enabled": onboarding_enabled,
            "no_redundant_dumping": "Do NOT restate full code listings. Keep items concise and actionable.",
            "onboarding_md_goal": (
                "Write onboarding_md as a practical architect handoff: quickstart, entrypoints, boundaries, "
                "data/state flow, configuration surface (env vars), and common change locations."
            ),
            "missing_claims_rule": (
                "Do NOT claim routes/pages are missing from the repo snapshot unless repo_presence_hints indicates absent. "
                "If missing from supporting_files, say it was not included in provided file contents."
            ),
        },
        "pass1": {
            "counts": repo_index.get("counts", {}),
            "path_aliases": path_aliases,
            "signals": {
                "entrypoints": signals.get("entrypoints", []),
                "env_vars": signals.get("env_vars", []),
                "package_json": signals.get("package_json", {}),
            },
            "read_plan": {
                "closure_seeds_sample": _read_plan_closure_seeds(repo_index)[:80],
                "candidates_sample": _read_plan_candidates(repo_index)[:40],
            },
        },
        "repo_presence_hints": presence_hints,
        "architecture_summary": {"modules": modules_summary, "uncertainties": arch_uncertainties},
        "deterministic_gaps_already_found": deterministic_gap_items,
        "supporting_files": [{"path": p, "content": c} for p, c in support_files.items()],
        "output_contract": {
            "return_json_object_with_keys": ["gaps", "onboarding_md"],
            "gaps": {"must_include_keys": ["generated_at", "job_id", "items"]},
            "onboarding_md": {"type": "string", "allow_empty_if_disabled": True},
        },
    }


def _gaps_onboarding_prompt_text(payload: dict[str, Any]) -> str:
    return (
            "Generate Pass 2 gaps + onboarding artifacts.\n"
            "Return ONLY a single JSON object (no markdown) with keys:\n"
            "  - gaps: object\n"
            "  - onboarding_md: string\n\n"
            "Hard requirements:\n"
            "1) gaps must include keys: generated_at, job_id, items (array).\n"
            "2) Do NOT duplicate deterministic gaps already listed in deterministic_gaps_already_found.\n"
            "3) If onboarding_enabled is false, set onboarding_md to an empty string.\n"
            "4) Keep it concise and self-auditing.\n"
            "5) IMPORTANT: Do NOT say 'missing from repo snapshot' unless repo_presence_hints indicates absent. "
            "If you didn't see a file in supporting_files, say it was not included in provided file contents.\n\n"
            "Onboarding MD requirements (if enabled):\n"
            "- Must include sections: Overview, Entry points, Configuration (env vars), Common tasks, Where to change code, Risks/footguns.\n"
            "- Use file paths when referencing code.\n\n"
            "INPUT PAYLOAD (JSON):\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


# -------------------------------------------------------------------
# Deterministic derivations (deps derived from pass1; no LLM deps)
# -------------------------------------------------------------------


def _derive_deps_from_anchor_paths(
        modules: list[dict[str, Any]],
        deps_by_file: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    for m in modules:
        anchors = m.get("anchor_paths")
        if not isinstance(anchors, list):
            anchors = []
        anchors = [p for p in anchors if isinstance(p, str) and p.strip()]

        internal: set[str] = set()
        external: set[str] = set()
        for p in anchors:
            info = deps_by_file.get(p)
            if not info:
                continue
            internal |= set(info["resolved_internal"])
            external |= set(info["external_specs"])

        m["anchor_paths"] = anchors
        m["evidence_paths"] = list(anchors)  # compat name for downstream consumers
        m["dependencies"] = sorted(internal) + sorted(external)

    return modules


# -------------------------------------------------------------------
# Deterministic gap scans (grounded, non-LLM)
# -------------------------------------------------------------------


def _deterministic_gap_scan(repo_index: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    files = _require_list(repo_index.get("files"), name="repo_index.files")
    sig = _signals_from_repo_index(repo_index)

    unresolved_paths: list[str] = []
    parse_failed: list[str] = []

    for f in files:
        if not isinstance(f, dict):
            raise Pass2SemanticError("Pass1 contract violation: repo_index.files must contain dict entries.")
        p = f.get("path")
        if not isinstance(p, str) or not p:
            raise Pass2SemanticError("Pass1 contract violation: each file must have a non-empty 'path'.")
        flags = f.get("flags")
        if flags is None:
            continue
        if not isinstance(flags, list):
            raise Pass2SemanticError(f"Pass1 contract violation: flags must be a list when present ({p}).")
        flags_s = [x for x in flags if isinstance(x, str)]
        if "import_unresolved" in flags_s:
            unresolved_paths.append(p)
        if "python_parse_failed" in flags_s or "js_ts_parse_failed" in flags_s:
            parse_failed.append(p)

    if unresolved_paths:
        items.append(
            {
                "type": "unresolved_internal_imports",
                "severity": "medium",
                "description": "Some files contain internal-looking imports that did not resolve to repo paths in pass1.",
                "files_involved": sorted(unresolved_paths)[:60],
                "suggested_questions": [
                    "Are path aliases/baseUrl configured correctly (tsconfig/jsconfig)?",
                    "Are imports pointing to generated files or omitted extensions not covered by resolver?",
                ],
            }
        )

    if parse_failed:
        items.append(
            {
                "type": "parser_failures",
                "severity": "low",
                "description": "Some files could not be parsed cleanly for defs/imports; pass1 fell back to best-effort.",
                "files_involved": sorted(parse_failed)[:60],
                "suggested_questions": ["Are these files syntactically valid? Are they templates or partials?"],
            }
        )

    eps = sig.get("entrypoints")
    if eps is not None and (not isinstance(eps, list) or len(eps) == 0):
        items.append(
            {
                "type": "no_entrypoints_detected",
                "severity": "low",
                "description": "Pass1 did not detect any entrypoint hints (Next.js pages/routes, python __main__, FastAPI app, etc.).",
                "files_involved": [],
                "suggested_questions": [
                    "Is the repo structure non-standard (not frontend/backend)?",
                    "Do we need additional heuristics for your framework conventions?",
                ],
            }
        )

    return items


# -------------------------------------------------------------------
# Post-enforcement (grounding only; deps derived deterministically later)
# -------------------------------------------------------------------


_ALLOWED_TYPES = {"frontend", "backend", "shared_lib", "db", "jobs", "infra", "tooling", "docs", "unknown"}


def _post_enforce_architecture_constraints(
        *,
        modules: Any,
        uncertainties: Any,
        allowed_paths: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_modules: list[dict[str, Any]] = []
    out_uncertainties: list[dict[str, Any]] = []

    if isinstance(uncertainties, list):
        for u in uncertainties:
            if isinstance(u, dict):
                out_uncertainties.append(u)

    if not isinstance(modules, list):
        raise Pass2SemanticError("Pass2 LLM contract violation: 'modules' must be a list.")

    for i, m in enumerate(modules):
        if not isinstance(m, dict):
            continue
        mm = dict(m)

        name = mm.get("name")
        if not isinstance(name, str) or not name.strip():
            mm["name"] = f"unknown_module_{i}"

        mtype = mm.get("type")
        if not isinstance(mtype, str) or not mtype.strip():
            mm["type"] = "unknown"
        else:
            mt = mtype.strip()
            mm["type"] = mt if mt in _ALLOWED_TYPES else "unknown"

        anchors = mm.get("anchor_paths")
        if not isinstance(anchors, list):
            anchors = []
        anchors = [p for p in anchors if isinstance(p, str) and p in allowed_paths]
        if len(anchors) > 10:
            anchors = anchors[:10]
        mm["anchor_paths"] = anchors

        summary = mm.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            resp = mm.get("responsibilities")
            if isinstance(resp, list):
                resp2 = [r for r in resp if isinstance(r, str) and r.strip()]
                if resp2:
                    summary = "; ".join(resp2[:3])
            mm["summary"] = summary.strip() if isinstance(summary, str) and summary.strip() else "unknown"

        resp = mm.get("responsibilities")
        if not isinstance(resp, list):
            resp = []
        resp = [r for r in resp if isinstance(r, str) and r.strip()]

        if not anchors:
            mm["responsibilities"] = ["unknown"]
            mm["summary"] = "unknown"
            out_uncertainties.append(
                {
                    "type": "ungrounded_module",
                    "description": (
                        f"Module '{mm.get('name','unknown')}' lacks anchor_paths in files_read; "
                        "summary/responsibilities set to unknown."
                    ),
                    "files_involved": [],
                    "suggested_questions": ["Which files define this module? Add representative files to pass2 selection."],
                }
            )
        else:
            mm["responsibilities"] = resp or ["unknown"]
            if not resp:
                out_uncertainties.append(
                    {
                        "type": "empty_responsibilities",
                        "description": f"Module '{mm.get('name','unknown')}' had no responsibilities listed; set to unknown.",
                        "files_involved": anchors,
                        "suggested_questions": ["What does this module do? Add explicit responsibilities."],
                    }
                )

        # Do NOT accept deps from LLM output (deterministic derivation later)
        mm["dependencies"] = []
        mm["evidence_paths"] = list(anchors)  # compat (filled again later)

        out_modules.append(mm)

    return out_modules, out_uncertainties


# -------------------------------------------------------------------
# Gaps normalization / dedupe + false-positive rewrite
# -------------------------------------------------------------------


def _normalize_gaps_object(gaps: Any, *, job_id: str) -> dict[str, Any]:
    if not isinstance(gaps, dict):
        raise Pass2SemanticError("Pass2 LLM contract violation: 'gaps' must be an object.")
    out = dict(gaps)
    out.setdefault("generated_at", None)
    out["job_id"] = job_id
    items = out.get("items")
    if not isinstance(items, list):
        raise Pass2SemanticError("Pass2 LLM contract violation: gaps.items must be a list.")
    cleaned: list[dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict):
            cleaned.append(it)
    out["items"] = cleaned
    return out


def _dedupe_gap_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        key = json.dumps(
            {
                "type": it.get("type"),
                "severity": it.get("severity"),
                "module": it.get("module"),
                "dependency": it.get("dependency"),
                "description": it.get("description"),
                "evidence_paths": it.get("evidence_paths"),
                "files_involved": it.get("files_involved"),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _fix_false_missing_route_gap_items(items: list[dict[str, Any]], allowed_paths: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        t = it.get("type")
        files_involved = it.get("files_involved")
        present: list[str] = []
        if isinstance(files_involved, list):
            present = [p for p in files_involved if isinstance(p, str) and p in allowed_paths]

        if t == "missing_route" and present:
            it2 = dict(it)
            it2["type"] = "route_implementation_check"
            it2["severity"] = it2.get("severity") or "medium"
            it2["description"] = (
                    (it2.get("description") or "").strip()
                    + " (Referenced route file(s) exist in snapshot; verify if implementation is placeholder or not wired.)"
            ).strip()
            out.append(it2)
        else:
            out.append(it)
    return out


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------


def generate_pass2_semantic_artifacts(
        *,
        repo_url: str,
        resolved_commit: str,
        job_id: str,
        repo_index: dict[str, Any],
        files_read: list[dict[str, Any]],
        files_not_read: list[dict[str, Any]],
        file_contents_map: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """
    Contracted behavior:
      - Pass2 LLM outputs ONLY: semantic modules + anchor_paths + short semantics
      - evidence_paths and dependencies are derived deterministically from pass1 after the fact
      - bounded read pack is selected deterministically from pass1 facts (closure seeds + read plan + deps expansion),
        constrained to provided file_contents_map (no hidden reads)
    """
    caps = _semantic_caps_from_env()

    # STRICT: validate contract essentials early
    _ = _read_plan_closure_seeds(repo_index)
    _ = _read_plan_candidates(repo_index)

    deps_by_file = _extract_pass1_deps(repo_index)

    # -------------------------
    # Architecture semantics
    # -------------------------
    arch_payload, pack_paths = _build_architecture_payload(
        repo_url=repo_url,
        resolved_commit=resolved_commit,
        job_id=job_id,
        repo_index=repo_index,
        file_contents_map=file_contents_map,
        deps_by_file=deps_by_file,
        caps=caps,
    )
    arch_prompt = _architecture_prompt_text(arch_payload)

    arch_obj = _openai_call_json(
        prompt=arch_prompt,
        model=caps.model,
        max_output_tokens=caps.max_output_tokens,
        system="You are a precise code architect. Output JSON only.",
    )

    allowed_paths = set(file_contents_map.keys())
    repo_paths = _repo_paths_set(repo_index)

    enforced_modules, enforced_uncertainties = _post_enforce_architecture_constraints(
        modules=arch_obj.get("modules"),
        uncertainties=arch_obj.get("uncertainties", []),
        allowed_paths=allowed_paths,
    )

    # deterministically attach deps (and evidence_paths := anchor_paths for compat)
    enforced_modules = _derive_deps_from_anchor_paths(enforced_modules, deps_by_file)

    arch_out: dict[str, Any] = {"modules": enforced_modules, "uncertainties": enforced_uncertainties}

    # -------------------------
    # Deterministic gaps (non-LLM)
    # -------------------------
    deterministic_gap_items = _deterministic_gap_scan(repo_index)

    # -------------------------
    # LLM gaps + onboarding (bounded supporting pack)
    # -------------------------
    gaps_payload = _build_gaps_onboarding_payload(
        repo_url=repo_url,
        resolved_commit=resolved_commit,
        job_id=job_id,
        repo_index=repo_index,
        onboarding_enabled=caps.onboarding_enabled,
        arch_modules=enforced_modules,
        arch_uncertainties=enforced_uncertainties,
        deterministic_gap_items=deterministic_gap_items,
        file_contents_map=file_contents_map,
        caps=caps,
    )
    gaps_prompt = _gaps_onboarding_prompt_text(gaps_payload)

    gaps_obj = _openai_call_json(
        prompt=gaps_prompt,
        model=caps.model,
        max_output_tokens=caps.max_output_tokens,
        system="You are a precise repo auditor. Output JSON only.",
    )

    gaps_raw = _normalize_gaps_object(gaps_obj.get("gaps"), job_id=job_id)
    onboarding_md = gaps_obj.get("onboarding_md")
    if onboarding_md is None:
        onboarding_md = ""
    if not isinstance(onboarding_md, str):
        raise Pass2SemanticError("Pass2 LLM contract violation: onboarding_md must be a string.")
    if not caps.onboarding_enabled:
        onboarding_md = ""

    merged_items = list(deterministic_gap_items)
    for it in gaps_raw["items"]:
        if isinstance(it, dict):
            merged_items.append(it)

    # keep: correctness over “missing” language if file was present in pack
    merged_items = _fix_false_missing_route_gap_items(merged_items, allowed_paths)

    gaps_out = dict(gaps_raw)
    gaps_out["items"] = _dedupe_gap_items(merged_items)

    return arch_out, gaps_out, onboarding_md
