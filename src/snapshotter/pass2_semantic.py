# snapshotter/pass2_semantic.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from snapshotter.job import Job
from snapshotter.utils import sha256_bytes, utc_ts

# --------------------------------------------------------------------------------------
# Pass 2 Semantic Contract (LOCKED, strict; aligned with pass1.py)
#
# Inputs (STRICT):
# - pass1_repo_index MUST be the dict produced by pass1.build_repo_index / pass1.generate_pass1_artifacts
# - We DO NOT accept legacy/back-compat shapes (no alternate field names).
#
# Pass1 -> Pass2 schema assumptions (STRICT):
# - pass1_repo_index["schema_version"] == "pass1_repo_index.v1"
# - pass1_repo_index["job"]["resolved_commit"] is present and not "unknown"
# - pass1_repo_index["files"] is a list of dicts, each with:
#     - path: str
#     - deps: dict with:
#         - import_edges: list[dict] (may be empty)
#           * Each edge MUST contain:
#               - kind: str
#               - spec: str
#               - lineno: int
#               - resolved_path: str
#               - is_external: False
#         - internal_unresolved_specs: list[str] (may be empty)
#
# Caps source of truth (aligned with Pass1 read-plan cap rule):
# - Pass2 caps come from Job.pass2.* (primary).
# - Environment variables are allowed ONLY as a temporary back-compat shim when Job schema
#   does not yet contain the field (mirrors pass1.py fallback pattern). No multi-location probing.
#
# Outputs (LOCKED):
# - PASS2_SEMANTIC.json         (model output + pack fingerprints + strict metadata)
# - PASS2_ARCH_PACK.json        (bounded architecture evidence pack: path->content)
# - PASS2_SUPPORT_PACK.json     (bounded supporting pack: path->content)
# - PASS2_LLM_RAW.txt           (raw model text for inspection on failure)
# - PASS2_LLM_REPAIRED.txt      (optional: repaired JSON text if repair was used)
# --------------------------------------------------------------------------------------

PASS2_SEMANTIC_FILENAME = "PASS2_SEMANTIC.json"
PASS2_ARCH_PACK_FILENAME = "PASS2_ARCH_PACK.json"
PASS2_SUPPORT_PACK_FILENAME = "PASS2_SUPPORT_PACK.json"
PASS2_LLM_RAW_FILENAME = "PASS2_LLM_RAW.txt"
PASS2_LLM_REPAIRED_FILENAME = "PASS2_LLM_REPAIRED.txt"

PASS1_REPO_INDEX_SCHEMA_VERSION = "pass1_repo_index.v1"
PASS2_SEMANTIC_SCHEMA_VERSION = "pass2_semantic.v1"
PASS2_ARCH_PACK_SCHEMA_VERSION = "pass2_arch_pack.v1"
PASS2_SUPPORT_PACK_SCHEMA_VERSION = "pass2_support_pack.v1"


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


# -------------------------------------------------------------------
# Caps: Job-first (aligned), env-only fallback for missing Job fields
# -------------------------------------------------------------------


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


def _job_get(job: Job, dotted: str) -> Any:
    cur: Any = job
    for part in dotted.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return cur


def _caps_from_job_or_env(job: Job) -> SemanticCaps:
    """
    Contract alignment with pass1.py:
    - Prefer Job.pass2.* as the only intended cap source.
    - Env vars are used ONLY as fallback when the Job schema doesn't yet expose the field.
    """
    # Defaults are deliberately consistent with prior behavior.
    def b(field: str, env: str, default: bool) -> bool:
        v = _job_get(job, field)
        if isinstance(v, bool):
            return v
        # fallback only if absent / wrong type
        return _bool_from_env(env, default)

    def i(field: str, env: str, default: int) -> int:
        v = _job_get(job, field)
        try:
            if v is not None:
                n = int(v)
                return n
        except Exception:
            pass
        return _int_from_env(env, default)

    def s(field: str, env: str, default: str) -> str:
        v = _job_get(job, field)
        if isinstance(v, str) and v.strip():
            return v.strip()
        w = os.environ.get(env, "").strip()
        return w or default

    onboarding_enabled = b("pass2.onboarding_enabled", "SNAPSHOTTER_PASS2_ONBOARDING", True)
    model = s("pass2.model", "SNAPSHOTTER_LLM_MODEL", "gpt-4.1-mini")
    max_output_tokens = i("pass2.max_output_tokens", "SNAPSHOTTER_LLM_MAX_OUTPUT_TOKENS", 2000)

    max_arch_input_chars = i("pass2.max_arch_input_chars", "SNAPSHOTTER_PASS2_MAX_ARCH_INPUT_CHARS", 240_000)
    max_arch_files = i("pass2.max_files", "SNAPSHOTTER_PASS2_MAX_ARCH_FILES", 120)
    max_arch_chars_per_file = i("pass2.max_arch_chars_per_file", "SNAPSHOTTER_PASS2_MAX_ARCH_CHARS_PER_FILE", 9000)

    max_support_files = i("pass2.max_support_files", "SNAPSHOTTER_PASS2_MAX_SUPPORT_FILES", 28)
    max_support_chars = i("pass2.max_support_chars", "SNAPSHOTTER_PASS2_MAX_SUPPORT_CHARS", 120_000)
    max_support_chars_per_file = i(
        "pass2.max_support_chars_per_file",
        "SNAPSHOTTER_PASS2_MAX_SUPPORT_CHARS_PER_FILE",
        9000,
    )

    pack_dep_hops = i("pass2.pack_dep_hops", "SNAPSHOTTER_PASS2_PACK_DEP_HOPS", 1)
    pack_max_dep_edges_per_file = i(
        "pass2.pack_max_dep_edges_per_file",
        "SNAPSHOTTER_PASS2_PACK_MAX_DEP_EDGES_PER_FILE",
        12,
    )

    # guardrails (deterministic clamping)
    max_output_tokens = max(256, min(max_output_tokens, 20_000))
    max_arch_files = max(1, min(max_arch_files, 240))
    max_arch_input_chars = max(10_000, min(max_arch_input_chars, 500_000))
    max_arch_chars_per_file = max(500, min(max_arch_chars_per_file, 60_000))

    max_support_files = max(1, min(max_support_files, 120))
    max_support_chars = max(5_000, min(max_support_chars, 300_000))
    max_support_chars_per_file = max(500, min(max_support_chars_per_file, 60_000))

    pack_dep_hops = max(0, min(pack_dep_hops, 4))
    pack_max_dep_edges_per_file = max(0, min(pack_max_dep_edges_per_file, 100))

    return SemanticCaps(
        onboarding_enabled=bool(onboarding_enabled),
        model=model,
        max_output_tokens=int(max_output_tokens),
        max_arch_input_chars=int(max_arch_input_chars),
        max_arch_files=int(max_arch_files),
        max_arch_chars_per_file=int(max_arch_chars_per_file),
        max_support_files=int(max_support_files),
        max_support_chars=int(max_support_chars),
        max_support_chars_per_file=int(max_support_chars_per_file),
        pack_dep_hops=int(pack_dep_hops),
        pack_max_dep_edges_per_file=int(pack_max_dep_edges_per_file),
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

    # STRICT: expect a single JSON object. Allow extracting the first object span only if
    # the SDK returns extra wrapper text.
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


def _openai_call_json(*, prompt: str, model: str, max_output_tokens: int, system: str) -> tuple[dict[str, Any], str, str | None]:
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

    raw_text: str
    try:
        raw_text = _responses_create_text(input_payload)
    except Exception as e:
        raise Pass2SemanticError(f"OpenAI Responses API call failed: {e}") from e

    try:
        obj = _try_parse_json(raw_text)
        return obj, raw_text, None
    except Exception as parse_err:
        if _looks_truncated(raw_text):
            raise Pass2SemanticLLMOutputError(
                "OpenAI returned truncated/incomplete JSON (likely hit max output tokens). "
                "Increase pass2.max_output_tokens (Job) and retry.\n"
                f"Parse error: {parse_err}\n"
                f"First 400 chars:\n{raw_text[:400]}",
                raw_text=raw_text,
            ) from parse_err

        # Repair allowed (artifact salvage).
        repair_prompt = _build_json_repair_prompt(raw_text)
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
                f"First 400 chars of original:\n{raw_text[:400]}",
                raw_text=raw_text,
            ) from e

        try:
            obj2 = _try_parse_json(repaired_text)
            return obj2, raw_text, repaired_text
        except Exception as e:
            raise Pass2SemanticLLMOutputError(
                "Failed to parse OpenAI JSON response (including repair attempt).\n"
                f"Original first 400 chars:\n{raw_text[:400]}",
                raw_text=raw_text,
                repaired_text=repaired_text,
            ) from e


# -------------------------------------------------------------------
# Pass1 contract validation + strict extractors
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


def _assert_pass1_repo_index_contract(repo_index: dict[str, Any]) -> None:
    if not isinstance(repo_index, dict):
        raise Pass2SemanticError("Pass1 contract violation: repo_index must be a dict.")

    if repo_index.get("schema_version") != PASS1_REPO_INDEX_SCHEMA_VERSION:
        raise Pass2SemanticError("Pass1 contract violation: repo_index.schema_version mismatch.")

    job = _require_dict(repo_index.get("job"), name="repo_index.job")
    rc = job.get("resolved_commit")
    if not isinstance(rc, str) or not rc.strip() or rc.strip() == "unknown":
        raise Pass2SemanticError("Pass1 contract violation: job.resolved_commit missing/invalid.")

    rp = _require_dict(repo_index.get("read_plan"), name="repo_index.read_plan")
    _require_list_str(rp.get("closure_seeds"), name="repo_index.read_plan.closure_seeds")
    _require_list(rp.get("candidates"), name="repo_index.read_plan.candidates")

    files = _require_list(repo_index.get("files"), name="repo_index.files")
    for f in files:
        if not isinstance(f, dict):
            raise Pass2SemanticError("Pass1 contract violation: repo_index.files must contain dict entries.")
        path = f.get("path")
        if not isinstance(path, str) or not path:
            raise Pass2SemanticError("Pass1 contract violation: each file must have non-empty 'path'.")
        deps = _require_dict(f.get("deps"), name=f"file.deps ({path})")
        edges = _require_list(deps.get("import_edges"), name=f"file.deps.import_edges ({path})")
        for e in edges:
            if not isinstance(e, dict):
                raise Pass2SemanticError(f"Pass1 contract violation: import_edges must contain dict entries ({path}).")
            spec = e.get("spec")
            if not isinstance(spec, str) or not spec.strip():
                raise Pass2SemanticError(f"Pass1 contract violation: import edge missing spec ({path}).")
            # Pass1 guarantees internal edges only and is_external=False
            if bool(e.get("is_external", False)) is not False:
                raise Pass2SemanticError(f"Pass1 contract violation: import edge is_external must be False ({path}).")
            rp2 = e.get("resolved_path")
            if not isinstance(rp2, str) or not rp2.strip():
                raise Pass2SemanticError(f"Pass1 contract violation: internal edge missing resolved_path ({path}).")
        iu = deps.get("internal_unresolved_specs", [])
        if not isinstance(iu, list):
            raise Pass2SemanticError(f"Pass1 contract violation: internal_unresolved_specs must be list ({path}).")
        for s in iu:
            if not isinstance(s, str) or not s.strip():
                raise Pass2SemanticError(f"Pass1 contract violation: internal_unresolved_specs must be strings ({path}).")


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
    return sig


def _read_plan_candidates(repo_index: dict[str, Any]) -> list[str]:
    rp = _require_dict(repo_index.get("read_plan"), name="repo_index.read_plan")
    cands = _require_list(rp.get("candidates"), name="repo_index.read_plan.candidates")

    out: list[str] = []
    for it in cands:
        if not isinstance(it, dict):
            raise Pass2SemanticError("Pass1 contract violation: read_plan.candidates must contain dict entries.")
        p = it.get("path")
        if not isinstance(p, str) or not p.strip():
            raise Pass2SemanticError("Pass1 contract violation: read_plan.candidate.path must be a non-empty string.")
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
    rp = _require_dict(repo_index.get("read_plan"), name="repo_index.read_plan")
    seeds = _require_list_str(rp.get("closure_seeds"), name="repo_index.read_plan.closure_seeds")

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

    Output per file:
      {
        "resolved_internal": set[str],
        "import_edges": list[dict],
        "flags": set[str],
        "language": str|None,
        "top_level_defs": list[str],
        "internal_unresolved_specs": list[str],
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
        edges_raw = _require_list(deps.get("import_edges"), name=f"file.deps.import_edges ({path})")

        import_edges: list[dict[str, Any]] = []
        resolved_internal: set[str] = set()
        for e in edges_raw:
            if not isinstance(e, dict):
                raise Pass2SemanticError(f"Pass1 contract violation: import_edges must contain dict entries ({path}).")
            spec = e.get("spec")
            if not isinstance(spec, str) or not spec.strip():
                raise Pass2SemanticError(f"Pass1 contract violation: import edge missing spec ({path}).")
            if bool(e.get("is_external", False)) is not False:
                raise Pass2SemanticError(f"Pass1 contract violation: import edge is_external must be False ({path}).")
            rp = e.get("resolved_path")
            if not isinstance(rp, str) or not rp.strip():
                raise Pass2SemanticError(f"Pass1 contract violation: internal edge missing resolved_path ({path}).")
            import_edges.append(dict(e))
            resolved_internal.add(rp.strip())

        iu0 = deps.get("internal_unresolved_specs", [])
        if not isinstance(iu0, list):
            raise Pass2SemanticError(f"Pass1 contract violation: internal_unresolved_specs must be list ({path}).")
        internal_unresolved_specs: list[str] = []
        for s in iu0:
            if not isinstance(s, str) or not s.strip():
                raise Pass2SemanticError(f"Pass1 contract violation: internal_unresolved_specs must be strings ({path}).")
            internal_unresolved_specs.append(s.strip())
        internal_unresolved_specs = sorted(set(internal_unresolved_specs))

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
            "import_edges": import_edges,
            "flags": flags_set,
            "language": lang_str,
            "top_level_defs": top_defs,
            "internal_unresolved_specs": internal_unresolved_specs,
        }

    return out


# -------------------------------------------------------------------
# Deterministic pack selection helpers
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
    prefixes = ("", "frontend/", "apps/web/", "apps/frontend/")
    out: list[str] = []

    def add(p: str) -> None:
        if p in available_paths and p not in out:
            out.append(p)

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

    for p in ("pyproject.toml", "uv.lock", "alembic.ini", "package.json", "tsconfig.json", "README.md", "readme.md"):
        add(p)

    for p in ("backend/main.py", "backend/app.py", "backend/server.py", "backend/security.py", "backend/config.py"):
        add(p)

    return out


def _compute_available_dep_graph(
        *,
        available_paths: set[str],
        deps_by_file: dict[str, dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    out_edges: dict[str, set[str]] = {p: set() for p in available_paths}
    in_edges: dict[str, set[str]] = {p: set() for p in available_paths}

    for p in available_paths:
        info = deps_by_file[p]
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

    # STRICT seed order (deterministic):
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

        c = file_contents_map.get(p, "")
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

    # ensure minimum breadth deterministically
    floor = min(12, caps.max_arch_files)
    if len(out) < floor:
        for p in ordered_paths:
            if len(out) >= min(24, caps.max_arch_files):
                break
            if p in out:
                continue
            c = file_contents_map.get(p, "")
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
        c = file_contents_map.get(p, "")
        if not isinstance(c, str) or not c:
            continue
        remaining = max_total_chars - total
        if remaining <= 0:
            break
        c2 = _truncate_with_tail(c, max_chars_per_file)
        if len(c2) > remaining:
            c2 = _truncate_with_tail(c2, remaining)
        if not c2:
            continue
        out[p] = c2
        total += len(c2)

    return out


# -------------------------------------------------------------------
# Repo file reading (deterministic, pass1-driven)
# -------------------------------------------------------------------


_BINARY_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".zip",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".mp4",
    ".mov",
    ".mp3",
    ".wav",
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
}


def _read_repo_file_text(repo_dir: str, rel_path: str, *, max_bytes: int) -> str | None:
    p = Path(repo_dir) / rel_path
    if not p.exists() or not p.is_file():
        return None
    if p.suffix.lower() in _BINARY_EXTS:
        return None
    try:
        raw = p.read_bytes()
    except Exception:
        return None
    if max_bytes > 0 and len(raw) > max_bytes:
        raw = raw[:max_bytes]
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return None


def _build_file_contents_map(repo_dir: str, repo_index: dict[str, Any], job: Job) -> dict[str, str]:
    """
    Deterministic:
    - Only consider pass1_repo_index.files paths (source of truth).
    - Read text for those files (best-effort), bounded by job.limits.max_file_bytes where available.
    """
    max_file_bytes = _job_get(job, "limits.max_file_bytes")
    try:
        maxb = int(max_file_bytes) if max_file_bytes is not None else 0
    except Exception:
        maxb = 0
    if maxb <= 0:
        maxb = 512_000  # safe fallback; pass1 already bounded scanning; pass2 pack caps also bound.

    files = _require_list(repo_index.get("files"), name="repo_index.files")
    out: dict[str, str] = {}
    for f in files:
        if not isinstance(f, dict):
            continue
        rp = f.get("path")
        if not isinstance(rp, str) or not rp:
            continue
        txt = _read_repo_file_text(repo_dir, rp, max_bytes=maxb)
        if isinstance(txt, str) and txt:
            out[rp] = txt
    return out


# -------------------------------------------------------------------
# Prompting: strict JSON object output
# -------------------------------------------------------------------


def _build_system_prompt() -> str:
    return (
        "You are Snapshotter Pass2 Semantic.\n"
        "You must output ONLY a single JSON object (no markdown, no commentary).\n"
        "The JSON must follow the requested schema strictly.\n"
        "If you are unsure, use nulls and empty arrays, but keep keys present.\n"
    )


def _build_user_prompt(
        *,
        repo_meta: dict[str, Any],
        caps: SemanticCaps,
        pass1_repo_index: dict[str, Any],
        arch_pack: dict[str, str],
        support_pack: dict[str, str],
        deps_by_file: dict[str, dict[str, Any]],
) -> str:
    """
    Keep the prompt deterministic:
    - stable ordering
    - explicit schema
    - include pass1 signals and the two packs
    """
    # lightweight structured deps summary to avoid dumping huge edge lists
    dep_summary: dict[str, Any] = {}
    for p in sorted(deps_by_file.keys()):
        info = deps_by_file[p]
        resolved = sorted([x for x in info["resolved_internal"] if isinstance(x, str)])
        unresolved = info.get("internal_unresolved_specs", [])
        dep_summary[p] = {
            "resolved_internal_count": len(resolved),
            "resolved_internal_sample": resolved[:12],
            "internal_unresolved_specs": unresolved[:12] if isinstance(unresolved, list) else [],
            "flags": sorted(list(info.get("flags", set())))[:12],
            "language": info.get("language"),
            "top_level_defs": info.get("top_level_defs", [])[:20],
        }

    schema = {
        "schema_version": PASS2_SEMANTIC_SCHEMA_VERSION,
        "generated_at": "ISO8601",
        "repo": {"repo_url": "string|null", "resolved_commit": "string"},
        "caps": {
            "model": "string",
            "max_output_tokens": "int",
            "max_arch_files": "int",
            "max_support_files": "int",
        },
        "summary": {
            "primary_stack": "string|null",
            "architecture_overview": "string",
            "key_components": ["string"],
            "data_flows": ["string"],
            "auth_and_routing_notes": ["string"],
            "risks_or_gaps": ["string"],
        },
        "evidence": {
            "arch_pack_paths": ["string"],
            "support_pack_paths": ["string"],
            "notable_files": [{"path": "string", "why": "string"}],
        },
    }

    sig = pass1_repo_index.get("signals", {})
    resolver_inputs = pass1_repo_index.get("resolver_inputs", {})

    payload = {
        "repo_meta": repo_meta,
        "schema": schema,
        "pass1_signals": sig,
        "pass1_resolver_inputs": resolver_inputs,
        "deps_summary": dep_summary,
        "arch_pack": {k: arch_pack[k] for k in sorted(arch_pack.keys())},
        "support_pack": {k: support_pack[k] for k in sorted(support_pack.keys())},
        "rules": [
            "Output JSON only.",
            "Do not invent files or paths; reference only paths present in the packs.",
            "Keep schema keys present.",
            "Prefer short, high-signal bullets in arrays.",
        ],
    }

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)


# -------------------------------------------------------------------
# Artifact writers (atomic; deterministic formatting)
# -------------------------------------------------------------------


def _write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = (text or "")
    if payload and not payload.endswith("\n"):
        payload += "\n"
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(tmp, p)


def _write_json(path: str | Path, obj: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(tmp, p)


def _stable_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8", errors="replace")


def _fingerprint_pack(pack_obj: dict[str, Any]) -> str:
    return sha256_bytes(_stable_json_bytes(pack_obj))


# -------------------------------------------------------------------
# Public entrypoint
# -------------------------------------------------------------------


def generate_pass2_semantic_artifacts(
        *,
        repo_dir: str,
        job: Job,
        out_dir: str | Path,
        pass1_repo_index: dict[str, Any],
        repo_url: str | None = None,  # <-- compat with caller; optional override
) -> dict[str, Any]:
    """
    Pass 2 "proper contract" entrypoint.

    Writes:
      - PASS2_ARCH_PACK.json
      - PASS2_SUPPORT_PACK.json
      - PASS2_SEMANTIC.json
      - PASS2_LLM_RAW.txt (always)
      - PASS2_LLM_REPAIRED.txt (only if repair happened)

    Strict:
    - assumes pass1 contract (validated)
    - no back-compat schema paths

    Note:
    - `repo_url` is accepted for compatibility with callers that still pass it.
      If provided, it overrides job.repo_url.
    """
    _assert_pass1_repo_index_contract(pass1_repo_index)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    caps = _caps_from_job_or_env(job)

    # Repo URL resolution (deterministic precedence):
    # 1) explicit kwarg from caller (compat)
    # 2) job.repo_url (if present)
    if isinstance(repo_url, str):
        repo_url = repo_url.strip() or None
    if repo_url is None:
        try:
            repo_url = getattr(job, "repo_url", None)
        except Exception:
            repo_url = None
        if isinstance(repo_url, str):
            repo_url = repo_url.strip() or None

    resolved_commit = _require_dict(pass1_repo_index.get("job"), name="repo_index.job").get("resolved_commit")
    assert isinstance(resolved_commit, str) and resolved_commit.strip()

    deps_by_file = _extract_pass1_deps(pass1_repo_index)
    file_contents_map = _build_file_contents_map(repo_dir, pass1_repo_index, job)

    ordered_paths, selection_debug = _select_pack_paths_for_architecture(
        file_contents_map=file_contents_map,
        repo_index=pass1_repo_index,
        deps_by_file=deps_by_file,
        caps=caps,
    )

    arch_files = _build_arch_files_pack(
        ordered_paths=ordered_paths,
        file_contents_map=file_contents_map,
        caps=caps,
    )

    support_files = _select_supporting_files_for_gaps_and_onboarding(
        file_contents_map,
        pass1_repo_index,
        max_files=caps.max_support_files,
        max_total_chars=caps.max_support_chars,
        max_chars_per_file=caps.max_support_chars_per_file,
    )

    arch_pack_obj = {
        "schema_version": PASS2_ARCH_PACK_SCHEMA_VERSION,
        "generated_at": utc_ts(),
        "repo": {"repo_url": repo_url, "resolved_commit": resolved_commit},
        "caps": {
            "max_arch_files": caps.max_arch_files,
            "max_arch_input_chars": caps.max_arch_input_chars,
            "max_arch_chars_per_file": caps.max_arch_chars_per_file,
            "pack_dep_hops": caps.pack_dep_hops,
            "pack_max_dep_edges_per_file": caps.pack_max_dep_edges_per_file,
        },
        "selection_debug": selection_debug,
        "files": {k: arch_files[k] for k in sorted(arch_files.keys())},
    }
    arch_pack_obj["fingerprint_sha256"] = _fingerprint_pack(
        {"repo": arch_pack_obj["repo"], "caps": arch_pack_obj["caps"], "files": arch_pack_obj["files"]}
    )

    support_pack_obj = {
        "schema_version": PASS2_SUPPORT_PACK_SCHEMA_VERSION,
        "generated_at": utc_ts(),
        "repo": {"repo_url": repo_url, "resolved_commit": resolved_commit},
        "caps": {
            "max_support_files": caps.max_support_files,
            "max_support_chars": caps.max_support_chars,
            "max_support_chars_per_file": caps.max_support_chars_per_file,
        },
        "files": {k: support_files[k] for k in sorted(support_files.keys())},
    }
    support_pack_obj["fingerprint_sha256"] = _fingerprint_pack(
        {"repo": support_pack_obj["repo"], "caps": support_pack_obj["caps"], "files": support_pack_obj["files"]}
    )

    _write_json(out_root / PASS2_ARCH_PACK_FILENAME, arch_pack_obj)
    _write_json(out_root / PASS2_SUPPORT_PACK_FILENAME, support_pack_obj)

    # LLM call
    system = _build_system_prompt()
    user_prompt = _build_user_prompt(
        repo_meta={"repo_url": repo_url, "resolved_commit": resolved_commit},
        caps=caps,
        pass1_repo_index=pass1_repo_index,
        arch_pack=arch_pack_obj["files"],
        support_pack=support_pack_obj["files"],
        deps_by_file=deps_by_file,
    )

    obj, raw_text, repaired_text = _openai_call_json(
        prompt=user_prompt,
        model=caps.model,
        max_output_tokens=caps.max_output_tokens,
        system=system,
    )

    _write_text(out_root / PASS2_LLM_RAW_FILENAME, raw_text)
    if repaired_text is not None:
        _write_text(out_root / PASS2_LLM_REPAIRED_FILENAME, repaired_text)

    pass2_semantic = {
        "schema_version": PASS2_SEMANTIC_SCHEMA_VERSION,
        "generated_at": utc_ts(),
        "repo": {"repo_url": repo_url, "resolved_commit": resolved_commit},
        "caps": {
            "onboarding_enabled": caps.onboarding_enabled,
            "model": caps.model,
            "max_output_tokens": caps.max_output_tokens,
            "max_arch_input_chars": caps.max_arch_input_chars,
            "max_arch_files": caps.max_arch_files,
            "max_arch_chars_per_file": caps.max_arch_chars_per_file,
            "max_support_files": caps.max_support_files,
            "max_support_chars": caps.max_support_chars,
            "max_support_chars_per_file": caps.max_support_chars_per_file,
            "pack_dep_hops": caps.pack_dep_hops,
            "pack_max_dep_edges_per_file": caps.pack_max_dep_edges_per_file,
        },
        "inputs": {
            "pass1_repo_index_schema_version": pass1_repo_index.get("schema_version"),
            "pass1_repo_index_fingerprint_sha256": sha256_bytes(_stable_json_bytes(pass1_repo_index)),
            "arch_pack_fingerprint_sha256": arch_pack_obj.get("fingerprint_sha256"),
            "support_pack_fingerprint_sha256": support_pack_obj.get("fingerprint_sha256"),
        },
        "llm_output": obj,
        "llm_raw_paths": {
            "raw_text": PASS2_LLM_RAW_FILENAME,
            "repaired_text": PASS2_LLM_REPAIRED_FILENAME if repaired_text is not None else None,
        },
    }

    fp_obj = {
        "repo": pass2_semantic["repo"],
        "caps": pass2_semantic["caps"],
        "inputs": pass2_semantic["inputs"],
        "llm_output": pass2_semantic["llm_output"],
    }
    pass2_semantic["fingerprint_sha256"] = sha256_bytes(_stable_json_bytes(fp_obj))

    _write_json(out_root / PASS2_SEMANTIC_FILENAME, pass2_semantic)

    return {
        "pass2_semantic_path": str(out_root / PASS2_SEMANTIC_FILENAME),
        "pass2_arch_pack_path": str(out_root / PASS2_ARCH_PACK_FILENAME),
        "pass2_support_pack_path": str(out_root / PASS2_SUPPORT_PACK_FILENAME),
        "pass2_semantic": pass2_semantic,
    }
