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
    # input caps (defensive; prevents accidental huge prompts)
    max_arch_input_chars: int
    max_arch_files: int
    max_arch_chars_per_file: int


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
    # Lower default: we are no longer asking the model to emit dependency/evidence appendices.
    max_output_tokens = _int_from_env("SNAPSHOTTER_LLM_MAX_OUTPUT_TOKENS", 2000)

    max_arch_input_chars = _int_from_env("SNAPSHOTTER_PASS2_MAX_ARCH_INPUT_CHARS", 240_000)
    max_arch_files = _int_from_env("SNAPSHOTTER_PASS2_MAX_ARCH_FILES", 120)
    max_arch_chars_per_file = _int_from_env("SNAPSHOTTER_PASS2_MAX_ARCH_CHARS_PER_FILE", 9000)

    return SemanticCaps(
        onboarding_enabled=onboarding_enabled,
        model=model,
        max_output_tokens=max_output_tokens,
        max_arch_input_chars=max_arch_input_chars,
        max_arch_files=max_arch_files,
        max_arch_chars_per_file=max_arch_chars_per_file,
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
                    try:
                        chunks.append(json.dumps(c_text, ensure_ascii=False))
                    except Exception:
                        pass

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

    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise Pass2SemanticError("OpenAI response parsed but is not a JSON object.")
        return obj
    except Exception:
        candidate = _extract_first_json_object_span(text)
        if not candidate:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                raise Pass2SemanticError(f"OpenAI response was not valid JSON. First 400 chars:\n{text[:400]}")
            candidate = m.group(0)

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
            "- Only fix syntax (missing commas, quotes, escaping, trailing commas, etc.).\n"
            "- If the input is clearly truncated/incomplete and cannot be repaired losslessly, still output the best-effort JSON object you can, "
            "but NEVER invent new fields beyond what is implied.\n\n"
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
# Pass1 -> deterministic facts (deps/env/signals)
# -------------------------------------------------------------------


def _repo_paths_set(repo_index: dict[str, Any]) -> set[str]:
    s: set[str] = set()
    for f in repo_index.get("files", []) or []:
        if isinstance(f, dict):
            p = f.get("path")
            if isinstance(p, str) and p:
                s.add(p)
    return s


def _language_by_path_from_repo_index(repo_index: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for f in repo_index.get("files", []) or []:
        if not isinstance(f, dict):
            continue
        p = f.get("path")
        lang = f.get("language")
        if isinstance(p, str) and p and isinstance(lang, str) and lang:
            out[p] = lang
    return out


def _signals_from_repo_index(repo_index: dict[str, Any]) -> dict[str, Any]:
    sig = repo_index.get("signals", {})
    return sig if isinstance(sig, dict) else {}


def _extract_pass1_deps(repo_index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Canonical extraction based on Pass1 v2 schema:
      file["deps"] + file["import_edges"].
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
    out: dict[str, dict[str, Any]] = {}
    files = repo_index.get("files", []) or []
    for f in files:
        if not isinstance(f, dict):
            continue
        path = f.get("path")
        if not isinstance(path, str) or not path:
            continue

        resolved_internal: set[str] = set()
        external_specs: set[str] = set()
        import_edges: list[dict[str, Any]] = []
        flags_set: set[str] = set()

        deps = f.get("deps")
        if isinstance(deps, dict):
            internal = deps.get("internal")
            external = deps.get("external")
            edges = deps.get("import_edges")
            if isinstance(internal, list):
                for x in internal:
                    if isinstance(x, str) and x.strip():
                        resolved_internal.add(x.strip())
            if isinstance(external, list):
                for x in external:
                    if isinstance(x, str) and x.strip():
                        external_specs.add(x.strip())
            if isinstance(edges, list):
                for e in edges:
                    if isinstance(e, dict):
                        import_edges.append(dict(e))

        if not import_edges:
            edges2 = f.get("import_edges")
            if isinstance(edges2, list):
                for e in edges2:
                    if isinstance(e, dict):
                        import_edges.append(dict(e))

        # Fallback old fields if needed
        if not resolved_internal:
            v = f.get("imports_resolved_internal")
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, str) and x.strip():
                        resolved_internal.add(x.strip())

        if not external_specs:
            v = f.get("imports_external")
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, str) and x.strip():
                        external_specs.add(x.strip())

        fl = f.get("flags")
        if isinstance(fl, list):
            for x in fl:
                if isinstance(x, str) and x.strip():
                    flags_set.add(x.strip())

        lang = f.get("language") if isinstance(f.get("language"), str) else None
        tdefs = f.get("top_level_defs")
        top_defs: list[str] = []
        if isinstance(tdefs, list):
            top_defs = [x for x in tdefs if isinstance(x, str) and x.strip()]

        out[path] = {
            "resolved_internal": resolved_internal,
            "external_specs": external_specs,
            "import_edges": import_edges,
            "flags": flags_set,
            "language": lang,
            "top_level_defs": top_defs,
        }

    return out


def _known_present_route_hints(repo_paths: set[str]) -> dict[str, Any]:
    """
    Tiny, high-signal presence hints to prevent 'not found in snapshot' hallucinations
    when the file exists but wasn't included in the LLM pack.
    Kept intentionally small to avoid prompt bloat.
    """
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
# File selection (architecture + onboarding)
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


def _select_files_for_architecture(
        *,
        file_contents_map: dict[str, str],
        repo_index: dict[str, Any],
        caps: SemanticCaps,
) -> dict[str, str]:
    lang_by_path = _language_by_path_from_repo_index(repo_index)
    sig = _signals_from_repo_index(repo_index)

    entrypoints: set[str] = set()
    ep = sig.get("entrypoints")
    if isinstance(ep, list):
        for it in ep:
            if isinstance(it, dict):
                p = it.get("path")
                if isinstance(p, str) and p:
                    entrypoints.add(p)

    keys = sorted(file_contents_map.keys())

    def score(p: str) -> int:
        pl = p.lower()
        s = 0

        # highest priority: discovered entrypoints
        if p in entrypoints:
            s += 1000
        if pl.endswith(("main.py", "app.py", "server.py")):
            s += 220
        if pl.endswith(("/route.ts", "/route.js", "/page.tsx", "/layout.tsx")):
            s += 220

        # known “spines”
        if pl in ("backend/main.py", "frontend/middleware.ts", "frontend/middleware.js"):
            s += 700
        if pl in ("frontend/app/layout.tsx", "frontend/app/layout.ts"):
            s += 620

        # ---- guardrails for common false "missing route" hallucinations ----
        # login route: ensure included if present in file_contents_map
        if pl.startswith("frontend/app/login/"):
            s += 900
        if pl == "frontend/app/login/page.tsx":
            s += 1200
        if pl == "frontend/app/login/loginpageclient.tsx":
            s += 800

        # case-studies route: ensure included if present in file_contents_map
        if "frontend/app/case-studies/" in pl:
            s += 800
        if pl == "frontend/app/case-studies/page.tsx":
            s += 1100
        # -------------------------------------------------------------------

        # backend clusters
        if pl.startswith("backend/routers/"):
            s += 480
        if pl in ("backend/security.py", "backend/models.py", "backend/db.py", "backend/config.py"):
            s += 420
        if pl.startswith("backend/migrations/"):
            s += 190
        if pl.startswith("backend/scripts/"):
            s += 140

        # frontend clusters
        if "/app/api/" in pl and pl.endswith(("/route.ts", "/route.js")):
            s += 420
        if "/app/" in pl and pl.endswith("/layout.tsx"):
            s += 360
        if "/app/" in pl and pl.endswith("/page.tsx"):
            s += 300
        if pl.startswith("frontend/lib/"):
            s += 260
        if pl.startswith("frontend/components/"):
            s += 200

        # docs + project metadata
        if pl.endswith("readme.md") or pl == "readme.md":
            s += 320
        if pl.startswith("docs/"):
            s += 260
        if pl.endswith(".md"):
            s += 120

        # runtime/config files
        if pl.endswith(("pyproject.toml", "alembic.ini", "package.json", "next.config.ts", "next.config.js")):
            s += 220
        if "eslint" in pl or pl.endswith(("makefile", "uv.lock", "package-lock.json", "tsconfig.json", "jsconfig.json")):
            s += 120
        if pl.endswith((".env.example", ".env", "dockerfile", "docker-compose.yml", "docker-compose.yaml")):
            s += 140

        lang = lang_by_path.get(p, "")
        if lang in ("python", "typescript", "javascript"):
            s += 15

        return s

    ranked = sorted(keys, key=lambda p: (-score(p), p))

    out: dict[str, str] = {}
    total = 0

    for p in ranked:
        if len(out) >= caps.max_arch_files:
            break

        c = file_contents_map.get(p, "")
        if not isinstance(c, str):
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

    # floor: ensure we have enough breadth to summarize architecture
    if len(out) < 12:
        for p in keys:
            if len(out) >= min(24, caps.max_arch_files):
                break
            if p in out:
                continue
            c = file_contents_map.get(p, "")
            if not isinstance(c, str):
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
        file_contents_map: dict[str, str], *, max_files: int = 28, max_total_chars: int = 120_000
) -> dict[str, str]:
    keys = sorted(file_contents_map.keys())

    def score(p: str) -> int:
        p_low = p.lower()
        s = 0
        if p_low.endswith("readme.md") or p_low == "readme.md":
            s += 220
        if p_low.startswith("docs/") or "/docs/" in p_low:
            s += 180
        if p_low.endswith(".md"):
            s += 140
        if p_low.endswith(("pyproject.toml", "alembic.ini")):
            s += 130
        if p_low.endswith(("makefile", "uv.lock")):
            s += 110
        if "next.config" in p_low or "eslint" in p_low:
            s += 100
        if p_low.endswith(("backend/main.py", "backend/config.py", "backend/security.py")):
            s += 85
        if p_low.endswith(("frontend/app/layout.tsx", "frontend/middleware.ts")):
            s += 80
        if p_low.endswith(("package.json", "tsconfig.json", "jsconfig.json")):
            s += 75

        # keep these high so gaps pass has grounding if it wants to mention them
        if p_low.startswith("frontend/app/login/"):
            s += 120
        if "frontend/app/case-studies/" in p_low:
            s += 110

        if p_low.endswith((".ts", ".tsx")):
            s += 10
        if p_low.endswith((".py",)):
            s += 10
        return s

    ranked = sorted(keys, key=lambda p: (-score(p), p))

    out: dict[str, str] = {}
    total = 0
    for p in ranked:
        if len(out) >= max_files:
            break
        c = file_contents_map.get(p, "")
        if not isinstance(c, str):
            continue
        remaining = max_total_chars - total
        if remaining <= 0:
            break
        if len(c) > remaining:
            c = c[:remaining]
        out[p] = c
        total += len(c)

    return out


# -------------------------------------------------------------------
# Prompt payloads (LLM output is SMALL: modules + anchor_paths + short semantics)
# -------------------------------------------------------------------


def _build_architecture_payload(
        *,
        repo_url: str,
        resolved_commit: str,
        job_id: str,
        repo_index: dict[str, Any],
        file_contents_map: dict[str, str],
        caps: SemanticCaps,
) -> dict[str, Any]:
    arch_files = _select_files_for_architecture(file_contents_map=file_contents_map, repo_index=repo_index, caps=caps)
    files_pack: list[dict[str, Any]] = [{"path": p, "content": c} for p, c in arch_files.items()]

    # signals are useful, cheap, and deterministic
    signals = _signals_from_repo_index(repo_index)
    path_aliases = repo_index.get("path_aliases", {})
    if not isinstance(path_aliases, dict):
        path_aliases = {}

    repo_paths = _repo_paths_set(repo_index)
    presence_hints = _known_present_route_hints(repo_paths)

    # Keep pass1 counts/signals only; do NOT inject deps_by_file into the prompt.
    return {
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
            # critical: stop claiming 'missing from snapshot' when it's only missing from the pack
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
        },
        # tiny presence hints only (do not dump full repo path list)
        "repo_presence_hints": presence_hints,
        "pass2": {"files": files_pack},
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
    }


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
) -> dict[str, Any]:
    support_files = _select_supporting_files_for_gaps_and_onboarding(file_contents_map)

    modules_summary: list[dict[str, Any]] = []
    for m in arch_modules:
        if not isinstance(m, dict):
            continue
        modules_summary.append(
            {
                "name": m.get("name"),
                "type": m.get("type"),
                "summary": m.get("summary"),
                "anchor_paths": m.get("anchor_paths", m.get("evidence_paths", [])),
                "entrypoints": m.get("entrypoints", []),
                "where_to_change": m.get("where_to_change", []),
                "risk_notes": m.get("risk_notes", []),
            }
        )

    signals = _signals_from_repo_index(repo_index)
    path_aliases = repo_index.get("path_aliases", {})
    if not isinstance(path_aliases, dict):
        path_aliases = {}

    env_vars = signals.get("env_vars", [])
    pkg = signals.get("package_json", {})

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
                "env_vars": env_vars,
                "package_json": pkg,
            },
        },
        "repo_presence_hints": presence_hints,
        "architecture_summary": {
            "modules": modules_summary,
            "uncertainties": arch_uncertainties,
        },
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
# Deterministic derivations (NO LLM dependency enforcement)
# -------------------------------------------------------------------


def _derive_deps_from_anchor_paths(
        modules: list[dict[str, Any]],
        deps_by_file: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Deterministically attach dependencies to modules from pass1 deps_by_file, based on anchor_paths.

    deps = sorted(unique internal repo paths) + sorted(unique external specs)

    Also sets evidence_paths := anchor_paths for downstream compatibility.
    """
    for m in modules:
        anchors = m.get("anchor_paths")
        if not isinstance(anchors, list):
            anchors = m.get("evidence_paths")
        if not isinstance(anchors, list):
            anchors = []
        anchors = [p for p in anchors if isinstance(p, str) and p.strip()]

        internal: set[str] = set()
        external: set[str] = set()
        for p in anchors:
            info = deps_by_file.get(p)
            if not info:
                continue
            internal |= set(info.get("resolved_internal", set()) or [])
            external |= set(info.get("external_specs", set()) or [])

        m["anchor_paths"] = anchors
        m["evidence_paths"] = list(anchors)  # compat
        m["dependencies"] = sorted(internal) + sorted(external)

    return modules


# -------------------------------------------------------------------
# Deterministic gap scans (grounded, non-LLM)
# -------------------------------------------------------------------


def _deterministic_gap_scan(repo_index: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    files = repo_index.get("files", []) or []
    sig = _signals_from_repo_index(repo_index)

    # (1) unresolved internal imports flagged by pass1
    unresolved_paths: list[str] = []
    parse_failed: list[str] = []
    for f in files:
        if not isinstance(f, dict):
            continue
        p = f.get("path")
        flags = f.get("flags")
        if not isinstance(p, str) or not p:
            continue
        if isinstance(flags, list):
            if "import_unresolved" in flags:
                unresolved_paths.append(p)
            if "python_parse_failed" in flags or "js_ts_parse_failed" in flags:
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

    # (2) missing entrypoint hints
    eps = sig.get("entrypoints", [])
    if not isinstance(eps, list) or len(eps) == 0:
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
        modules = []

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

        # enforce 3–10 anchor paths if possible
        if len(anchors) > 10:
            anchors = anchors[:10]
        mm["anchor_paths"] = anchors

        # summary (required)
        summary = mm.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            # salvage from responsibilities if present
            resp = mm.get("responsibilities")
            if isinstance(resp, list):
                resp2 = [r for r in resp if isinstance(r, str) and r.strip()]
                if resp2:
                    summary = "; ".join(resp2[:3])
            mm["summary"] = summary.strip() if isinstance(summary, str) and summary.strip() else "unknown"

        # responsibilities (optional but useful)
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
                    "description": f"Module '{mm.get('name','unknown')}' lacks anchor_paths in files_read; summary/responsibilities set to unknown.",
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

        # Do NOT accept/require deps from LLM output
        mm["dependencies"] = []
        # downstream compat (populated later, but keep field stable)
        mm["evidence_paths"] = list(anchors)

        out_modules.append(mm)

    return out_modules, out_uncertainties


# -------------------------------------------------------------------
# Gaps normalization / dedupe + false-positive rewrite
# -------------------------------------------------------------------


def _normalize_gaps_object(gaps: Any, *, job_id: str) -> dict[str, Any]:
    if not isinstance(gaps, dict):
        gaps = {}
    out = dict(gaps)
    out.setdefault("generated_at", None)
    out["job_id"] = job_id
    items = out.get("items")
    if not isinstance(items, list):
        items = []
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
    """
    If an item claims route/docs missing but those referenced files exist (in allowed_paths),
    rewrite it to a check-style item instead of 'missing'.
    """
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


def _rewrite_known_false_missing_items(items: list[dict[str, Any]], repo_paths: set[str]) -> list[dict[str, Any]]:
    """
    Downgrade/convert known false positives caused by prompt coverage rather than repo reality,
    notably /login and /case-studies.
    """
    has_login = ("frontend/app/login/page.tsx" in repo_paths) or ("frontend/app/login/LoginPageClient.tsx" in repo_paths)
    has_case_studies = "frontend/app/case-studies/page.tsx" in repo_paths

    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        desc = (it.get("description") or "")
        desc_l = desc.lower()
        t = (it.get("type") or "").strip()

        def _files_list() -> list[str]:
            fi = it.get("files_involved")
            if not isinstance(fi, list):
                return []
            return [p for p in fi if isinstance(p, str) and p]

        # /login false "missing"
        if has_login and ("/login" in desc_l or "login page" in desc_l):
            if "not found" in desc_l or "no route" in desc_l or "missing" in desc_l or t in ("incomplete_extraction", "missing_docs"):
                it2 = dict(it)
                it2["type"] = "prompt_coverage_gap"
                it2["severity"] = "low"
                it2["description"] = (
                    "Login route exists in snapshot (frontend/app/login/page.tsx + LoginPageClient.tsx). "
                    "Prior pass likely omitted these files from the LLM evidence pack; treat as prompt coverage issue, not missing implementation."
                )
                files = _files_list()
                for p in ("frontend/app/login/page.tsx", "frontend/app/login/LoginPageClient.tsx"):
                    if p in repo_paths and p not in files:
                        files.append(p)
                it2["files_involved"] = files
                out.append(it2)
                continue

        # /case-studies false "missing"
        if has_case_studies and ("/case-studies" in desc_l or "case studies" in desc_l):
            if "not found" in desc_l or "no route" in desc_l or "missing" in desc_l or t in ("missing_docs", "incomplete_extraction"):
                it2 = dict(it)
                it2["type"] = "prompt_coverage_gap"
                it2["severity"] = "low"
                it2["description"] = (
                    "Case studies route exists in snapshot (frontend/app/case-studies/page.tsx). "
                    "Prior pass likely omitted this file from the LLM evidence pack; treat as prompt coverage issue, not missing route."
                )
                files = _files_list()
                if "frontend/app/case-studies/page.tsx" not in files:
                    files.append("frontend/app/case-studies/page.tsx")
                it2["files_involved"] = files
                out.append(it2)
                continue

        out.append(it)

    return out


def _rewrite_known_false_missing_uncertainties(uncertainties: list[dict[str, Any]], repo_paths: set[str]) -> list[dict[str, Any]]:
    """
    Same as gaps rewrite but for architecture 'uncertainties' objects.
    """
    has_login = ("frontend/app/login/page.tsx" in repo_paths) or ("frontend/app/login/LoginPageClient.tsx" in repo_paths)
    has_case_studies = "frontend/app/case-studies/page.tsx" in repo_paths

    out: list[dict[str, Any]] = []
    for u in uncertainties:
        if not isinstance(u, dict):
            continue
        desc = (u.get("description") or "")
        desc_l = desc.lower()
        typ = (u.get("type") or "").strip()

        files_involved = u.get("files_involved")
        if not isinstance(files_involved, list):
            files_involved = []
        files_involved = [p for p in files_involved if isinstance(p, str) and p]

        # login uncertainty false positive
        if has_login and ("/login" in desc_l or "login page" in desc_l) and ("not found" in desc_l or "unknown" in desc_l):
            u2 = dict(u)
            u2["type"] = "prompt_coverage_gap"
            u2["description"] = (
                "Login route exists in snapshot (frontend/app/login/page.tsx + LoginPageClient.tsx). "
                "If this was flagged as missing/unknown, it was likely not included in the provided file contents."
            )
            for p in ("frontend/app/login/page.tsx", "frontend/app/login/LoginPageClient.tsx"):
                if p in repo_paths and p not in files_involved:
                    files_involved.append(p)
            u2["files_involved"] = files_involved
            u2["suggested_questions"] = [
                "Ensure login route files are included in pass2 selection and cited as anchor_paths where relevant."
            ]
            out.append(u2)
            continue

        # case-studies uncertainty false positive
        if has_case_studies and ("/case-studies" in desc_l or "case studies" in desc_l) and ("not found" in desc_l or "unknown" in desc_l):
            u2 = dict(u)
            u2["type"] = "prompt_coverage_gap"
            u2["description"] = (
                "Case studies route exists in snapshot (frontend/app/case-studies/page.tsx). "
                "If this was flagged as missing/unknown, it was likely not included in the provided file contents."
            )
            if "frontend/app/case-studies/page.tsx" not in files_involved:
                files_involved.append("frontend/app/case-studies/page.tsx")
            u2["files_involved"] = files_involved
            u2["suggested_questions"] = [
                "Ensure case-studies route file is included in pass2 selection and cited where navigation references it."
            ]
            out.append(u2)
            continue

        out.append(u)

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
    Drop-in replacement behavior:
      - Pass2 LLM outputs ONLY: semantic modules + anchor_paths + short text
      - evidence_paths and dependencies are derived deterministically after the fact
      - no LLM dependency reconciliation, no dependency_mismatch gap spam

    Fixes included:
      - Boost selection for /login and /case-studies files so they land in the LLM pack
      - Explicit prompt rule: don't claim "missing from snapshot" if merely missing from provided contents
      - Deterministic rewrite of known false-positive 'missing' gaps/uncertainties for /login + /case-studies
    """
    caps = _semantic_caps_from_env()

    # -------------------------
    # Architecture semantics
    # -------------------------
    arch_payload = _build_architecture_payload(
        repo_url=repo_url,
        resolved_commit=resolved_commit,
        job_id=job_id,
        repo_index=repo_index,
        file_contents_map=file_contents_map,
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
        modules=arch_obj.get("modules", []),
        uncertainties=arch_obj.get("uncertainties", []),
        allowed_paths=allowed_paths,
    )

    # deterministically attach deps (and evidence_paths := anchor_paths for compat)
    deps_by_file = _extract_pass1_deps(repo_index)
    enforced_modules = _derive_deps_from_anchor_paths(enforced_modules, deps_by_file)

    # rewrite known false-positive uncertainties (prompt coverage vs repo reality)
    enforced_uncertainties = _rewrite_known_false_missing_uncertainties(enforced_uncertainties, repo_paths)

    arch_out: dict[str, Any] = {
        "modules": enforced_modules,
        "uncertainties": enforced_uncertainties,
    }

    # -------------------------
    # Deterministic gaps (non-LLM)
    # -------------------------
    deterministic_gap_items = _deterministic_gap_scan(repo_index)

    # -------------------------
    # LLM gaps + onboarding (no deterministic dumping)
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
    )
    gaps_prompt = _gaps_onboarding_prompt_text(gaps_payload)

    gaps_obj = _openai_call_json(
        prompt=gaps_prompt,
        model=caps.model,
        max_output_tokens=caps.max_output_tokens,
        system="You are a precise repo auditor. Output JSON only.",
    )

    gaps_raw = _normalize_gaps_object(gaps_obj.get("gaps"), job_id=job_id)
    onboarding_md = gaps_obj.get("onboarding_md") or ""
    if not caps.onboarding_enabled:
        onboarding_md = ""

    merged_items = list(deterministic_gap_items)
    llm_items = gaps_raw.get("items", [])
    if isinstance(llm_items, list):
        for it in llm_items:
            if isinstance(it, dict):
                merged_items.append(it)

    # rewrite known false positives caused by prompt coverage
    merged_items = _rewrite_known_false_missing_items(merged_items, repo_paths)

    # keep existing rewrite for 'missing_route' items (but operate on allowed_paths)
    merged_items = _fix_false_missing_route_gap_items(merged_items, allowed_paths)

    gaps_out = dict(gaps_raw)
    gaps_out["items"] = _dedupe_gap_items(merged_items)

    if not isinstance(onboarding_md, str):
        raise Pass2SemanticError("onboarding_md is not a string after processing.")
    if not isinstance(gaps_out, dict):
        raise Pass2SemanticError("gaps output has invalid type.")
    if not isinstance(arch_out, dict):
        raise Pass2SemanticError("architecture output has invalid type.")

    return arch_out, gaps_out, onboarding_md