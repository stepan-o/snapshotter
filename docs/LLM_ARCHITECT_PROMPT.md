## Snapshotter — LLM Architect Handoff (Current State Locked)

### What Snapshotter is for (do not relitigate)
Snapshotter is a deterministic repo scanner + semantic summarizer that produces:
- **Pass 1 (deterministic):** repo index + dependency evidence + read-plan seeds (+ now a deterministic dependency graph artifact)
- **Pass 2 (LLM-assisted):** architecture snapshot + gaps report + onboarding doc, grounded on Pass 1 facts + targeted file packs

The pipeline is already running locally and the recent investigations/fixes are **done** (server-side admin gate recognition, correct branch cloning, Pass1/Pass2 updates, graph output persisted). Continue from *current state only*.

---

## Current Pipeline Shape (High-level)

### Pass 0 — Git clone/checkout
- Clone repo, checkout requested ref (branch/tag/sha).
- Output: **resolved commit SHA** used for the run (must be correct, branch must not “mysteriously” equal main).

### Pass 1 — Deterministic scan (“source of truth”)
Produces a `repo_index.json` containing:
- `files[]` records with:
    - `import_edges[]` (evidence: kind/spec/lineno + best-effort resolution fields)
    - `deps{...}` canonical dependency buckets (internal resolved paths, unresolved specs, external specs, ambiguous specs)
    - resource edges (currently env vars), routing/auth signals, entrypoint hints, etc.
- `import_index` forward + reverse maps (derived only from Pass1 facts)
- read plan suggestions + closure seeds
- `path_aliases` rules + fingerprint

Also produces: **DEPENDENCY_GRAPH.json** (deterministic, evidence-bearing graph derived solely from Pass1 facts).

### validate_basic (server-side gate reconciliation)
- Validates that “dependencies” and module reconciliation match evidence paths (Pass1 import/index/graph), and gaps report flags missing/unsupported edges correctly.
- This was the core area of recent investigation (admin gate recognition & dependency reconciliation).

### Pass 2 — Semantic pack + LLM outputs
- Builds a selected file pack (`pass2.files[]`) from Pass1 read plan + closure.
- LLM generates:
    - `ARCHITECTURE_SUMMARY_SNAPSHOT.json`
    - `GAPS_AND_INCONSISTENCIES.json`
    - `ONBOARDING.md`
- Must be grounded on:
    - Pass1 facts + signals + dependency graph
    - The selected pack for richer explanations, not for inventing dependency structure

---

## “Source of Truth” Rules (Non-negotiable)

### Determinism contract
- Pass1 outputs must be reproducible for the same repo commit + job config.
- Pass2 must not invent structure that should come from Pass1 (especially dependencies).

### Dependency truth
- **Actionable dependency data must come from Pass1 artifacts**, not from LLM summarization.
- If the architecture snapshot wants “dependencies”, it should reference / summarize **DEPENDENCY_GRAPH.json** (or derived summaries) rather than free-text LLM guesses.

### Grounding
- The LLM should be grounded on:
    1) compact Pass1 digests (signals/facts/import_index summaries)
    2) DEPENDENCY_GRAPH.json (or a compact derived subset)
    3) the selected pack of file contents for explanation and nuance

---

## How to Run Locally (Dev Workflow)

### Typical run
- `uv run snapshotter`
- For dry-run (no uploads / faster iteration):
    - `uv run snapshotter --dry-run`

### Capture machine-readable output
- `uv run snapshotter 2>&1 | tee /tmp/snapshotter_run.json`
- Check pipeline exit status:
    - `echo $pipestatus`

### Where artifacts land (local)
- Outputs are written under:
    - `out/<repo-slug>/<timestamp>/<job-id>/...`
- Pass2 raw LLM output is saved as:
    - `PASS2_LLM_RAW_OUTPUT.txt`  
      (use this when JSON truncation happens)

---

## Running on a Different Branch / Ref (The “branch != main” rule)

### What must be true
- `requested_ref` must be applied during clone/checkout
- `resolved_commit` must reflect that ref (not silently default to main)

### Expected behavior
- A run against `branch-X` must show a different `resolved_commit` if the branch differs.
- If the UI/logs show branch selection but commit is identical, treat as a bug in ref handling or reporting.

### What to verify quickly
- In run result JSON:
    - `requested_ref`
    - `resolved_commit`
- In the checked-out repo directory:
    - `git rev-parse HEAD`
    - `git branch --show-current` (or detached HEAD expected if sha)

---

## Known Recent Failure Mode: Pass2 Truncated JSON

### Symptom
Pass2 fails with:
- “OpenAI returned truncated/incomplete JSON (likely hit max output tokens). Increase SNAPSHOTTER_LLM_MAX_OUTPUT_TOKENS and retry.”
- JSON parse error mid-output
- Raw output saved to `PASS2_LLM_RAW_OUTPUT.txt`

### What this means architecturally
- The output schema + included context is too large for the current output token cap.
- Fix should prefer **structure changes** over “just crank tokens forever”:
    - Split Pass2 generation into multiple smaller calls (modules first, then gaps, then onboarding)
    - Or make Pass2 output minimal and push heavy data to deterministic artifacts (graph, indices)

### Allowed knobs
- `SNAPSHOTTER_LLM_MAX_OUTPUT_TOKENS` (increase cautiously)
- Reduce prompt payload size:
    - limit per-file content inclusion in pack
    - include dependency graph summaries rather than full edge lists

---

## What’s “Wrong” With Dependencies (and the intended fix direction)

### Current issue framing
- The architecture snapshot “dependencies” field became garbage-valued when it was being produced by the model.
- Hard caps can hide real dependencies; generic deps aren’t useful; we want *complete + actionable* dependency data.

### Intended direction
- Make **dependency truth deterministic**:
    - Pass1 produces `DEPENDENCY_GRAPH.json`
    - Pass2 references it (or a derived compact summary), and only adds interpretation

### What “actionable” means here
- For internal code navigation:
    - internal resolved edges with evidence (from/ to / lineno / spec / kind)
- For gaps:
    - unresolved internal edges grouped by importer, with candidate hints where possible
- For external:
    - keep as a separate bucket, not mixed into architecture structure

---

## Files & Modules You’ll Touch Most

### Pass1 + graph artifacts
- `snapshotter/pass1.py` (scan + evidence + indexes + graph builder functions)
- `snapshotter/graph.py` (or orchestrator that writes/exports DEPENDENCY_GRAPH.json)
- `snapshotter/validate_basic.py` (dependency reconciliation rules)

### Pass2
- `snapshotter/pass2_semantic.py` (pack selection, grounding payload, schema enforcement, multi-call strategy if needed)

### Git ops / job wiring
- `snapshotter/git_ops.py` (clone + checkout correctness, error handling)
- `snapshotter/job.py` (job config shape)
- `snapshotter/utils.py` (hashing, timestamps, helpers)

---

## Guardrails for Future Changes (So We Don’t Regress)

### Don’t “fix” things by inventing content in Pass2
- If data is structural (imports/deps/routing/gates), it belongs in Pass1 artifacts.

### Don’t hide errors
- Checkout failures, reconciliation failures, parse failures: they must surface loudly with enough context to debug.

### Keep outputs stable
- Anything persisted as a JSON artifact should have:
    - deterministic ordering
    - stable fingerprints where useful (graph_fingerprint, alias rules fingerprint)

### Keep payload-only configuration
- Snapshotter configuration is driven only by `SNAPSHOTTER_JOB_JSON` (no convenience env vars sneaking back in).

---

## Next Work Priorities (Concrete)

### P0 — Pass2 output-token robustness
- Implement a multi-stage Pass2 generation strategy (split outputs) OR reduce payload/verbosity so JSON never truncates.
- Add sanity checks:
    - validate JSON completeness before writing final artifacts
    - if truncated, persist raw + a minimal error report (already persists raw)

### P0 — Make “dependencies” in architecture snapshot non-garbage by design
- If the architecture snapshot has a `dependencies` field:
    - it should be derived from DEPENDENCY_GRAPH.json summaries
    - not a free-form model field

### P1 — Improve unresolved internal edges usefulness
- For unresolved internal edges:
    - include “resolution candidates tried” + alias rule context (already present in edge fields)
    - optionally add a deterministic “candidate path hints” list when safe (bounded)

### P1 — Validate_basic reconciliation clarity
- When reconciliation fails:
    - error should name:
        - module
        - dependency
        - the exact evidence paths that were considered
        - what gap flag was expected but missing

---

## “Definition of Done” for Your Next PR

- Running `uv run snapshotter --dry-run` on:
    - main
    - a clearly different branch
      yields:
    - correct `resolved_commit` for each
    - Pass2 completes without JSON truncation
    - `ARCHITECTURE_SUMMARY_SNAPSHOT.json` dependencies are derived from deterministic graph artifacts
    - validate_basic passes and failure messages are precise when intentionally broken

---

## Working Style / Expectations

- Ship deterministic artifacts first; only then add LLM interpretation.
- Keep changes minimal and testable:
    - small diffs, one invariant per commit if possible
- If you change schemas:
    - bump/record schema version
    - keep backward-compat in readers (or explicitly break with a migration note)

## Prompt — For the Next LLM Architect

You are the LLM Architect working on **Snapshotter**, a deterministic repo scanner + semantic summarizer.

### Mission
1) Preserve Pass1 determinism as the single source of truth for structure (especially dependencies).
2) Make Pass2 robust (no truncated JSON) and grounded on Pass1 artifacts + a small targeted file pack.
3) Ensure branch/ref runs produce correct `resolved_commit` and artifact diffs.

### Current State (assume true; do not revisit old context)
- The pipeline runs locally via `uv run snapshotter` and `uv run snapshotter --dry-run`.
- Pass1 produces `repo_index.json` with evidence-bearing import edges, indexes, read-plan suggestions, routing/auth signals.
- A deterministic `DEPENDENCY_GRAPH.json` artifact exists and must be treated as the canonical dependency truth.
- validate_basic enforces dependency reconciliation using evidence paths/gaps rules.
- Pass2 can fail due to truncated JSON output; raw output is saved to `PASS2_LLM_RAW_OUTPUT.txt`.

### Your Tasks (do in order)
1) Remove the possibility of Pass2 JSON truncation:
    - either split Pass2 into multiple smaller LLM calls (recommended),
    - or reduce the Pass2 payload/output size while preserving required schema.
2) Fix architecture snapshot `dependencies` so it’s never garbage:
    - derive dependency summaries from `DEPENDENCY_GRAPH.json` (or a compact deterministic summary),
    - do not let the model invent dependency structure.
3) Add/Improve diagnostics:
    - if Pass2 fails JSON parse, persist raw + a compact error metadata JSON explaining where it failed.
    - if validate_basic fails, error message must include module, dependency, evidence_paths, and expected gap flags.
4) Verify branch/ref correctness:
    - run on main and a different branch and confirm `resolved_commit` differs when it should.

### Constraints
- No silent failure swallowing (especially git checkout, reconciliation, parse errors).
- Keep deterministic ordering in JSON artifacts.
- Config is payload-only (`SNAPSHOTTER_JOB_JSON`).

### Local test commands
- `uv run snapshotter --dry-run`
- `uv run snapshotter 2>&1 | tee /tmp/snapshotter_run.json`
- `echo $pipestatus`

Deliverables:
- PR with focused diffs
- brief run notes showing main vs branch commit + artifact success
- evidence that Pass2 never truncates and dependencies are derived from graph artifacts

## Runbook Appendix (Quick Debug Checklist)

### If branch output “reads fewer files”
- Check `job.filters` allow/deny rules (deny dirs, deny_file_regex, allow_exts)
- Check `job.limits` (max_files, max_total_bytes, max_file_bytes)
- Verify the branch commit actually contains the same directory structure
- Confirm `resolved_commit` is correct (not accidentally main)

### If validate_basic fails “dependency reconciliation”
- Identify the “module” and “dependency” named in the error
- Look up:
    - Pass1 `repo_index.files[].deps.import_edges` for evidence
    - `import_index.reverse_internal` for reverse evidence
    - `DEPENDENCY_GRAPH.json` for from/to/lineno evidence
- Confirm whether the dependency is:
    - internal resolved edge
    - unresolved internal candidate
    - external spec
    - or points to a file excluded by filters/limits (common source of “unsupported”)

### If Pass2 fails with truncated JSON
- Inspect `PASS2_LLM_RAW_OUTPUT.txt`
- Prefer: split Pass2 into multiple calls (modules/gaps/onboarding)
- Reduce payload size:
    - shrink file pack
    - provide graph summaries instead of full edge lists
- Increase output tokens only as a last resort