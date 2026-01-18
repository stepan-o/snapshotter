# Snapshotter System Prompt for LLM Architects

## **Overview & Purpose**

**Project**: `snapshotter` - A deterministic 2-pass repository analysis system

**Core Architecture**:
```
Repo → Clone → Pass1 (Deterministic Scan) → Read Plan → Pass2 (LLM Semantic Analysis) → Artifacts
```

**Primary Goal**: Generate an "onboarding package" for LLM architects to understand evolving codebases through structured artifacts:
1. `PASS1_REPO_INDEX.json` - Deterministic scan results
2. `DEPENDENCY_GRAPH.json` - Internal dependency relationships
3. `PASS2_SEMANTIC.json` - LLM-generated architecture analysis
4. `ARCHITECTURE_SUMMARY_SNAPSHOT.json` - Normalized architecture view
5. `ONBOARDING.md` - Human-readable summary

---

## **System Contract & Design Philosophy**

### **Core Principles**
1. **Deterministic First Pass**: Pass1 produces identical output for identical repo state
2. **LLM-Augmented Second Pass**: Pass2 uses LLM only where human-like understanding adds value
3. **Strict Artifact Contracts**: Schemas are locked, no back-compat, fail-fast validation
4. **Single Source of Truth**: Artifact paths defined once in `graph.py`'s `build_local_paths()`
5. **Production-Ready**: S3 upload, error staging, configurable caps

### **Locked Contracts**
- `PASS1_REPO_INDEX.json` schema version: `"pass1_repo_index.v1"`
- `PASS2_SEMANTIC.json` schema version: `"pass2_semantic.v1"`
- `graph.py` stages are canonical: `STAGE_PARSE_JOB`, `STAGE_CLONE`, etc.
- `local_paths` keys are defined in `build_local_paths()` and used everywhere

---

## **Architecture Components**

### **1. Orchestration Layer (`graph.py`)**

**State Machine** (LangGraph-based):
```python
SnapshotterState = TypedDict with:
    payload: dict               # Raw job JSON
    config: RuntimeConfig       # dry_run, aws_region
    job: Job                    # Validated job object
    local_paths: dict[str, str] # SINGLE SOURCE OF TRUTH for artifact paths
    repo_index: dict            # Pass1 output
    read_plan: list[str]        # Files for Pass2 analysis
    file_contents_map: dict[str, str]  # Actual file content for Pass2
```

**Key Functions**:
- `build_local_paths(out_dir)` - Defines ALL artifact file paths
- `get_validation_paths(local_paths)` - Extracts validation subset
- `build_artifact_manifest(local_paths)` - Creates fingerprint manifest
- Stage-aware error handling via `SnapshotterStageError`

**Node Flow**:
```
load_job → clone_repo → pass1_build_index → pass2_make_read_plan → 
pass2_fetch_files → pass2_generate_outputs → pass1_manifest → 
validate_basic → upload_artifacts → emit_result
```

---

### **2. Pass1: Deterministic Repo Scanner (`pass1.py`)**

**Input**: Repo directory, Job config
**Output**: `PASS1_REPO_INDEX.json`, `DEPENDENCY_GRAPH.json`

**Core Capabilities**:
- Language detection by extension (Python, TypeScript, JavaScript, etc.)
- AST parsing for Python imports
- Regex-based parsing for JS/TS imports
- TSConfig alias resolution with extends chain support
- Workspace package detection (monorepo support)
- Python import root discovery (pyproject.toml, src layout)
- Entrypoint signal detection (FastAPI, Next.js routes, etc.)
- Environment variable extraction (os.getenv, process.env)

**Import Resolution Rules**:
1. **Python**: Relative imports (`.`), absolute via discovered roots
2. **JS/TS**: Relative (`./`), rooted (`/`), aliases, workspace packages, baseUrl
3. **Internal Only**: Only repo-internal resolved paths become dependency edges
4. **External Classification**: Non-resolvable imports marked external (architecture out of scope)

**Read Plan Generation**:
- Uses `read_plan.py`'s `suggest_files_to_read()`
- Returns `{"closure_seeds": [...], "candidates": [...]}` structure
- Closure seeds: routing/auth files, entrypoints, dependency-adjacent files
- Candidates: scored by patterns, import graph position, file type

---

### **3. Read Plan Logic (`read_plan.py`)**

**Deterministic Scoring**:
```python
# High-priority patterns (architecture signals)
HIGH_PATTERNS = [
    (r"app/(layout|page)\.(t|j)sx?", 80, "next_root_layout_or_page"),
    (r"middleware\.(t|j)s$", 80, "middleware"),
    (r"(^|/)(main|app|server)\.py$", 60, "backend_entrypoint"),
    (r"(^|/)schema\.prisma$", 70, "prisma_schema"),
]

# Deprioritized patterns
NEG_PATTERNS = [
    (r"(^|/)__tests__(/|$)", 60, "tests"),
    (r"\.(spec|test)\.(t|j)sx?$", 60, "tests"),
]

# Graph signals
score += 2.0 * imports_count   # Fan-out
score += 10.0 * imported_by_count  # Fan-in
```

**Bucket Caps System**:
```python
DEFAULT_CAPS = {
    "app/": 35,
    "pages/": 20,
    "src/": 25,
    "backend/": 25,
    "other": 9999,
}
```

**Output Contract**:
```json
{
  "max_files_to_read_default": 120,
  "candidates": [
    {"path": "frontend/app/page.tsx", "score": 182.5, "reasons": [...]}
  ]
}
```

---

### **4. Pass2: Semantic Analysis (`pass2_semantic.py`)**

**Input**: Pass1 repo_index, file contents map
**Output**: `PASS2_SEMANTIC.json`, `PASS2_ARCH_PACK.json`, `PASS2_SUPPORT_PACK.json`

**Evidence Pack Strategy**:
1. **Architecture Pack**: Files needed to understand system architecture
    - Selected by: closure_seeds, entrypoints, dependency hops
    - Caps: `max_arch_files=120`, `max_arch_chars_per_file=9000`

2. **Support Pack**: Files for gaps/onboarding context
    - Selected by: READMEs, configs, docs, supplemental files
    - Caps: `max_support_files=28`, `max_support_chars_per_file=9000`

**LLM Integration**:
- Uses OpenAI Responses API with `text={"format": {"type": "json_object"}}`
- JSON repair system for malformed responses
- Prompt includes: repo metadata, pass1 signals, dependency summary, both packs
- Output schema enforced in prompt

**Generated Schema** (`PASS2_SEMANTIC.json`):
```json
{
  "schema_version": "pass2_semantic.v1",
  "generated_at": "ISO8601",
  "repo": {"repo_url": "...", "resolved_commit": "..."},
  "caps": {...},
  "inputs": {
    "pass1_repo_index_schema_version": "...",
    "pass1_repo_index_fingerprint_sha256": "...",
    "arch_pack_fingerprint_sha256": "...",
    "support_pack_fingerprint_sha256": "..."
  },
  "llm_output": {
    "summary": {
      "primary_stack": "string|null",
      "architecture_overview": "string",
      "key_components": ["string"],
      "data_flows": ["string"],
      "auth_and_routing_notes": ["string"],
      "risks_or_gaps": ["string"]
    },
    "evidence": {
      "arch_pack_paths": ["string"],
      "support_pack_paths": ["string"],
      "notable_files": [{"path": "string", "why": "string"}]
    }
  },
  "fingerprint_sha256": "..."
}
```

---

### **5. Artifact Generation & Normalization**

**Architecture Snapshot Normalization** (`graph.py`):
```python
def _normalize_architecture_snapshot(...):
    # Ensures every module has:
    # - evidence_paths (non-empty list from files_read)
    # - responsibilities (non-empty list or ["unknown"])
    # - dependencies (list)
    # Adds uncertainties for ungrounded modules
```

**Onboarding Generation**:
- Extracted from `pass2_semantic.llm_output.summary`
- Markdown structure: Overview, Key Components, Data Flows, Auth Notes
- Always generated (caps.onboarding_enabled defaults True)

**Gaps Extraction**:
- Direct from `pass2_semantic.llm_output.summary.risks_or_gaps`
- Preserves LLM's risk assessment

---

### **6. Validation System (`validate_basic.py`)**

**Hard Contract Validation**:
```python
def validate_basic_artifacts(local_paths: dict[str, str | None]):
    # Required keys (from graph.py's get_validation_paths):
    required = ["repo_index", "artifact_manifest", "architecture_snapshot", 
                "gaps", "onboarding", "pass2_semantic"]
    
    # Each artifact has schema-specific validation:
    _validate_repo_index(obj)        # Files list, read_plan structure
    _validate_architecture_snapshot(obj)  # Coverage stats, files_read
    _validate_pass2_semantic(obj)    # Full llm_output schema
    _validate_artifact_manifest(obj) # Includes other artifacts
    _validate_onboarding(path)       # Non-empty markdown
    _validate_cross_artifact_consistency(...)  # Repo/commit consistency
```

**Key Validation Rules**:
1. `pass2_semantic.llm_output.summary` must have all 6 expected keys
2. `architecture_snapshot.coverage` must have integer stats
3. `artifact_manifest` must include all core artifacts
4. Cross-artifact: repo URLs and commit hashes must match
5. `onboarding.md` must have ≥50 characters of content

---

## **Configuration & Environment**

### **Job Schema (`job.py`)**
```python
class Job(BaseModel):
    job_id: str | None            # Auto-generated from fingerprint
    repo_url: str
    ref: str
    mode: Literal["full", "light"]
    limits: Limits                # File/byte caps
    filters: Filters              # Deny dirs, extensions
    output: Output               # S3 bucket/prefix
    metadata: Metadata
    
    # Derived
    repo_slug: str               # org__repo from URL
    timestamp_utc: str          # Generation time
    s3_job_prefix() -> str      # output.s3_prefix/repo_slug/timestamp/job_id
```

### **Environment Variables**
```bash
# Required
OPENAI_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

# Pass2 Configuration (Job-first, env fallback)
SNAPSHOTTER_LLM_MODEL=gpt-4.1-mini
SNAPSHOTTER_PASS2_MAX_FILES=120
SNAPSHOTTER_PASS2_MAX_TOTAL_CHARS=250000
SNAPSHOTTER_PASS2_MAX_CHARS_PER_FILE=0  # 0 = disabled
SNAPSHOTTER_PASS2_ONBOARDING=1

# Job Payload (can be in .env)
SNAPSHOTTER_JOB_JSON='{"repo_url": "...", "ref": "...", ...}'
```

### **CLI Interface (`cli.py`)**
```bash
# Basic usage
snapshotter --dotenv --dry-run

# With explicit .env
snapshotter --dotenv .env.local --dotenv-override

# AWS region override
snapshotter --dotenv --aws-region eu-west-1
```

**Dotenv Parsing Rules**:
- Minimal parser, no variable expansion (`${...}`)
- Supports `KEY=VALUE`, `export KEY=VALUE`
- Quotes preserved, inline comments stripped
- No shell-like escaping beyond basic quotes

---

## **Artifact Ecosystem**

### **Canonical Artifacts (Uploaded to S3)**
1. `PASS1_REPO_INDEX.json` - Primary scan output
2. `DEPENDENCY_GRAPH.json` - Derived from import_index.by_path
3. `PASS2_SEMANTIC.json` - LLM analysis with fingerprints
4. `ARCHITECTURE_SUMMARY_SNAPSHOT.json` - Normalized view
5. `GAPS_AND_INCONSISTENCIES.json` - Risks assessment
6. `ONBOARDING.md` - Human-readable summary
7. `artifact_manifest.json` - Fingerprint manifest

### **Debug Artifacts (Local Only)**
1. `PASS2_LLM_RAW_OUTPUT.txt` - Raw LLM response
2. `PASS2_LLM_REPAIRED_OUTPUT.txt` - Repaired JSON if needed
3. `PASS2_ARCH_PACK.json` - Architecture evidence pack
4. `PASS2_SUPPORT_PACK.json` - Supporting context pack
5. `repo_index.json` - Convenience alias for PASS1_REPO_INDEX

### **Artifact Relationships**
```
PASS1_REPO_INDEX.json
    ├── import_index.by_path → DEPENDENCY_GRAPH.json
    ├── files[] → evidence for read_plan
    └── read_plan → input to pass2 selection

PASS2_SEMANTIC.json
    ├── llm_output.summary → ONBOARDING.md (extracted)
    ├── llm_output.summary.risks_or_gaps → GAPS_AND_INCONSISTENCIES.json
    └── llm_output → ARCHITECTURE_SUMMARY_SNAPSHOT.json (normalized)

artifact_manifest.json
    └── fingerprints all canonical artifacts
```

---

## **Error Handling & Recovery**

### **Stage-Aware Errors**
```python
class SnapshotterStageError(RuntimeError):
    def __init__(self, stage: str, inner: Exception):
        # Reported as: {"stage": "validate_basic", "error_code": "SNAPSHOTTER_FAILED_VALIDATE_BASIC"}
```

**Stage Pipeline**:
```
STAGE_INIT → STAGE_PARSE_JOB → STAGE_CLONE → STAGE_PASS1_REPO_INDEX → 
STAGE_PASS2_MAKE_READ_PLAN → STAGE_PASS2_FETCH_FILES → 
STAGE_PASS2_GENERATE_OUTPUTS → STAGE_PASS1_MANIFEST → 
STAGE_VALIDATE_BASIC → STAGE_UPLOAD → STAGE_EMIT_RESULT → 
STAGE_DONE / STAGE_DONE_DRY_RUN
```

### **Recovery Strategies**
1. **Dry-run first**: Always test with `--dry-run` before actual upload
2. **Artifact validation**: `validate_basic` catches schema issues early
3. **LLM JSON repair**: Automatic repair attempt for malformed responses
4. **Deterministic retry**: Same input → same output (except timestamps)

---

## **Implementation Details & Patterns**

### **Deterministic Patterns**
1. **Sorting**: Always sort collections before serialization
2. **JSON Canonicalization**: `json.dumps(..., sort_keys=True, separators=(",", ":"))`
3. **Path Normalization**: `norm_relpath()` converts `\` to `/`, strips `./`
4. **Fingerprinting**: `stable_json_fingerprint_sha256()` strips volatile keys

### **File Reading Strategies**
1. **Pass1**: Bounded by `job.limits.max_file_bytes`, binary detection
2. **Pass2**: Stream reading with UTF-8 replacement decoder
3. **Truncation**: Head + tail strategy for large files
4. **Binary Avoidance**: Extension-based and content-based detection

### **Dependency Resolution**
1. **Python**: Multi-root discovery (pyproject.toml, src layout)
2. **JS/TS**: Config chain resolution (tsconfig extends), workspace packages
3. **Internal Only**: External dependencies filtered out at architecture level
4. **Unresolved Tracking**: Internal unresolved specs recorded separately

---

## **Extension Points & Future Work**

### **Planned Enhancements**
1. **Multi-LLM Provider**: Abstract LLM interface for DeepSeek, Anthropic, etc.
2. **Incremental Analysis**: Git diff-based updates for changed files only
3. **Artifact Diffing**: Compare fingerprints between snapshots
4. **Quality Metrics**: Score LLM outputs for consistency/completeness
5. **Web Dashboard**: S3-hosted HTML viewer for snapshots

### **Configuration Evolution**
```python
# Future Job schema extension
class Pass2Config(BaseModel):
    model: str = "gpt-4.1-mini"
    max_output_tokens: int = 2000
    provider: Literal["openai", "deepseek", "anthropic"] = "openai"
    temperature: float = 0.0

class Job(BaseModel):
    # ... existing fields ...
    pass2: Pass2Config = Field(default_factory=Pass2Config)
```

### **Architecture Improvements**
1. **Caching Layer**: Cache dependency graph computations
2. **Parallel Processing**: Concurrent file reading where safe
3. **Streaming Uploads**: Direct S3 streaming for large artifacts
4. **Plugin System**: Language-specific parsers as plugins

---

## **Usage Examples**

### **Basic Run**
```bash
# 1. Create .env with credentials and job
cat > .env << 'EOF'
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
OPENAI_API_KEY=...
SNAPSHOTTER_JOB_JSON='{
  "repo_url": "https://github.com/org/repo.git",
  "ref": "main",
  "mode": "full",
  "output": {
    "s3_bucket": "my-snapshots",
    "s3_prefix": "scans"
  }
}'
EOF

# 2. Dry run test
snapshotter --dotenv --dry-run

# 3. Actual run
snapshotter --dotenv
```

### **Artifact Inspection**
```python
# Load and examine artifacts
import json

with open('out/.../PASS1_REPO_INDEX.json') as f:
    pass1 = json.load(f)
    print(f"Scanned {pass1['counts']['files_included']} files")
    print(f"Read plan: {len(pass1['read_plan']['candidates'])} candidates")

with open('out/.../PASS2_SEMANTIC.json') as f:
    pass2 = json.load(f)
    summary = pass2['llm_output']['summary']
    print(f"Primary stack: {summary['primary_stack']}")
    print(f"Key components: {len(summary['key_components'])}")
```

---

## **Troubleshooting Guide**

### **Common Issues**
1. **Validation fails**: Check `validate_basic.py` error for exact missing key
2. **LLM timeouts**: Increase `SNAPSHOTTER_LLM_MAX_OUTPUT_TOKENS`
3. **Missing files**: Verify `job.filters.allow_exts` includes needed extensions
4. **S3 upload fails**: Check bucket policy requires `ServerSideEncryption: AES256`

### **Debug Commands**
```bash
# Check artifact structure
python -c "import json; d=json.load(open('out/.../PASS2_SEMANTIC.json')); print(list(d['llm_output']['summary'].keys()))"

# Verify local_paths
python -c "from snapshotter.graph import build_local_paths; print(build_local_paths('out/test'))"

# Test validation
python -c "from snapshotter.validate_basic import validate_basic_artifacts; validate_basic_artifacts({'pass2_semantic': 'out/.../PASS2_SEMANTIC.json', ...})"
```

---

## **Contributor Guidelines**

### **Code Standards**
1. **Type Hints**: Required for all function signatures
2. **Error Messages**: Clear, actionable, include context
3. **Determinism**: No randomness, stable sorts, canonical JSON
4. **Contract First**: Define schema before implementation
5. **Single Source**: No duplicate definitions (paths, constants, schemas)

### **Testing Requirements**
1. **New Artifacts**: Update `graph.py` `build_local_paths()` and `validate_basic.py`
2. **Schema Changes**: Update both producer and validator
3. **Integration Test**: Full pipeline dry-run before merge
4. **Edge Cases**: Empty repos, binary files, malformed configs

### **Change Workflow**
```
1. Update contract/schema definition
2. Update producer (pass1/pass2/graph)
3. Update consumer (validate_basic)
4. Test with --dry-run
5. Update system prompt (this document)
```

---

## **System Limitations & Assumptions**

### **Current Limitations**
1. **Single Repo**: No cross-repo dependency analysis
2. **Language Scope**: Primary support for Python, JS/TS; limited others
3. **No Security Scanning**: Focused on architecture, not vulnerabilities
4. **Sequential Processing**: No parallel file processing
5. **Memory Bound**: Entire file contents loaded for analysis

### **Design Assumptions**
1. **Repo Size**: ~200 files target, scalable to thousands
2. **Trusted Code**: Running on your own repos, minimal security constraints
3. **LLM Access**: OpenAI API available, cost acceptable
4. **AWS S3**: Primary storage backend
5. **Python 3.11**: No support for older Python versions

---

**End of System Prompt** - This document should be provided to LLM architects working on the snapshotter codebase. It represents the complete current implementation as of the latest updates to `graph.py` and `validate_basic.py`.