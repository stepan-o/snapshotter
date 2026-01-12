import secrets
from typing import Literal

from pydantic import BaseModel, Field

from snapshotter.utils import repo_slug_from_url, utc_ts


class Limits(BaseModel):
    max_file_bytes: int = 10 * 1024 * 1024
    max_total_bytes: int = 250 * 1024 * 1024
    max_files: int = 20000


class Filters(BaseModel):
    deny_dirs: list[str] = ["node_modules", ".git", ".next", "dist", "build", ".venv"]
    deny_file_regex: list[str] = [
        # key material
        r"(?i).*\.pem$",
        r"(?i).*\.key$",
        r"(?i).*id_rsa$",
        # env secrets (".env", ".env.local", ".env.production", etc.)
        r"(?i)(^|/)\.env(\..*)?$",
        # common credential filenames (requested)
        r"(?i)(^|/)credentials\.json$",
        r"(?i)(^|/)service[_-]?account.*\.json$",
    ]
    allow_exts: list[str] = ["*"]

    # v0.1 safety: binary files are skipped unless explicitly allowed
    allow_binary: bool = False


class Output(BaseModel):
    s3_bucket: str
    s3_prefix: str


class Metadata(BaseModel):
    triggered_by: Literal["manual", "langgraph", "cron"] = "manual"
    notes: str | None = None


class Job(BaseModel):
    job_id: str | None = None
    repo_url: str
    ref: str
    mode: Literal["full", "light"] = "full"
    limits: Limits = Field(default_factory=Limits)
    filters: Filters = Field(default_factory=Filters)
    output: Output
    metadata: Metadata = Field(default_factory=Metadata)

    # derived at runtime
    repo_slug: str | None = None
    timestamp_utc: str | None = None

    def finalize(self):
        self.repo_slug = repo_slug_from_url(self.repo_url)
        self.timestamp_utc = utc_ts()

        # IMPORTANT:
        # timestamp_utc is already a path segment, so job_id must NOT equal timestamp_utc
        # or you get out/<ts>/<ts> and s3/<ts>/<ts>.
        if not self.job_id:
            # short-ish, unique per run; stable enough for v0.1
            self.job_id = f"{self.timestamp_utc}-{secrets.token_hex(4)}"

        return self

    def s3_job_prefix(self) -> str:
        assert self.repo_slug and self.timestamp_utc and self.job_id
        return f"{self.output.s3_prefix}/{self.repo_slug}/{self.timestamp_utc}/{self.job_id}"