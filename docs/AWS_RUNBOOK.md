AWS Runbook: Repo Snapshotter v0.1 (Scaffold)

Overview
This document describes how to run the Snapshotter container on AWS using ECS Fargate or AWS Batch. The application is configured entirely via a single JSON payload provided at runtime.

Key principles
- Payload-only configuration: provide the job JSON via the SNAPSHOTTER_JOB_JSON environment variable. For large payloads, consider storing JSON in S3 and passing an S3 URI (future enhancement; not implemented yet).
- No static AWS keys: use task/job IAM roles for accessing AWS services (e.g., S3).
- No hardcoded secrets: use AWS Secrets Manager for sensitive values such as OPENAI_API_KEY (to be used by future logic).
- Logging: rely on container stdout/stderr streaming to CloudWatch Logs.
- Data protection: enforce server-side encryption (SSE) for any S3 objects this job reads/writes (operational policy; not enforced in code at this time).

Container image
- Build locally: make docker-build (image tag defaults to snapshotter:0.1.0)
- Run locally: make docker-run (ensure SNAPSHOTTER_JOB_JSON is set or provide JOB_FILE)

Runtime contract
- CLI: python -m snapshotter.cli
- Env var: SNAPSHOTTER_JOB_JSON (string). Example: '{"repo_url": "https://github.com/org/repo", "ref": "main"}'
- Exit codes: 0 on success; non-zero on error. Errors are also printed as a JSON object to stdout.

ECS Fargate (generic steps)
1) Create task execution role (AWS managed policy: service-role/AmazonECSTaskExecutionRolePolicy) and a task role with least-privilege permissions your job needs (e.g., S3 read). Attach any required KMS permissions if accessing SSE-KMS objects.
2) Create a CloudWatch Logs log group (e.g., /ecs/snapshotter). Configure retention and encryption as needed.
3) Register a Task Definition:
   - Launch type: Fargate
   - CPU/Memory: choose appropriate (e.g., 0.5 vCPU, 1GB)
   - Container image: your ECR image URI
   - Command: ["python", "-m", "snapshotter.cli"]
   - Environment:
       - Name: SNAPSHOTTER_JOB_JSON, Value: <your JSON string>
     Note: If the JSON is large, consider putting it in S3 and pass a small JSON that includes an s3:// URI (to be handled by future logic).
   - Logging: awslogs driver to your log group
   - Task Role: the role that grants runtime permissions (e.g., S3 access)
   - Execution Role: for pulling from ECR and publishing logs
4) Run Task:
   - Use the same VPC/Subnets/Security Groups as your environment standards
   - Observe logs in CloudWatch for job output

AWS Batch (generic steps)
1) Create a Job Role with least-privilege permissions required by the job (e.g., S3 read/write). Create a Job Queue and Compute Environment (Fargate or EC2 as per your standards).
2) Register a Job Definition:
   - Container image: your ECR image URI
   - Command: ["python", "-m", "snapshotter.cli"]
   - Environment: include SNAPSHOTTER_JOB_JSON with the JSON payload
   - Logging: ensure CloudWatch Logs is enabled for your environment
3) Submit a Job:
   - Provide job name, job queue, and job definition
   - Optionally override environment variables including SNAPSHOTTER_JOB_JSON per job

Security and secrets
- IAM roles: Always rely on the task/job role for access to AWS resources; do not embed access keys.
- Secrets Manager: Store sensitive values like OPENAI_API_KEY (used by future logic). Inject via environment variables at runtime using ECS/Batch native secret support.
- KMS/SSE: Ensure that any S3 buckets involved enforce SSE (AES-256 or KMS) and that roles have permissions to use the relevant KMS keys when needed.

Operational notes
- Env var size limits: ECS and Batch have limits for environment variable sizes. If your payload is large, store it in S3 and pass a minimal reference (e.g., an S3 URI) as SNAPSHOTTER_JOB_JSON content. Future code can fetch and parse it.
- Observability: All logs are written to stdout/stderr. Ensure your log group retention and metric filters meet your requirements.
- Retries: Use ECS or Batch-level retry policies where appropriate.
- Timeouts: Configure container/step timeouts at the scheduler level to prevent runaway tasks.

Example payloads
Small inline payload (for testing):
{
  "repo_url": "https://github.com/example/repo",
  "ref": "main"
}

Larger payloads: place JSON in S3 and (later) pass a reference such as:
{"payload_s3_uri": "s3://your-bucket/path/to/job.json"}
