# CLAUDE.md - Claude Code Implementation Guide

**Project:** MNEMOS Platform  
**Environment:** Docker Compose on Ubuntu WSL  
**Target:** Production-Ready AI Operations Platform  
**Implementation Partner:** Claude Code

-----

## Overview

You are building MNEMOS, a production-grade AI operations platform for orchestrating LLM workloads. This document provides step-by-step instructions for implementing all components using Claude Code as your AI pair programmer.

**Key Principles:**

- Security first: TLS, secrets management, least privilege
- Observable by design: Metrics, logs, traces for everything
- Fail-safe: Graceful degradation, retries, circuit breakers
- Production-ready: No shortcuts, no placeholders, real implementation

-----

## Prerequisites

Before starting implementation:

```bash
# Verify environment
docker --version          # 24.0+
docker compose version    # 2.20+
python3 --version        # 3.11+
git --version            # 2.40+

# Install Claude Code
curl -fsSL https://install.claude.com/code | sh
claude-code auth login

# Create workspace
mkdir -p ~/mnemos && cd ~/mnemos
git init
```

-----

## Phase 1: Project Structure & Foundation

### Step 1.1: Initialize Project Structure

```bash
# Create directory tree
mkdir -p {src,docker,config,scripts,data,tests,docs}
mkdir -p src/{soul,genome,cortex,neuron,engram,wraith,relay,synkron,trace,cradle}
mkdir -p config/{traefik,prometheus,grafana,loki,vault}
mkdir -p data/{models,artifacts,logs,backups}
mkdir -p tests/{unit,integration,e2e}
```

**Claude Code Prompt:**

```
Create a comprehensive .gitignore file for a Python/Docker project including:
- Python artifacts (__pycache__, *.pyc, *.pyo, venv/, .env)
- Docker volumes and data (data/*, *.log)
- Secrets (.vault-keys.json, *.pem, *.key)
- IDE files (.vscode/, .idea/)
- OS files (.DS_Store, Thumbs.db)
```

### Step 1.2: Create Root docker-compose.yml

**Claude Code Prompt:**

```
Create a production-ready docker-compose.yml file for MNEMOS with:
- Three isolated networks: frontend (172.20.0.0/24), backend (172.21.0.0/24, internal), data (172.22.0.0/24, internal)
- Named volumes: vault-data, postgres-data, redis-data, engram-data, prometheus-data, loki-data, grafana-data
- Services: vault, postgres, redis, minio, traefik, prometheus, loki, grafana, otel-collector
- Health checks for all services
- Resource limits (CPU, memory)
- Restart policies (unless-stopped)
- Logging configuration (json-file driver with rotation)
- Security options (no-new-privileges, read-only root filesystem where possible)

Use environment variables from .env file for secrets.
Include comprehensive labels for service discovery.
```

### Step 1.3: Create Environment Configuration

**Claude Code Prompt:**

```
Create .env.example file with all required environment variables:
- ENVIRONMENT (development/staging/production)
- Database credentials (POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB)
- Redis password (REDIS_PASSWORD)
- MinIO credentials (MINIO_ROOT_USER, MINIO_ROOT_PASSWORD)
- JWT secrets (JWT_SIGNING_KEY, JWT_ALGORITHM=RS256)
- TLS configuration (TLS_CERT_PATH, TLS_KEY_PATH)
- Service URLs (VAULT_ADDR, CORTEX_URL, GENOME_URL, etc.)
- Observability (OTEL_ENDPOINT, PROMETHEUS_URL, LOKI_URL)
- Feature flags (ENABLE_AUTH, ENABLE_RATE_LIMIT, ENABLE_WAF)

Add detailed comments explaining each variable.
Include secure default values where appropriate.
```

### Step 1.4: Create Makefile

**Claude Code Prompt:**

```
Create a comprehensive Makefile with targets for:
- init: Initialize project (copy .env.example, create directories)
- build: Build all Docker images
- up: Start all services
- down: Stop all services
- clean: Remove volumes and data
- logs: Tail logs for all services
- health: Check health of all services
- test: Run test suite
- bootstrap: Run CRADLE bootstrap
- backup: Create backup of all data
- restore: Restore from backup
- shell-{service}: Open shell in service container
- restart-{service}: Restart specific service
- migrate: Run database migrations
- seed: Seed initial data

Include help target that explains all targets.
Use .PHONY declarations.
Add error handling and colored output.
```

-----

## Phase 2: Infrastructure Services

### Step 2.1: PostgreSQL Setup

**File:** `docker/postgres/Dockerfile`

**Claude Code Prompt:**

```
Create a Dockerfile for PostgreSQL 16 with:
- Official postgres:16-alpine base image
- Custom postgresql.conf for performance tuning
- Initialization scripts for creating databases and schemas
- pg_stat_statements extension enabled
- Backup and restore scripts
- Health check script

Include optimizations for:
- Shared buffers, work memory, maintenance work memory
- WAL configuration for durability
- Connection pooling parameters
- Query performance monitoring
```

**File:** `docker/postgres/init-db.sh`

**Claude Code Prompt:**

```
Create PostgreSQL initialization script that:
- Creates databases: mnemos_cortex, mnemos_genome, mnemos_synkron
- Creates users with appropriate permissions
- Creates schemas: jobs, workers, pipelines, schemas, audit
- Creates tables with proper indexes and constraints
- Enables required extensions (uuid-ossp, pg_trgm, pg_stat_statements)
- Sets up audit logging triggers
- Creates initial admin user

Use proper error handling and idempotency.
```

### Step 2.2: Redis Setup

**File:** `docker/redis/redis.conf`

**Claude Code Prompt:**

```
Create production-ready redis.conf with:
- Authentication enabled (requirepass from environment)
- Persistence configuration (RDB + AOF)
- Memory limits and eviction policies (allkeys-lru)
- Connection limits and timeouts
- Slow log configuration
- Pub/sub configuration for job queues
- TLS configuration for secure connections
- Cluster-ready settings (even if starting with single instance)

Optimize for:
- Job queue workloads (high write throughput)
- Cache efficiency (fast reads)
- Durability (AOF every second)
```

### Step 2.3: Vault (SOUL) Setup

**File:** `src/soul/Dockerfile`

**Claude Code Prompt:**

```
Create Dockerfile for HashiCorp Vault with:
- Official vault:1.15 base image
- Custom entrypoint script for initialization
- TLS certificate generation on first run
- Health check script
- Backup automation script

Include configuration for:
- File storage backend (for Docker environment)
- TLS listener on 8200
- Audit logging enabled
- UI enabled
- High availability preparation (even if single node initially)
```

**File:** `src/soul/config/vault.hcl`

**Claude Code Prompt:**

```
Create comprehensive vault.hcl configuration:
- Storage: file backend with encryption
- Listener: TCP with TLS 1.3, strong cipher suites
- API and cluster addresses
- Telemetry export to Prometheus
- Audit logging to file and syslog
- UI enabled with CSP headers
- Plugin directory configuration
- Performance tuning (cache size, lease durations)

Include detailed comments explaining each section.
```

**File:** `src/soul/scripts/init-vault.sh`

**Claude Code Prompt:**

```
Create Vault initialization script that:
- Checks if Vault is already initialized
- Initializes with 5 key shares, threshold of 3
- Saves keys securely to encrypted file
- Auto-unseals Vault
- Creates root token
- Enables secrets engines (kv-v2, pki, transit, database)
- Creates initial policies for each MNEMOS component
- Stores initial secrets (database passwords, API keys)
- Configures PKI for internal certificate authority
- Sets up JWT auth backend
- Creates service accounts for each component

Include error handling, logging, and rollback on failure.
```

-----

## Phase 3: Core Application Services

### Step 3.1: GENOME (Schema Registry)

**File:** `src/genome/Dockerfile`

**Claude Code Prompt:**

```
Create production Dockerfile for GENOME service:
- Base: python:3.11-slim
- Multi-stage build (builder + runtime)
- Non-root user (uid 1000)
- Dependencies installed via pip with hash verification
- Application code copied with proper permissions
- Health check endpoint exposed
- Startup optimization (pre-compiled Python files)
- Security hardening (no shells, minimal packages)

Optimize image size (<200MB final).
```

**File:** `src/genome/requirements.txt`

**Claude Code Prompt:**

```
Create requirements.txt for GENOME with pinned versions:
- fastapi==0.104.1
- uvicorn[standard]==0.24.0
- pydantic==2.5.0
- redis==5.0.1
- asyncpg==0.29.0
- httpx==0.25.1
- jsonschema==4.20.0
- python-multipart==0.0.6
- python-jose[cryptography]==3.3.0
- opentelemetry-api==1.21.0
- opentelemetry-sdk==1.21.0
- opentelemetry-instrumentation-fastapi==0.42b0
- opentelemetry-exporter-otlp==1.21.0
- prometheus-client==0.19.0
- structlog==23.2.0
- hvac==2.0.0 (Vault client)

Include security dependencies (python-dotenv, cryptography).
Add development dependencies in separate section.
```

**File:** `src/genome/main.py`

**Claude Code Prompt:**

```
Create comprehensive FastAPI application for GENOME:

Core functionality:
- Schema CRUD operations (Model, Pipeline, Policy)
- JSON Schema validation using jsonschema library
- Admission webhook endpoint (Kubernetes-style)
- Policy enforcement (quotas, PII detection, license validation)
- Schema versioning with compatibility checking
- Redis caching for frequently accessed schemas
- PostgreSQL persistence for all schemas

Features to include:
- OpenAPI documentation with examples
- CORS middleware with configurable origins
- Request ID tracking middleware
- Structured logging with correlation IDs
- OpenTelemetry instrumentation for traces
- Prometheus metrics endpoint
- Health check endpoints (liveness, readiness)
- Graceful shutdown handling
- Rate limiting middleware
- Authentication via JWT tokens from Vault
- Input validation with Pydantic models
- Comprehensive error handling with proper status codes
- API versioning (v1)

Database schema:
- schemas table: id, kind, name, version, spec, metadata, created_at, updated_at
- schema_versions table: schema_id, version, spec, created_at, created_by
- policies table: id, name, spec, enabled, created_at
- audit_log table: id, action, schema_id, user_id, timestamp, changes

Include:
- Async database operations using asyncpg
- Connection pooling
- Transaction management
- Database migration support (Alembic)

Security:
- Input sanitization
- SQL injection prevention (parameterized queries)
- XSS prevention in responses
- Rate limiting per API key
- Audit logging of all mutations

Performance:
- Response compression
- ETag support for caching
- Pagination for list endpoints
- Background tasks for async operations
- Circuit breaker for external dependencies

Make this production-ready, well-documented, and fully type-hinted.
```

**File:** `src/genome/models/schema.py`

**Claude Code Prompt:**

```
Create comprehensive Pydantic models for GENOME schemas:

Base models:
- Schema: Base schema object with metadata, spec, validation
- SchemaMetadata: Name, namespace, labels, annotations, timestamps
- ModelSpec: LLM model specification with artifact, tokenizer, runtime, policy
- PipelineSpec: Multi-step pipeline with DAG dependencies
- PolicySpec: Governance policies (quotas, PII, licensing)

Validation models:
- ValidationRequest: Input for validation endpoint
- ValidationResponse: Validation results with errors
- AdmissionRequest: Kubernetes-style admission webhook request
- AdmissionResponse: Allow/deny with reasons

Support models:
- ArtifactSpec: URI, hash, format
- RuntimeSpec: Engine type, parameters
- QuotaSpec: Rate limits, token limits
- RetryPolicy: Max retries, backoff strategy

Include:
- Field validation with Pydantic validators
- Custom validation logic (regex patterns, range checks)
- JSON Schema generation
- Example values in docstrings
- Serialization/deserialization helpers
- Immutable models where appropriate (frozen=True)

All models should be:
- Fully type-hinted
- Well-documented
- Validated on construction
- Serializable to JSON
```

**File:** `src/genome/services/validator.py`

**Claude Code Prompt:**

```
Create SchemaValidator service class:

Functionality:
- Validate specs against registered JSON schemas
- Schema compilation and caching
- Custom validation rules beyond JSON Schema
- Validation error formatting (user-friendly messages)
- Schema compatibility checking (backward/forward)
- Draft 2020-12 JSON Schema support

Methods:
- async validate(kind: str, name: str, spec: dict) -> ValidationResult
- async validate_batch(items: List[tuple]) -> List[ValidationResult]
- async check_compatibility(old_schema: dict, new_schema: dict) -> CompatibilityResult
- _compile_schema(schema: dict) -> Validator
- _format_errors(errors: List) -> List[str]

Features:
- Redis caching of compiled schemas
- Async validation for large specs
- Streaming validation for very large documents
- Custom format validators (URI, email, etc.)
- Reference resolution ($ref support)

Error handling:
- Detailed error paths (JSON Pointer)
- Suggestion engine for common mistakes
- Multiple error collection (don't fail on first error)

Include comprehensive unit tests.
```

**File:** `src/genome/services/policy_enforcer.py`

**Claude Code Prompt:**

```
Create PolicyEnforcer service class:

Functionality:
- Enforce policies on API requests
- PII detection and handling (redact/block/allow)
- Quota enforcement (rate limits, token limits)
- License compliance checking
- Cost attribution and tracking
- Access control (user allowlists)

Methods:
- async enforce(policy_name: str, context: dict) -> PolicyResult
- async check_quota(user: str, action: str) -> QuotaStatus
- async detect_pii(text: str) -> List[PIIMatch]
- async redact_pii(text: str) -> str
- async check_license(license: str, usage: str) -> bool

PII Detection:
- Email addresses (regex + validation)
- Phone numbers (international formats)
- SSN, credit cards, passport numbers
- IP addresses
- Names (NER model - optional)
- Addresses (regex patterns)

Quota Management:
- Redis-based counters with expiration
- Sliding window rate limiting
- Token bucket algorithm for burst handling
- Cost tracking (token count * model price)

License Checking:
- SPDX license validation
- Commercial vs open source
- Derivative work restrictions
- Citation requirements

Include:
- Async operations
- Caching of policy decisions
- Audit logging of policy violations
- Metrics export (policy hits, violations)
```

### Step 3.2: CORTEX (Orchestrator)

**File:** `src/cortex/Dockerfile`

**Claude Code Prompt:**

```
Create production Dockerfile for CORTEX:
- Base: python:3.11-slim
- Multi-stage build
- Non-root user
- Install dependencies with hash verification
- Copy application code
- Health check on ports 8080 (HTTP) and 9090 (gRPC)
- Optimize for fast startup
- Security hardening

Final image size target: <250MB
```

**File:** `src/cortex/requirements.txt`

**Claude Code Prompt:**

```
Create requirements.txt for CORTEX:
- fastapi==0.104.1
- uvicorn[standard]==0.24.0
- grpcio==1.60.0
- grpcio-tools==1.60.0
- pydantic==2.5.0
- redis==5.0.1
- asyncpg==0.29.0
- httpx==0.25.1
- apscheduler==3.10.4
- tenacity==8.2.3 (retry logic)
- circuitbreaker==1.4.0
- opentelemetry-* (full suite)
- prometheus-client==0.19.0
- structlog==23.2.0
- psutil==5.9.6

Add type stubs where available.
```

**File:** `src/cortex/main.py`

**Claude Code Prompt:**

```
Create comprehensive FastAPI + gRPC application for CORTEX orchestrator:

HTTP API (FastAPI):
- POST /v1/jobs - Submit job
- GET /v1/jobs/{job_id} - Get job status
- GET /v1/jobs/{job_id}/stream - Stream job output (SSE)
- DELETE /v1/jobs/{job_id} - Cancel job
- GET /v1/jobs - List jobs (with pagination, filtering)
- GET /v1/workers - List registered workers
- GET /v1/workers/{worker_id} - Get worker details
- POST /v1/workers/{worker_id}/drain - Drain worker
- GET /v1/queue - Get queue statistics
- GET /v1/metrics - Prometheus metrics
- GET /v1/health - Health check

gRPC API (for internal communication):
- RegisterWorker(WorkerInfo) -> WorkerID
- Heartbeat(WorkerID) -> Ack
- PollJob(WorkerID) -> Job
- UpdateJobStatus(JobStatus) -> Ack
- StreamJobOutput(JobID) -> stream<Output>

Core Components:
1. Job Scheduler:
   - Priority queue (critical > high > normal > low)
   - Resource-aware scheduling (CPU, GPU, memory)
   - Worker affinity and anti-affinity
   - Job dependencies (wait for job X before starting job Y)
   - Deadline scheduling
   - Fair-share scheduling across users

2. Worker Pool Manager:
   - Worker registration and deregistration
   - Health monitoring (heartbeat every 30s)
   - Capacity tracking (available resources)
   - Worker draining for maintenance
   - Auto-scaling signals (when queue depth > threshold)

3. Retry Handler:
   - Exponential backoff (1s, 2s, 4s, 8s, ..., max 5min)
   - Max retry limit (configurable per job)
   - Retry queue in Redis (ZSET with retry timestamp)
   - Dead letter queue for permanently failed jobs

4. Circuit Breaker:
   - Per-worker circuit breaker
   - Fail-fast when worker is unhealthy
   - Auto-recovery after cooldown period
   - Metrics on circuit state changes

5. Job State Machine:
   - pending -> scheduled -> running -> completed/failed
   - Atomic state transitions with PostgreSQL
   - Event emission on state changes
   - Audit trail of all transitions

Database Schema:
- jobs: id, kind, status, priority, model_ref, spec, result, error, resources, 
        user, namespace, created_at, scheduled_at, started_at, completed_at, 
        retry_count, max_retries, timeout, worker_id, metadata
- workers: id, hostname, capabilities, resources, status, last_heartbeat, 
          registered_at, metadata
- job_events: id, job_id, event, timestamp, details
- queue_stats: timestamp, priority, depth, avg_wait_time

Features:
- Request ID tracking
- Structured logging (structlog)
- OpenTelemetry instrumentation
- Prometheus metrics (queue depth, job durations, success rate)
- Graceful shutdown (complete current operations)
- Configuration hot-reload (SIGHUP)
- Admin endpoints (protected by auth)

Security:
- JWT authentication
- RBAC (users can only see their own jobs, admins see all)
- Rate limiting per user
- Input validation
- Audit logging

Performance:
- Connection pooling (Redis, PostgreSQL)
- Async operations throughout
- Batch operations where possible
- Efficient queue data structures
- Lazy loading of job details

Make this production-ready with comprehensive error handling and logging.
```

**File:** `src/cortex/models/job.py`

**Claude Code Prompt:**

```
Create comprehensive Pydantic models for CORTEX jobs:

Core Models:
- Job: Complete job specification with all fields
- JobStatus: Enum (pending, scheduled, running, completed, failed, cancelled, retrying)
- JobPriority: Enum (low, normal, high, critical)
- ResourceRequirements: CPU, memory, GPU, storage
- InferenceJobSpec: LLM inference specific fields
- TrainingJobSpec: Model training specific fields
- PipelineJobSpec: Reference to SYNKRON pipeline

Worker Models:
- Worker: Worker registration info
- WorkerCapabilities: GPU types, CPU cores, memory
- WorkerStatus: Enum (available, busy, draining, offline)
- WorkerResources: Current resource availability

Request/Response Models:
- JobSubmitRequest: API request to submit job
- JobSubmitResponse: Job ID and initial status
- JobListRequest: Pagination and filters
- JobListResponse: List of jobs with metadata
- WorkerListResponse: List of workers

Statistics Models:
- QueueStats: Depth per priority, average wait time
- WorkerStats: Utilization, job count, success rate
- SystemStats: Overall system health

Include:
- Validation on all fields
- Defaults where appropriate
- Documentation in docstrings
- JSON schema generation
- Example values
```

**File:** `src/cortex/services/scheduler.py`

**Claude Code Prompt:**

```
Create Scheduler service class:

Core Algorithm:
- Multi-level priority queue (heapq)
- Resource-aware scheduling
- Fair-share across users
- Deadline-aware scheduling
- Worker affinity/anti-affinity

Class Structure:
```python
class Scheduler:
    def __init__(self, redis, db, worker_pool):
        self.redis = redis
        self.db = db
        self.worker_pool = worker_pool
        self.queues = {priority: [] for priority in JobPriority}
    
    async def submit_job(self, job: Job) -> UUID:
        """Submit job to scheduler"""
        pass
    
    async def schedule_loop(self):
        """Main scheduling loop"""
        pass
    
    async def _find_suitable_worker(self, job: Job) -> Optional[Worker]:
        """Find worker that can run job"""
        pass
    
    async def _assign_job(self, job: Job, worker: Worker):
        """Assign job to worker"""
        pass
    
    async def cancel_job(self, job_id: UUID):
        """Cancel pending or running job"""
        pass
```

Scheduling Logic:

1. Get available workers from pool
2. For each priority (high to low):
- Pop jobs from queue
- Find suitable worker (resources match)
- Assign job to worker
- Update job status
- Notify worker via Redis pub/sub
1. Sleep briefly, repeat

Resource Matching:

- CPU cores (worker.available >= job.required)
- Memory (with safety margin)
- GPU count and type
- Storage space

Fairness:

- Track jobs per user
- Deprioritize users with many running jobs
- Configurable fairness window

Metrics:

- Time in queue per priority
- Scheduling decisions per second
- Failed scheduling attempts
- Worker utilization

Include comprehensive tests for scheduling logic.

```
**File:** `src/cortex/services/worker_pool.py`

**Claude Code Prompt:**
```

Create WorkerPool service class:

Functionality:

- Worker registration and deregistration
- Heartbeat monitoring
- Health checks
- Capacity tracking
- Worker draining

Class Structure:

```python
class WorkerPool:
    def __init__(self, redis, db):
        self.redis = redis
        self.db = db
        self.workers = {}  # In-memory cache
    
    async def register_worker(self, worker_info: WorkerInfo) -> str:
        """Register new worker"""
        pass
    
    async def deregister_worker(self, worker_id: str):
        """Deregister worker"""
        pass
    
    async def heartbeat(self, worker_id: str):
        """Process worker heartbeat"""
        pass
    
    async def get_available_workers(self) -> List[Worker]:
        """Get workers ready to accept jobs"""
        pass
    
    async def reserve_resources(self, worker_id: str, resources: ResourceRequirements):
        """Reserve resources on worker"""
        pass
    
    async def release_resources(self, worker_id: str, resources: ResourceRequirements):
        """Release resources on worker"""
        pass
    
    async def health_check_loop(self):
        """Monitor worker health"""
        pass
    
    async def drain_worker(self, worker_id: str):
        """Drain worker for maintenance"""
        pass
```

Heartbeat Monitoring:

- Expect heartbeat every 30 seconds
- Mark worker offline after 2 missed heartbeats (60s)
- Re-schedule jobs from offline workers
- Alert on worker failures

Health Checks:

- Periodic ping to worker /health endpoint
- Check resource utilization
- Verify GPU availability (if applicable)
- Check disk space

Worker States:

- available: Ready for work
- busy: Has running jobs
- draining: No new jobs, completing existing
- offline: Not responding to heartbeats

Metrics:

- Worker count by status
- Average worker utilization
- Job distribution across workers
- Worker uptime

Include tests for heartbeat timeout and recovery.

```
### Step 3.3: NEURON (Worker Runtime)

**File:** `src/neuron/Dockerfile`

**Claude Code Prompt:**
```

Create production Dockerfile for NEURON worker with GPU support:

- Base: nvidia/cuda:12.1.0-runtime-ubuntu22.04
- Install Python 3.11
- Install vLLM dependencies
- Install PyTorch with CUDA support
- Copy application code
- Non-root user
- Health check endpoint
- Optimize for image caching (layer ordering)

Create two variants:

1. Dockerfile (CPU-only, smaller)
2. Dockerfile.gpu (NVIDIA GPU support)

Final GPU image target: <8GB

```
**File:** `src/neuron/requirements.txt`

**Claude Code Prompt:**
```

Create requirements.txt for NEURON:

Core:

- torch==2.1.0+cu121
- vllm==0.2.7
- transformers==4.35.0
- accelerate==0.24.0
- bitsandbytes==0.41.0 (quantization)
- sentencepiece==0.1.99
- protobuf==4.25.0

Infrastructure:

- fastapi==0.104.1
- uvicorn[standard]==0.24.0
- redis==5.0.1
- httpx==0.25.1
- boto3==1.34.0 (S3 client)

Monitoring:

- psutil==5.9.6
- nvidia-ml-py3==7.352.0 (GPU monitoring)
- opentelemetry-* (full suite)
- prometheus-client==0.19.0

Utilities:

- tenacity==8.2.3
- structlog==23.2.0
- python-dotenv==1.0.0

Pin all versions for reproducibility.

```
**File:** `src/neuron/worker.py`

**Claude Code Prompt:**
```

Create NeuronWorker class - the main worker daemon:

Core Structure:

```python
class NeuronWorker:
    def __init__(self, worker_id: str, config: WorkerConfig):
        self.worker_id = worker_id
        self.config = config
        self.current_job = None
        self.shutdown_requested = False
        self.runtime = None  # VLLMRuntime or PyTorchRuntime
    
    async def start(self):
        """Start worker main loop"""
        pass
    
    async def _register_worker(self):
        """Register with CORTEX"""
        pass
    
    async def _initialize_runtime(self):
        """Initialize ML runtime (vLLM, PyTorch)"""
        pass
    
    async def _poll_job(self) -> Optional[Job]:
        """Poll CORTEX for next job"""
        pass
    
    async def _execute_job(self, job: Job):
        """Execute job based on kind"""
        pass
    
    async def _run_inference(self, job: Job) -> dict:
        """Run LLM inference"""
        pass
    
    async def _run_training(self, job: Job) -> dict:
        """Run model training"""
        pass
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat to CORTEX"""
        pass
    
    def _detect_capabilities(self) -> dict:
        """Detect worker capabilities (GPU, CPU, memory)"""
        pass
    
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        pass
    
    async def _cleanup(self):
        """Cleanup resources"""
        pass
```

Worker Lifecycle:

1. Start -> Register with CORTEX
2. Initialize runtime (load default model)
3. Enter polling loop:
- Poll for jobs (blocking with timeout)
- Execute job
- Stream results
- Update job status
- Release resources
1. Send heartbeat every 30s
2. On SIGTERM: finish current job, cleanup, exit

Job Execution:

- Load model from ENGRAM if needed
- Validate inputs with GENOME
- Execute workload (inference, training, etc.)
- Stream outputs to Redis for client consumption
- Store final results in ENGRAM
- Update CORTEX with job status

Error Handling:

- Catch all exceptions
- Report errors to CORTEX
- Mark job as failed
- Continue with next job (don’t crash)
- Circuit breaker for repeated failures

Resource Management:

- Monitor GPU memory usage
- Unload models when memory low
- Implement model cache (LRU)
- Graceful degradation (CPU fallback)

Metrics:

- Jobs completed
- Average job duration
- GPU utilization
- Memory usage
- Error rate

Security:

- Validate job payloads
- Sandboxing (Docker isolation)
- Resource limits (ulimits)
- No arbitrary code execution

Make this robust for 24/7 operation.

```
**File:** `src/neuron/runtimes/vllm_runtime.py`

**Claude Code Prompt:**
```

Create VLLMRuntime class for LLM inference:

```python
class VLLMRuntime:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.engine = None
        self.loaded_models = {}
    
    async def initialize(self):
        """Initialize vLLM engine"""
        pass
    
    async def load_model(self, model_ref: str):
        """Load model weights"""
        pass
    
    def is_model_loaded(self, model_ref: str) -> bool:
        """Check if model is loaded"""
        pass
    
    async def generate(self, prompts: List[str], params: SamplingParams) -> List[RequestOutput]:
        """Generate completions (batch)"""
        pass
    
    async def generate_stream(self, prompt: str, params: SamplingParams) -> AsyncIterator:
        """Generate completions (streaming)"""
        pass
    
    async def _ensure_model_local(self, model_info: dict) -> str:
        """Download model from ENGRAM if not local"""
        pass
    
    async def cleanup(self):
        """Cleanup resources"""
        pass
```

vLLM Configuration:

- Tensor parallelism (multi-GPU)
- GPU memory utilization (0.9 default)
- Max model length
- Quantization (AWQ, GPTQ, bitsandbytes)
- KV cache configuration
- Prefix caching

Model Loading:

- Download from ENGRAM (S3/MinIO)
- Verify checksums
- Load weights into GPU memory
- Warm up (run dummy inference)
- Cache for reuse

Inference:

- Batch inference for efficiency
- Streaming for real-time responses
- Sampling parameters (temperature, top_p, top_k)
- Stop sequences
- Logprobs export (optional)

Performance:

- Continuous batching
- PagedAttention
- Quantization for memory efficiency
- Multi-GPU inference

Error Handling:

- OOM recovery
- Corrupted model handling
- Timeout for long generations

Include comprehensive tests with mock models.

```
**File:** `src/neuron/utils/gpu_monitor.py`

**Claude Code Prompt:**
```

Create GPUMonitor utility class:

Functionality:

- Query GPU information (nvidia-smi or NVML)
- Monitor GPU memory usage
- Monitor GPU utilization
- Temperature monitoring
- Power draw monitoring
- Export metrics to Prometheus

```python
class GPUMonitor:
    def __init__(self):
        import pynvml
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
    
    def get_gpu_info(self, gpu_id: int) -> dict:
        """Get comprehensive GPU information"""
        pass
    
    def get_memory_info(self, gpu_id: int) -> dict:
        """Get memory usage (used, free, total)"""
        pass
    
    def get_utilization(self, gpu_id: int) -> dict:
        """Get GPU and memory utilization %"""
        pass
    
    def get_temperature(self, gpu_id: int) -> int:
        """Get GPU temperature in Celsius"""
        pass
    
    def get_power_usage(self, gpu_id: int) -> float:
        """Get power draw in watts"""
        pass
    
    async def monitor_loop(self, interval: int = 5):
        """Continuous monitoring loop"""
        pass
    
    def export_metrics(self) -> str:
        """Export Prometheus metrics"""
        pass
```

Metrics to Export:

- nvidia_gpu_utilization{gpu=“0”}
- nvidia_gpu_memory_used_bytes{gpu=“0”}
- nvidia_gpu_memory_total_bytes{gpu=“0”}
- nvidia_gpu_temperature_celsius{gpu=“0”}
- nvidia_gpu_power_watts{gpu=“0”}
- nvidia_gpu_fan_speed_percent{gpu=“0”}

Alerts:

- High temperature (>80°C)
- High memory usage (>95%)
- GPU errors
- Throttling detected

Include graceful handling of missing NVIDIA drivers (CPU-only mode).

```
### Step 3.4: ENGRAM (Storage Layer)

**File:** `docker/minio/Dockerfile`

**Claude Code Prompt:**
```

Create Dockerfile for MinIO with custom configuration:

- Base: minio/minio:latest
- Custom entrypoint script
- Automated bucket creation
- Lifecycle policy configuration
- TLS certificate setup
- Health check script

Include initialization script that creates:

- mnemos-artifacts bucket (for general artifacts)
- mnemos-models bucket (for model weights)
- mnemos-backups bucket (for backups)

```
**File:** `config/minio/lifecycle.json`

**Claude Code Prompt:**
```

Create MinIO lifecycle policy JSON:

Rules:

1. Transition job outputs to GLACIER after 90 days
2. Expire job outputs after 365 days
3. Delete incomplete multipart uploads after 7 days
4. Keep model artifacts indefinitely
5. Keep backups for 90 days then delete

Include versioning rules:

- Keep last 5 versions of models
- Expire old versions after 180 days

```
**File:** `src/engram/client.py`

**Claude Code Prompt:**
```

Create EngramClient - comprehensive S3 client wrapper:

```python
class EngramClient:
    def __init__(self, endpoint: str, access_key: str, secret_key: str):
        self.s3 = boto3.client(...)
        self.bucket = "mnemos-artifacts"
    
    async def upload_artifact(self, key: str, data: BinaryIO, metadata: dict = None) -> str:
        """Upload artifact and return SHA256 hash"""
        pass
    
    async def download_artifact(self, key: str, dest_path: str):
        """Download artifact to local path"""
        pass
    
    async def stream_artifact(self, key: str) -> AsyncIterator[bytes]:
        """Stream artifact for large files"""
        pass
    
    async def list_artifacts(self, prefix: str, max_age: datetime = None) -> List[dict]:
        """List artifacts with optional age filter"""
        pass
    
    async def get_metadata(self, key: str) -> dict:
        """Get artifact metadata"""
        pass
    
    async def delete_artifact(self, key: str):
        """Delete artifact"""
        pass
    
    async def copy_artifact(self, source_key: str, dest_key: str):
        """Copy artifact within bucket"""
        pass
    
    async def create_backup(self, prefix: str, backup_name: str):
        """Create backup of artifacts with prefix"""
        pass
    
    async def restore_backup(self, backup_name: str, dest_prefix: str):
        """Restore from backup"""
        pass
    
    async def get_presigned_url(self, key: str, expiration: int = 3600) -> str:
        """Generate presigned URL for temporary access"""
        pass
```

Features:

- Multipart upload for large files (>100MB)
- Resume interrupted uploads
- Parallel downloads for speed
- Checksum verification (SHA256)
- Metadata tagging
- Server-side encryption
- Versioning support
- Lifecycle policy application

Error Handling:

- Retry with exponential backoff
- Handle network errors
- Handle access denied errors
- Validate bucket existence

Metrics:

- Upload/download throughput
- Operation latency
- Error rates
- Storage usage

Include comprehensive unit tests with moto (S3 mocking).

```
### Step 3.5: WRAITH (Background Jobs)

**File:** `src/wraith/main.py`

**Claude Code Prompt:**
```

Create WRAITH daemon with APScheduler:

Job Definitions:

1. compact_logs (daily at 3 AM):
- Compress logs older than 7 days
- Archive to ENGRAM
- Delete local copies
1. replay_failed_events (every 15 minutes):
- Read from dead letter queue
- Replay failed events
- Move to permanent failure after 10 attempts
1. cleanup_old_artifacts (every 6 hours):
- Find artifacts older than retention policy
- Archive or delete based on policy
- Update storage metrics
1. vacuum_databases (daily at 2 AM):
- PostgreSQL VACUUM ANALYZE
- Redis MEMORY PURGE
- Update database statistics
1. aggregate_metrics (every 5 minutes):
- Calculate success rates
- Worker utilization
- Queue depths
- Store in Redis for fast access
1. health_check_services (every 10 minutes):
- Check all service health endpoints
- Alert if any service is down
- Update service status dashboard
1. warm_model_cache (daily at 1 AM):
- Identify most-used models
- Pre-load into NEURON workers
- Improve first-request latency
1. rotate_logs (daily at 4 AM):
- Rotate application logs
- Compress old logs
- Delete logs older than 30 days
1. backup_databases (daily at 5 AM):
- Create PostgreSQL backup
- Upload to ENGRAM
- Verify backup integrity
- Delete backups older than 30 days
1. prune_completed_jobs (daily at 6 AM):
- Archive completed jobs older than 90 days
- Keep only metadata in main database
- Maintain audit trail

Main Structure:

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

scheduler = AsyncIOScheduler()

@scheduler.scheduled_job(CronTrigger(hour=3, minute=0))
async def compact_logs():
    """Implementation here"""
    pass

# ... more jobs

def start_wraith():
    logger.info("Starting WRAITH daemon")
    scheduler.start()
    try:
        asyncio.get_event_loop().run_forever()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
```

Features:

- Job persistence (survive restarts)
- Job execution history
- Failed job retry
- Job concurrency control
- Job chaining (job B after job A)
- Manual job trigger via API

Metrics:

- Job execution count
- Job duration
- Job success/failure rate
- Next scheduled run time

Include comprehensive logging for all jobs.

```
### Step 3.6: RELAY (API Gateway)

**File:** `config/traefik/traefik.yml`

**Claude Code Prompt:**
```

Create comprehensive Traefik static configuration:

Features:

- HTTP to HTTPS redirect
- TLS 1.3 only with strong ciphers
- Let’s Encrypt ACME for certificates (or self-signed for dev)
- Dashboard enabled (secured)
- Docker provider with label-based configuration
- File provider for static routes
- Access logs in JSON format
- Metrics endpoint for Prometheus
- Health check endpoint

Entry Points:

- web: :80 (redirect to websecure)
- websecure: :443 (TLS)
- traefik: :8080 (dashboard, internal only)

Certificate Resolvers:

- letsencrypt (for production)
- selfsigned (for development)

Providers:

- docker (main provider)
- file (static configuration)

Logging:

- JSON format
- Include request ID
- Log level: INFO

Metrics:

- Prometheus format
- Include entry points and services labels

```
**File:** `config/traefik/dynamic.yml`

**Claude Code Prompt:**
```

Create Traefik dynamic configuration:

Routers:

- cortex-api: api.mnemos.local/v1/jobs -> cortex:8080
- genome-api: api.mnemos.local/v1/schemas -> genome:8081
- synkron-api: api.mnemos.local/v1/pipelines -> synkron:8082
- grafana: monitoring.mnemos.local -> grafana:3000

Middlewares:

1. auth: JWT validation via forward auth to Vault
2. rate-limit: 100 req/min per API key, burst 50
3. circuit-breaker: Open circuit if error rate > 30%
4. security-headers: HSTS, CSP, XFO, etc.
5. compress: Gzip compression
6. cors: CORS headers for web clients
7. ip-whitelist: Restrict admin endpoints
8. retry: Retry failed requests (max 3)

Services:

- cortex: Load balancer with health checks
- genome: Load balancer with health checks
- synkron: Load balancer with health checks

TLS Options:

- Minimum version: TLS 1.3
- Cipher suites: Modern, secure ciphers only
- ALPN protocols: h2, http/1.1

Include comprehensive comments explaining each section.

```
**File:** `config/waf/modsecurity.conf`

**Claude Code Prompt:**
```

Create ModSecurity WAF rules (if using Traefik ModSecurity plugin):

Core Rule Set:

- SQL injection protection
- XSS protection
- Command injection protection
- Path traversal protection
- File upload restrictions
- Request size limits
- Rate limiting

Custom Rules:

1. Block requests with SQL keywords in parameters
2. Block requests with script tags
3. Block requests with suspicious user agents
4. Block requests with ../ in path
5. Limit request body size to 10MB
6. Limit query string length to 4KB
7. Block requests from known bad IPs

Response Rules:

- Remove server headers (hide tech stack)
- Block information disclosure
- Sanitize error messages

Audit Logging:

- Log blocked requests
- Include request details
- Send to TRACE for analysis

Use OWASP CRS (Core Rule Set) as base.
Tune for minimal false positives.

```
### Step 3.7: SYNKRON (Pipeline Orchestrator)

**File:** `src/synkron/main.py`

**Claude Code Prompt:**
```

Create SYNKRON FastAPI application for pipeline orchestration:

API Endpoints:

- POST /v1/pipelines - Create pipeline definition
- GET /v1/pipelines/{id} - Get pipeline
- PUT /v1/pipelines/{id} - Update pipeline
- DELETE /v1/pipelines/{id} - Delete pipeline
- GET /v1/pipelines - List pipelines
- POST /v1/pipelines/{id}/run - Execute pipeline
- GET /v1/pipelines/{id}/runs/{run_id} - Get run status
- POST /v1/pipelines/{id}/runs/{run_id}/cancel - Cancel run
- GET /v1/pipelines/{id}/runs - List runs
- GET /v1/health - Health check

Core Features:

1. DAG Validation:
- Check for cycles
- Verify dependencies exist
- Validate resource requirements
1. DAG Execution:
- Topological sort
- Parallel execution of independent steps
- Sequential execution of dependent steps
- Resource-aware scheduling
1. Step Management:
- Submit steps as jobs to CORTEX
- Wait for completion
- Handle step failures
- Retry failed steps
1. Artifact Passing:
- Resolve input references (${step.output})
- Pass artifacts between steps via ENGRAM
- Validate artifact schemas
1. Pipeline Versioning:
- Store pipeline versions
- Track which version was executed
- Allow rollback to previous version

Database Schema:

- pipelines: id, name, version, spec, created_at, updated_at
- pipeline_runs: id, pipeline_id, status, started_at, completed_at, trigger, parameters
- step_runs: id, run_id, step_name, status, job_id, started_at, completed_at, outputs, error, retry_count

Include comprehensive error handling and logging.

```
**File:** `src/synkron/services/dag_executor.py`

**Claude Code Prompt:**
```

Create DAGExecutor class using NetworkX:

Implementation of the examples from MNEMOS.md, including:

- _build_dag: Create NetworkX DiGraph from pipeline steps
- _execute_dag: Execute with parallelism and dependency tracking
- _execute_step: Run individual step via CORTEX
- _resolve_inputs: Replace variable references with actual URIs
- _wait_for_job: Poll CORTEX for job completion

Key Features:

- Parallel execution where possible
- Dependency tracking (completed, failed, running sets)
- Failed step propagation (mark descendants as skipped)
- Retry logic with exponential backoff
- Circuit breaker for repeatedly failing steps
- Progress tracking and status updates
- Detailed error messages

Include comprehensive tests with various DAG topologies.

```
### Step 3.8: TRACE (Observability)

**File:** `config/otel/otel-collector.yml`

**Claude Code Prompt:**
```

Create comprehensive OTEL Collector configuration from MNEMOS.md:

Receivers:

- otlp (gRPC and HTTP)
- prometheus (scrape metrics)

Processors:

- batch (10s timeout, 1024 batch size)
- memory_limiter (512MB limit)
- resource (add environment attributes)

Exporters:

- logging (for debugging)
- prometheus (expose metrics)
- prometheusremotewrite (to Prometheus)
- loki (for logs)
- jaeger (for traces)

Pipelines:

- traces: otlp -> [memory_limiter, batch, resource] -> [logging, jaeger]
- metrics: [otlp, prometheus] -> [memory_limiter, batch, resource] -> [logging, prometheus, prometheusremotewrite]
- logs: otlp -> [memory_limiter, batch, resource] -> [logging, loki]

Telemetry:

- Collector metrics on :8888
- Health check on :13133

```
**File:** `config/prometheus/prometheus.yml`

**Claude Code Prompt:**
```

Create Prometheus configuration from MNEMOS.md:

Global:

- scrape_interval: 15s
- evaluation_interval: 15s
- external_labels (cluster, environment)

Scrape Configs:

- cortex:8080/metrics
- genome:8081/metrics
- neuron:8000/metrics
- synkron:8082/metrics
- otel-collector:8889
- node-exporter:9100
- dcgm-exporter:9400 (NVIDIA GPU metrics)

Alert Manager:

- alertmanager:9093

Rule Files:

- /etc/prometheus/rules/*.yml

Remote Write (optional):

- Mimir or Cortex for long-term storage

```
**File:** `config/prometheus/rules/alerts.yml`

**Claude Code Prompt:**
```

Create alert rules from MNEMOS.md:

Groups:

1. mnemos_core:
- HighErrorRate (>5% for 5min)
- JobQueueBacklog (>1000 for 10min)
- WorkerDown (for 2min)
- GPUMemoryHigh (>95% for 5min)
- ServiceDown (for 1min)
- HighLatency (P95 >1s for 10min)
- DiskSpaceLow (<10% for 5min)
1. mnemos_business:
- JobSuccessRateLow (<95% for 15min)
- ModelLoadFailure (any)
- PipelineFailureRate (>10% for 30min)
1. mnemos_infrastructure:
- PostgreSQLDown
- RedisDown
- VaultSealed
- MinIODown

Include appropriate severity levels (critical, warning, info).
Add descriptive annotations with runbook links.

```
**File:** `config/grafana/dashboards/overview.json`

**Claude Code Prompt:**
```

Create Grafana dashboard JSON for MNEMOS overview:

Panels:

1. Job Success Rate (graph, 5m rate)
2. Active Workers (stat)
3. Queue Depth by Priority (graph)
4. GPU Utilization (graph, per GPU)
5. API Latency P50/P95/P99 (graph)
6. Error Rate by Service (graph)
7. Worker Map (world map or table)
8. Top Models by Usage (bar chart)
9. Storage Usage (gauge)
10. System Health (alert list)

Time range: Last 6 hours
Refresh: 30s
Variables: Environment, Namespace, Service

Use Prometheus as data source.
Include drill-down links to detailed dashboards.

```
### Step 3.9: CRADLE (Bootstrap)

**File:** `scripts/bootstrap.sh`

**Claude Code Prompt:**
```

Create comprehensive bootstrap script from MNEMOS.md:

Steps:

1. Check prerequisites (docker, docker-compose, jq, curl)
2. Start infrastructure (vault, postgres, redis, minio)
3. Initialize Vault:
- Check if already initialized
- Initialize with 5 shares, threshold 3
- Save keys to .vault-keys.json (encrypted)
- Unseal Vault
- Export root token
1. Create Vault policies (cortex, genome, neuron, etc.)
2. Store initial secrets:
- Database passwords
- Redis password
- MinIO credentials
- JWT signing key
- Grafana password
- API keys
1. Initialize databases:
- Run PostgreSQL init scripts
- Create schemas and tables
- Create indexes
- Seed initial data
1. Create MinIO buckets:
- mnemos-artifacts
- mnemos-models
- mnemos-backups
- Apply lifecycle policies
1. Load initial schemas into GENOME:
- Model schema
- Pipeline schema
- Policy schema
1. Start application services (genome, cortex, neuron, etc.)
2. Health check all services
3. Display summary and access info

Features:

- Colored output (red, green, yellow)
- Progress indicators
- Error handling with rollback
- Idempotent (can run multiple times)
- Dry-run mode
- Verbose mode for debugging

Exit codes:

- 0: Success
- 1: Prerequisites missing
- 2: Service failed to start
- 3: Health check failed

```
**File:** `scripts/init-databases.sql`

**Claude Code Prompt:**
```

Create PostgreSQL initialization SQL:

Databases:

- mnemos_cortex (jobs, workers, events)
- mnemos_genome (schemas, policies, audit)
- mnemos_synkron (pipelines, runs, steps)

Tables for mnemos_cortex:

- jobs: Full schema from MNEMOS.md models/job.py
- workers: Full schema from MNEMOS.md
- job_events: Audit trail of job state changes
- queue_stats: Historical queue metrics

Tables for mnemos_genome:

- schemas: id, kind, name, version, spec, metadata, created_at, updated_at
- schema_versions: Version history
- policies: Policy definitions
- validation_cache: Cache validation results

Tables for mnemos_synkron:

- pipelines: Pipeline definitions
- pipeline_runs: Run history
- step_runs: Step execution details

Indexes:

- Jobs: status, created_at, user, model_ref
- Workers: status, last_heartbeat
- Schemas: (kind, name), created_at
- Pipelines: name, created_at

Constraints:

- Foreign keys with CASCADE delete
- Check constraints on enums
- Unique constraints where appropriate

Extensions:

- uuid-ossp (UUID generation)
- pg_trgm (text search)
- pg_stat_statements (query performance)

Partitioning:

- Partition jobs table by created_at (monthly)
- Partition job_events by timestamp (weekly)

Include comments explaining each table and column.

```
**File:** `scripts/load-schemas.sh`

**Claude Code Prompt:**
```

Create script to load initial schemas into GENOME:

Load these JSON Schema definitions:

1. Model schema (model-v1.json)
2. Pipeline schema (pipeline-v1.json)
3. Policy schema (policy-v1.json)

For each schema:

- POST to GENOME /v1/schemas endpoint
- Verify success
- Handle errors gracefully

Example model-v1.json:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://mnemos.dev/schemas/model/v1",
  "title": "Model",
  "description": "LLM model specification",
  "type": "object",
  "required": ["family", "version", "artifact", "runtime"],
  "properties": {
    "family": {
      "type": "string",
      "enum": ["qwen2", "llama", "mistral", "gpt"]
    },
    "version": {
      "type": "string",
      "pattern": "^[0-9]+(b|B)-.*$"
    },
    "artifact": {
      "type": "object",
      "required": ["uri", "sha256"],
      "properties": {
        "uri": {"type": "string", "format": "uri"},
        "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"}
      }
    },
    // ... more properties
  }
}
```

Create comprehensive schemas for all three kinds.

```
---

## Phase 4: Advanced Features

### Step 4.1: Authentication System

**File:** `src/soul/auth/jwt_manager.py`

**Claude Code Prompt:**
```

Create JWTManager class for token issuance and validation:

```python
class JWTManager:
    def __init__(self, vault_client, algorithm="RS256"):
        self.vault = vault_client
        self.algorithm = algorithm
        self.public_key = None
        self.private_key = None
    
    async def initialize(self):
        """Load keys from Vault"""
        pass
    
    def create_token(self, user_id: str, roles: List[str], ttl: int = 3600) -> str:
        """Create JWT token"""
        pass
    
    def verify_token(self, token: str) -> dict:
        """Verify and decode JWT token"""
        pass
    
    def refresh_token(self, refresh_token: str) -> str:
        """Issue new access token from refresh token"""
        pass
    
    async def revoke_token(self, token_id: str):
        """Revoke token (add to blocklist)"""
        pass
    
    def _is_token_revoked(self, token_id: str) -> bool:
        """Check if token is revoked"""
        pass
```

Token Claims:

- sub: User ID
- roles: List of roles
- exp: Expiration timestamp
- iat: Issued at timestamp
- jti: Token ID (for revocation)
- iss: “mnemos”

Features:

- RSA key pair (2048-bit minimum)
- Token rotation (refresh tokens)
- Token revocation (Redis blocklist)
- Claims validation
- Signature verification

Include tests for token lifecycle.

```
**File:** `src/soul/auth/middleware.py`

**Claude Code Prompt:**
```

Create FastAPI authentication middleware:

```python
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token from Authorization header"""
    token = credentials.credentials
    try:
        payload = jwt_manager.verify_token(token)
        return payload
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def require_role(required_role: str):
    """Require specific role"""
    def role_checker(payload: dict = Depends(verify_token)):
        if required_role not in payload.get("roles", []):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return payload
    return role_checker
```

Usage in endpoints:

```python
@app.get("/v1/jobs")
async def list_jobs(user: dict = Depends(verify_token)):
    # user dict contains claims
    pass

@app.delete("/v1/workers/{worker_id}")
async def delete_worker(
    worker_id: str,
    user: dict = Depends(require_role("admin"))
):
    # Only admins can delete workers
    pass
```

Include rate limiting per user.

```
### Step 4.2: WebUI (React Dashboard)

**File:** `src/web/package.json`

**Claude Code Prompt:**
```

Create package.json for React dashboard:

Dependencies:

- react 18+
- react-router-dom (routing)
- @tanstack/react-query (data fetching)
- axios (HTTP client)
- zustand (state management)
- recharts (charts)
- shadcn/ui (UI components)
- lucide-react (icons)
- date-fns (date formatting)
- react-hook-form (forms)
- zod (validation)

Dev Dependencies:

- vite (build tool)
- typescript
- tailwindcss
- eslint
- prettier

Scripts:

- dev: Run dev server
- build: Production build
- preview: Preview build
- lint: ESLint
- format: Prettier

```
**File:** `src/web/src/App.tsx`

**Claude Code Prompt:**
```

Create React application with routes:

Pages:

1. Dashboard (/):
- System overview
- Job statistics
- Worker status
- Recent activity
1. Jobs (/jobs):
- List all jobs
- Filter by status, user, model
- Search by ID or keyword
- Pagination
1. Job Detail (/jobs/:id):
- Full job details
- Logs streaming
- Results display
- Cancel button
- Retry button
1. Workers (/workers):
- List all workers
- Status indicators
- Resource usage
- Drain/undrain actions
1. Models (/models):
- Registered models
- Model details
- Upload new model
- Delete model
1. Pipelines (/pipelines):
- List pipelines
- Create pipeline (DAG builder)
- Run pipeline
- Pipeline history
1. Monitoring (/monitoring):
- Embedded Grafana dashboards
- Metrics charts
- Alert status
1. Settings (/settings):
- User profile
- API keys
- Preferences

Layout:

- Sidebar navigation
- Top bar (user menu, notifications)
- Breadcrumbs
- Dark mode toggle

Authentication:

- Login page
- Token storage (localStorage)
- Auto-refresh tokens
- Logout

Features:

- Real-time updates (WebSocket or polling)
- Responsive design
- Loading states
- Error boundaries
- Toast notifications

Use TypeScript throughout.
Implement proper error handling.

```
### Step 4.3: CLI Tool

**File:** `cli/mnemos`

**Claude Code Prompt:**
```

Create CLI tool using Click or Typer:

Commands:

- mnemos init: Initialize configuration
- mnemos login: Authenticate with username/password
- mnemos jobs list: List jobs
- mnemos jobs submit <spec.json>: Submit job
- mnemos jobs get <job-id>: Get job details
- mnemos jobs cancel <job-id>: Cancel job
- mnemos jobs logs <job-id>: Stream job logs
- mnemos workers list: List workers
- mnemos workers drain <worker-id>: Drain worker
- mnemos models list: List models
- mnemos models register <model.json>: Register model
- mnemos pipelines list: List pipelines
- mnemos pipelines create <pipeline.json>: Create pipeline
- mnemos pipelines run <pipeline-id>: Run pipeline
- mnemos config: Show current configuration
- mnemos config set <key> <value>: Set config value
- mnemos version: Show version

Features:

- Configuration file (~/.mnemos/config.yaml)
- Multiple profiles (dev, staging, prod)
- Pretty-printed output
- JSON output mode (–json flag)
- Verbose mode (-v, -vv, -vvv)
- Color output (with –no-color option)
- Progress bars for long operations
- Interactive prompts (with –yes to skip)

Authentication:

- Store tokens in keyring
- Auto-refresh expired tokens
- Handle auth errors gracefully

Example usage:

```bash
mnemos login --profile prod
mnemos jobs submit inference.json
mnemos jobs logs job-123 --follow
mnemos workers list --status available
```

Implement with proper error handling and help text.

```
### Step 4.4: Model Fine-Tuning Pipeline

**File:** `examples/pipelines/lora-finetuning.yaml`

**Claude Code Prompt:**
```

Create example pipeline for LoRA fine-tuning:

```yaml
apiVersion: synkron.mnemos.dev/v1
kind: Pipeline
metadata:
  name: lora-finetuning
  version: v1
spec:
  steps:
    - name: prepare-data
      kind: data_prep
      runtime: custom
      image: ghcr.io/n1x-mnemos/data-prep:latest
      spec:
        dataset_uri: s3://mnemos-datasets/training/my-dataset.jsonl
        validation_split: 0.1
        max_samples: 10000
      outputs:
        - name: train_data
          uri: s3://mnemos-artifacts/${RUN_ID}/train.jsonl
        - name: val_data
          uri: s3://mnemos-artifacts/${RUN_ID}/val.jsonl
    
    - name: train-lora
      depends_on: [prepare-data]
      kind: training
      runtime: neuron
      spec:
        model_ref: llama-7b
        training_type: lora
        hyperparameters:
          learning_rate: 2e-4
          batch_size: 4
          num_epochs: 3
          lora_r: 8
          lora_alpha: 16
          lora_dropout: 0.05
        train_data: ${prepare-data.train_data}
        val_data: ${prepare-data.val_data}
      outputs:
        - name: lora_weights
          uri: s3://mnemos-models/${RUN_ID}/lora-adapter
    
    - name: merge-weights
      depends_on: [train-lora]
      kind: custom
      image: ghcr.io/n1x-mnemos/model-merge:latest
      spec:
        base_model: llama-7b
        lora_weights: ${train-lora.lora_weights}
      outputs:
        - name: merged_model
          uri: s3://mnemos-models/${RUN_ID}/merged-model
    
    - name: evaluate
      depends_on: [merge-weights]
      kind: inference
      runtime: neuron
      spec:
        model_uri: ${merge-weights.merged_model}
        eval_dataset: s3://mnemos-datasets/eval/benchmark.jsonl
        metrics: [perplexity, accuracy]
      outputs:
        - name: eval_results
          uri: s3://mnemos-artifacts/${RUN_ID}/eval-results.json
    
    - name: register-model
      depends_on: [evaluate]
      kind: custom
      image: ghcr.io/n1x-mnemos/model-register:latest
      spec:
        model_uri: ${merge-weights.merged_model}
        eval_results: ${evaluate.eval_results}
        model_name: "my-finetuned-model"
```

This pipeline demonstrates:

- Multi-step workflow
- Data preparation
- LoRA training
- Model merging
- Evaluation
- Automated model registration

Create similar examples for:

- Full fine-tuning
- RAG indexing pipeline
- Multi-model ensemble

```
### Step 4.5: RAG Support (Vector Database)

**File:** `docker-compose.rag.yml`

**Claude Code Prompt:**
```

Create docker-compose override for RAG components:

Services:

- qdrant: Vector database
  - Port 6333
  - Persistent volume
  - Resource limits
- embedding-service:
  - Custom service for generating embeddings
  - Uses sentence-transformers
  - Batch processing
  - Queue-based

Configuration:

- Connect embedding service to CORTEX
- Store vectors in Qdrant
- Integrate with NEURON for retrieval

Usage:
docker-compose -f docker-compose.yml -f docker-compose.rag.yml up

```
**File:** `src/rag/embedding_service.py`

**Claude Code Prompt:**
```

Create embedding generation service:

Features:

- Generate embeddings using sentence-transformers
- Batch processing for efficiency
- Support multiple embedding models
- Store embeddings in Qdrant
- API endpoints for:
  - Generate embedding for text
  - Index document (chunk + embed + store)
  - Search similar documents
  - Delete document

Models to support:

- all-MiniLM-L6-v2 (lightweight, fast)
- all-mpnet-base-v2 (higher quality)
- intfloat/e5-large-v2 (best quality)

API:

- POST /v1/embed: Generate embedding
- POST /v1/index: Index document
- POST /v1/search: Similarity search
- DELETE /v1/documents/{id}: Delete document

Integration with NEURON:

- NEURON can call embedding service for RAG
- Retrieve context before inference
- Augment prompts with retrieved docs

```
### Step 4.6: Auto-Scaling

**File:** `scripts/autoscale.py`

**Claude Code Prompt:**
```

Create auto-scaling script for NEURON workers:

Metrics to Monitor:

- Queue depth (from CORTEX)
- Worker utilization
- Average wait time
- Failed scheduling attempts

Scaling Logic:

- Scale up if:
  - Queue depth > 100 for >5 minutes
  - Worker utilization > 80% for >10 minutes
  - Wait time > 60 seconds
- Scale down if:
  - Queue depth < 10 for >15 minutes
  - Worker utilization < 20% for >30 minutes
  - No failed scheduling attempts

Constraints:

- Min workers: 1
- Max workers: 10 (configurable)
- Cool-down period: 5 minutes between scale events
- Graceful scale-down (drain workers first)

Implementation:

- Query Prometheus for metrics
- Calculate scaling decision
- Call Docker Compose or Kubernetes API to scale
- Update worker pool configuration
- Log scaling events

Run as:

- WRAITH background job (every minute)
- Or standalone daemon

Include dry-run mode for testing.

```
### Step 4.7: Disaster Recovery

**File:** `scripts/backup.sh`

**Claude Code Prompt:**
```

Create comprehensive backup script:

Backup Components:

1. PostgreSQL databases:
- pg_dump with custom format
- Include all schemas
- Compress with gzip
1. Redis (if persistence enabled):
- SAVE command to trigger snapshot
- Copy RDB file
1. MinIO (ENGRAM):
- mc mirror to backup location
- Verify checksums
1. Vault:
- Export secrets (encrypted)
- Backup encryption key separately
1. Configuration files:
- All config/ directory
- docker-compose.yml

Backup Strategy:

- Daily full backups
- Hourly incremental (if changed)
- Retain: 7 daily, 4 weekly, 12 monthly

Backup Location:

- Local: /backups directory
- Remote: S3-compatible storage
- Encrypted with GPG

Verification:

- Test restore on separate environment weekly
- Checksum verification
- Completeness checks

Alerts:

- Notify on backup failure
- Notify on verification failure
- Metrics to Prometheus

Usage:
./scripts/backup.sh –full
./scripts/backup.sh –incremental
./scripts/backup.sh –verify

```
**File:** `scripts/restore.sh`

**Claude Code Prompt:**
```

Create restore script:

Steps:

1. Stop all services
2. Restore PostgreSQL:
- Drop existing databases
- Restore from backup
- Verify data
1. Restore Redis:
- Copy RDB file
- Restart Redis
1. Restore MinIO:
- mc mirror from backup
- Verify objects
1. Restore Vault:
- Import secrets
- Verify access
1. Restore configuration files
2. Start services
3. Run health checks
4. Verify system functionality

Safety:

- Prompt for confirmation
- Create backup before restore
- Dry-run mode
- Rollback on failure

Usage:
./scripts/restore.sh –from /backups/2024-01-15
./scripts/restore.sh –dry-run
./scripts/restore.sh –full

```
---

## Phase 5: Testing & Validation

### Step 5.1: Unit Tests

**For each component, create comprehensive unit tests:**

**Claude Code Prompt:**
```

Create pytest test suite for GENOME:

Test files:

- test_schema.py: Test Pydantic models
- test_validator.py: Test schema validation
- test_policy_enforcer.py: Test policy enforcement
- test_api.py: Test API endpoints
- test_database.py: Test database operations

Use fixtures for:

- Database connection (with rollback)
- Redis connection (with flush)
- Mock Vault client
- Sample schemas

Test coverage target: >80%

Example structure:

```python
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.fixture
async def db_conn():
    # Setup test database
    conn = await create_test_db()
    yield conn
    # Teardown
    await conn.close()

@pytest.mark.asyncio
async def test_schema_validation():
    validator = SchemaValidator(redis_mock)
    result = await validator.validate("Model", "qwen2-7b", spec)
    assert result.valid == True

@pytest.mark.asyncio
async def test_pii_detection():
    enforcer = PolicyEnforcer(redis_mock)
    text = "My email is test@example.com"
    pii_matches = await enforcer.detect_pii(text)
    assert len(pii_matches) == 1
    assert pii_matches[0].type == "email"
```

Repeat for all components.

```
### Step 5.2: Integration Tests

**Claude Code Prompt:**
```

Create integration test suite (tests/integration/):

Test scenarios:

1. test_job_submission:
- Submit job to CORTEX
- Verify job appears in queue
- Verify worker picks up job
- Verify job completes
- Verify results in ENGRAM
1. test_pipeline_execution:
- Create pipeline in SYNKRON
- Run pipeline
- Verify all steps execute
- Verify artifacts passed correctly
- Verify final result
1. test_model_inference:
- Register model in GENOME
- Submit inference job
- Verify NEURON loads model
- Verify inference completes
- Verify output correctness
1. test_authentication:
- Login with credentials
- Get JWT token
- Use token to access API
- Verify token expiration
- Refresh token
1. test_failure_recovery:
- Simulate worker failure
- Verify job retry
- Verify job completes on different worker
1. test_graceful_shutdown:
- Start job
- Send SIGTERM to worker
- Verify job completes
- Verify worker exits cleanly

Setup:

- Use docker-compose for test environment
- Populate test data
- Run tests in isolation
- Clean up after each test

Use pytest-docker plugin for container management.

```
### Step 5.3: End-to-End Tests

**Claude Code Prompt:**
```

Create E2E test suite (tests/e2e/):

Full workflow tests:

1. test_complete_llm_workflow:
- Bootstrap system
- Register model
- Submit inference job
- Wait for completion
- Verify results
- Check observability data
1. test_finetuning_pipeline:
- Upload training data
- Create pipeline
- Run pipeline
- Monitor progress
- Verify trained model
- Run inference with new model
1. test_multi_user:
- Create multiple users
- Submit jobs from each user
- Verify isolation
- Verify quotas
- Verify fair scheduling
1. test_failure_scenarios:
- Kill random services
- Verify recovery
- Verify no data loss
- Verify continued operation
1. test_scale_up_down:
- Start with 1 worker
- Submit many jobs
- Verify auto-scale up
- Wait for queue drain
- Verify auto-scale down

Use Playwright or Selenium for WebUI tests.
Use pytest for backend tests.

```
### Step 5.4: Load Testing

**Claude Code Prompt:**
```

Create load testing suite using Locust:

Scenarios:

1. Sustained load:
- 100 RPS for 1 hour
- Mix of job types
- Verify no degradation
1. Burst load:
- Ramp from 0 to 1000 RPS in 1 minute
- Sustain for 5 minutes
- Verify graceful handling
1. Stress test:
- Increase load until failure
- Identify bottlenecks
- Measure breaking point

Metrics to track:

- Response times (P50, P95, P99)
- Error rate
- Throughput
- Resource utilization

File: tests/load/locustfile.py

```python
from locust import HttpUser, task, between

class MNEMOSUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def submit_job(self):
        self.client.post("/v1/jobs", json={
            "kind": "inference",
            "model_ref": "qwen2-7b-instruct",
            "spec": {
                "prompt": "Hello, world!",
                "params": {"max_tokens": 100}
            }
        })
    
    @task(1)
    def list_jobs(self):
        self.client.get("/v1/jobs")
    
    @task(1)
    def get_worker_status(self):
        self.client.get("/v1/workers")
```

Run with:
locust -f tests/load/locustfile.py –host https://api.mnemos.local

```
### Step 5.5: Security Testing

**Claude Code Prompt:**
```

Create security test checklist:

Automated tests:

1. SQL injection:
- Test all input fields
- Use sqlmap
1. XSS:
- Test all user inputs
- Test stored XSS
1. Authentication bypass:
- Test JWT validation
- Test token expiration
- Test role enforcement
1. Authorization:
- Test user isolation
- Test admin endpoints
- Test RBAC
1. Secrets exposure:
- Check error messages
- Check logs
- Check API responses
1. Rate limiting:
- Test rate limits
- Test burst handling
1. DOS protection:
- Test large payloads
- Test many connections
- Test slow requests

Tools:

- OWASP ZAP (automated scanning)
- Burp Suite (manual testing)
- nmap (port scanning)
- sqlmap (SQL injection)

Manual review:

- Code review for security issues
- Dependency vulnerability scan
- Container image scanning (Trivy)
- Secrets scanning (detect-secrets)

Create security test report template.

```
---

## Phase 6: Documentation & Deployment

### Step 6.1: Documentation

**Claude Code Prompt:**
```

Create comprehensive documentation:

1. README.md:
- Project overview
- Quick start
- Architecture diagram
- Key features
- License
1. docs/installation.md:
- Prerequisites
- Installation steps
- Configuration
- Bootstrap process
- Verification
1. docs/user-guide.md:
- Submitting jobs
- Managing workers
- Creating pipelines
- Monitoring system
- Troubleshooting
1. docs/api-reference.md:
- OpenAPI spec for each service
- Authentication
- Request/response examples
- Error codes
1. docs/operations.md:
- Backup and restore
- Upgrading
- Scaling
- Monitoring
- Alerting
- Disaster recovery
1. docs/development.md:
- Setting up dev environment
- Running tests
- Contributing guidelines
- Code style
- Git workflow
1. docs/architecture.md:
- Component details
- Data flow
- Security model
- Scalability considerations
- Technology choices
1. docs/faq.md:
- Common issues
- Best practices
- Performance tuning

Use MkDocs or Docusaurus for static site generation.
Include diagrams using Mermaid or PlantUML.

```
### Step 6.2: Production Deployment Guide

**Claude Code Prompt:**
```

Create production deployment guide:

docs/production-deployment.md:

1. Infrastructure Requirements:
- Minimum: 4 CPU, 16GB RAM, 500GB storage
- Recommended: 16 CPU, 64GB RAM, 2TB NVMe SSD
- GPU: NVIDIA A100/H100 for LLM inference
- Network: 10Gbps minimum
1. Pre-deployment Checklist:
- [ ] DNS records configured
- [ ] TLS certificates obtained
- [ ] Secrets prepared (passwords, keys)
- [ ] Backup storage configured
- [ ] Monitoring alerts configured
- [ ] Firewall rules set
- [ ] Load balancer configured
1. Deployment Steps:
- Clone repository
- Configure environment variables
- Run bootstrap script
- Verify all services healthy
- Load initial schemas
- Register first model
- Run smoke tests
- Configure backups
- Set up monitoring
- Document access credentials
1. Post-deployment:
- Monitor for 24 hours
- Review logs for errors
- Test disaster recovery
- Train operations team
- Document runbooks
1. Security Hardening:
- Enable firewall (UFW)
- Configure fail2ban
- Set up intrusion detection
- Enable audit logging
- Restrict SSH access
- Regular security updates

Include infrastructure-as-code templates:

- Terraform for cloud resources
- Ansible playbooks for configuration
- Kubernetes manifests (optional)

```
### Step 6.3: CI/CD Pipeline

**File:** `.github/workflows/ci.yml`

**Claude Code Prompt:**
```

Create GitHub Actions CI/CD pipeline:

Triggers:

- push to main/develop branches
- pull requests
- manual dispatch

Jobs:

1. lint:
- Python (flake8, black, mypy)
- Shell scripts (shellcheck)
- YAML (yamllint)
- Dockerfile (hadolint)
1. test:
- Run unit tests
- Run integration tests
- Generate coverage report
- Upload to Codecov
1. security:
- Scan dependencies (safety, pip-audit)
- Scan containers (Trivy)
- Scan secrets (detect-secrets)
- SAST (Semgrep)
1. build:
- Build Docker images
- Tag with git SHA
- Push to GHCR
- Create multi-arch images (amd64, arm64)
1. deploy-staging:
- Deploy to staging environment
- Run smoke tests
- Notify team
1. deploy-production:
- Manual approval required
- Deploy to production
- Run smoke tests
- Monitor for 1 hour
- Rollback on failure

Matrix strategy for Python versions (3.11, 3.12).
Caching for dependencies.
Artifacts for test results and coverage.

```
### Step 6.4: Monitoring & Alerting Setup

**Claude Code Prompt:**
```

Create monitoring setup guide:

1. Prometheus:
- Deploy Prometheus server
- Configure scrape targets
- Load alert rules
- Set up remote write (optional)
1. Grafana:
- Deploy Grafana
- Add Prometheus data source
- Import dashboards
- Configure alerting
- Set up users and teams
1. Alert Manager:
- Deploy Alert Manager
- Configure routing
- Set up receivers:
  - Slack
  - PagerDuty
  - Email
  - Webhook
- Configure alert grouping
- Set up silences
1. Loki:
- Deploy Loki
- Configure retention
- Set up log aggregation
- Create Grafana queries
1. Jaeger (optional):
- Deploy Jaeger
- Configure trace storage
- Set up Grafana integration
1. Uptime Monitoring:
- Deploy Uptime Kuma or similar
- Monitor all service endpoints
- Configure status page

Alert Channels:

- Critical: PagerDuty (on-call)
- Warning: Slack channel
- Info: Email digest

Include:

- Dashboard JSON exports
- Alert rule YAML
- Alert Manager config
- Grafana provisioning configs

```
---

## Success Criteria

### Phase Completion Checklist

**Phase 1 - Foundation:**
- [ ] Project structure created
- [ ] Docker Compose configured
- [ ] Makefile with all targets
- [ ] Environment variables documented

**Phase 2 - Infrastructure:**
- [ ] PostgreSQL running with schemas
- [ ] Redis running and configured
- [ ] Vault initialized and unsealed
- [ ] MinIO running with buckets

**Phase 3 - Core Services:**
- [ ] GENOME running, schemas loaded
- [ ] CORTEX running, accepting jobs
- [ ] NEURON running, executing jobs
- [ ] ENGRAM storing artifacts
- [ ] WRAITH running background jobs
- [ ] RELAY routing requests
- [ ] SYNKRON executing pipelines
- [ ] TRACE collecting telemetry

**Phase 4 - Advanced Features:**
- [ ] Authentication working (JWT)
- [ ] WebUI accessible
- [ ] CLI tool functional
- [ ] RAG support added (optional)
- [ ] Auto-scaling implemented

**Phase 5 - Testing:**
- [ ] Unit tests passing (>80% coverage)
- [ ] Integration tests passing
- [ ] E2E tests passing
- [ ] Load tests completed
- [ ] Security tests passed

**Phase 6 - Production Ready:**
- [ ] Documentation complete
- [ ] CI/CD pipeline working
- [ ] Monitoring configured
- [ ] Backups automated
- [ ] Disaster recovery tested

### Final Validation

Run this command to validate full system:
```bash
make test-all
```

This should:

1. Start all services
2. Run bootstrap
3. Submit test job
4. Execute test pipeline
5. Verify results
6. Check observability data
7. Test backup/restore
8. Run security scan
9. Generate report

Expected output:

```
✓ All services healthy
✓ Test job completed successfully
✓ Test pipeline executed
✓ Observability data present
✓ Backup/restore successful
✓ Security scan passed
✓ System ready for production
```

-----

## Tips for Using Claude Code

1. **Start Small**: Implement one component at a time
2. **Test Often**: Run tests after each component
3. **Ask for Help**: Use Claude Code to debug errors
4. **Iterate**: Refine based on test results
5. **Document**: Keep track of decisions and changes

**Example Claude Code Commands:**

```bash
# Create a component
claude-code create --template python-fastapi --name genome

# Debug an error
claude-code debug --error "Connection refused to PostgreSQL"

# Optimize code
claude-code optimize --file src/cortex/scheduler.py

# Generate tests
claude-code test generate --file src/genome/validator.py

# Review code
claude-code review --file src/cortex/main.py

# Explain code
claude-code explain --file src/neuron/worker.py
```

-----

## Conclusion

You now have everything needed to build MNEMOS from scratch using Claude Code. Follow the phases sequentially, test thoroughly, and don’t skip the production hardening steps.

The result will be a production-ready AI operations platform capable of orchestrating LLM workloads at scale with enterprise-grade reliability and security.

Good luck! 🚀