# MNEMOS Phase 4: CORTEX Service (Job Orchestrator)

**Complete Implementation Specification**

## Overview

CORTEX is the central job orchestration service for the MNEMOS platform. It manages job scheduling, worker coordination, priority queues, and execution state tracking. CORTEX acts as the brain of the system, intelligently distributing work across available NEURON workers.

### Key Responsibilities

- Job submission and validation
- Priority-based queue management
- Worker pool coordination
- Job scheduling and assignment
- Retry logic with exponential backoff
- Job lifecycle management
- Execution monitoring
- Resource allocation

### Architecture Position

```
┌─────────────────────────────────────────────────────────┐
│                   External Systems                       │
│              (API Clients, SYNKRON, etc.)               │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ REST/gRPC
                     ▼
         ┌──────────────────────────┐
         │   CORTEX (This Phase)    │
         │   ┌──────────────────┐   │
         │   │   HTTP Server    │   │ Port 8080
         │   └────────┬─────────┘   │
         │   ┌────────▼─────────┐   │
         │   │   gRPC Server    │   │ Port 9090
         │   └────────┬─────────┘   │
         │            │              │
         │   ┌────────▼─────────┐   │
         │   │    Scheduler     │   │
         │   └────┬─────────┬───┘   │
         │        │         │        │
         │   ┌────▼──┐  ┌───▼────┐  │
         │   │Worker │  │ Retry  │  │
         │   │Manager│  │Handler │  │
         │   └───┬───┘  └────────┘  │
         └───────┼──────────────────┘
                 │
        ┌────────┼────────┐
        │        │        │
    ┌───▼───┐ ┌─▼──┐ ┌──▼────┐
    │ Redis │ │ PG │ │ GENOME│
    │ Queue │ │ DB │ │       │
    └───┬───┘ └────┘ └───────┘
        │
        │ Job Polling
        ▼
   ┌─────────┐
   │ NEURON  │ (Workers)
   │ Workers │
   └─────────┘
```

### Technology Stack

- **Language:** Python 3.11+
- **Framework:** FastAPI 0.104+ (HTTP), gRPC 1.59+ (Worker API)
- **Database:** PostgreSQL 16 (Job state)
- **Queue:** Redis 7 (Priority queues)
- **Validation:** Pydantic 2.5+
- **Observability:** OpenTelemetry + Prometheus
- **Serialization:** msgpack, JSON
- **Authentication:** JWT (RS256)

## Project Structure

```
src/cortex/
├── main.py                      # FastAPI application entry
├── grpc_server.py              # gRPC server for workers
├── config.py                    # Configuration management
├── dependencies.py              # FastAPI dependencies
├── middleware.py                # Custom middleware
│
├── models/                      # Pydantic models
│   ├── __init__.py
│   ├── job.py                  # Job models
│   ├── worker.py               # Worker models
│   ├── execution.py            # Execution models
│   └── retry.py                # Retry policy models
│
├── api/                         # REST API endpoints
│   ├── __init__.py
│   ├── v1/
│   │   ├── __init__.py
│   │   ├── jobs.py            # Job CRUD operations
│   │   ├── workers.py         # Worker management
│   │   ├── queues.py          # Queue operations
│   │   └── health.py          # Health checks
│   └── errors.py               # Error handlers
│
├── services/                    # Business logic
│   ├── __init__.py
│   ├── scheduler.py            # Job scheduling logic
│   ├── queue_manager.py        # Redis queue management
│   ├── worker_manager.py       # Worker pool management
│   ├── retry_handler.py        # Retry logic
│   ├── job_service.py          # Job lifecycle
│   └── metrics_collector.py    # Metrics aggregation
│
├── repository/                  # Data access layer
│   ├── __init__.py
│   ├── job_repository.py       # Job CRUD
│   ├── worker_repository.py    # Worker CRUD
│   └── execution_repository.py # Execution history
│
├── workers/                     # Worker-related code
│   ├── __init__.py
│   ├── heartbeat.py            # Heartbeat monitoring
│   ├── assignment.py           # Job assignment logic
│   └── health_checker.py       # Worker health checks
│
├── grpc/                        # gRPC definitions
│   ├── __init__.py
│   ├── cortex.proto            # Protocol buffer definitions
│   ├── cortex_pb2.py           # Generated protobuf code
│   ├── cortex_pb2_grpc.py      # Generated gRPC code
│   └── server.py               # gRPC service implementation
│
├── utils/                       # Utilities
│   ├── __init__.py
│   ├── logger.py               # Structured logging
│   ├── metrics.py              # Prometheus metrics
│   ├── tracing.py              # OpenTelemetry tracing
│   ├── redis_client.py         # Redis connection pool
│   └── db_client.py            # PostgreSQL connection
│
└── alembic/                     # Database migrations
    ├── env.py
    ├── script.py.mako
    └── versions/
        ├── 001_initial_schema.py
        ├── 002_add_retry_fields.py
        └── 003_add_worker_metrics.py

tests/
├── conftest.py                  # Pytest fixtures
├── unit/
│   ├── test_scheduler.py
│   ├── test_queue_manager.py
│   ├── test_retry_handler.py
│   └── test_worker_manager.py
├── integration/
│   ├── test_job_flow.py
│   ├── test_worker_lifecycle.py
│   └── test_api_endpoints.py
└── e2e/
    └── test_complete_orchestration.py

docker/
├── Dockerfile                   # Multi-stage build
└── docker-compose.cortex.yml   # Service-specific compose

config/
├── cortex.yaml                 # Service configuration
└── scheduler.yaml              # Scheduler tuning

scripts/
├── generate_proto.sh           # Generate gRPC code
└── run_migrations.sh           # Database migrations

docs/
├── API.md                      # API documentation
├── ARCHITECTURE.md             # Architecture details
└── OPERATIONS.md               # Operations guide

requirements.txt                 # Python dependencies
pyproject.toml                  # Project metadata
README.md                       # Service documentation
```

## Database Schema

### Jobs Table

```sql
-- Job state and metadata
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    
    -- Status tracking
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    -- Values: pending, queued, assigned, running, completed, failed, cancelled
    
    -- Priority and scheduling
    priority INTEGER NOT NULL DEFAULT 5,
    -- Range: 1 (lowest) to 10 (highest)
    max_retries INTEGER NOT NULL DEFAULT 3,
    retry_count INTEGER NOT NULL DEFAULT 0,
    timeout_seconds INTEGER NOT NULL DEFAULT 3600,
    
    -- Payload
    input_data JSONB NOT NULL,
    output_data JSONB,
    error_message TEXT,
    error_stack TEXT,
    
    -- Worker assignment
    assigned_worker_id UUID REFERENCES workers(id),
    assigned_at TIMESTAMP WITH TIME ZONE,
    
    -- Timing
    submitted_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    queued_at TIMESTAMP WITH TIME ZONE,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Metrics
    execution_duration_ms INTEGER,
    queue_wait_time_ms INTEGER,
    
    -- Metadata
    tags JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_by VARCHAR(255),
    
    -- Retry tracking
    next_retry_at TIMESTAMP WITH TIME ZONE,
    retry_backoff_seconds INTEGER,
    
    -- Indexes
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_priority ON jobs(priority DESC, submitted_at ASC);
CREATE INDEX idx_jobs_worker ON jobs(assigned_worker_id) WHERE assigned_worker_id IS NOT NULL;
CREATE INDEX idx_jobs_model ON jobs(model_name, model_version);
CREATE INDEX idx_jobs_submitted ON jobs(submitted_at DESC);
CREATE INDEX idx_jobs_retry ON jobs(next_retry_at) WHERE next_retry_at IS NOT NULL;
CREATE INDEX idx_jobs_tags ON jobs USING gin(tags);

-- Partial index for active jobs
CREATE INDEX idx_jobs_active ON jobs(priority DESC, submitted_at ASC) 
WHERE status IN ('pending', 'queued', 'assigned', 'running');
```

### Workers Table

```sql
-- Worker registration and status
CREATE TABLE workers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    hostname VARCHAR(255) NOT NULL,
    ip_address INET NOT NULL,
    port INTEGER NOT NULL,
    
    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'idle',
    -- Values: idle, busy, offline, unhealthy
    
    -- Capabilities
    capabilities JSONB NOT NULL DEFAULT '{}',
    -- Example: {"gpu": true, "gpu_count": 2, "gpu_memory": "24GB", "models": ["llama2", "mistral"]}
    
    -- Resource limits
    max_concurrent_jobs INTEGER NOT NULL DEFAULT 1,
    current_job_count INTEGER NOT NULL DEFAULT 0,
    
    -- Metrics
    total_jobs_completed INTEGER NOT NULL DEFAULT 0,
    total_jobs_failed INTEGER NOT NULL DEFAULT 0,
    average_execution_time_ms INTEGER,
    
    -- Health
    last_heartbeat TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    heartbeat_interval_seconds INTEGER NOT NULL DEFAULT 30,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    
    -- Metadata
    version VARCHAR(50),
    metadata JSONB DEFAULT '{}',
    
    -- Timing
    registered_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_active_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    UNIQUE(hostname, port)
);

-- Indexes
CREATE INDEX idx_workers_status ON workers(status);
CREATE INDEX idx_workers_heartbeat ON workers(last_heartbeat DESC);
CREATE INDEX idx_workers_capabilities ON workers USING gin(capabilities);

-- Partial index for available workers
CREATE INDEX idx_workers_available ON workers(current_job_count, max_concurrent_jobs) 
WHERE status = 'idle' AND current_job_count < max_concurrent_jobs;
```

### Executions Table

```sql
-- Execution history and audit trail
CREATE TABLE executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    worker_id UUID NOT NULL REFERENCES workers(id),
    
    -- Execution details
    attempt_number INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL,
    -- Values: started, running, completed, failed, timeout, cancelled
    
    -- Timing
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER,
    
    -- Results
    output_data JSONB,
    error_message TEXT,
    error_stack TEXT,
    error_code VARCHAR(50),
    
    -- Metrics
    cpu_usage_percent DECIMAL(5,2),
    memory_usage_mb INTEGER,
    gpu_usage_percent DECIMAL(5,2),
    
    -- Logs
    log_file_path TEXT,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_executions_job ON executions(job_id, attempt_number);
CREATE INDEX idx_executions_worker ON executions(worker_id, started_at DESC);
CREATE INDEX idx_executions_status ON executions(status);
CREATE INDEX idx_executions_started ON executions(started_at DESC);
```

### Queue Metrics Table

```sql
-- Queue performance metrics
CREATE TABLE queue_metrics (
    id BIGSERIAL PRIMARY KEY,
    
    -- Queue stats
    queue_depth INTEGER NOT NULL,
    pending_jobs INTEGER NOT NULL,
    running_jobs INTEGER NOT NULL,
    
    -- Priority breakdown
    priority_1_count INTEGER NOT NULL DEFAULT 0,
    priority_2_count INTEGER NOT NULL DEFAULT 0,
    priority_3_count INTEGER NOT NULL DEFAULT 0,
    priority_4_count INTEGER NOT NULL DEFAULT 0,
    priority_5_count INTEGER NOT NULL DEFAULT 0,
    priority_6_count INTEGER NOT NULL DEFAULT 0,
    priority_7_count INTEGER NOT NULL DEFAULT 0,
    priority_8_count INTEGER NOT NULL DEFAULT 0,
    priority_9_count INTEGER NOT NULL DEFAULT 0,
    priority_10_count INTEGER NOT NULL DEFAULT 0,
    
    -- Worker stats
    total_workers INTEGER NOT NULL,
    idle_workers INTEGER NOT NULL,
    busy_workers INTEGER NOT NULL,
    
    -- Performance
    average_wait_time_ms INTEGER,
    average_execution_time_ms INTEGER,
    
    -- Throughput
    jobs_per_minute DECIMAL(10,2),
    
    -- Timestamp
    recorded_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Index for time-series queries
CREATE INDEX idx_queue_metrics_time ON queue_metrics(recorded_at DESC);
```

## Data Models (Pydantic)

### models/job.py

```python
"""Job data models."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, validator


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class JobPriority(int, Enum):
    """Job priority levels."""
    LOWEST = 1
    LOW = 3
    NORMAL = 5
    HIGH = 7
    HIGHEST = 10


class JobCreate(BaseModel):
    """Job creation request."""
    name: str = Field(..., min_length=1, max_length=255)
    model_name: str = Field(..., min_length=1, max_length=255)
    model_version: str = Field(..., min_length=1, max_length=50)
    input_data: Dict[str, Any] = Field(...)
    priority: int = Field(default=5, ge=1, le=10)
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout_seconds: int = Field(default=3600, ge=1, le=86400)
    tags: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_by: Optional[str] = None

    @validator("input_data")
    def validate_input_data(cls, v):
        """Validate input data is not empty."""
        if not v:
            raise ValueError("input_data cannot be empty")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "name": "text-generation-task",
                "model_name": "llama2-7b",
                "model_version": "v1.0.0",
                "input_data": {
                    "prompt": "Write a poem about AI",
                    "max_tokens": 100
                },
                "priority": 7,
                "max_retries": 3,
                "timeout_seconds": 1800,
                "tags": {"project": "demo", "env": "prod"},
                "metadata": {"user_id": "user123"}
            }
        }


class JobUpdate(BaseModel):
    """Job update request."""
    status: Optional[JobStatus] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_stack: Optional[str] = None
    execution_duration_ms: Optional[int] = None
    assigned_worker_id: Optional[UUID] = None


class Job(BaseModel):
    """Job model."""
    id: UUID
    name: str
    model_name: str
    model_version: str
    status: JobStatus
    priority: int
    max_retries: int
    retry_count: int
    timeout_seconds: int
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_stack: Optional[str] = None
    assigned_worker_id: Optional[UUID] = None
    assigned_at: Optional[datetime] = None
    submitted_at: datetime
    queued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_duration_ms: Optional[int] = None
    queue_wait_time_ms: Optional[int] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_by: Optional[str] = None
    next_retry_at: Optional[datetime] = None
    retry_backoff_seconds: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class JobList(BaseModel):
    """Paginated job list response."""
    jobs: List[Job]
    total: int
    page: int
    page_size: int
    pages: int


class JobStats(BaseModel):
    """Job statistics."""
    total_jobs: int
    pending: int
    queued: int
    assigned: int
    running: int
    completed: int
    failed: int
    cancelled: int
    average_execution_time_ms: Optional[float] = None
    average_queue_time_ms: Optional[float] = None
    success_rate: Optional[float] = None
```

### models/worker.py

```python
"""Worker data models."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, IPvAnyAddress, validator


class WorkerStatus(str, Enum):
    """Worker status enumeration."""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    UNHEALTHY = "unhealthy"


class WorkerRegister(BaseModel):
    """Worker registration request."""
    name: str = Field(..., min_length=1, max_length=255)
    hostname: str = Field(..., min_length=1, max_length=255)
    ip_address: str = Field(...)
    port: int = Field(..., ge=1, le=65535)
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    max_concurrent_jobs: int = Field(default=1, ge=1, le=100)
    heartbeat_interval_seconds: int = Field(default=30, ge=5, le=300)
    version: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("capabilities")
    def validate_capabilities(cls, v):
        """Validate worker capabilities."""
        if not v:
            return {"gpu": False}
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "name": "neuron-worker-01",
                "hostname": "neuron-01.mnemos.local",
                "ip_address": "10.0.2.10",
                "port": 8000,
                "capabilities": {
                    "gpu": True,
                    "gpu_count": 2,
                    "gpu_memory": "24GB",
                    "models": ["llama2-7b", "mistral-7b"]
                },
                "max_concurrent_jobs": 2,
                "version": "1.0.0"
            }
        }


class WorkerHeartbeat(BaseModel):
    """Worker heartbeat."""
    worker_id: UUID
    status: WorkerStatus
    current_job_count: int = Field(ge=0)
    current_jobs: List[UUID] = Field(default_factory=list)
    cpu_usage_percent: Optional[float] = Field(None, ge=0, le=100)
    memory_usage_mb: Optional[int] = Field(None, ge=0)
    gpu_usage_percent: Optional[float] = Field(None, ge=0, le=100)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Worker(BaseModel):
    """Worker model."""
    id: UUID
    name: str
    hostname: str
    ip_address: str
    port: int
    status: WorkerStatus
    capabilities: Dict[str, Any]
    max_concurrent_jobs: int
    current_job_count: int
    total_jobs_completed: int
    total_jobs_failed: int
    average_execution_time_ms: Optional[int] = None
    last_heartbeat: datetime
    heartbeat_interval_seconds: int
    consecutive_failures: int
    version: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    registered_at: datetime
    last_active_at: datetime
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class WorkerList(BaseModel):
    """Worker list response."""
    workers: List[Worker]
    total: int


class WorkerStats(BaseModel):
    """Worker statistics."""
    total_workers: int
    idle_workers: int
    busy_workers: int
    offline_workers: int
    unhealthy_workers: int
    total_capacity: int
    current_utilization: int
    utilization_percent: float
```

### models/execution.py

```python
"""Execution data models."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    """Execution status enumeration."""
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ExecutionCreate(BaseModel):
    """Execution creation request."""
    job_id: UUID
    worker_id: UUID
    attempt_number: int = Field(ge=1)
    status: ExecutionStatus = ExecutionStatus.STARTED


class ExecutionUpdate(BaseModel):
    """Execution update request."""
    status: Optional[ExecutionStatus] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_stack: Optional[str] = None
    error_code: Optional[str] = None
    cpu_usage_percent: Optional[float] = Field(None, ge=0, le=100)
    memory_usage_mb: Optional[int] = Field(None, ge=0)
    gpu_usage_percent: Optional[float] = Field(None, ge=0, le=100)
    log_file_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Execution(BaseModel):
    """Execution model."""
    id: UUID
    job_id: UUID
    worker_id: UUID
    attempt_number: int
    status: ExecutionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_stack: Optional[str] = None
    error_code: Optional[str] = None
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[int] = None
    gpu_usage_percent: Optional[float] = None
    log_file_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    class Config:
        from_attributes = True
```

### models/retry.py

```python
"""Retry policy models."""
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class RetryStrategy(str, Enum):
    """Retry strategy types."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"


class RetryPolicy(BaseModel):
    """Retry policy configuration."""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_retries: int = Field(default=3, ge=0, le=10)
    initial_delay_seconds: int = Field(default=1, ge=1)
    max_delay_seconds: int = Field(default=300, ge=1)
    backoff_multiplier: float = Field(default=2.0, ge=1.0, le=10.0)
    retry_on_errors: Optional[List[str]] = None  # Error codes to retry
    jitter: bool = Field(default=True)

    class Config:
        json_schema_extra = {
            "example": {
                "strategy": "exponential",
                "max_retries": 3,
                "initial_delay_seconds": 1,
                "max_delay_seconds": 60,
                "backoff_multiplier": 2.0,
                "retry_on_errors": ["timeout", "worker_unavailable"],
                "jitter": True
            }
        }


class RetryState(BaseModel):
    """Current retry state for a job."""
    attempt_number: int = Field(ge=0)
    next_retry_at: Optional[str] = None
    backoff_seconds: Optional[int] = None
    can_retry: bool
    reason: Optional[str] = None
```

## Core Services

### services/scheduler.py

```python
"""Job scheduler implementation."""
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

import structlog
from prometheus_client import Counter, Gauge, Histogram

from cortex.models.job import Job, JobStatus
from cortex.models.worker import Worker, WorkerStatus
from cortex.repository.job_repository import JobRepository
from cortex.repository.worker_repository import WorkerRepository
from cortex.services.queue_manager import QueueManager
from cortex.services.worker_manager import WorkerManager
from cortex.utils.tracing import trace_async

logger = structlog.get_logger()

# Metrics
jobs_scheduled = Counter(
    "cortex_jobs_scheduled_total",
    "Total number of jobs scheduled",
    ["priority"]
)
jobs_assigned = Counter(
    "cortex_jobs_assigned_total",
    "Total number of jobs assigned to workers"
)
scheduling_duration = Histogram(
    "cortex_scheduling_duration_seconds",
    "Time spent scheduling jobs"
)
queue_depth = Gauge(
    "cortex_queue_depth",
    "Current queue depth by priority",
    ["priority"]
)


class Scheduler:
    """
    Job scheduler that assigns jobs to workers based on priority and availability.
    
    Scheduling algorithm:
    1. Poll queue for highest priority job
    2. Find available worker with required capabilities
    3. Assign job to worker
    4. Update job and worker state
    5. Repeat
    """

    def __init__(
        self,
        job_repo: JobRepository,
        worker_repo: WorkerRepository,
        queue_manager: QueueManager,
        worker_manager: WorkerManager,
        poll_interval: float = 1.0,
        batch_size: int = 10
    ):
        self.job_repo = job_repo
        self.worker_repo = worker_repo
        self.queue_manager = queue_manager
        self.worker_manager = worker_manager
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.logger = logger.bind(component="scheduler")

    async def start(self):
        """Start the scheduler."""
        if self._running:
            self.logger.warning("scheduler_already_running")
            return

        self._running = True
        self._task = asyncio.create_task(self._scheduling_loop())
        self.logger.info("scheduler_started")

    async def stop(self):
        """Stop the scheduler."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("scheduler_stopped")

    @trace_async("scheduler.scheduling_loop")
    async def _scheduling_loop(self):
        """Main scheduling loop."""
        self.logger.info("scheduling_loop_started")

        while self._running:
            try:
                with scheduling_duration.time():
                    await self._schedule_batch()
            except Exception as e:
                self.logger.error(
                    "scheduling_error",
                    error=str(e),
                    exc_info=True
                )

            # Update queue depth metrics
            await self._update_metrics()

            await asyncio.sleep(self.poll_interval)

    @trace_async("scheduler.schedule_batch")
    async def _schedule_batch(self):
        """Schedule a batch of jobs."""
        # Get available workers
        available_workers = await self.worker_manager.get_available_workers()
        
        if not available_workers:
            self.logger.debug("no_available_workers")
            return

        # Schedule jobs for each available worker slot
        jobs_scheduled_count = 0
        
        for worker in available_workers:
            # Calculate how many jobs this worker can accept
            available_slots = worker.max_concurrent_jobs - worker.current_job_count
            
            for _ in range(available_slots):
                if jobs_scheduled_count >= self.batch_size:
                    break

                # Get next job from queue
                job_id = await self.queue_manager.dequeue()
                
                if not job_id:
                    break

                # Assign job to worker
                success = await self._assign_job(job_id, worker.id)
                
                if success:
                    jobs_scheduled_count += 1
                    jobs_assigned.inc()
                else:
                    # Re-queue the job
                    job = await self.job_repo.get(job_id)
                    if job:
                        await self.queue_manager.enqueue(job)

        if jobs_scheduled_count > 0:
            self.logger.info(
                "batch_scheduled",
                jobs_count=jobs_scheduled_count,
                workers_used=len(available_workers)
            )

    @trace_async("scheduler.assign_job")
    async def _assign_job(self, job_id: UUID, worker_id: UUID) -> bool:
        """
        Assign a job to a worker.
        
        Args:
            job_id: Job ID
            worker_id: Worker ID
            
        Returns:
            True if assignment succeeded
        """
        try:
            # Get job and worker
            job = await self.job_repo.get(job_id)
            worker = await self.worker_repo.get(worker_id)

            if not job or not worker:
                self.logger.error(
                    "assignment_failed_not_found",
                    job_id=str(job_id),
                    worker_id=str(worker_id)
                )
                return False

            # Validate worker can handle job
            if not self._can_worker_handle_job(worker, job):
                self.logger.warning(
                    "worker_cannot_handle_job",
                    job_id=str(job_id),
                    worker_id=str(worker_id),
                    job_model=job.model_name,
                    worker_capabilities=worker.capabilities
                )
                return False

            # Update job status
            await self.job_repo.update(
                job_id,
                {
                    "status": JobStatus.ASSIGNED,
                    "assigned_worker_id": worker_id,
                    "assigned_at": datetime.utcnow(),
                    "queued_at": job.queued_at or datetime.utcnow()
                }
            )

            # Update worker
            await self.worker_repo.increment_job_count(worker_id)

            # Notify worker via gRPC (handled by WorkerManager)
            await self.worker_manager.notify_job_assignment(worker_id, job_id)

            self.logger.info(
                "job_assigned",
                job_id=str(job_id),
                worker_id=str(worker_id),
                job_name=job.name,
                worker_name=worker.name
            )

            return True

        except Exception as e:
            self.logger.error(
                "assignment_error",
                job_id=str(job_id),
                worker_id=str(worker_id),
                error=str(e),
                exc_info=True
            )
            return False

    def _can_worker_handle_job(self, worker: Worker, job: Job) -> bool:
        """
        Check if worker can handle the job.
        
        Args:
            worker: Worker instance
            job: Job instance
            
        Returns:
            True if worker can handle job
        """
        capabilities = worker.capabilities

        # Check if worker supports the model
        if "models" in capabilities:
            supported_models = capabilities["models"]
            if job.model_name not in supported_models:
                return False

        # Check GPU requirement
        if job.metadata.get("requires_gpu", False):
            if not capabilities.get("gpu", False):
                return False

        # Check memory requirement
        required_memory = job.metadata.get("required_memory_gb")
        if required_memory:
            worker_memory = capabilities.get("memory_gb")
            if not worker_memory or worker_memory < required_memory:
                return False

        return True

    async def _update_metrics(self):
        """Update queue depth metrics."""
        try:
            for priority in range(1, 11):
                depth = await self.queue_manager.get_queue_depth(priority)
                queue_depth.labels(priority=str(priority)).set(depth)
        except Exception as e:
            self.logger.error("metrics_update_error", error=str(e))

    @trace_async("scheduler.schedule_job")
    async def schedule_job(self, job: Job):
        """
        Manually schedule a specific job (bypasses queue).
        
        Args:
            job: Job to schedule
        """
        # Find available worker
        available_workers = await self.worker_manager.get_available_workers()
        
        for worker in available_workers:
            if self._can_worker_handle_job(worker, job):
                success = await self._assign_job(job.id, worker.id)
                if success:
                    jobs_scheduled.labels(priority=str(job.priority)).inc()
                    return

        # No available worker, enqueue it
        await self.queue_manager.enqueue(job)
        self.logger.info(
            "job_queued",
            job_id=str(job.id),
            priority=job.priority
        )
```

### services/queue_manager.py

```python
"""Redis-based priority queue manager."""
import json
from datetime import datetime
from typing import List, Optional
from uuid import UUID

import redis.asyncio as aioredis
import structlog
from prometheus_client import Counter, Gauge

from cortex.models.job import Job
from cortex.utils.tracing import trace_async

logger = structlog.get_logger()

# Metrics
jobs_enqueued = Counter(
    "cortex_jobs_enqueued_total",
    "Total jobs enqueued",
    ["priority"]
)
jobs_dequeued = Counter(
    "cortex_jobs_dequeued_total",
    "Total jobs dequeued",
    ["priority"]
)


class QueueManager:
    """
    Redis-based priority queue manager.
    
    Uses Redis sorted sets with job priority as score.
    Higher priority = lower score for proper ordering.
    
    Queue structure:
    - cortex:queue:{priority} - sorted set of job IDs
    - cortex:job:{job_id} - job metadata
    """

    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.logger = logger.bind(component="queue_manager")
        self.queue_prefix = "cortex:queue"
        self.job_prefix = "cortex:job"

    @trace_async("queue.enqueue")
    async def enqueue(self, job: Job):
        """
        Add job to priority queue.
        
        Args:
            job: Job to enqueue
        """
        try:
            # Calculate score (lower = higher priority, earlier timestamp)
            # Score = (11 - priority) * 1000000 + timestamp
            priority_score = (11 - job.priority) * 1000000
            timestamp_score = int(job.submitted_at.timestamp())
            score = priority_score + timestamp_score

            queue_key = f"{self.queue_prefix}:{job.priority}"
            job_key = f"{self.job_prefix}:{job.id}"

            # Store job metadata
            job_data = {
                "id": str(job.id),
                "name": job.name,
                "model_name": job.model_name,
                "model_version": job.model_version,
                "priority": job.priority,
                "submitted_at": job.submitted_at.isoformat(),
                "metadata": json.dumps(job.metadata)
            }

            async with self.redis.pipeline() as pipe:
                # Add to sorted set
                pipe.zadd(queue_key, {str(job.id): score})
                # Store job metadata with 24h TTL
                pipe.hset(job_key, mapping=job_data)
                pipe.expire(job_key, 86400)
                await pipe.execute()

            jobs_enqueued.labels(priority=str(job.priority)).inc()

            self.logger.info(
                "job_enqueued",
                job_id=str(job.id),
                job_name=job.name,
                priority=job.priority,
                score=score
            )

        except Exception as e:
            self.logger.error(
                "enqueue_error",
                job_id=str(job.id),
                error=str(e),
                exc_info=True
            )
            raise

    @trace_async("queue.dequeue")
    async def dequeue(self) -> Optional[UUID]:
        """
        Get next job from queue (highest priority, earliest submission).
        
        Returns:
            Job ID or None if queue is empty
        """
        try:
            # Check all priority queues from highest (10) to lowest (1)
            for priority in range(10, 0, -1):
                queue_key = f"{self.queue_prefix}:{priority}"

                # Get job with lowest score (ZPOPMIN)
                result = await self.redis.zpopmin(queue_key, count=1)

                if result:
                    job_id_str, score = result[0]
                    job_id = UUID(job_id_str)

                    jobs_dequeued.labels(priority=str(priority)).inc()

                    self.logger.info(
                        "job_dequeued",
                        job_id=str(job_id),
                        priority=priority,
                        score=score
                    )

                    return job_id

            return None

        except Exception as e:
            self.logger.error("dequeue_error", error=str(e), exc_info=True)
            raise

    @trace_async("queue.peek")
    async def peek(self, priority: Optional[int] = None) -> List[UUID]:
        """
        Peek at jobs in queue without removing them.
        
        Args:
            priority: Specific priority to peek, or None for all
            
        Returns:
            List of job IDs
        """
        try:
            job_ids = []

            if priority:
                priorities = [priority]
            else:
                priorities = range(10, 0, -1)

            for p in priorities:
                queue_key = f"{self.queue_prefix}:{p}"
                # Get top 10 jobs from this priority
                results = await self.redis.zrange(queue_key, 0, 9)
                job_ids.extend([UUID(jid) for jid in results])

            return job_ids

        except Exception as e:
            self.logger.error("peek_error", error=str(e), exc_info=True)
            return []

    @trace_async("queue.remove")
    async def remove(self, job_id: UUID, priority: int):
        """
        Remove specific job from queue.
        
        Args:
            job_id: Job ID to remove
            priority: Job priority
        """
        try:
            queue_key = f"{self.queue_prefix}:{priority}"
            job_key = f"{self.job_prefix}:{job_id}"

            async with self.redis.pipeline() as pipe:
                pipe.zrem(queue_key, str(job_id))
                pipe.delete(job_key)
                await pipe.execute()

            self.logger.info(
                "job_removed",
                job_id=str(job_id),
                priority=priority
            )

        except Exception as e:
            self.logger.error(
                "remove_error",
                job_id=str(job_id),
                error=str(e),
                exc_info=True
            )

    @trace_async("queue.get_depth")
    async def get_queue_depth(self, priority: Optional[int] = None) -> int:
        """
        Get queue depth.
        
        Args:
            priority: Specific priority or None for total
            
        Returns:
            Queue depth
        """
        try:
            if priority:
                queue_key = f"{self.queue_prefix}:{priority}"
                return await self.redis.zcard(queue_key)
            else:
                total = 0
                for p in range(1, 11):
                    queue_key = f"{self.queue_prefix}:{p}"
                    total += await self.redis.zcard(queue_key)
                return total

        except Exception as e:
            self.logger.error("depth_error", error=str(e))
            return 0

    @trace_async("queue.clear")
    async def clear(self, priority: Optional[int] = None):
        """
        Clear queue.
        
        Args:
            priority: Specific priority or None for all
        """
        try:
            if priority:
                priorities = [priority]
            else:
                priorities = range(1, 11)

            for p in priorities:
                queue_key = f"{self.queue_prefix}:{p}"
                await self.redis.delete(queue_key)

            self.logger.info(
                "queue_cleared",
                priority=priority or "all"
            )

        except Exception as e:
            self.logger.error("clear_error", error=str(e), exc_info=True)
```

### services/retry_handler.py

```python
"""Retry logic with exponential backoff."""
import random
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

import structlog
from prometheus_client import Counter

from cortex.models.job import Job, JobStatus
from cortex.models.retry import RetryPolicy, RetryState, RetryStrategy
from cortex.repository.job_repository import JobRepository
from cortex.services.queue_manager import QueueManager
from cortex.utils.tracing import trace_async

logger = structlog.get_logger()

# Metrics
jobs_retried = Counter(
    "cortex_jobs_retried_total",
    "Total jobs retried",
    ["attempt"]
)


class RetryHandler:
    """
    Handles job retry logic with configurable strategies.
    
    Supports:
    - Exponential backoff
    - Linear backoff
    - Fixed delay
    - Jitter
    """

    def __init__(
        self,
        job_repo: JobRepository,
        queue_manager: QueueManager,
        default_policy: Optional[RetryPolicy] = None
    ):
        self.job_repo = job_repo
        self.queue_manager = queue_manager
        self.default_policy = default_policy or RetryPolicy()
        self.logger = logger.bind(component="retry_handler")

    @trace_async("retry.should_retry")
    async def should_retry(
        self,
        job: Job,
        error_code: Optional[str] = None
    ) -> RetryState:
        """
        Determine if job should be retried.
        
        Args:
            job: Job to check
            error_code: Error code from failure
            
        Returns:
            Retry state with decision
        """
        # Check retry count
        if job.retry_count >= job.max_retries:
            return RetryState(
                attempt_number=job.retry_count,
                can_retry=False,
                reason="max_retries_exceeded"
            )

        # Check if error is retryable
        policy = self._get_policy(job)
        if policy.retry_on_errors and error_code:
            if error_code not in policy.retry_on_errors:
                return RetryState(
                    attempt_number=job.retry_count,
                    can_retry=False,
                    reason=f"error_not_retryable: {error_code}"
                )

        # Calculate next retry time
        next_retry_at, backoff_seconds = self._calculate_backoff(
            job, policy
        )

        return RetryState(
            attempt_number=job.retry_count + 1,
            next_retry_at=next_retry_at.isoformat(),
            backoff_seconds=backoff_seconds,
            can_retry=True
        )

    @trace_async("retry.schedule_retry")
    async def schedule_retry(
        self,
        job_id: UUID,
        error_message: Optional[str] = None,
        error_code: Optional[str] = None
    ):
        """
        Schedule job for retry.
        
        Args:
            job_id: Job ID
            error_message: Error message
            error_code: Error code
        """
        try:
            job = await self.job_repo.get(job_id)
            if not job:
                self.logger.error("job_not_found", job_id=str(job_id))
                return

            # Check if should retry
            retry_state = await self.should_retry(job, error_code)

            if not retry_state.can_retry:
                self.logger.info(
                    "retry_not_allowed",
                    job_id=str(job_id),
                    reason=retry_state.reason
                )
                # Mark as permanently failed
                await self.job_repo.update(
                    job_id,
                    {
                        "status": JobStatus.FAILED,
                        "error_message": error_message
                    }
                )
                return

            # Update job for retry
            next_retry_at = datetime.fromisoformat(retry_state.next_retry_at)
            
            await self.job_repo.update(
                job_id,
                {
                    "status": JobStatus.PENDING,
                    "retry_count": retry_state.attempt_number,
                    "next_retry_at": next_retry_at,
                    "retry_backoff_seconds": retry_state.backoff_seconds,
                    "assigned_worker_id": None,
                    "assigned_at": None
                }
            )

            # Re-queue job (will be picked up after backoff)
            updated_job = await self.job_repo.get(job_id)
            if updated_job:
                await self.queue_manager.enqueue(updated_job)

            jobs_retried.labels(
                attempt=str(retry_state.attempt_number)
            ).inc()

            self.logger.info(
                "retry_scheduled",
                job_id=str(job_id),
                attempt=retry_state.attempt_number,
                backoff_seconds=retry_state.backoff_seconds,
                next_retry_at=next_retry_at.isoformat()
            )

        except Exception as e:
            self.logger.error(
                "retry_schedule_error",
                job_id=str(job_id),
                error=str(e),
                exc_info=True
            )

    def _calculate_backoff(
        self,
        job: Job,
        policy: RetryPolicy
    ) -> tuple[datetime, int]:
        """
        Calculate backoff delay.
        
        Args:
            job: Job instance
            policy: Retry policy
            
        Returns:
            (next_retry_time, backoff_seconds)
        """
        if policy.strategy == RetryStrategy.EXPONENTIAL:
            # Exponential: delay = initial * (multiplier ^ attempt)
            delay = policy.initial_delay_seconds * (
                policy.backoff_multiplier ** job.retry_count
            )
        elif policy.strategy == RetryStrategy.LINEAR:
            # Linear: delay = initial + (attempt * multiplier)
            delay = policy.initial_delay_seconds + (
                job.retry_count * policy.backoff_multiplier
            )
        else:  # FIXED
            delay = policy.initial_delay_seconds

        # Cap at max delay
        delay = min(delay, policy.max_delay_seconds)

        # Add jitter if enabled
        if policy.jitter:
            jitter_range = delay * 0.1  # ±10%
            jitter = random.uniform(-jitter_range, jitter_range)
            delay += jitter

        delay_seconds = int(delay)
        next_retry_at = datetime.utcnow() + timedelta(seconds=delay_seconds)

        return next_retry_at, delay_seconds

    def _get_policy(self, job: Job) -> RetryPolicy:
        """
        Get retry policy for job.
        
        Args:
            job: Job instance
            
        Returns:
            Retry policy
        """
        # Check if job has custom policy in metadata
        if "retry_policy" in job.metadata:
            try:
                return RetryPolicy(**job.metadata["retry_policy"])
            except Exception as e:
                self.logger.warning(
                    "invalid_retry_policy",
                    job_id=str(job.id),
                    error=str(e)
                )

        return self.default_policy
```

## API Endpoints

### api/v1/jobs.py

```python
"""Job management API endpoints."""
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.dependencies import get_db, get_job_service
from cortex.models.job import (
    Job,
    JobCreate,
    JobList,
    JobStats,
    JobStatus,
    JobUpdate
)
from cortex.services.job_service import JobService

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.post(
    "",
    response_model=Job,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a new job"
)
async def create_job(
    job_create: JobCreate,
    service: JobService = Depends(get_job_service)
) -> Job:
    """
    Submit a new job for execution.
    
    The job will be validated, queued, and assigned to an available worker.
    """
    try:
        return await service.create_job(job_create)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job: {str(e)}"
        )


@router.get(
    "",
    response_model=JobList,
    summary="List jobs"
)
async def list_jobs(
    status_filter: Optional[JobStatus] = Query(None, description="Filter by status"),
    priority: Optional[int] = Query(None, ge=1, le=10, description="Filter by priority"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Page size"),
    service: JobService = Depends(get_job_service)
) -> JobList:
    """
    List jobs with optional filtering and pagination.
    """
    return await service.list_jobs(
        status=status_filter,
        priority=priority,
        model_name=model_name,
        page=page,
        page_size=page_size
    )


@router.get(
    "/{job_id}",
    response_model=Job,
    summary="Get job details"
)
async def get_job(
    job_id: UUID,
    service: JobService = Depends(get_job_service)
) -> Job:
    """
    Get detailed information about a specific job.
    """
    job = await service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    return job


@router.patch(
    "/{job_id}",
    response_model=Job,
    summary="Update job"
)
async def update_job(
    job_id: UUID,
    job_update: JobUpdate,
    service: JobService = Depends(get_job_service)
) -> Job:
    """
    Update job status or results.
    
    Typically called by workers to update execution status.
    """
    job = await service.update_job(job_id, job_update)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    return job


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel job"
)
async def cancel_job(
    job_id: UUID,
    service: JobService = Depends(get_job_service)
):
    """
    Cancel a pending or running job.
    """
    success = await service.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found or cannot be cancelled"
        )


@router.post(
    "/{job_id}/retry",
    response_model=Job,
    summary="Retry failed job"
)
async def retry_job(
    job_id: UUID,
    service: JobService = Depends(get_job_service)
) -> Job:
    """
    Manually retry a failed job.
    """
    job = await service.retry_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found or cannot be retried"
        )
    return job


@router.get(
    "/stats/summary",
    response_model=JobStats,
    summary="Get job statistics"
)
async def get_job_stats(
    service: JobService = Depends(get_job_service)
) -> JobStats:
    """
    Get aggregate statistics about jobs.
    """
    return await service.get_stats()
```

### Remaining API files

Due to length constraints, I'll provide the structure for the remaining key files. The full implementation would follow similar patterns:

- **api/v1/workers.py**: Worker registration, heartbeat, status endpoints
- **api/v1/queues.py**: Queue depth, peek, clear operations
- **api/v1/health.py**: Health check endpoints

## gRPC Service

### grpc/cortex.proto

```protobuf
syntax = "proto3";

package cortex;

// Job service for worker communication
service JobService {
    // Worker registers with CORTEX
    rpc RegisterWorker(WorkerRegistration) returns (WorkerRegistrationResponse);
    
    // Worker sends heartbeat
    rpc Heartbeat(HeartbeatRequest) returns (HeartbeatResponse);
    
    // Worker polls for jobs
    rpc PollJob(PollJobRequest) returns (PollJobResponse);
    
    // Worker updates job status
    rpc UpdateJobStatus(JobStatusUpdate) returns (JobStatusUpdateResponse);
    
    // Worker reports job completion
    rpc CompleteJob(JobCompletion) returns (JobCompletionResponse);
}

message WorkerRegistration {
    string name = 1;
    string hostname = 2;
    string ip_address = 3;
    int32 port = 4;
    map<string, string> capabilities = 5;
    int32 max_concurrent_jobs = 6;
    string version = 7;
}

message WorkerRegistrationResponse {
    string worker_id = 1;
    bool success = 2;
    string message = 3;
}

message HeartbeatRequest {
    string worker_id = 1;
    string status = 2;
    int32 current_job_count = 3;
    repeated string current_jobs = 4;
    map<string, double> metrics = 5;
}

message HeartbeatResponse {
    bool acknowledged = 1;
    repeated string pending_jobs = 2;
}

message PollJobRequest {
    string worker_id = 1;
}

message PollJobResponse {
    bool has_job = 1;
    string job_id = 2;
    string job_name = 3;
    string model_name = 4;
    string model_version = 5;
    bytes input_data = 6;
    int32 timeout_seconds = 7;
}

message JobStatusUpdate {
    string job_id = 1;
    string worker_id = 2;
    string status = 3;
    string message = 4;
    map<string, double> metrics = 5;
}

message JobStatusUpdateResponse {
    bool acknowledged = 1;
}

message JobCompletion {
    string job_id = 1;
    string worker_id = 2;
    bool success = 3;
    bytes output_data = 4;
    string error_message = 5;
    int64 duration_ms = 6;
    map<string, double> metrics = 7;
}

message JobCompletionResponse {
    bool acknowledged = 1;
    string message = 2;
}
```

## Dockerfile

```dockerfile
# Multi-stage build for CORTEX service

# Stage 1: Build
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 cortex && \
    chown -R cortex:cortex /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/cortex/.local

# Copy application code
COPY --chown=cortex:cortex src/cortex /app/cortex
COPY --chown=cortex:cortex config /app/config
COPY --chown=cortex:cortex alembic.ini /app/

# Set Python path
ENV PYTHONPATH=/app
ENV PATH=/home/cortex/.local/bin:$PATH

# Switch to non-root user
USER cortex

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 9090

# Run application
CMD ["python", "-m", "uvicorn", "cortex.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

## requirements.txt

```txt
# Core framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Database
asyncpg==0.29.0
sqlalchemy[asyncio]==2.0.23
alembic==1.13.0

# Redis
redis[hiredis]==5.0.1

# gRPC
grpcio==1.59.3
grpcio-tools==1.59.3
protobuf==4.25.1

# Observability
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0
opentelemetry-instrumentation-fastapi==0.42b0
opentelemetry-instrumentation-sqlalchemy==0.42b0
opentelemetry-instrumentation-redis==0.42b0
opentelemetry-exporter-otlp==1.21.0
prometheus-client==0.19.0
structlog==23.2.0

# HTTP client
httpx==0.25.2

# Serialization
msgpack==1.0.7
orjson==3.9.10

# Utilities
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dateutil==2.8.2

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-mock==3.12.0
httpx==0.25.2
faker==20.1.0

# Development
black==23.12.0
isort==5.13.2
flake8==6.1.0
mypy==1.7.1
```

## Configuration Files

### config/cortex.yaml

```yaml
# CORTEX Service Configuration

server:
  host: "0.0.0.0"
  http_port: 8080
  grpc_port: 9090
  workers: 4
  reload: false
  log_level: "info"

scheduler:
  enabled: true
  poll_interval: 1.0  # seconds
  batch_size: 10
  worker_timeout: 300  # seconds
  job_timeout_default: 3600  # seconds

queue:
  redis_url: "${REDIS_URL}"
  max_queue_size: 10000
  queue_prefix: "cortex:queue"
  job_prefix: "cortex:job"

retry:
  strategy: "exponential"
  max_retries: 3
  initial_delay: 1
  max_delay: 300
  backoff_multiplier: 2.0
  jitter: true

database:
  url: "${DATABASE_URL}"
  pool_size: 20
  max_overflow: 10
  pool_timeout: 30
  pool_recycle: 3600
  echo: false

security:
  jwt_secret_key: "${JWT_SECRET_KEY}"
  jwt_algorithm: "RS256"
  jwt_expiration: 3600
  require_auth: true

observability:
  otlp_endpoint: "${OTLP_ENDPOINT}"
  service_name: "cortex"
  service_version: "1.0.0"
  environment: "${ENVIRONMENT}"
  metrics_enabled: true
  tracing_enabled: true
  logging_level: "info"

cors:
  enabled: true
  origins:
    - "http://localhost:3000"
    - "https://mnemos.local"
  allow_credentials: true
  allow_methods: ["*"]
  allow_headers: ["*"]
```

## Testing

### tests/conftest.py

```python
"""Pytest configuration and fixtures."""
import asyncio
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from cortex.main import app
from cortex.models.base import Base
from cortex.utils.db_client import get_db

# Test database URL
TEST_DATABASE_URL = "postgresql+asyncpg://cortex:cortex@localhost:5432/cortex_test"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture
def client(db_session: AsyncSession) -> TestClient:
    """Create test client."""
    def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture
def sample_job_create():
    """Sample job creation data."""
    from cortex.models.job import JobCreate
    
    return JobCreate(
        name="test-job",
        model_name="test-model",
        model_version="v1.0.0",
        input_data={"prompt": "test"},
        priority=5,
        max_retries=3,
        timeout_seconds=3600
    )
```

### tests/integration/test_job_flow.py

```python
"""Integration tests for complete job flow."""
import pytest
from uuid import uuid4

from cortex.models.job import JobCreate, JobStatus


@pytest.mark.asyncio
async def test_complete_job_flow(client, sample_job_create):
    """Test complete job lifecycle."""
    # 1. Submit job
    response = client.post("/api/v1/jobs", json=sample_job_create.dict())
    assert response.status_code == 201
    job = response.json()
    job_id = job["id"]
    assert job["status"] == "pending"
    
    # 2. Check job was queued
    response = client.get(f"/api/v1/jobs/{job_id}")
    assert response.status_code == 200
    assert response.json()["status"] in ["pending", "queued"]
    
    # 3. Simulate worker assignment
    # (In real scenario, scheduler would do this)
    
    # 4. Update job to running
    response = client.patch(
        f"/api/v1/jobs/{job_id}",
        json={"status": "running"}
    )
    assert response.status_code == 200
    
    # 5. Complete job
    response = client.patch(
        f"/api/v1/jobs/{job_id}",
        json={
            "status": "completed",
            "output_data": {"result": "success"},
            "execution_duration_ms": 1000
        }
    )
    assert response.status_code == 200
    job = response.json()
    assert job["status"] == "completed"
    assert job["output_data"]["result"] == "success"


@pytest.mark.asyncio
async def test_job_retry_on_failure(client, sample_job_create):
    """Test job retry logic."""
    # Submit job
    response = client.post("/api/v1/jobs", json=sample_job_create.dict())
    job_id = response.json()["id"]
    
    # Mark as failed
    response = client.patch(
        f"/api/v1/jobs/{job_id}",
        json={
            "status": "failed",
            "error_message": "Worker crashed"
        }
    )
    
    # Retry job
    response = client.post(f"/api/v1/jobs/{job_id}/retry")
    assert response.status_code == 200
    job = response.json()
    assert job["retry_count"] == 1
    assert job["status"] == "pending"
```

## Documentation

### README.md

```markdown
# CORTEX - Job Orchestrator Service

CORTEX is the central orchestration service for MNEMOS, managing job scheduling, worker coordination, and execution monitoring.

## Features

- Priority-based job scheduling (1-10)
- Redis-backed job queues
- Worker pool management
- Automatic retry with exponential backoff
- gRPC API for workers
- REST API for job management
- Comprehensive observability

## Quick Start

### Using Docker Compose

```bash
docker-compose up cortex
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Start server
uvicorn cortex.main:app --reload
```

## API Documentation

### REST API

- **POST /api/v1/jobs** - Submit job
- **GET /api/v1/jobs** - List jobs
- **GET /api/v1/jobs/{id}** - Get job
- **PATCH /api/v1/jobs/{id}** - Update job
- **DELETE /api/v1/jobs/{id}** - Cancel job
- **POST /api/v1/jobs/{id}/retry** - Retry job

### gRPC API

- **RegisterWorker** - Worker registration
- **Heartbeat** - Worker heartbeat
- **PollJob** - Poll for jobs
- **UpdateJobStatus** - Update job status
- **CompleteJob** - Report completion

## Configuration

See `config/cortex.yaml` for configuration options.

## Monitoring

Metrics available at http://localhost:8080/metrics

Key metrics:
- `cortex_jobs_queued_total`
- `cortex_jobs_scheduled_total`
- `cortex_queue_depth`
- `cortex_scheduling_duration_seconds`

## Testing

```bash
pytest tests/ -v --cov=cortex
```

## Architecture

CORTEX uses a scheduler that continuously polls the Redis priority queue and assigns jobs to available workers. Jobs are tracked in PostgreSQL for durability and audit purposes.
```

---

## Implementation Notes

### Priority Guidelines

1. **Implement core services first:**
   - QueueManager (Redis operations)
   - Scheduler (Job assignment)
   - RetryHandler (Retry logic)
   - JobService (Business logic)

2. **Then add repositories:**
   - JobRepository
   - WorkerRepository
   - ExecutionRepository

3. **Build APIs:**
   - REST endpoints
   - gRPC server

4. **Add observability:**
   - Metrics
   - Tracing
   - Logging

### Testing Strategy

- Unit tests: Test individual components in isolation
- Integration tests: Test component interactions
- E2E tests: Test complete job flows
- Load tests: Test with 100+ concurrent jobs

### Performance Considerations

- Use connection pooling for PostgreSQL and Redis
- Implement batch processing in scheduler
- Use async operations throughout
- Add caching for frequently accessed data
- Monitor queue depth and worker utilization

### Security

- Validate all inputs with Pydantic
- Use parameterized SQL queries
- Implement rate limiting
- Add authentication/authorization
- Encrypt sensitive data

---

**Phase 4 Status:** Complete specification ready for implementation  
**Estimated Lines:** ~2,500  
**Estimated Time:** 8-12 hours with Claude Code  
**Dependencies:** Phase 1 (Infrastructure), Phase 2 (Configuration), Phase 3 (GENOME)

