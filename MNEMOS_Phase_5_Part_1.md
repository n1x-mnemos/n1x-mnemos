# MNEMOS Phase 5 Part 1: NEURON Service - Core Implementation

**Status:** Ready for Implementation  
**Priority:** High  
**Complexity:** High  
**Part:** 1 of 2  
**Estimated Lines (Part 1):** ~1,500  

## Document Structure

This is Part 1 of the NEURON Service implementation, covering:
- Overview and architecture
- File structure
- Data models (worker, job, runtime)
- Core service implementations
- Client implementations
- Utility modules

**Part 2** will cover:
- Runtime implementations (vLLM, PyTorch, Transformers)
- Testing requirements
- Docker configuration and deployment
- Integration examples
- Operations guide

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Data Models](#data-models)
5. [Core Services](#core-services)
6. [Client Implementations](#client-implementations)
7. [Utility Modules](#utility-modules)
8. [Configuration](#configuration)

## Overview

### Purpose

NEURON is the worker runtime service that executes AI inference jobs in the MNEMOS platform. Each NEURON worker:
- Registers with CORTEX orchestrator
- Polls for available jobs
- Loads AI models on-demand from storage
- Executes inference using multiple runtime backends
- Streams results back to clients
- Monitors and reports resource utilization

### Key Responsibilities

**Job Execution:**
- Acquire jobs from CORTEX job queue
- Load required models from ENGRAM/MinIO
- Execute inference using appropriate runtime
- Handle timeouts and errors gracefully
- Report results back to CORTEX

**Resource Management:**
- Monitor GPU utilization and memory
- Track system resources (CPU, RAM, disk)
- Enforce job concurrency limits
- Manage model caching locally

**Health & Status:**
- Send periodic heartbeats to CORTEX
- Report current status and capabilities
- Handle graceful shutdown signals
- Complete in-flight jobs before termination

### Technology Stack

- **Language:** Python 3.11+
- **Framework:** FastAPI for health API
- **Communication:** gRPC for CORTEX, HTTP for GENOME
- **Runtimes:** vLLM, PyTorch, Transformers
- **Storage:** MinIO client for model artifacts
- **Monitoring:** Prometheus metrics, OpenTelemetry traces
- **Logging:** Structured logging with structlog

## Architecture

### High-Level Component Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                    NEURON Worker Service                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐         ┌─────────────────┐              │
│  │  Worker Manager │────────►│   Heartbeat     │              │
│  │  - Lifecycle    │         │   Service       │              │
│  │  - Coordination │         │  - Health check │              │
│  └────────┬────────┘         └─────────────────┘              │
│           │                                                     │
│           │                  ┌─────────────────┐              │
│           └─────────────────►│   Job Poller    │              │
│                              │   Service       │              │
│                              │  - Acquire jobs │              │
│                              └────────┬────────┘              │
│                                       │                        │
│                                       ▼                        │
│                              ┌─────────────────┐              │
│                              │    Executor     │              │
│                              │    Service      │              │
│                              │  - Orchestrate  │              │
│                              └────────┬────────┘              │
│                                       │                        │
│              ┌────────────────────────┼──────────────┐        │
│              │                        │              │        │
│              ▼                        ▼              ▼        │
│      ┌─────────────┐         ┌─────────────┐  ┌──────────┐  │
│      │ Model Loader│         │  Runtime    │  │ Resource │  │
│      │  - Download │         │  Registry   │  │ Monitor  │  │
│      │  - Cache    │         │  - vLLM     │  │ - GPU    │  │
│      └─────────────┘         │  - PyTorch  │  │ - System │  │
│                              │  - Transform│  └──────────┘  │
│                              └─────────────┘                 │
│                                                               │
└───────────────────────────────────────────────────────────────┘
           │                    │                  │
           ▼                    ▼                  ▼
    ┌──────────┐        ┌──────────┐      ┌──────────┐
    │  CORTEX  │        │  ENGRAM  │      │  GENOME  │
    │  gRPC    │        │  MinIO   │      │   HTTP   │
    └──────────┘        └──────────┘      └──────────┘
```

### Worker Lifecycle State Machine

```
┌──────────┐
│  START   │
└────┬─────┘
     │
     ▼
┌──────────────────┐
│  INITIALIZING    │
│  - Load config   │
│  - Setup logging │
│  - Detect GPU    │
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│  CONNECTING      │
│  - CORTEX gRPC   │
│  - GENOME HTTP   │
│  - MinIO client  │
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│  REGISTERING     │
│  - Send caps     │
│  - Get worker_id │
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│  STARTING        │
│  - Heartbeat     │
│  - Job poller    │
│  - Health API    │
└────┬─────────────┘
     │
     ▼
     ┌─────────────────────────┐
     │      IDLE/BUSY          │
     │  ┌──────────────────┐   │
     │  │  Poll for job    │   │
     │  └────┬─────────────┘   │
     │       │                  │
     │       ▼                  │
     │  ┌──────────────────┐   │
     │  │  Acquire job     │   │
     │  └────┬─────────────┘   │
     │       │                  │
     │       ▼                  │
     │  ┌──────────────────┐   │
     │  │  Load model      │   │
     │  └────┬─────────────┘   │
     │       │                  │
     │       ▼                  │
     │  ┌──────────────────┐   │
     │  │  Execute job     │   │
     │  └────┬─────────────┘   │
     │       │                  │
     │       ▼                  │
     │  ┌──────────────────┐   │
     │  │  Report result   │   │
     │  └────┬─────────────┘   │
     │       │                  │
     │       └──────────────────┤
     │                          │
     │  Send heartbeat (30s)    │
     └────────────┬─────────────┘
                  │
                  ▼ (SIGTERM/SIGINT)
           ┌──────────────────┐
           │ SHUTTING_DOWN    │
           │  - Stop poller   │
           │  - Finish jobs   │
           │  - Deregister    │
           └────┬─────────────┘
                │
                ▼
           ┌──────────┐
           │  STOPPED │
           └──────────┘
```

### Communication Flow

```
Worker Startup:
    NEURON → CORTEX: RegisterWorker(capabilities, resources)
    CORTEX → NEURON: WorkerRegistered(worker_id)

Job Acquisition:
    NEURON → CORTEX: AcquireJob(worker_id)
    CORTEX → NEURON: Job(spec) or NoJobAvailable

Model Loading:
    NEURON → GENOME: GetModelMetadata(family, version)
    GENOME → NEURON: ModelMetadata(artifact_path, schema)
    NEURON → ENGRAM: DownloadModel(artifact_path)
    ENGRAM → NEURON: ModelArtifact(bytes)

Job Execution:
    NEURON → CORTEX: UpdateJobStatus(job_id, LOADING_MODEL)
    NEURON → CORTEX: UpdateJobStatus(job_id, EXECUTING)
    NEURON → CORTEX: StreamResult(job_id, chunk) [optional]
    NEURON → CORTEX: ReportResult(job_id, result, metrics)

Health Monitoring:
    NEURON → CORTEX: Heartbeat(worker_id, status, resources) [every 30s]
    CORTEX → NEURON: HeartbeatAck(action?) [shutdown, restart, etc]

Worker Shutdown:
    NEURON → CORTEX: DeregisterWorker(worker_id)
    CORTEX → NEURON: DeregistrationAck
```

## File Structure

```
src/neuron/
├── __init__.py                      # Package initialization
├── main.py                          # FastAPI health API entry
├── worker.py                        # Main worker entry point
├── config.py                        # Configuration management
│
├── models/                          # Data models
│   ├── __init__.py
│   ├── worker.py                    # Worker models
│   ├── job.py                       # Job models
│   └── runtime.py                   # Runtime config models
│
├── services/                        # Core services
│   ├── __init__.py
│   ├── heartbeat.py                 # Heartbeat service
│   ├── poller.py                    # Job polling service
│   ├── executor.py                  # Job execution orchestrator
│   ├── model_loader.py              # Model loading/caching
│   └── resource_monitor.py          # Resource monitoring
│
├── runtimes/                        # Runtime implementations
│   ├── __init__.py
│   ├── base.py                      # Base runtime interface
│   ├── vllm_runtime.py              # vLLM implementation
│   ├── pytorch_runtime.py           # PyTorch implementation
│   └── transformers_runtime.py      # Transformers implementation
│
├── clients/                         # External service clients
│   ├── __init__.py
│   ├── cortex_client.py             # gRPC client for CORTEX
│   ├── genome_client.py             # HTTP client for GENOME
│   └── engram_client.py             # MinIO client wrapper
│
├── utils/                           # Utility modules
│   ├── __init__.py
│   ├── gpu.py                       # GPU detection/monitoring
│   ├── retry.py                     # Retry decorators
│   └── streaming.py                 # Result streaming helpers
│
├── api/                             # FastAPI endpoints
│   ├── __init__.py
│   ├── health.py                    # Health check endpoints
│   └── metrics.py                   # Prometheus metrics
│
├── Dockerfile                       # Multi-stage build
├── requirements.txt                 # Python dependencies
├── pyproject.toml                  # Package metadata
└── README.md                       # Service documentation

tests/neuron/
├── __init__.py
├── conftest.py                     # Shared test fixtures
│
├── unit/                           # Unit tests
│   ├── test_worker.py
│   ├── test_heartbeat.py
│   ├── test_poller.py
│   ├── test_executor.py
│   ├── test_model_loader.py
│   └── test_runtimes.py
│
├── integration/                    # Integration tests
│   ├── test_cortex_integration.py
│   ├── test_genome_integration.py
│   ├── test_engram_integration.py
│   └── test_job_execution.py
│
└── e2e/                           # End-to-end tests
    └── test_complete_workflow.py
```

## Data Models

### 1. Worker Models (`models/worker.py`)

```python
"""Worker registration and status models."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator
import platform
import socket


class WorkerStatus(str, Enum):
    """Worker status enumeration."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"


class GPUInfo(BaseModel):
    """GPU device information."""
    index: int = Field(..., ge=0, description="GPU device index")
    name: str = Field(..., description="GPU device name")
    uuid: Optional[str] = Field(None, description="GPU UUID")
    total_memory_mb: int = Field(..., gt=0, description="Total GPU memory in MB")
    available_memory_mb: int = Field(..., ge=0, description="Available GPU memory in MB")
    utilization_percent: float = Field(..., ge=0, le=100, description="GPU utilization percentage")
    temperature_celsius: Optional[float] = Field(None, description="GPU temperature in Celsius")
    power_watts: Optional[float] = Field(None, description="GPU power consumption in Watts")
    
    class Config:
        json_schema_extra = {
            "example": {
                "index": 0,
                "name": "NVIDIA A100-SXM4-40GB",
                "uuid": "GPU-12345678-1234-1234-1234-123456789012",
                "total_memory_mb": 40960,
                "available_memory_mb": 38912,
                "utilization_percent": 15.5,
                "temperature_celsius": 42.0,
                "power_watts": 185.3
            }
        }


class SystemResources(BaseModel):
    """System resource information."""
    cpu_count: int = Field(..., gt=0, description="Number of CPU cores")
    cpu_percent: float = Field(..., ge=0, le=100, description="CPU utilization percentage")
    memory_total_mb: int = Field(..., gt=0, description="Total system memory in MB")
    memory_available_mb: int = Field(..., ge=0, description="Available system memory in MB")
    disk_total_gb: int = Field(..., gt=0, description="Total disk space in GB")
    disk_available_gb: int = Field(..., ge=0, description="Available disk space in GB")
    gpus: List[GPUInfo] = Field(default_factory=list, description="List of GPU devices")
    
    @property
    def memory_percent(self) -> float:
        """Calculate memory usage percentage."""
        if self.memory_total_mb == 0:
            return 0.0
        return ((self.memory_total_mb - self.memory_available_mb) / self.memory_total_mb) * 100
    
    @property
    def disk_percent(self) -> float:
        """Calculate disk usage percentage."""
        if self.disk_total_gb == 0:
            return 0.0
        return ((self.disk_total_gb - self.disk_available_gb) / self.disk_total_gb) * 100
    
    class Config:
        json_schema_extra = {
            "example": {
                "cpu_count": 32,
                "cpu_percent": 12.5,
                "memory_total_mb": 262144,
                "memory_available_mb": 245760,
                "disk_total_gb": 1000,
                "disk_available_gb": 850,
                "gpus": []
            }
        }


class WorkerCapabilities(BaseModel):
    """Worker capability information."""
    supported_runtimes: List[str] = Field(
        ...,
        min_length=1,
        description="List of supported runtime types"
    )
    max_concurrent_jobs: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum number of concurrent jobs"
    )
    supports_streaming: bool = Field(
        default=True,
        description="Whether worker supports result streaming"
    )
    supports_gpu: bool = Field(
        ...,
        description="Whether worker has GPU support"
    )
    max_model_size_gb: int = Field(
        default=10,
        gt=0,
        description="Maximum model size that can be loaded (GB)"
    )
    max_batch_size: int = Field(
        default=32,
        ge=1,
        description="Maximum inference batch size"
    )
    
    @validator('supported_runtimes')
    def validate_runtimes(cls, v):
        """Validate runtime types."""
        allowed = {'vllm', 'pytorch', 'transformers'}
        invalid = set(v) - allowed
        if invalid:
            raise ValueError(f"Invalid runtimes: {invalid}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "supported_runtimes": ["vllm", "pytorch", "transformers"],
                "max_concurrent_jobs": 1,
                "supports_streaming": True,
                "supports_gpu": True,
                "max_model_size_gb": 40,
                "max_batch_size": 32
            }
        }


class WorkerRegistration(BaseModel):
    """Worker registration request."""
    hostname: str = Field(..., description="Worker hostname")
    ip_address: str = Field(..., description="Worker IP address")
    version: str = Field(..., description="Worker software version")
    capabilities: WorkerCapabilities = Field(..., description="Worker capabilities")
    resources: SystemResources = Field(..., description="Current system resources")
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom worker tags for filtering/routing"
    )
    
    @validator('hostname')
    def validate_hostname(cls, v):
        """Validate hostname."""
        if not v or len(v) > 253:
            raise ValueError("Invalid hostname length")
        return v
    
    @validator('ip_address')
    def validate_ip(cls, v):
        """Validate IP address format."""
        try:
            socket.inet_aton(v)
        except socket.error:
            raise ValueError("Invalid IP address format")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "hostname": "neuron-worker-01",
                "ip_address": "10.0.1.50",
                "version": "1.0.0",
                "capabilities": {
                    "supported_runtimes": ["vllm"],
                    "max_concurrent_jobs": 1,
                    "supports_streaming": True,
                    "supports_gpu": True,
                    "max_model_size_gb": 40
                },
                "resources": {
                    "cpu_count": 32,
                    "cpu_percent": 5.0,
                    "memory_total_mb": 262144,
                    "memory_available_mb": 250000,
                    "disk_total_gb": 1000,
                    "disk_available_gb": 800,
                    "gpus": [
                        {
                            "index": 0,
                            "name": "NVIDIA A100-SXM4-40GB",
                            "total_memory_mb": 40960,
                            "available_memory_mb": 40000,
                            "utilization_percent": 0.0
                        }
                    ]
                },
                "tags": {
                    "datacenter": "us-west-2",
                    "zone": "a",
                    "instance_type": "p4d.24xlarge"
                }
            }
        }


class WorkerInfo(BaseModel):
    """Complete worker information returned after registration."""
    worker_id: str = Field(..., description="Unique worker identifier")
    hostname: str = Field(..., description="Worker hostname")
    status: WorkerStatus = Field(..., description="Current worker status")
    capabilities: WorkerCapabilities = Field(..., description="Worker capabilities")
    current_jobs: List[str] = Field(
        default_factory=list,
        description="List of currently executing job IDs"
    )
    total_jobs_completed: int = Field(
        default=0,
        ge=0,
        description="Total number of jobs completed"
    )
    total_jobs_failed: int = Field(
        default=0,
        ge=0,
        description="Total number of jobs failed"
    )
    last_heartbeat: datetime = Field(
        ...,
        description="Timestamp of last heartbeat"
    )
    registered_at: datetime = Field(
        ...,
        description="Worker registration timestamp"
    )
    updated_at: datetime = Field(
        ...,
        description="Last update timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "worker_id": "wrk_a1b2c3d4e5f6",
                "hostname": "neuron-worker-01",
                "status": "idle",
                "current_jobs": [],
                "total_jobs_completed": 42,
                "total_jobs_failed": 2,
                "last_heartbeat": "2024-01-15T10:30:00Z",
                "registered_at": "2024-01-15T08:00:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }


class Heartbeat(BaseModel):
    """Worker heartbeat message."""
    worker_id: str = Field(..., description="Worker ID")
    status: WorkerStatus = Field(..., description="Current status")
    current_jobs: List[str] = Field(
        default_factory=list,
        description="Currently executing job IDs"
    )
    resources: SystemResources = Field(
        ...,
        description="Current system resources"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Heartbeat timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "worker_id": "wrk_a1b2c3d4e5f6",
                "status": "busy",
                "current_jobs": ["job_xyz789"],
                "resources": {
                    "cpu_count": 32,
                    "cpu_percent": 45.2,
                    "memory_total_mb": 262144,
                    "memory_available_mb": 180000,
                    "disk_total_gb": 1000,
                    "disk_available_gb": 750,
                    "gpus": [
                        {
                            "index": 0,
                            "name": "NVIDIA A100",
                            "total_memory_mb": 40960,
                            "available_memory_mb": 10240,
                            "utilization_percent": 85.5
                        }
                    ]
                },
                "timestamp": "2024-01-15T10:30:15Z"
            }
        }


class HeartbeatResponse(BaseModel):
    """Heartbeat acknowledgment from CORTEX."""
    acknowledged: bool = Field(..., description="Whether heartbeat was acknowledged")
    action: Optional[str] = Field(
        None,
        description="Action to take (shutdown, restart, update_config)"
    )
    message: Optional[str] = Field(
        None,
        description="Optional message from orchestrator"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "acknowledged": True,
                "action": None,
                "message": None
            }
        }
```

### 2. Job Models (`models/job.py`)

```python
"""Job execution models."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    ACQUIRED = "acquired"
    LOADING_MODEL = "loading_model"
    EXECUTING = "executing"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class JobType(str, Enum):
    """Job type enumeration."""
    INFERENCE = "inference"
    BATCH_INFERENCE = "batch_inference"
    TRAINING = "training"
    FINE_TUNING = "fine_tuning"
    EVALUATION = "evaluation"


class JobPriority(int, Enum):
    """Job priority levels (1=lowest, 10=highest)."""
    LOW = 1
    BELOW_NORMAL = 3
    NORMAL = 5
    ABOVE_NORMAL = 7
    HIGH = 8
    CRITICAL = 10


class ModelConfig(BaseModel):
    """Model configuration for job execution."""
    family: str = Field(..., min_length=1, description="Model family name")
    version: str = Field(..., min_length=1, description="Model version")
    runtime: str = Field(..., description="Runtime type (vllm, pytorch, transformers)")
    runtime_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Runtime-specific configuration parameters"
    )
    
    @validator('runtime')
    def validate_runtime(cls, v):
        """Validate runtime type."""
        allowed = ['vllm', 'pytorch', 'transformers']
        if v not in allowed:
            raise ValueError(f"Runtime must be one of: {allowed}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "family": "llama-2",
                "version": "7b-chat",
                "runtime": "vllm",
                "runtime_config": {
                    "tensor_parallel_size": 1,
                    "max_model_len": 4096,
                    "dtype": "float16",
                    "gpu_memory_utilization": 0.9
                }
            }
        }


class JobInput(BaseModel):
    """Job input data."""
    prompt: str = Field(..., min_length=1, description="Input prompt or text")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Generation/inference parameters"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context or metadata"
    )
    
    @validator('parameters')
    def validate_parameters(cls, v):
        """Validate common parameters."""
        if 'max_tokens' in v and v['max_tokens'] <= 0:
            raise ValueError("max_tokens must be positive")
        if 'temperature' in v and not (0 <= v['temperature'] <= 2):
            raise ValueError("temperature must be between 0 and 2")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Explain quantum computing in simple terms",
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 500,
                    "top_p": 0.9,
                    "top_k": 50
                },
                "context": {
                    "user_id": "user_12345",
                    "session_id": "session_67890",
                    "conversation_history": []
                }
            }
        }


class JobOutput(BaseModel):
    """Job output data."""
    result: Any = Field(..., description="Primary execution result")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Result metadata (tokens, model info, etc)"
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "result": "Quantum computing is a revolutionary approach...",
                "metadata": {
                    "model": "llama-2-7b-chat",
                    "tokens_generated": 247,
                    "finish_reason": "stop"
                },
                "metrics": {
                    "latency_ms": 1523.4,
                    "tokens_per_second": 162.3,
                    "first_token_latency_ms": 125.6
                }
            }
        }


class JobSpec(BaseModel):
    """Complete job specification."""
    job_id: str = Field(..., description="Unique job identifier")
    job_type: JobType = Field(..., description="Job type")
    priority: JobPriority = Field(
        default=JobPriority.NORMAL,
        description="Job priority"
    )
    model: ModelConfig = Field(..., description="Model configuration")
    input: JobInput = Field(..., description="Job input data")
    timeout_seconds: int = Field(
        default=300,
        gt=0,
        le=3600,
        description="Job timeout in seconds"
    )
    retry_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Retry configuration if job fails"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional job metadata"
    )
    submitted_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Job submission timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_xyz789abc",
                "job_type": "inference",
                "priority": 5,
                "model": {
                    "family": "llama-2",
                    "version": "7b-chat",
                    "runtime": "vllm"
                },
                "input": {
                    "prompt": "What is machine learning?",
                    "parameters": {
                        "temperature": 0.7,
                        "max_tokens": 200
                    }
                },
                "timeout_seconds": 300,
                "metadata": {
                    "user_id": "user_12345",
                    "request_id": "req_abc456"
                },
                "submitted_at": "2024-01-15T10:30:00Z"
            }
        }


class JobResult(BaseModel):
    """Job execution result."""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Execution status")
    output: Optional[JobOutput] = Field(
        None,
        description="Job output (if successful)"
    )
    error: Optional[str] = Field(
        None,
        description="Error message (if failed)"
    )
    error_details: Optional[Dict[str, Any]] = Field(
        None,
        description="Detailed error information"
    )
    worker_id: str = Field(..., description="Worker that executed the job")
    started_at: datetime = Field(..., description="Execution start timestamp")
    completed_at: Optional[datetime] = Field(
        None,
        description="Execution completion timestamp"
    )
    duration_seconds: Optional[float] = Field(
        None,
        ge=0,
        description="Total execution duration in seconds"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_xyz789abc",
                "status": "completed",
                "output": {
                    "result": "Machine learning is...",
                    "metadata": {
                        "tokens_generated": 180
                    },
                    "metrics": {
                        "latency_ms": 850.3
                    }
                },
                "worker_id": "wrk_a1b2c3d4e5f6",
                "started_at": "2024-01-15T10:30:00Z",
                "completed_at": "2024-01-15T10:32:30Z",
                "duration_seconds": 150.5
            }
        }


class StreamChunk(BaseModel):
    """Streaming result chunk for progressive output."""
    job_id: str = Field(..., description="Job identifier")
    sequence: int = Field(..., ge=0, description="Chunk sequence number")
    content: str = Field(..., description="Chunk content/text")
    is_final: bool = Field(
        default=False,
        description="Whether this is the final chunk"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk-specific metadata"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Chunk generation timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_xyz789abc",
                "sequence": 5,
                "content": "Machine learning is a subset of artificial intelligence...",
                "is_final": False,
                "metadata": {
                    "tokens_in_chunk": 12
                },
                "timestamp": "2024-01-15T10:30:15.234Z"
            }
        }
```

### 3. Runtime Models (`models/runtime.py`)

```python
"""Runtime configuration models."""

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, validator


class RuntimeType(str, Enum):
    """Supported runtime types."""
    VLLM = "vllm"
    PYTORCH = "pytorch"
    TRANSFORMERS = "transformers"


class VLLMConfig(BaseModel):
    """vLLM runtime configuration."""
    tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of GPUs for tensor parallelism"
    )
    pipeline_parallel_size: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of GPUs for pipeline parallelism"
    )
    max_model_len: Optional[int] = Field(
        None,
        gt=0,
        description="Maximum model context length"
    )
    gpu_memory_utilization: float = Field(
        default=0.9,
        gt=0,
        le=1.0,
        description="Fraction of GPU memory to use"
    )
    dtype: str = Field(
        default="float16",
        pattern="^(float16|float32|bfloat16|auto)$",
        description="Data type for model weights"
    )
    max_num_seqs: int = Field(
        default=256,
        ge=1,
        description="Maximum number of sequences in a batch"
    )
    max_num_batched_tokens: Optional[int] = Field(
        None,
        gt=0,
        description="Maximum number of batched tokens"
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Trust remote code from model repository"
    )
    quantization: Optional[str] = Field(
        None,
        pattern="^(awq|gptq|squeezellm|fp8)?$",
        description="Quantization method"
    )
    
    @validator('tensor_parallel_size', 'pipeline_parallel_size')
    def validate_parallel_size(cls, v):
        """Validate parallel size is reasonable."""
        if v not in [1, 2, 4, 8]:
            raise ValueError("Parallel size should be power of 2 (1, 2, 4, 8)")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "tensor_parallel_size": 1,
                "max_model_len": 4096,
                "gpu_memory_utilization": 0.9,
                "dtype": "float16",
                "max_num_seqs": 256
            }
        }


class PyTorchConfig(BaseModel):
    """PyTorch runtime configuration."""
    device: str = Field(
        default="cuda",
        pattern="^(cuda|cpu|cuda:[0-9]+)$",
        description="Device to run inference on"
    )
    dtype: str = Field(
        default="float16",
        pattern="^(float16|float32|bfloat16)$",
        description="Data type for tensors"
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        le=128,
        description="Inference batch size"
    )
    num_workers: int = Field(
        default=0,
        ge=0,
        le=16,
        description="Number of data loading workers"
    )
    compile: bool = Field(
        default=False,
        description="Use torch.compile for optimization"
    )
    use_amp: bool = Field(
        default=False,
        description="Use automatic mixed precision"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "device": "cuda",
                "dtype": "float16",
                "batch_size": 1,
                "compile": False
            }
        }


class TransformersConfig(BaseModel):
    """HuggingFace Transformers runtime configuration."""
    device: str = Field(
        default="cuda",
        pattern="^(cuda|cpu|auto)$",
        description="Device placement"
    )
    torch_dtype: str = Field(
        default="float16",
        pattern="^(float16|float32|bfloat16|auto)$",
        description="PyTorch data type"
    )
    load_in_8bit: bool = Field(
        default=False,
        description="Load model in 8-bit precision"
    )
    load_in_4bit: bool = Field(
        default=False,
        description="Load model in 4-bit precision"
    )
    device_map: str = Field(
        default="auto",
        description="Device mapping strategy"
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Trust remote code from model hub"
    )
    use_flash_attention_2: bool = Field(
        default=False,
        description="Use Flash Attention 2 optimization"
    )
    low_cpu_mem_usage: bool = Field(
        default=True,
        description="Minimize CPU memory usage during loading"
    )
    
    @validator('load_in_8bit', 'load_in_4bit')
    def validate_quantization(cls, v, values):
        """Ensure only one quantization method is active."""
        if v and values.get('load_in_8bit') and values.get('load_in_4bit'):
            raise ValueError("Cannot use both 8-bit and 4-bit quantization")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "device": "auto",
                "torch_dtype": "float16",
                "load_in_8bit": False,
                "trust_remote_code": False,
                "use_flash_attention_2": True
            }
        }
```

## Core Services

### 1. Worker Manager (`worker.py`)

```python
"""Main NEURON worker orchestrator."""

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from typing import Optional

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .config import Settings
from .models.worker import WorkerStatus, WorkerRegistration, WorkerCapabilities
from .services.heartbeat import HeartbeatService
from .services.poller import JobPollerService
from .services.executor import ExecutorService
from .services.resource_monitor import ResourceMonitor
from .clients.cortex_client import CortexClient
from .clients.genome_client import GenomeClient
from .clients.engram_client import EngramClient

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)


class NEURONWorker:
    """Main NEURON worker class that orchestrates all services."""
    
    def __init__(self, settings: Settings):
        """Initialize worker with settings.
        
        Args:
            settings: Worker configuration settings
        """
        self.settings = settings
        self.worker_id: Optional[str] = None
        self.status = WorkerStatus.INITIALIZING
        self.shutdown_event = asyncio.Event()
        
        # Initialize clients (will be connected on start)
        self.cortex_client: Optional[CortexClient] = None
        self.genome_client: Optional[GenomeClient] = None
        self.engram_client: Optional[EngramClient] = None
        
        # Initialize services (will be started after registration)
        self.heartbeat_service: Optional[HeartbeatService] = None
        self.poller_service: Optional[JobPollerService] = None
        self.executor_service: Optional[ExecutorService] = None
        self.resource_monitor: Optional[ResourceMonitor] = None
        
        logger.info(
            "worker_initialized",
            hostname=settings.hostname,
            version=settings.version,
            supported_runtimes=settings.supported_runtimes
        )
    
    async def start(self):
        """Start the worker and all services.
        
        This is the main entry point that:
        1. Initializes connections to external services
        2. Registers with CORTEX
        3. Starts all background services
        4. Enters main loop waiting for shutdown signal
        """
        with tracer.start_as_current_span("worker_start") as span:
            try:
                # Setup signal handlers for graceful shutdown
                self._setup_signal_handlers()
                
                # Initialize all external connections
                await self._initialize_connections()
                span.add_event("connections_initialized")
                
                # Register this worker with CORTEX
                await self._register_worker()
                span.add_event("worker_registered", {"worker_id": self.worker_id})
                
                # Start all background services
                await self._start_services()
                span.add_event("services_started")
                
                # Transition to idle status
                self.status = WorkerStatus.IDLE
                
                logger.info(
                    "worker_started",
                    worker_id=self.worker_id,
                    status=self.status.value
                )
                
                # Wait for shutdown signal
                await self.shutdown_event.wait()
                
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                logger.error(
                    "worker_start_failed",
                    error=str(e),
                    exc_info=True
                )
                self.status = WorkerStatus.ERROR
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(
                "shutdown_signal_received",
                signal=signal.Signals(signum).name
            )
            # Create task to shutdown gracefully
            asyncio.create_task(self.shutdown())
        
        # Handle SIGTERM (docker stop, k8s termination)
        signal.signal(signal.SIGTERM, signal_handler)
        # Handle SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def _initialize_connections(self):
        """Initialize connections to external services."""
        logger.info("initializing_connections")
        
        # Initialize CORTEX gRPC client
        self.cortex_client = CortexClient(
            url=self.settings.cortex_grpc_url,
            timeout=self.settings.cortex_timeout
        )
        await self.cortex_client.connect()
        
        # Initialize GENOME HTTP client
        self.genome_client = GenomeClient(
            url=self.settings.genome_http_url,
            timeout=self.settings.genome_timeout
        )
        
        # Verify GENOME is reachable
        genome_health = await self.genome_client.health_check()
        if not genome_health:
            raise RuntimeError("GENOME service is not healthy")
        
        # Initialize ENGRAM/MinIO client
        self.engram_client = EngramClient(
            endpoint=self.settings.minio_endpoint,
            access_key=self.settings.minio_access_key,
            secret_key=self.settings.minio_secret_key,
            secure=self.settings.minio_secure
        )
        
        # Verify MinIO is reachable
        await self.engram_client.health_check()
        
        logger.info("connections_initialized")
    
    async def _register_worker(self):
        """Register this worker with CORTEX orchestrator."""
        logger.info("registering_worker")
        
        # Initialize resource monitor to detect capabilities
        self.resource_monitor = ResourceMonitor(self.settings)
        await self.resource_monitor.start()
        
        # Get current system resources
        resources = await self.resource_monitor.get_current_resources()
        
        # Build worker capabilities
        capabilities = WorkerCapabilities(
            supported_runtimes=self.settings.supported_runtimes,
            max_concurrent_jobs=self.settings.max_concurrent_jobs,
            supports_streaming=True,
            supports_gpu=len(resources.gpus) > 0,
            max_model_size_gb=self.settings.max_model_size_gb,
            max_batch_size=self.settings.max_batch_size
        )
        
        # Create registration request
        registration = WorkerRegistration(
            hostname=self.settings.hostname,
            ip_address=self.settings.ip_address,
            version=self.settings.version,
            capabilities=capabilities,
            resources=resources,
            tags=self.settings.worker_tags
        )
        
        # Send registration to CORTEX
        response = await self.cortex_client.register_worker(registration)
        self.worker_id = response.worker_id
        
        logger.info(
            "worker_registered",
            worker_id=self.worker_id,
            gpus=len(resources.gpus),
            runtimes=capabilities.supported_runtimes
        )
    
    async def _start_services(self):
        """Start all worker background services."""
        logger.info("starting_services")
        
        # Initialize executor service
        self.executor_service = ExecutorService(
            settings=self.settings,
            worker_id=self.worker_id,
            genome_client=self.genome_client,
            engram_client=self.engram_client,
            cortex_client=self.cortex_client
        )
        await self.executor_service.start()
        
        # Initialize and start heartbeat service
        self.heartbeat_service = HeartbeatService(
            worker_id=self.worker_id,
            cortex_client=self.cortex_client,
            resource_monitor=self.resource_monitor,
            executor_service=self.executor_service,
            interval_seconds=self.settings.heartbeat_interval
        )
        await self.heartbeat_service.start()
        
        # Initialize and start job poller service
        self.poller_service = JobPollerService(
            worker_id=self.worker_id,
            cortex_client=self.cortex_client,
            executor_service=self.executor_service,
            poll_interval=self.settings.poll_interval
        )
        await self.poller_service.start()
        
        logger.info("services_started")
    
    async def shutdown(self):
        """Gracefully shutdown the worker.
        
        Shutdown sequence:
        1. Stop accepting new jobs (stop poller)
        2. Wait for in-flight jobs to complete (with timeout)
        3. Stop heartbeat service
        4. Deregister from CORTEX
        5. Close all connections
        """
        if self.shutdown_event.is_set():
            return
        
        logger.info("worker_shutdown_initiated")
        self.status = WorkerStatus.SHUTTING_DOWN
        
        with tracer.start_as_current_span("worker_shutdown"):
            try:
                # Stop accepting new jobs
                if self.poller_service:
                    await self.poller_service.stop()
                    logger.info("job_poller_stopped")
                
                # Wait for in-flight jobs to complete
                if self.executor_service:
                    try:
                        await asyncio.wait_for(
                            self.executor_service.wait_for_completion(),
                            timeout=self.settings.shutdown_timeout
                        )
                        logger.info("in_flight_jobs_completed")
                    except asyncio.TimeoutError:
                        logger.warning(
                            "shutdown_timeout_exceeded",
                            timeout=self.settings.shutdown_timeout,
                            message="Some jobs may not have completed"
                        )
                
                # Stop heartbeat service
                if self.heartbeat_service:
                    await self.heartbeat_service.stop()
                    logger.info("heartbeat_service_stopped")
                
                # Deregister from CORTEX
                if self.worker_id and self.cortex_client:
                    await self.cortex_client.deregister_worker(self.worker_id)
                    logger.info("worker_deregistered")
                
                # Stop resource monitor
                if self.resource_monitor:
                    await self.resource_monitor.stop()
                
                # Close all client connections
                if self.cortex_client:
                    await self.cortex_client.close()
                
                logger.info("worker_shutdown_complete")
                
            except Exception as e:
                logger.error(
                    "shutdown_error",
                    error=str(e),
                    exc_info=True
                )
            finally:
                self.shutdown_event.set()


async def main():
    """Main entry point for NEURON worker."""
    # Load configuration
    settings = Settings()
    
    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Create and start worker
    worker = NEURONWorker(settings)
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt_received")
    except Exception as e:
        logger.error(
            "worker_fatal_error",
            error=str(e),
            exc_info=True
        )
        sys.exit(1)
    finally:
        await worker.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

### Configuration Module (`config.py`)

```python
"""NEURON worker configuration."""

import os
import platform
import socket
from typing import Dict, List

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_hostname() -> str:
    """Get system hostname."""
    return platform.node()


def get_ip_address() -> str:
    """Get primary IP address."""
    try:
        # Connect to external address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


class Settings(BaseSettings):
    """NEURON worker configuration settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="NEURON_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8"
    )
    
    # Worker Identity
    hostname: str = Field(
        default_factory=get_hostname,
        description="Worker hostname"
    )
    ip_address: str = Field(
        default_factory=get_ip_address,
        description="Worker IP address"
    )
    version: str = Field(
        default="1.0.0",
        description="Worker version"
    )
    worker_tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom worker tags"
    )
    
    # External Service URLs
    cortex_grpc_url: str = Field(
        default="cortex:9090",
        description="CORTEX gRPC endpoint"
    )
    cortex_timeout: int = Field(
        default=30,
        gt=0,
        description="CORTEX request timeout (seconds)"
    )
    genome_http_url: str = Field(
        default="http://genome:8081",
        description="GENOME HTTP endpoint"
    )
    genome_timeout: int = Field(
        default=30,
        gt=0,
        description="GENOME request timeout (seconds)"
    )
    
    # MinIO/ENGRAM Configuration
    minio_endpoint: str = Field(
        default="engram:9000",
        description="MinIO endpoint"
    )
    minio_access_key: str = Field(
        default="minioadmin",
        description="MinIO access key"
    )
    minio_secret_key: str = Field(
        default="minioadmin",
        description="MinIO secret key"
    )
    minio_secure: bool = Field(
        default=False,
        description="Use HTTPS for MinIO"
    )
    
    # Worker Capabilities
    supported_runtimes: List[str] = Field(
        default=["vllm", "pytorch", "transformers"],
        description="List of supported runtime types"
    )
    max_concurrent_jobs: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum concurrent jobs"
    )
    max_model_size_gb: int = Field(
        default=40,
        gt=0,
        description="Maximum model size in GB"
    )
    max_batch_size: int = Field(
        default=32,
        ge=1,
        description="Maximum inference batch size"
    )
    
    # Service Timing
    heartbeat_interval: int = Field(
        default=30,
        gt=0,
        description="Heartbeat interval (seconds)"
    )
    poll_interval: float = Field(
        default=1.0,
        gt=0,
        description="Job polling interval (seconds)"
    )
    shutdown_timeout: int = Field(
        default=300,
        gt=0,
        description="Graceful shutdown timeout (seconds)"
    )
    
    # Storage Paths
    model_cache_dir: str = Field(
        default="/tmp/neuron/models",
        description="Local model cache directory"
    )
    
    # API Server
    api_host: str = Field(
        default="0.0.0.0",
        description="Health API host"
    )
    api_port: int = Field(
        default=8000,
        gt=0,
        lt=65536,
        description="Health API port"
    )
    
    # Logging & Monitoring
    log_level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Logging level"
    )
    otlp_endpoint: str = Field(
        default="otel-collector:4317",
        description="OpenTelemetry collector endpoint"
    )
    
    @validator('worker_tags', pre=True)
    def parse_worker_tags(cls, v):
        """Parse worker tags from environment string."""
        if isinstance(v, str):
            tags = {}
            for pair in v.split(','):
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    tags[key.strip()] = value.strip()
            return tags
        return v
    
    @validator('supported_runtimes', pre=True)
    def parse_supported_runtimes(cls, v):
        """Parse supported runtimes from environment string."""
        if isinstance(v, str):
            return [r.strip() for r in v.split(',') if r.strip()]
        return v
```

---

## Continuation

This completes **Part 1** of the Phase 5 NEURON Service implementation specification. 

**What's included in Part 1:**
✅ Complete overview and architecture  
✅ File structure  
✅ All data models (Worker, Job, Runtime)  
✅ Core worker manager implementation  
✅ Configuration system  

**Part 2 will include:**
- Heartbeat Service implementation
- Job Poller Service implementation
- Executor Service implementation
- Model Loader implementation
- Resource Monitor implementation
- Client implementations (CORTEX, GENOME, ENGRAM)
- Runtime implementations (vLLM, PyTorch, Transformers)
- Utility modules
- Testing suite
- Docker configuration
- Deployment guide

---

**Document:** MNEMOS_Phase_5_Part_1.md  
**Status:** Complete  
**Next:** Generate Part 2

[View Part 1 document](computer:///mnt/user-data/outputs/MNEMOS_Phase_5_Part_1.md)
