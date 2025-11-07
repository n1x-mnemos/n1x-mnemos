# MNEMOS Phase 3: GENOME Service (Schema Registry)

**Schema Validation and Registry Service**

## Overview

GENOME is the schema registry service for the MNEMOS platform, responsible for validating and managing:
- Model schemas (AI model configurations)
- Pipeline definitions (DAG-based workflows)
- Policy definitions (RBAC, quotas, rate limits)

This service ensures that all components in the MNEMOS ecosystem operate with validated, versioned, and consistent schemas.

**Status:** Phase 3 of 13  
**Priority:** High  
**Complexity:** Medium  
**Estimated Lines:** ~2,000  
**Dependencies:** PostgreSQL, Vault (from Phase 2)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GENOME Service                        │
│                                                          │
│  ┌──────────────┐      ┌──────────────┐                │
│  │   REST API   │      │   Validator  │                │
│  │   (FastAPI)  │─────▶│   Service    │                │
│  └──────┬───────┘      └──────┬───────┘                │
│         │                     │                         │
│         │                     │                         │
│  ┌──────▼──────────────────────▼──────┐                │
│  │      Registry Service                │                │
│  │  - Model Registry                    │                │
│  │  - Pipeline Registry                 │                │
│  │  - Policy Registry                   │                │
│  └──────┬──────────────────────────────┘                │
│         │                                                │
│  ┌──────▼───────┐      ┌──────────────┐                │
│  │  Repository  │─────▶│  PostgreSQL  │                │
│  │    Layer     │      │   Database   │                │
│  └──────────────┘      └──────────────┘                │
│                                                          │
│  Observability: OpenTelemetry + Prometheus + Logs       │
└─────────────────────────────────────────────────────────┘
```

## File Structure

```
src/genome/
├── __init__.py
├── main.py                      # FastAPI application entry
├── config.py                    # Configuration management
├── models/                      # Pydantic data models
│   ├── __init__.py
│   ├── base.py                 # Base model classes
│   ├── model.py                # Model schema definitions
│   ├── pipeline.py             # Pipeline schema definitions
│   ├── policy.py               # Policy schema definitions
│   └── validation.py           # Validation results
├── api/                        # REST API endpoints
│   ├── __init__.py
│   ├── deps.py                 # API dependencies
│   ├── models.py               # Model endpoints
│   ├── pipelines.py            # Pipeline endpoints
│   ├── policies.py             # Policy endpoints
│   └── health.py               # Health check endpoints
├── services/                   # Business logic
│   ├── __init__.py
│   ├── validator.py            # Schema validation service
│   ├── model_registry.py       # Model registry service
│   ├── pipeline_registry.py    # Pipeline registry service
│   └── policy_registry.py      # Policy registry service
├── repository/                 # Database access layer
│   ├── __init__.py
│   ├── base.py                 # Base repository
│   ├── models.py               # Model repository
│   ├── pipelines.py            # Pipeline repository
│   └── policies.py             # Policy repository
├── schemas/                    # JSON schemas for validation
│   ├── model_schema.json
│   ├── pipeline_schema.json
│   └── policy_schema.json
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── logging.py              # Structured logging
│   ├── metrics.py              # Prometheus metrics
│   └── tracing.py              # OpenTelemetry tracing
└── exceptions.py               # Custom exceptions

tests/
├── __init__.py
├── conftest.py                 # Pytest fixtures
├── unit/
│   ├── test_validator.py
│   ├── test_model_registry.py
│   ├── test_pipeline_registry.py
│   └── test_policy_registry.py
├── integration/
│   ├── test_models_api.py
│   ├── test_pipelines_api.py
│   └── test_policies_api.py
└── fixtures/
    ├── sample_model.json
    ├── sample_pipeline.json
    └── sample_policy.json

config/
└── genome/
    ├── config.yaml             # Service configuration
    └── logging.yaml            # Logging configuration

docker/
└── genome/
    ├── Dockerfile              # Multi-stage build
    └── .dockerignore

migrations/
└── genome/
    ├── V001__create_models_table.sql
    ├── V002__create_pipelines_table.sql
    ├── V003__create_policies_table.sql
    └── V004__create_indexes.sql

docs/
└── genome/
    ├── README.md               # Service documentation
    ├── API.md                  # API documentation
    └── SCHEMAS.md              # Schema documentation

requirements.txt                # Python dependencies
pyproject.toml                  # Project configuration
Makefile                        # Build and run tasks
```

## Implementation Summary

### Core Components

1. **Data Models** (`models/`)
   - Base schemas with timestamp mixins
   - Model schemas with family, runtime, artifact configs
   - Pipeline schemas with DAG validation
   - Policy schemas with rule validation
   - Validation result models

2. **API Layer** (`api/`)
   - REST endpoints for models, pipelines, policies
   - Full CRUD operations
   - Query filtering and pagination
   - Health check endpoints
   - OpenAPI documentation

3. **Services** (`services/`)
   - Validator service (JSON Schema + custom validation)
   - Registry services (model, pipeline, policy)
   - Business logic and validation rules
   - Vault integration for secrets

4. **Repository Layer** (`repository/`)
   - Async database operations
   - PostgreSQL queries with SQLAlchemy
   - CRUD operations
   - Version management

5. **Database** (`migrations/`)
   - Models table with JSONB columns
   - Pipelines table with constraints
   - Policies table with priority
   - Indexes for performance

## Key Features

### Schema Validation
- **Pydantic Models**: Type-safe validation with field validators
- **JSON Schema**: Additional validation layer
- **Custom Rules**: Business logic validation
- **DAG Validation**: Cycle detection for pipelines
- **Version Checking**: Semantic versioning enforcement

### Version Management
- Semantic versioning (MAJOR.MINOR.PATCH)
- Multiple versions per schema
- Latest version retrieval
- Version listing and comparison

### API Features
- RESTful design with proper HTTP methods
- Pagination support (skip/limit)
- Filtering by attributes
- Comprehensive error handling
- OpenAPI/Swagger documentation

### Database Design
- JSONB columns for flexible schema storage
- GIN indexes for JSONB queries
- Composite indexes for common queries
- Triggers for automatic timestamps
- Materialized views for statistics

### Observability
- OpenTelemetry distributed tracing
- Prometheus metrics export
- Structured logging with structlog
- Health check endpoints
- Performance monitoring

## Quick Start

### 1. Database Setup

```bash
# Run migrations
cd migrations/genome
psql -U mnemos -d mnemos -f V001__create_models_table.sql
psql -U mnemos -d mnemos -f V002__create_pipelines_table.sql
psql -U mnemos -d mnemos -f V003__create_policies_table.sql
psql -U mnemos -d mnemos -f V004__create_indexes.sql
```

### 2. Build Docker Image

```bash
cd docker/genome
docker build -t mnemos/genome:1.0.0 .
```

### 3. Start Service

```bash
docker-compose up -d genome
```

### 4. Verify Health

```bash
curl http://localhost:8081/health
```

### 5. Register Sample Model

```bash
curl -X POST http://localhost:8081/api/v1/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "llama2-7b-chat",
    "version": "1.0.0",
    "family": "llama",
    "description": "Llama 2 7B Chat model",
    "artifact": {
      "bucket": "mnemos-models",
      "path": "llama2/7b-chat/v1.0.0",
      "format": "safetensors",
      "size_bytes": 13483847680
    },
    "runtime": {
      "type": "vllm",
      "gpu_memory_utilization": 0.9,
      "max_model_len": 4096,
      "tensor_parallel_size": 1
    },
    "requirements": {
      "min_gpu_memory_gb": 16,
      "min_ram_gb": 32
    },
    "capabilities": ["chat", "completion"],
    "tags": ["production", "english"]
  }'
```

## API Reference

### Models Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/v1/models | Register new model |
| GET | /api/v1/models | List all models |
| GET | /api/v1/models/{name} | Get model by name |
| GET | /api/v1/models/id/{id} | Get model by ID |
| PUT | /api/v1/models/{name} | Update model |
| DELETE | /api/v1/models/{name} | Delete model |
| POST | /api/v1/models/validate | Validate model |
| GET | /api/v1/models/{name}/versions | List versions |

### Pipelines Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/v1/pipelines | Register pipeline |
| GET | /api/v1/pipelines | List pipelines |
| GET | /api/v1/pipelines/{name} | Get pipeline |
| PUT | /api/v1/pipelines/{name} | Update pipeline |
| DELETE | /api/v1/pipelines/{name} | Delete pipeline |
| POST | /api/v1/pipelines/validate | Validate pipeline |
| GET | /api/v1/pipelines/{name}/execution-order | Get execution order |

### Policies Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/v1/policies | Register policy |
| GET | /api/v1/policies | List policies |
| GET | /api/v1/policies/{name} | Get policy |
| PUT | /api/v1/policies/{name} | Update policy |
| DELETE | /api/v1/policies/{name} | Delete policy |
| POST | /api/v1/policies/validate | Validate policy |

### Health Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Service health |
| GET | /health/ready | Readiness probe |
| GET | /health/live | Liveness probe |

## Data Model Examples

### Model Schema Example

```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "name": "llama2-7b-chat",
  "version": "1.0.0",
  "family": "llama",
  "description": "Llama 2 7B Chat model optimized for conversational AI",
  "artifact": {
    "bucket": "mnemos-models",
    "path": "llama2/7b-chat/v1.0.0",
    "format": "safetensors",
    "size_bytes": 13483847680,
    "checksum": "sha256:abc123..."
  },
  "runtime": {
    "type": "vllm",
    "gpu_memory_utilization": 0.9,
    "max_model_len": 4096,
    "tensor_parallel_size": 1,
    "dtype": "float16",
    "quantization": null
  },
  "requirements": {
    "min_gpu_memory_gb": 16,
    "min_ram_gb": 32,
    "cuda_version": "11.8"
  },
  "capabilities": ["chat", "completion", "instruction-following"],
  "metadata": {
    "training_date": "2023-07-01",
    "license": "llama2-license",
    "languages": ["en"]
  },
  "tags": ["production", "chat", "english"],
  "active": true,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

### Pipeline Schema Example

```json
{
  "id": "456e7890-e89b-12d3-a456-426614174001",
  "name": "text-generation-pipeline",
  "version": "1.0.0",
  "description": "Complete text generation pipeline with preprocessing and postprocessing",
  "steps": [
    {
      "id": "validate",
      "type": "validation",
      "config": {
        "max_length": 10000,
        "check_language": true
      },
      "depends_on": [],
      "timeout_seconds": 5,
      "retry_count": 1,
      "optional": false
    },
    {
      "id": "preprocess",
      "type": "preprocessing",
      "config": {
        "clean_text": true,
        "remove_stopwords": false,
        "lowercase": false
      },
      "depends_on": ["validate"],
      "timeout_seconds": 10,
      "retry_count": 2,
      "optional": false
    },
    {
      "id": "generate",
      "type": "inference",
      "model": "llama2-7b-chat",
      "config": {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 500
      },
      "depends_on": ["preprocess"],
      "timeout_seconds": 300,
      "retry_count": 3,
      "optional": false
    },
    {
      "id": "postprocess",
      "type": "postprocessing",
      "config": {
        "trim_whitespace": true,
        "format_markdown": false
      },
      "depends_on": ["generate"],
      "timeout_seconds": 10,
      "retry_count": 2,
      "optional": false
    }
  ],
  "timeout_seconds": 600,
  "max_parallel": 2,
  "on_failure": "abort",
  "metadata": {
    "author": "ml-team",
    "use_case": "chat-completion"
  },
  "tags": ["production", "text-generation"],
  "active": true,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

### Policy Schema Example

```json
{
  "id": "789e1011-e89b-12d3-a456-426614174002",
  "name": "user-rate-limits",
  "version": "1.0.0",
  "type": "rate_limit",
  "description": "Rate limits for regular users",
  "rules": [
    {
      "entity": "user:*",
      "resource": "api:inference",
      "requests": 100,
      "window_seconds": 60,
      "burst": 120
    },
    {
      "entity": "user:*",
      "resource": "api:models",
      "requests": 50,
      "window_seconds": 60,
      "burst": 60
    }
  ],
  "enforcement": "hard",
  "priority": 100,
  "metadata": {
    "team": "platform",
    "review_date": "2024-06-01"
  },
  "tags": ["rate-limiting", "users"],
  "active": true,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

## Configuration

### Environment Variables

```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mnemos
POSTGRES_USER=mnemos
POSTGRES_PASSWORD=<secure-password>

# Vault
VAULT_ADDR=http://localhost:8200
VAULT_TOKEN=<vault-token>

# Service
GENOME_PORT=8081
GENOME_WORKERS=4
GENOME_LOG_LEVEL=INFO

# Observability
OTEL_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=genome
```

### Configuration File (config.yaml)

```yaml
service:
  name: genome
  version: 1.0.0
  environment: production

server:
  host: 0.0.0.0
  port: 8081
  workers: 4
  timeout: 60

database:
  pool_size: 20
  max_overflow: 10
  pool_timeout: 30
  pool_recycle: 3600

validation:
  max_model_size_gb: 500
  max_pipeline_steps: 50
  max_policy_rules: 100

api:
  rate_limit:
    enabled: true
    requests_per_minute: 100
    burst: 150
```

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=genome --cov-report=html

# Specific test file
pytest tests/unit/test_validator.py -v

# Integration tests only
pytest tests/integration/ -v
```

### Code Quality

```bash
# Format code
black src/genome tests/

# Lint
ruff check src/genome tests/

# Type checking
mypy src/genome

# All quality checks
make lint
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run service
uvicorn genome.main:app --reload --port 8081

# Access docs
open http://localhost:8081/docs
```

## Monitoring

### Metrics

Available at `http://localhost:9091/metrics`:

```
# Request metrics
genome_http_requests_total{method="GET",endpoint="/api/v1/models",status="200"} 1234
genome_http_request_duration_seconds{method="POST",endpoint="/api/v1/models"} 0.045

# Registry metrics
genome_models_total{family="llama"} 15
genome_pipelines_total{active="true"} 8
genome_policies_total{type="rate_limit"} 12

# Validation metrics
genome_validations_total{type="model",result="success"} 156
genome_validation_duration_seconds{type="pipeline"} 0.012

# Database metrics
genome_db_connections_active 8
genome_db_query_duration_seconds{operation="select"} 0.003
```

### Health Checks

```bash
# Service health
curl http://localhost:8081/health

# Readiness (for K8s)
curl http://localhost:8081/health/ready

# Liveness (for K8s)
curl http://localhost:8081/health/live
```

### Logs

Structured JSON logs:

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "info",
  "service": "genome",
  "event": "model_registered",
  "model_name": "llama2-7b-chat",
  "version": "1.0.0",
  "family": "llama",
  "request_id": "req-123",
  "duration_ms": 45
}
```

## Security

### Authentication
- JWT token validation (coming in Phase 4)
- API key support
- Role-based access control

### Data Protection
- Sensitive config in Vault
- Encrypted database connections
- TLS for all communications

### Input Validation
- Pydantic field validation
- JSON Schema validation
- SQL injection prevention (SQLAlchemy)
- XSS prevention (FastAPI)

## Performance

### Optimization Strategies

1. **Database**
   - Connection pooling (20 connections)
   - Indexes on common queries
   - JSONB GIN indexes
   - Materialized views for stats

2. **API**
   - Response caching (5 min TTL)
   - Pagination for large results
   - Async operations
   - Connection keep-alive

3. **Validation**
   - Cached JSON schemas
   - Parallel validation
   - Early exit on errors

### Expected Performance

- Model registration: < 50ms
- Model lookup: < 10ms
- Pipeline validation: < 20ms
- List operations: < 15ms (100 items)

## Troubleshooting

### Common Issues

**Issue: Service won't start**
```bash
# Check database connection
psql -U mnemos -d mnemos -c "SELECT 1"

# Check logs
docker logs genome

# Verify environment variables
docker exec genome env | grep POSTGRES
```

**Issue: Validation failing**
```bash
# Check JSON schemas
ls -la /app/schemas/

# Test validation endpoint
curl -X POST http://localhost:8081/api/v1/models/validate \
  -H "Content-Type: application/json" \
  -d @test_model.json
```

**Issue: Slow queries**
```bash
# Check database indexes
psql -U mnemos -d mnemos -c "\d models"

# Analyze query performance
psql -U mnemos -d mnemos -c "EXPLAIN ANALYZE SELECT * FROM models WHERE name = 'llama2-7b-chat'"
```

## Integration with Other Services

### CORTEX (Job Orchestrator)
- Validates model schemas before job submission
- Validates pipeline definitions
- Enforces policies during scheduling

### NEURON (Worker)
- Fetches model schemas for execution
- Validates runtime configurations
- Checks resource requirements

### SYNKRON (Pipeline Orchestrator)
- Retrieves pipeline definitions
- Validates pipeline steps
- Gets execution order

## Future Enhancements

### Phase 3.1: Advanced Features
- [ ] Schema versioning with migration paths
- [ ] Schema comparison and diff tools
- [ ] Automated schema testing
- [ ] Schema documentation generation

### Phase 3.2: Performance
- [ ] Redis caching layer
- [ ] GraphQL API support
- [ ] Bulk operations
- [ ] Schema compression

### Phase 3.3: Enterprise Features
- [ ] Multi-tenancy support
- [ ] Audit logging
- [ ] Schema approval workflows
- [ ] Import/export tools

## Testing Strategy

### Unit Tests
- Model validation logic
- Pipeline DAG validation
- Policy rule validation
- Repository operations

### Integration Tests
- API endpoint testing
- Database operations
- Error handling
- Authentication/authorization

### E2E Tests
- Complete registration flows
- Version management
- Cross-service integration

### Load Tests
- 1000 concurrent requests
- Large schema payloads
- Bulk operations
- Sustained load

## Deployment

### Docker Compose

```yaml
services:
  genome:
    image: mnemos/genome:1.0.0
    container_name: genome
    ports:
      - "8081:8081"
      - "9091:9091"
    environment:
      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
      - POSTGRES_DB=mnemos
      - POSTGRES_USER=mnemos
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - VAULT_ADDR=http://vault:8200
      - VAULT_TOKEN=${VAULT_TOKEN}
    depends_on:
      - postgres
      - vault
    networks:
      - mnemos-backend
      - mnemos-data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8081/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes (Future)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genome
spec:
  replicas: 3
  selector:
    matchLabels:
      app: genome
  template:
    metadata:
      labels:
        app: genome
    spec:
      containers:
      - name: genome
        image: mnemos/genome:1.0.0
        ports:
        - containerPort: 8081
        - containerPort: 9091
        env:
        - name: POSTGRES_HOST
          valueFrom:
            configMapKeyRef:
              name: genome-config
              key: postgres.host
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8081
          initialDelaySeconds: 10
          periodSeconds: 5
```

## Success Criteria

### Phase 3 Complete When:
- [x] All source files generated
- [x] Data models with validation
- [x] Complete REST API
- [x] Database migrations
- [x] Service layer implementation
- [x] Repository layer
- [x] Validation service
- [x] Tests (unit + integration)
- [x] Docker configuration
- [x] Documentation
- [ ] Tests passing (>80% coverage)
- [ ] Docker image builds
- [ ] Service starts healthy
- [ ] API endpoints respond
- [ ] Metrics exported

### Integration Tests Pass:
- [ ] Register model
- [ ] Update model
- [ ] Delete model
- [ ] Register pipeline
- [ ] Validate pipeline DAG
- [ ] Register policy
- [ ] Query with filters
- [ ] Version management

## Next Steps

After Phase 3 completion:

1. **Test GENOME Service**
   ```bash
   pytest tests/ -v --cov=genome
   ```

2. **Build Docker Image**
   ```bash
   docker build -t mnemos/genome:1.0.0 -f docker/genome/Dockerfile .
   ```

3. **Deploy Service**
   ```bash
   docker-compose up -d genome
   ```

4. **Load Sample Data**
   ```bash
   python scripts/seed_genome.py
   ```

5. **Proceed to Phase 4: CORTEX**
   - Job orchestration service
   - Depends on GENOME for schema validation
   - Uses PostgreSQL + Redis
   - gRPC API for workers

---

**Phase 3 Status:** ✅ Specification Complete

**Next Phase:** Phase 4 - CORTEX Service (Job Orchestrator)  
**Estimated Implementation Time:** 8-12 hours with Claude Code  
**Dependencies:** PostgreSQL ✅, Vault ✅

**Total Phase 3 Lines:** ~2,600 lines of specification  
**Implementation Lines:** ~2,000 lines of production code expected

---

**Generated by:** Nix with Claude (Anthropic)  
**For:** MNEMOS AI Operations Platform  
**Version:** 1.0.0  
**Date:** November 2024
