# Building MNEMOS on Docker with Claude Code

Alright, let’s build this thing from scratch. We’re going WSL Ubuntu, Docker-based (no full K8s locally), and using Claude Code as our AI pair programmer.

## Phase 0: Environment Setup

### WSL Ubuntu Baseline

```bash
# Make sure you're on WSL2, not WSL1
wsl --set-default-version 2

# Inside WSL
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  build-essential \
  git \
  curl \
  wget \
  jq \
  python3-pip \
  python3-venv
```

### Docker Desktop + WSL Integration

```bash
# Install Docker Desktop on Windows (with WSL2 backend enabled)
# Then verify from WSL:
docker --version
docker compose version

# Test it
docker run hello-world
```

### Claude Code Setup

```bash
# Install Claude Code CLI
curl -fsSL https://install.claude.com/code | sh

# Or via npm if that's your vibe
npm install -g @anthropic-ai/claude-code

# Authenticate
claude-code auth login

# Verify
claude-code --version
```

## Phase 1: Project Structure

```bash
# Create workspace
mkdir -p ~/mnemos && cd ~/mnemos

# Initialize git
git init
git branch -M main

# Directory structure
mkdir -p {src,docker,config,scripts,data}
mkdir -p src/{soul,genome,cortex,neuron,engram,wraith,relay,synkron,trace,cradle}
```

### Core Docker Compose Scaffold

```bash
# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.9'

networks:
  mnemos:
    driver: bridge

volumes:
  engram-data:
  vault-data:
  loki-data:
  prometheus-data:

services:
  # SOUL - Secrets & Identity
  vault:
    image: hashicorp/vault:latest
    container_name: mnemos-vault
    ports:
      - "8200:8200"
    environment:
      VAULT_DEV_ROOT_TOKEN_ID: "dev-only-token"
      VAULT_DEV_LISTEN_ADDRESS: "0.0.0.0:8200"
    networks:
      - mnemos
    volumes:
      - vault-data:/vault/data
    cap_add:
      - IPC_LOCK

  # GENOME - Config Registry
  genome:
    build:
      context: ./src/genome
      dockerfile: Dockerfile
    container_name: mnemos-genome
    ports:
      - "8081:8080"
    environment:
      VAULT_ADDR: "http://vault:8200"
      VAULT_TOKEN: "dev-only-token"
    networks:
      - mnemos
    depends_on:
      - vault

  # CORTEX - Orchestrator
  cortex:
    build:
      context: ./src/cortex
      dockerfile: Dockerfile
    container_name: mnemos-cortex
    ports:
      - "8080:8080"
      - "9090:9090"
    environment:
      GENOME_URL: "http://genome:8080"
      ENGRAM_URL: "http://engram:8080"
      OTEL_EXPORTER_OTLP_ENDPOINT: "http://trace:4317"
    networks:
      - mnemos
    depends_on:
      - genome
      - engram

  # NEURON - Worker Runtime
  neuron:
    build:
      context: ./src/neuron
      dockerfile: Dockerfile
    container_name: mnemos-neuron
    ports:
      - "8000:8000"
    environment:
      CORTEX_URL: "http://cortex:8080"
      ENGRAM_URL: "http://engram:8080"
    networks:
      - mnemos
    volumes:
      - ./data/models:/models:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    depends_on:
      - cortex

  # ENGRAM - State Store
  engram:
    image: minio/minio:latest
    container_name: mnemos-engram
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: "mnemos"
      MINIO_ROOT_PASSWORD: "mnemos-dev-password"
    networks:
      - mnemos
    volumes:
      - engram-data:/data
    command: server /data --console-address ":9001"

  # RELAY - API Gateway
  relay:
    image: traefik:v2.10
    container_name: mnemos-relay
    ports:
      - "80:80"
      - "443:443"
      - "8888:8080"  # Traefik dashboard
    networks:
      - mnemos
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./config/traefik.yml:/etc/traefik/traefik.yml:ro
    depends_on:
      - cortex

  # TRACE - Observability Stack
  trace:
    image: grafana/otel-collector:latest
    container_name: mnemos-trace
    ports:
      - "4317:4317"  # OTLP gRPC
      - "4318:4318"  # OTLP HTTP
    networks:
      - mnemos
    volumes:
      - ./config/otel-collector.yml:/etc/otel-collector.yml:ro

  prometheus:
    image: prom/prometheus:latest
    container_name: mnemos-prometheus
    ports:
      - "9091:9090"
    networks:
      - mnemos
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus

  loki:
    image: grafana/loki:latest
    container_name: mnemos-loki
    ports:
      - "3100:3100"
    networks:
      - mnemos
    volumes:
      - loki-data:/loki

  grafana:
    image: grafana/grafana:latest
    container_name: mnemos-grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: "admin"
    networks:
      - mnemos
    volumes:
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
    depends_on:
      - prometheus
      - loki

EOF
```

## Phase 2: Building Components with Claude Code

### Start with CORTEX (the brain)

```bash
cd src/cortex

# Use Claude Code to generate the service
claude-code create \
  --template "python-fastapi-service" \
  --name "cortex" \
  --description "MNEMOS orchestrator that schedules jobs to NEURON workers, enforces policies from GENOME, and coordinates with ENGRAM for state"
```

**Prompt for Claude Code:**

```
Create a FastAPI service called CORTEX with:
- HTTP API on port 8080, gRPC on 9090
- Job queue management (in-memory Redis-compatible queue)
- REST endpoints: POST /jobs, GET /jobs/{id}, GET /health
- Job scheduling logic that routes to NEURON workers
- Integration with GENOME for config validation
- OTEL instrumentation for tracing
- Async job execution with status updates
- Exponential backoff retry (1s-30s)
- Queue concurrency limit (100 default)
```

### CORTEX Dockerfile

```dockerfile
# src/cortex/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080 9090

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### CORTEX requirements.txt

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
httpx==0.25.1
redis==5.0.1
pydantic==2.5.0
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0
opentelemetry-instrumentation-fastapi==0.42b0
opentelemetry-exporter-otlp==1.21.0
```

### Build GENOME (config/schema validator)

```bash
cd ../genome

claude-code create \
  --template "python-fastapi-service" \
  --name "genome" \
  --description "Schema registry and policy validator for MNEMOS. Validates Model, Pipeline, and Policy objects against JSON schemas. Webhooks for admission control."
```

**Prompt:**

```
Create GENOME as a FastAPI service with:
- Schema registry endpoints: POST /schemas, GET /schemas/{name}
- Validation endpoints: POST /validate/{kind} (Model, Pipeline, Policy)
- In-memory schema store (or Redis-backed)
- JSON Schema validation using jsonschema library
- Admission webhook endpoint: POST /admit
- Integration with Vault (SOUL) for secret validation
- Return 200 OK or 400 Bad Request with validation errors
```

### Build NEURON (worker runtime)

```bash
cd ../neuron

claude-code create \
  --template "python-worker-service" \
  --name "neuron" \
  --description "Runtime execution worker for ML inference. Supports vLLM and PyTorch runtimes. Polls CORTEX for jobs, executes them, streams results."
```

**Prompt:**

```
Create NEURON worker service with:
- Job polling from CORTEX (long-polling or SSE)
- vLLM runtime integration (subprocess or HTTP client)
- PyTorch runtime support
- Model loading from /models volume
- Result streaming to ENGRAM (S3-compatible)
- GPU detection and resource management
- Health check endpoint
- Job status updates back to CORTEX
```

### NEURON with vLLM support

```dockerfile
# src/neuron/Dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install vLLM
RUN pip3 install vllm==0.2.7

COPY . .

EXPOSE 8000

CMD ["python3", "worker.py"]
```

### Build ENGRAM (state/storage abstraction)

```bash
cd ../engram

# ENGRAM can be a thin wrapper around MinIO
# Or build a simple S3-compatible client service
```

**Use MinIO directly** in docker-compose (already done above), or build a thin abstraction layer:

```bash
claude-code create \
  --template "python-fastapi-service" \
  --name "engram" \
  --description "State and artifact storage abstraction. Wraps MinIO/S3 with MNEMOS-specific APIs for model artifacts, job outputs, and snapshots."
```

### Build RELAY (Traefik config)

```bash
# config/traefik.yml
api:
  dashboard: true
  insecure: true

entryPoints:
  web:
    address: ":80"
  websecure:
    address: ":443"

providers:
  docker:
    exposedByDefault: false
  file:
    filename: /etc/traefik/dynamic.yml

log:
  level: INFO
```

```bash
# config/traefik-dynamic.yml (optional for advanced routing)
http:
  routers:
    cortex:
      rule: "Host(`cortex.mnemos.local`)"
      service: cortex
      entryPoints:
        - web
  services:
    cortex:
      loadBalancer:
        servers:
          - url: "http://cortex:8080"
```

## Phase 3: Observability Configuration

### Prometheus config

```yaml
# config/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'cortex'
    static_configs:
      - targets: ['cortex:8080']
  
  - job_name: 'neuron'
    static_configs:
      - targets: ['neuron:8000']
  
  - job_name: 'genome'
    static_configs:
      - targets: ['genome:8080']
```

### OTEL Collector config

```yaml
# config/otel-collector.yml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:

exporters:
  logging:
    loglevel: debug
  prometheus:
    endpoint: "prometheus:9090"
  loki:
    endpoint: "http://loki:3100/loki/api/v1/push"

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [logging]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus]
    logs:
      receivers: [otlp]
      processors: [batch]
      exporters: [loki]
```

## Phase 4: Build & Launch

```bash
# From ~/mnemos root
docker compose build

# Launch infrastructure first
docker compose up -d vault engram prometheus loki grafana

# Wait 10 seconds, then launch app layer
sleep 10
docker compose up -d genome cortex neuron relay trace

# Check logs
docker compose logs -f cortex
```

## Phase 5: Test the Stack

### Health checks

```bash
# CORTEX
curl http://localhost:8080/health

# GENOME
curl http://localhost:8081/health

# Vault (SOUL)
curl http://localhost:8200/v1/sys/health

# MinIO (ENGRAM)
curl http://localhost:9000/minio/health/live

# Grafana
open http://localhost:3000  # admin/admin
```

### Submit a test job

```bash
curl -X POST http://localhost:8080/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "modelRef": "qwen2-7b-instruct",
    "prompt": "Hello, MNEMOS!",
    "params": {
      "temperature": 0.7,
      "max_tokens": 100
    }
  }'
```

## Phase 6: Claude Code Development Loop

### Iterative development with Claude Code

```bash
# Navigate to component
cd src/cortex

# Ask Claude to add features
claude-code edit \
  --file main.py \
  --instruction "Add job priority queue support with 3 priority levels: high, normal, low"

# Run tests
claude-code test --generate

# Rebuild
docker compose build cortex
docker compose up -d cortex
```

### Use Claude Code for debugging

```bash
# Tail logs and ask Claude to analyze
docker compose logs cortex | claude-code analyze \
  --prompt "Find the root cause of this error and suggest a fix"
```

## Phase 7: Add LLM Model Support

### Download a model

```bash
# Create models directory
mkdir -p ~/mnemos/data/models

# Download a small model (e.g., Qwen 7B)
cd ~/mnemos/data/models

# Use huggingface-cli or wget
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2-7B-Instruct \
  --local-dir ./qwen2-7b-instruct \
  --local-dir-use-symlinks False
```

### Register model in GENOME

```bash
curl -X POST http://localhost:8081/schemas \
  -H "Content-Type: application/json" \
  -d '{
    "kind": "Model",
    "name": "qwen2-7b-instruct",
    "spec": {
      "family": "qwen2",
      "version": "7b-instruct",
      "artifact": {
        "uri": "file:///models/qwen2-7b-instruct"
      },
      "runtime": {
        "engine": "vllm",
        "params": {
          "tensor_parallel": 1,
          "max_tokens": 2048
        }
      }
    }
  }'
```

## Phase 8: WRAITH (Background Jobs)

```bash
cd src/wraith

claude-code create \
  --template "python-cron-service" \
  --name "wraith" \
  --description "Background job daemon for MNEMOS. Handles log cleanup, event replay, and maintenance tasks."
```

**Prompt:**

```
Create WRAITH background service with:
- APScheduler for cron-like scheduling
- Jobs: log compaction (3 AM daily), event replay (every 15 min)
- Integration with ENGRAM for cleanup
- Integration with CORTEX for event replay
- Health check endpoint
- Configurable via environment variables
```

### Add to docker-compose

```yaml
  wraith:
    build:
      context: ./src/wraith
      dockerfile: Dockerfile
    container_name: mnemos-wraith
    environment:
      CORTEX_URL: "http://cortex:8080"
      ENGRAM_URL: "http://engram:9000"
    networks:
      - mnemos
    depends_on:
      - cortex
      - engram
```

## Phase 9: SYNKRON (Pipeline Orchestration)

```bash
cd src/synkron

claude-code create \
  --template "python-dag-orchestrator" \
  --name "synkron" \
  --description "Pipeline orchestrator for multi-step ML workflows. DAG execution with step dependencies."
```

**Prompt:**

```
Create SYNKRON pipeline orchestrator with:
- DAG-based pipeline execution (like Airflow but simpler)
- Pipeline CRD support: steps, dependencies, artifacts
- Integration with CORTEX for job submission
- Integration with ENGRAM for artifact passing between steps
- Pipeline status tracking
- REST API: POST /pipelines, GET /pipelines/{id}
```

## Phase 10: Development Workflow

### Daily development loop

```bash
# 1. Make changes with Claude Code
cd ~/mnemos/src/cortex
claude-code edit --interactive

# 2. Rebuild component
docker compose build cortex

# 3. Restart
docker compose restart cortex

# 4. Test
curl http://localhost:8080/health

# 5. Check logs
docker compose logs -f cortex

# 6. Commit
git add .
git commit -m "feat(cortex): add priority queue support"
```

### Debugging with Claude Code

```bash
# Stream logs to Claude for analysis
docker compose logs cortex --tail=100 | \
  claude-code chat --prompt "Analyze these logs and suggest improvements"

# Generate tests
claude-code test generate --file src/cortex/main.py

# Refactor code
claude-code refactor --file src/cortex/scheduler.py \
  --instruction "Optimize job scheduling algorithm for better throughput"
```

## Phase 11: Production Hardening

### Add health checks to docker-compose

```yaml
  cortex:
    # ... existing config
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Add restart policies

```yaml
  cortex:
    # ... existing config
    restart: unless-stopped
```

### Add resource limits

```yaml
  cortex:
    # ... existing config
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '0.5'
          memory: 1G
```

## Phase 12: Makefile for Convenience

```makefile
# Makefile
.PHONY: build up down logs clean test

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	docker compose down -v
	rm -rf data/engram/*

test:
	python -m pytest tests/

restart-cortex:
	docker compose restart cortex

restart-neuron:
	docker compose restart neuron

shell-cortex:
	docker compose exec cortex /bin/bash

health:
	@echo "=== Health Checks ==="
	@curl -s http://localhost:8080/health | jq .
	@curl -s http://localhost:8081/health | jq .
	@curl -s http://localhost:8000/health | jq .
```

Usage:

```bash
make build  # Build all services
make up     # Start stack
make logs   # Tail logs
make health # Check all health endpoints
```

## Next Steps

1. **Add authentication**: Integrate Vault properly for JWT tokens
1. **Add WebUI**: Build a React dashboard for job monitoring
1. **Add model registry UI**: Browse and manage models via web interface
1. **Add GPU pooling**: Multiple NEURON workers with GPU affinity
1. **Add CRADLE**: Bootstrap script to init Vault, seed schemas
1. **GitOps layer**: Add FluxCD or ArgoCD for declarative updates
1. **Model fine-tuning**: Add SYNKRON pipelines for LoRA training
1. **RAG support**: Add vector DB (Qdrant) for retrieval-augmented generation

You now have a Docker-based MNEMOS running on WSL with Claude Code as your AI pair programmer. Scale up from here!​​​​​​​​​​​​​​​​