# MNEMOS Phase 1: Foundation & Infrastructure Setup

**Version:** 1.0.0  
**Phase:** Foundation & Core Infrastructure  
**Target Environment:** Docker Compose on Ubuntu WSL  
**Implementation Partner:** Claude Code

═══════════════════════════════════════════════════════════════════

## Table of Contents

1. [Phase Overview](#phase-overview)
2. [Environment Setup](#environment-setup)
3. [Project Structure](#project-structure)
4. [Core Infrastructure Services](#core-infrastructure-services)
5. [Network Architecture](#network-architecture)
6. [Storage Architecture](#storage-architecture)
7. [Configuration Files](#configuration-files)
8. [Validation & Testing](#validation--testing)

═══════════════════════════════════════════════════════════════════

## Phase Overview

### Objectives

Phase 1 establishes the foundational infrastructure for MNEMOS:

- Set up development environment (WSL Ubuntu + Docker)
- Create project structure and repository
- Deploy core infrastructure services (PostgreSQL, Redis, Vault, MinIO)
- Configure networking with isolated network segments
- Establish storage layers and backup systems
- Create configuration management system
- Set up observability infrastructure (Prometheus, Loki, Grafana)

### Success Criteria

- [ ] WSL Ubuntu environment configured
- [ ] Docker and Docker Compose operational
- [ ] Project repository initialized with proper structure
- [ ] All infrastructure services running and healthy
- [ ] Network isolation validated
- [ ] Storage volumes created and accessible
- [ ] Observability stack operational
- [ ] Initial configuration files created
- [ ] Health checks passing for all services

═══════════════════════════════════════════════════════════════════

## Environment Setup

### System Requirements

**Minimum:**
- Windows 10/11 with WSL2 enabled
- 8GB RAM
- 4 CPU cores
- 100GB available disk space

**Recommended:**
- Windows 11 with WSL2
- 16GB RAM
- 8 CPU cores
- 500GB NVMe SSD
- NVIDIA GPU (for NEURON workers)

### WSL Ubuntu Installation

```bash
# Enable WSL2 (Run in PowerShell as Administrator)
wsl --install Ubuntu-22.04
wsl --set-default-version 2

# Verify WSL version
wsl -l -v

# Inside WSL Ubuntu
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y \
  build-essential \
  git \
  curl \
  wget \
  jq \
  vim \
  htop \
  net-tools \
  python3.11 \
  python3-pip \
  python3-venv \
  software-properties-common \
  apt-transport-https \
  ca-certificates \
  gnupg \
  lsb-release

# Verify installations
python3 --version  # Should be 3.11+
git --version
```

### Docker Installation

```bash
# Install Docker Desktop for Windows with WSL2 backend
# Download from: https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe

# After installation, enable WSL2 integration in Docker Desktop settings
# Settings → Resources → WSL Integration → Enable integration with Ubuntu

# Verify from WSL
docker --version          # 24.0+
docker compose version    # 2.20+

# Test Docker
docker run hello-world

# Configure Docker to use more resources (if needed)
# Docker Desktop → Settings → Resources:
#   - CPUs: 4-6
#   - Memory: 8-12 GB
#   - Swap: 2 GB
#   - Disk image size: 100 GB

# Enable Docker BuildKit for faster builds
echo 'export DOCKER_BUILDKIT=1' >> ~/.bashrc
source ~/.bashrc
```

### Claude Code Setup

```bash
# Install Claude Code CLI
curl -fsSL https://install.claude.com/code | sh

# Or via npm (alternative)
# npm install -g @anthropic-ai/claude-code

# Authenticate
claude-code auth login

# Verify installation
claude-code --version

# Configure Claude Code preferences
claude-code config set editor vim
claude-code config set auto-suggest true
```

### Additional Tools

```bash
# Install yq for YAML processing
sudo wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
sudo chmod +x /usr/local/bin/yq

# Install shellcheck for shell script validation
sudo apt install -y shellcheck

# Install hadolint for Dockerfile linting
sudo wget -O /usr/local/bin/hadolint https://github.com/hadolint/hadolint/releases/latest/download/hadolint-Linux-x86_64
sudo chmod +x /usr/local/bin/hadolint

# Install trivy for container security scanning
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | gpg --dearmor | sudo tee /usr/share/keyrings/trivy.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/trivy.gpg] https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
sudo apt update && sudo apt install -y trivy
```

═══════════════════════════════════════════════════════════════════

## Project Structure

### Directory Layout

```bash
# Navigate to workspace
mkdir -p ~/mnemos && cd ~/mnemos

# Initialize Git repository
git init
git branch -M main

# Create comprehensive directory structure
mkdir -p {src,docker,config,scripts,data,tests,docs,.github}

# Source directories (application code)
mkdir -p src/{soul,genome,cortex,neuron,engram,wraith,relay,synkron,trace,cradle}
mkdir -p src/common/{auth,logging,metrics,tracing,errors,models}

# Docker configurations
mkdir -p docker/{postgres,redis,vault,minio,traefik,otel}

# Configuration files
mkdir -p config/{traefik,prometheus,grafana,loki,vault,alertmanager}
mkdir -p config/grafana/{dashboards,datasources,provisioning}

# Data directories (excluded from git)
mkdir -p data/{models,artifacts,logs,backups,uploads}
mkdir -p data/postgres
mkdir -p data/redis
mkdir -p data/vault
mkdir -p data/minio
mkdir -p data/prometheus
mkdir -p data/loki
mkdir -p data/grafana

# Test directories
mkdir -p tests/{unit,integration,e2e,load,security}
mkdir -p tests/fixtures

# Documentation
mkdir -p docs/{api,architecture,operations,development}

# Scripts
mkdir -p scripts/{bootstrap,backup,migration,monitoring}

# GitHub Actions
mkdir -p .github/workflows

# Verify structure
tree -L 2 -d .
```

### Core Files Creation

```bash
# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
venv/
ENV/
env/

# Docker
**/.dockerignore

# Data directories
data/
!data/.gitkeep
*.log
*.db
*.sqlite

# Secrets
.env
.vault-keys.json
*.pem
*.key
*.crt
secrets/
*.secret

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# OS
Thumbs.db
.Trash-*

# Temporary
tmp/
temp/
*.tmp

# Backup files
*.bak
*.backup

# Test coverage
.coverage
htmlcov/
.pytest_cache/
.tox/

# MyPy
.mypy_cache/
.dmypy.json
dmypy.json
EOF

# Create placeholder files for data directories
find data -type d -exec touch {}/.gitkeep \;

# Create README.md
cat > README.md << 'EOF'
# MNEMOS - Modular Neural Memory & Orchestration System

**Version:** 1.0.0  
**Status:** Development

## Overview

MNEMOS is an intent-driven AI operations platform that orchestrates LLM workloads with enterprise-grade reliability, security, and observability.

## Architecture

See [docs/architecture/overview.md](docs/architecture/overview.md)

## Quick Start

```bash
# Initialize environment
make init

# Start all services
make up

# Check health
make health
```

## Documentation

- [Architecture](docs/architecture/)
- [API Reference](docs/api/)
- [Operations Guide](docs/operations/)
- [Development Guide](docs/development/)

## Components

- **SOUL**: Identity and secrets management
- **GENOME**: Configuration and schema registry
- **CORTEX**: Job orchestration and scheduling
- **NEURON**: Worker runtime execution
- **ENGRAM**: State and artifact storage
- **WRAITH**: Background job processing
- **RELAY**: API gateway and routing
- **SYNKRON**: Pipeline orchestration
- **TRACE**: Observability and telemetry
- **CRADLE**: Bootstrap and initialization

## License

Apache 2.0
EOF
```

═══════════════════════════════════════════════════════════════════

## Core Infrastructure Services

### Docker Compose Architecture

MNEMOS uses a multi-service Docker Compose architecture with three isolated networks:

```
┌─────────────────────────────────────────────────────────────┐
│                      FRONTEND NETWORK                        │
│                      (172.20.0.0/24)                        │
│                                                              │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐         │
│  │  RELAY   │      │  WebUI   │      │  CLI     │         │
│  │ (Traefik)│      │          │      │          │         │
│  └────┬─────┘      └────┬─────┘      └────┬─────┘         │
└───────┼──────────────────┼──────────────────┼──────────────┘
        │                  │                  │
┌───────┼──────────────────┼──────────────────┼──────────────┐
│       │        BACKEND NETWORK (INTERNAL)   │              │
│       │              (172.21.0.0/24)        │              │
│       │                                     │              │
│  ┌────▼─────┐   ┌──────────┐   ┌──────────▼────┐         │
│  │  CORTEX  │   │  GENOME  │   │    SYNKRON    │         │
│  │          │◄──┤          │◄──┤               │         │
│  └────┬─────┘   └────┬─────┘   └───────────────┘         │
│       │              │                                      │
│  ┌────▼─────┐   ┌───▼──────┐   ┌──────────┐              │
│  │  NEURON  │   │  WRAITH  │   │  CRADLE  │              │
│  │          │   │          │   │          │              │
│  └──────────┘   └──────────┘   └──────────┘              │
└─────────────────────┼──────────────────────────────────────┘
                      │
┌─────────────────────┼──────────────────────────────────────┐
│                     │   DATA NETWORK (INTERNAL)            │
│                     │      (172.22.0.0/24)                │
│                     │                                       │
│  ┌────▼─────┐   ┌──▼──────┐   ┌──────────┐   ┌────────┐ │
│  │PostgreSQL│   │  Redis  │   │  MinIO   │   │  Vault │ │
│  │          │   │         │   │ (ENGRAM) │   │ (SOUL) │ │
│  └──────────┘   └─────────┘   └──────────┘   └────────┘ │
│                                                            │
│  ┌──────────┐   ┌─────────┐   ┌──────────┐              │
│  │Prometheus│   │  Loki   │   │ Grafana  │              │
│  │          │   │         │   │ (TRACE)  │              │
│  └──────────┘   └─────────┘   └──────────┘              │
└────────────────────────────────────────────────────────────┘
```

### Root docker-compose.yml

```yaml
version: '3.9'

# Network definitions - Three isolated network segments
networks:
  frontend:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/24
          gateway: 172.20.0.1
  
  backend:
    driver: bridge
    internal: true  # No external internet access
    ipam:
      config:
        - subnet: 172.21.0.0/24
          gateway: 172.21.0.1
  
  data:
    driver: bridge
    internal: true  # No external internet access
    ipam:
      config:
        - subnet: 172.22.0.0/24
          gateway: 172.22.0.1

# Volume definitions - Persistent storage
volumes:
  postgres-data:
    driver: local
  redis-data:
    driver: local
  vault-data:
    driver: local
  minio-data:
    driver: local
  prometheus-data:
    driver: local
  loki-data:
    driver: local
  grafana-data:
    driver: local
  engram-data:
    driver: local

# Service definitions
services:
  
  # ═══════════════════════════════════════════════════════════
  # DATA LAYER SERVICES
  # ═══════════════════════════════════════════════════════════
  
  # PostgreSQL - Primary database
  postgres:
    image: postgres:16-alpine
    container_name: mnemos-postgres
    hostname: postgres
    restart: unless-stopped
    networks:
      - data
    ports:
      - "5432:5432"  # Exposed for development only
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./docker/postgres/init-db.sh:/docker-entrypoint-initdb.d/init-db.sh:ro
      - ./docker/postgres/postgresql.conf:/etc/postgresql/postgresql.conf:ro
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-mnemos}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?POSTGRES_PASSWORD required}
      POSTGRES_DB: ${POSTGRES_DB:-mnemos}
      POSTGRES_INITDB_ARGS: "-E UTF8 --locale=en_US.UTF-8"
      PGDATA: /var/lib/postgresql/data/pgdata
    command: 
      - "postgres"
      - "-c"
      - "config_file=/etc/postgresql/postgresql.conf"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-mnemos}"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '0.5'
          memory: 1G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    security_opt:
      - no-new-privileges:true
    labels:
      - "mnemos.service=postgres"
      - "mnemos.layer=data"
      - "mnemos.backup=true"
  
  # Redis - Cache and job queue
  redis:
    image: redis:7-alpine
    container_name: mnemos-redis
    hostname: redis
    restart: unless-stopped
    networks:
      - data
    ports:
      - "6379:6379"  # Exposed for development only
    volumes:
      - redis-data:/data
      - ./docker/redis/redis.conf:/usr/local/etc/redis/redis.conf:ro
    command: 
      - "redis-server"
      - "/usr/local/etc/redis/redis.conf"
      - "--requirepass"
      - "${REDIS_PASSWORD:?REDIS_PASSWORD required}"
    healthcheck:
      test: ["CMD", "redis-cli", "--no-auth-warning", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
      start_period: 5s
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.25'
          memory: 512M
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    security_opt:
      - no-new-privileges:true
    labels:
      - "mnemos.service=redis"
      - "mnemos.layer=data"
      - "mnemos.backup=true"
  
  # Vault (SOUL) - Secrets management
  vault:
    image: hashicorp/vault:1.15
    container_name: mnemos-vault
    hostname: vault
    restart: unless-stopped
    networks:
      - data
      - backend
    ports:
      - "8200:8200"
    volumes:
      - vault-data:/vault/data
      - ./docker/vault/config:/vault/config:ro
      - ./docker/vault/scripts:/vault/scripts:ro
    environment:
      VAULT_ADDR: "https://0.0.0.0:8200"
      VAULT_API_ADDR: "https://vault:8200"
      VAULT_CLUSTER_ADDR: "https://vault:8201"
    cap_add:
      - IPC_LOCK  # Required for mlock
    command: server
    healthcheck:
      test: ["CMD", "vault", "status"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 15s
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.25'
          memory: 512M
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    security_opt:
      - no-new-privileges:true
    labels:
      - "mnemos.service=vault"
      - "mnemos.component=soul"
      - "mnemos.layer=data"
      - "mnemos.critical=true"
  
  # MinIO (ENGRAM) - Object storage
  minio:
    image: minio/minio:latest
    container_name: mnemos-minio
    hostname: minio
    restart: unless-stopped
    networks:
      - data
      - backend
    ports:
      - "9000:9000"  # API
      - "9001:9001"  # Console
    volumes:
      - minio-data:/data
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER:-mnemos}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD:?MINIO_ROOT_PASSWORD required}
      MINIO_BROWSER: "on"
      MINIO_PROMETHEUS_AUTH_TYPE: "public"
    command: server /data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '0.5'
          memory: 1G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    security_opt:
      - no-new-privileges:true
    labels:
      - "mnemos.service=minio"
      - "mnemos.component=engram"
      - "mnemos.layer=data"
      - "mnemos.backup=true"
  
  # ═══════════════════════════════════════════════════════════
  # OBSERVABILITY SERVICES
  # ═══════════════════════════════════════════════════════════
  
  # OpenTelemetry Collector
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    container_name: mnemos-otel
    hostname: otel-collector
    restart: unless-stopped
    networks:
      - backend
      - data
    ports:
      - "4317:4317"  # OTLP gRPC
      - "4318:4318"  # OTLP HTTP
      - "8888:8888"  # Metrics
      - "8889:8889"  # Prometheus exporter
    volumes:
      - ./config/otel-collector.yaml:/etc/otel-collector-config.yaml:ro
    command: ["--config=/etc/otel-collector-config.yaml"]
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:13133/"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 10s
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.25'
          memory: 512M
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    labels:
      - "mnemos.service=otel-collector"
      - "mnemos.component=trace"
      - "mnemos.layer=observability"
  
  # Prometheus - Metrics collection
  prometheus:
    image: prom/prometheus:latest
    container_name: mnemos-prometheus
    hostname: prometheus
    restart: unless-stopped
    networks:
      - data
      - backend
    ports:
      - "9090:9090"
    volumes:
      - prometheus-data:/prometheus
      - ./config/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./config/prometheus/alerts:/etc/prometheus/alerts:ro
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--web.enable-lifecycle'
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '0.5'
          memory: 1G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    security_opt:
      - no-new-privileges:true
    labels:
      - "mnemos.service=prometheus"
      - "mnemos.component=trace"
      - "mnemos.layer=observability"
  
  # Loki - Log aggregation
  loki:
    image: grafana/loki:latest
    container_name: mnemos-loki
    hostname: loki
    restart: unless-stopped
    networks:
      - data
    ports:
      - "3100:3100"
    volumes:
      - loki-data:/loki
      - ./config/loki/loki-config.yaml:/etc/loki/loki-config.yaml:ro
    command: -config.file=/etc/loki/loki-config.yaml
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:3100/ready"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 10s
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.25'
          memory: 512M
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    security_opt:
      - no-new-privileges:true
    labels:
      - "mnemos.service=loki"
      - "mnemos.component=trace"
      - "mnemos.layer=observability"
  
  # Grafana - Visualization
  grafana:
    image: grafana/grafana:latest
    container_name: mnemos-grafana
    hostname: grafana
    restart: unless-stopped
    networks:
      - frontend
      - data
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./config/grafana/grafana.ini:/etc/grafana/grafana.ini:ro
      - ./config/grafana/datasources:/etc/grafana/provisioning/datasources:ro
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
    environment:
      GF_SECURITY_ADMIN_USER: ${GRAFANA_ADMIN_USER:-admin}
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD:?GRAFANA_ADMIN_PASSWORD required}
      GF_INSTALL_PLUGINS: "grafana-piechart-panel"
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:3000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.25'
          memory: 512M
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    security_opt:
      - no-new-privileges:true
    labels:
      - "mnemos.service=grafana"
      - "mnemos.component=trace"
      - "mnemos.layer=observability"
      - "traefik.enable=true"
      - "traefik.http.routers.grafana.rule=Host(`grafana.mnemos.local`)"
      - "traefik.http.services.grafana.loadbalancer.server.port=3000"
  
  # ═══════════════════════════════════════════════════════════
  # INFRASTRUCTURE SERVICES
  # ═══════════════════════════════════════════════════════════
  
  # Traefik (RELAY) - API Gateway
  traefik:
    image: traefik:v2.10
    container_name: mnemos-traefik
    hostname: traefik
    restart: unless-stopped
    networks:
      - frontend
      - backend
    ports:
      - "80:80"      # HTTP
      - "443:443"    # HTTPS
      - "8080:8080"  # Dashboard
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./config/traefik/traefik.yml:/etc/traefik/traefik.yml:ro
      - ./config/traefik/dynamic:/etc/traefik/dynamic:ro
      - ./config/traefik/certs:/etc/traefik/certs:ro
    environment:
      TRAEFIK_LOG_LEVEL: ${TRAEFIK_LOG_LEVEL:-INFO}
    healthcheck:
      test: ["CMD", "traefik", "healthcheck", "--ping"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 10s
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.25'
          memory: 256M
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    security_opt:
      - no-new-privileges:true
    labels:
      - "mnemos.service=traefik"
      - "mnemos.component=relay"
      - "mnemos.layer=infrastructure"
      - "traefik.enable=true"
      - "traefik.http.routers.dashboard.rule=Host(`traefik.mnemos.local`)"
      - "traefik.http.routers.dashboard.service=api@internal"

EOF

# Save the docker-compose.yml
cat > ~/mnemos/docker-compose.yml << 'DOCKER_COMPOSE_CONTENT'
# [Content above - too long, save separately]
DOCKER_COMPOSE_CONTENT
```

═══════════════════════════════════════════════════════════════════

## Network Architecture

### Network Segmentation

MNEMOS employs a three-tier network architecture for security and isolation:

**1. Frontend Network (172.20.0.0/24)**
- Publicly accessible services
- Traefik API Gateway
- WebUI (when implemented)
- Grafana dashboards

**2. Backend Network (172.21.0.0/24 - Internal Only)**
- Application services (CORTEX, GENOME, NEURON, etc.)
- Service-to-service communication
- No direct internet access
- Isolated from data layer except through controlled connections

**3. Data Network (172.22.0.0/24 - Internal Only)**
- Database services (PostgreSQL, Redis)
- Storage services (MinIO, Vault)
- Observability services (Prometheus, Loki)
- No direct internet access
- Most restricted network segment

### Network Security Rules

```yaml
# Example network policy (conceptual for Docker)
# In production K8s, these would be NetworkPolicy resources

Frontend Network:
  Ingress:
    - From: Internet
    - Ports: 80, 443
  Egress:
    - To: Backend Network
    - Ports: 8080, 9090

Backend Network:
  Ingress:
    - From: Frontend Network
    - Ports: 8080, 9090, 8000
  Egress:
    - To: Data Network
    - Ports: 5432, 6379, 8200, 9000

Data Network:
  Ingress:
    - From: Backend Network
    - Ports: 5432, 6379, 8200, 9000, 9090, 3100
  Egress:
    - None (except internal)
```

═══════════════════════════════════════════════════════════════════

## Storage Architecture

### Volume Strategy

```yaml
Persistent Volumes:
  postgres-data:
    Type: Local Docker volume
    Size: ~10GB (grows as needed)
    Backup: Daily (see backup scripts)
    Retention: 30 days
    
  redis-data:
    Type: Local Docker volume
    Size: ~5GB
    Backup: Daily snapshot
    Retention: 7 days
    
  vault-data:
    Type: Local Docker volume
    Size: ~1GB
    Backup: Encrypted backup every 6 hours
    Retention: 90 days (critical)
    
  minio-data:
    Type: Local Docker volume
    Size: 100GB+ (for models and artifacts)
    Backup: Incremental daily
    Retention: 60 days
    
  prometheus-data:
    Type: Local Docker volume
    Size: ~20GB
    Retention: 30 days (in-volume)
    
  loki-data:
    Type: Local Docker volume
    Size: ~30GB
    Retention: 30 days (in-volume)
    
  grafana-data:
    Type: Local Docker volume
    Size: ~1GB
    Backup: Configuration only
```

### Bind Mounts

```yaml
Configuration Files (Read-Only):
  - ./config/ → /etc/<service>/
  - Purpose: Configuration management
  - Security: Read-only mounts
  - Validation: On startup

Application Code:
  - ./src/ → /app/
  - Purpose: Development hot-reload
  - Security: Non-root user ownership
  - Build: Multi-stage Docker builds

Models and Artifacts:
  - ./data/models → /models
  - Purpose: Model storage and access
  - Security: Read-only for workers
  - Size: 50-500GB depending on models
```

═══════════════════════════════════════════════════════════════════

## Configuration Files

### Environment Variables (.env.example)

```bash
# =============================================================================
# MNEMOS ENVIRONMENT CONFIGURATION
# =============================================================================
# Copy this file to .env and fill in all required values
# Required variables are marked with (REQUIRED)
# =============================================================================

# -----------------------------------------------------------------------------
# ENVIRONMENT
# -----------------------------------------------------------------------------
ENVIRONMENT=development          # development, staging, production
DEBUG=true
LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# -----------------------------------------------------------------------------
# POSTGRESQL DATABASE
# -----------------------------------------------------------------------------
POSTGRES_USER=mnemos
POSTGRES_PASSWORD=                    # (REQUIRED) Strong password
POSTGRES_DB=mnemos
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_MAX_CONNECTIONS=100

# Database connection strings
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}
CORTEX_DB_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/mnemos_cortex
GENOME_DB_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/mnemos_genome
SYNKRON_DB_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/mnemos_synkron

# -----------------------------------------------------------------------------
# REDIS CACHE
# -----------------------------------------------------------------------------
REDIS_PASSWORD=                       # (REQUIRED) Strong password
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
REDIS_MAX_CONNECTIONS=50
REDIS_URL=redis://:${REDIS_PASSWORD}@${REDIS_HOST}:${REDIS_PORT}/${REDIS_DB}

# -----------------------------------------------------------------------------
# VAULT (SOUL) - SECRETS MANAGEMENT
# -----------------------------------------------------------------------------
VAULT_ADDR=https://vault:8200
VAULT_TOKEN=                          # (REQUIRED) Set after initialization
VAULT_SKIP_VERIFY=false              # Set to true only for development
VAULT_NAMESPACE=mnemos
VAULT_MOUNT_PATH=mnemos/

# -----------------------------------------------------------------------------
# MINIO (ENGRAM) - OBJECT STORAGE
# -----------------------------------------------------------------------------
MINIO_ROOT_USER=mnemos
MINIO_ROOT_PASSWORD=                  # (REQUIRED) Strong password (min 8 chars)
MINIO_ENDPOINT=minio:9000
MINIO_BUCKET=mnemos-artifacts
MINIO_REGION=us-east-1
MINIO_USE_SSL=false                  # Set to true in production

# S3-compatible access
AWS_ACCESS_KEY_ID=${MINIO_ROOT_USER}
AWS_SECRET_ACCESS_KEY=${MINIO_ROOT_PASSWORD}
AWS_ENDPOINT_URL=http://${MINIO_ENDPOINT}

# -----------------------------------------------------------------------------
# JWT AUTHENTICATION
# -----------------------------------------------------------------------------
JWT_SECRET_KEY=                       # (REQUIRED) Generate with: openssl rand -hex 32
JWT_ALGORITHM=HS256                  # or RS256 with key pairs
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
JWT_REFRESH_TOKEN_EXPIRE_DAYS=30

# For RS256 (asymmetric)
JWT_PRIVATE_KEY_PATH=/etc/mnemos/certs/jwt-private.pem
JWT_PUBLIC_KEY_PATH=/etc/mnemos/certs/jwt-public.pem

# -----------------------------------------------------------------------------
# TLS/SSL CONFIGURATION
# -----------------------------------------------------------------------------
TLS_ENABLED=true
TLS_CERT_PATH=/etc/mnemos/certs/server.crt
TLS_KEY_PATH=/etc/mnemos/certs/server.key
TLS_CA_PATH=/etc/mnemos/certs/ca.crt
TLS_VERIFY=true

# -----------------------------------------------------------------------------
# SERVICE URLS
# -----------------------------------------------------------------------------
CORTEX_URL=http://cortex:8080
CORTEX_GRPC_URL=cortex:9090
GENOME_URL=http://genome:8080
NEURON_URL=http://neuron:8000
ENGRAM_URL=http://minio:9000
RELAY_URL=http://traefik:80
SYNKRON_URL=http://synkron:8080
WRAITH_URL=http://wraith:8080
TRACE_URL=http://otel-collector:4318

# -----------------------------------------------------------------------------
# OPENTELEMETRY (OTEL)
# -----------------------------------------------------------------------------
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_SERVICE_NAME=mnemos
OTEL_RESOURCE_ATTRIBUTES=environment=${ENVIRONMENT}
OTEL_TRACES_EXPORTER=otlp
OTEL_METRICS_EXPORTER=otlp
OTEL_LOGS_EXPORTER=otlp

# -----------------------------------------------------------------------------
# PROMETHEUS
# -----------------------------------------------------------------------------
PROMETHEUS_URL=http://prometheus:9090
PROMETHEUS_SCRAPE_INTERVAL=15s
PROMETHEUS_EVALUATION_INTERVAL=15s
PROMETHEUS_RETENTION_TIME=30d

# -----------------------------------------------------------------------------
# LOKI
# -----------------------------------------------------------------------------
LOKI_URL=http://loki:3100
LOKI_RETENTION_PERIOD=744h  # 31 days

# -----------------------------------------------------------------------------
# GRAFANA
# -----------------------------------------------------------------------------
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=              # (REQUIRED) Strong password
GRAFANA_URL=http://grafana:3000

# -----------------------------------------------------------------------------
# TRAEFIK (RELAY)
# -----------------------------------------------------------------------------
TRAEFIK_LOG_LEVEL=INFO              # DEBUG, INFO, WARN, ERROR
TRAEFIK_ACCESS_LOG=true
TRAEFIK_DASHBOARD=true

# -----------------------------------------------------------------------------
# FEATURE FLAGS
# -----------------------------------------------------------------------------
ENABLE_AUTH=true
ENABLE_RATE_LIMIT=true
ENABLE_WAF=false                    # Set to true in production
ENABLE_METRICS=true
ENABLE_TRACING=true
ENABLE_PROFILING=false              # Enable only for debugging

# -----------------------------------------------------------------------------
# CORTEX CONFIGURATION
# -----------------------------------------------------------------------------
CORTEX_WORKER_CONCURRENCY=100
CORTEX_QUEUE_MAX_SIZE=10000
CORTEX_JOB_TIMEOUT=3600            # seconds
CORTEX_RETRY_MAX_ATTEMPTS=3
CORTEX_RETRY_BACKOFF_MIN=1         # seconds
CORTEX_RETRY_BACKOFF_MAX=30        # seconds

# -----------------------------------------------------------------------------
# NEURON CONFIGURATION
# -----------------------------------------------------------------------------
NEURON_WORKER_TYPE=cpu             # cpu, gpu, hybrid
NEURON_MAX_CONCURRENT_JOBS=10
NEURON_GPU_DEVICE_IDS=0,1          # Comma-separated GPU IDs
NEURON_MODEL_CACHE_SIZE=50G

# -----------------------------------------------------------------------------
# BACKUP CONFIGURATION
# -----------------------------------------------------------------------------
BACKUP_ENABLED=true
BACKUP_SCHEDULE=0 3 * * *          # Daily at 3 AM
BACKUP_RETENTION_DAYS=30
BACKUP_STORAGE_PATH=/backups
BACKUP_ENCRYPT=true
BACKUP_ENCRYPTION_KEY=              # (REQUIRED if BACKUP_ENCRYPT=true)

# -----------------------------------------------------------------------------
# SECURITY
# -----------------------------------------------------------------------------
SECURITY_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
SECURITY_ALLOWED_METHODS=GET,POST,PUT,DELETE,OPTIONS
SECURITY_ALLOWED_HEADERS=Content-Type,Authorization
SECURITY_MAX_REQUEST_SIZE=100M

# -----------------------------------------------------------------------------
# MONITORING & ALERTING
# -----------------------------------------------------------------------------
ALERT_ENABLE=true
ALERT_SLACK_WEBHOOK=                # Optional: Slack webhook URL
ALERT_EMAIL_SMTP_HOST=
ALERT_EMAIL_SMTP_PORT=587
ALERT_EMAIL_FROM=
ALERT_EMAIL_TO=

# -----------------------------------------------------------------------------
# DEVELOPER SETTINGS
# -----------------------------------------------------------------------------
DEV_HOT_RELOAD=true               # Auto-reload on code changes
DEV_PROFILING=false               # Enable profiling endpoints
DEV_MOCK_EXTERNAL_SERVICES=false  # Mock external APIs

# =============================================================================
# GENERATED VALUES (Do not edit manually)
# =============================================================================
COMPOSE_PROJECT_NAME=mnemos
COMPOSE_FILE=docker-compose.yml
```

═══════════════════════════════════════════════════════════════════

## Validation & Testing

### Health Check Script

Create `scripts/health-check.sh`:

```bash
#!/bin/bash
# Health check script for all MNEMOS services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

echo "============================================="
echo "MNEMOS Health Check"
echo "============================================="
echo ""

# Function to check service health
check_service() {
    local service_name=$1
    local health_url=$2
    local expected_status=${3:-200}
    
    echo -n "Checking $service_name... "
    
    if curl -sf -o /dev/null -w "%{http_code}" "$health_url" | grep -q "$expected_status"; then
        echo -e "${GREEN}✓ Healthy${NC}"
        return 0
    else
        echo -e "${RED}✗ Unhealthy${NC}"
        return 1
    fi
}

# Track failures
FAILURES=0

# Check PostgreSQL
echo -n "Checking PostgreSQL... "
if docker exec mnemos-postgres pg_isready -U ${POSTGRES_USER:-mnemos} > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Healthy${NC}"
else
    echo -e "${RED}✗ Unhealthy${NC}"
    ((FAILURES++))
fi

# Check Redis
echo -n "Checking Redis... "
if docker exec mnemos-redis redis-cli --no-auth-warning -a "${REDIS_PASSWORD}" ping | grep -q "PONG"; then
    echo -e "${GREEN}✓ Healthy${NC}"
else
    echo -e "${RED}✗ Unhealthy${NC}"
    ((FAILURES++))
fi

# Check Vault
check_service "Vault" "http://localhost:8200/v1/sys/health" || ((FAILURES++))

# Check MinIO
check_service "MinIO" "http://localhost:9000/minio/health/live" || ((FAILURES++))

# Check Prometheus
check_service "Prometheus" "http://localhost:9090/-/healthy" || ((FAILURES++))

# Check Loki
check_service "Loki" "http://localhost:3100/ready" || ((FAILURES++))

# Check Grafana
check_service "Grafana" "http://localhost:3000/api/health" || ((FAILURES++))

# Check Traefik
check_service "Traefik" "http://localhost:8080/ping" || ((FAILURES++))

# Check OTEL Collector
check_service "OTEL Collector" "http://localhost:13133/" || ((FAILURES++))

echo ""
echo "============================================="
if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}All services healthy!${NC}"
    exit 0
else
    echo -e "${RED}$FAILURES service(s) unhealthy${NC}"
    exit 1
fi
```

### Makefile

Create comprehensive Makefile:

```makefile
.PHONY: help init build up down restart logs clean test health backup restore

# Default target
.DEFAULT_GOAL := help

# Load environment variables
include .env
export

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
RED := \033[0;31m
RESET := \033[0m

help: ## Show this help message
	@echo "$(CYAN)MNEMOS Makefile Commands$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

init: ## Initialize the project (first-time setup)
	@echo "$(CYAN)Initializing MNEMOS project...$(RESET)"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)✓$(RESET) Created .env file from .env.example"; \
		echo "$(RED)!$(RESET) Please edit .env and fill in required values"; \
	else \
		echo "$(YELLOW)!$(RESET) .env file already exists"; \
	fi
	@mkdir -p data/{models,artifacts,logs,backups,uploads}
	@mkdir -p data/{postgres,redis,vault,minio,prometheus,loki,grafana}
	@echo "$(GREEN)✓$(RESET) Created data directories"
	@echo "$(GREEN)✓$(RESET) Initialization complete"

build: ## Build all Docker images
	@echo "$(CYAN)Building Docker images...$(RESET)"
	docker compose build
	@echo "$(GREEN)✓$(RESET) Build complete"

up: ## Start all services
	@echo "$(CYAN)Starting MNEMOS services...$(RESET)"
	docker compose up -d
	@echo "$(GREEN)✓$(RESET) Services started"
	@echo "Run 'make logs' to view logs or 'make health' to check status"

down: ## Stop all services
	@echo "$(CYAN)Stopping MNEMOS services...$(RESET)"
	docker compose down
	@echo "$(GREEN)✓$(RESET) Services stopped"

restart: ## Restart all services
	@echo "$(CYAN)Restarting MNEMOS services...$(RESET)"
	docker compose restart
	@echo "$(GREEN)✓$(RESET) Services restarted"

restart-%: ## Restart a specific service (e.g., make restart-cortex)
	@echo "$(CYAN)Restarting $* service...$(RESET)"
	docker compose restart $*
	@echo "$(GREEN)✓$(RESET) Service $* restarted"

logs: ## Tail logs for all services
	docker compose logs -f

logs-%: ## Tail logs for a specific service (e.g., make logs-cortex)
	docker compose logs -f $*

ps: ## Show running services
	docker compose ps

health: ## Run health checks on all services
	@bash scripts/health-check.sh

clean: ## Stop services and remove volumes (WARNING: deletes all data)
	@echo "$(RED)WARNING: This will delete all data!$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker compose down -v; \
		rm -rf data/postgres data/redis data/vault data/minio data/prometheus data/loki data/grafana; \
		echo "$(GREEN)✓$(RESET) Cleaned up"; \
	else \
		echo "$(YELLOW)Cancelled$(RESET)"; \
	fi

shell-%: ## Open shell in a service container (e.g., make shell-postgres)
	docker compose exec $* /bin/sh

test: ## Run test suite
	@echo "$(CYAN)Running tests...$(RESET)"
	python -m pytest tests/ -v
	@echo "$(GREEN)✓$(RESET) Tests complete"

test-unit: ## Run unit tests only
	python -m pytest tests/unit/ -v

test-integration: ## Run integration tests
	python -m pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests
	python -m pytest tests/e2e/ -v

backup: ## Create backup of all data
	@echo "$(CYAN)Creating backup...$(RESET)"
	@bash scripts/backup/backup-all.sh
	@echo "$(GREEN)✓$(RESET) Backup complete"

restore: ## Restore from backup (requires BACKUP_FILE variable)
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "$(RED)ERROR: BACKUP_FILE variable required$(RESET)"; \
		echo "Usage: make restore BACKUP_FILE=<path-to-backup>"; \
		exit 1; \
	fi
	@echo "$(CYAN)Restoring from backup...$(RESET)"
	@bash scripts/backup/restore-all.sh $(BACKUP_FILE)
	@echo "$(GREEN)✓$(RESET) Restore complete"

bootstrap: ## Run CRADLE bootstrap process
	@echo "$(CYAN)Bootstrapping MNEMOS...$(RESET)"
	docker compose exec cradle python bootstrap.py
	@echo "$(GREEN)✓$(RESET) Bootstrap complete"

migrate: ## Run database migrations
	@echo "$(CYAN)Running database migrations...$(RESET)"
	docker compose exec cortex alembic upgrade head
	docker compose exec genome alembic upgrade head
	docker compose exec synkron alembic upgrade head
	@echo "$(GREEN)✓$(RESET) Migrations complete"

seed: ## Seed initial data
	@echo "$(CYAN)Seeding initial data...$(RESET)"
	docker compose exec cortex python scripts/seed.py
	@echo "$(GREEN)✓$(RESET) Seeding complete"

lint: ## Run linters on all code
	@echo "$(CYAN)Running linters...$(RESET)"
	@echo "Python linting..."
	flake8 src/
	black --check src/
	mypy src/
	@echo "Shell script linting..."
	find scripts -name "*.sh" -exec shellcheck {} \;
	@echo "Dockerfile linting..."
	find docker -name "Dockerfile" -exec hadolint {} \;
	@echo "$(GREEN)✓$(RESET) Linting complete"

format: ## Auto-format code
	@echo "$(CYAN)Formatting code...$(RESET)"
	black src/
	isort src/
	@echo "$(GREEN)✓$(RESET) Formatting complete"

security-scan: ## Run security scans
	@echo "$(CYAN)Running security scans...$(RESET)"
	trivy image mnemos-cortex:latest
	trivy image mnemos-genome:latest
	trivy image mnemos-neuron:latest
	@echo "$(GREEN)✓$(RESET) Security scan complete"

dev-setup: init build ## Complete development setup
	@echo "$(GREEN)Development setup complete!$(RESET)"
	@echo "Next steps:"
	@echo "  1. Edit .env with your configuration"
	@echo "  2. Run 'make up' to start services"
	@echo "  3. Run 'make health' to verify"

prod-deploy: ## Deploy to production (with safety checks)
	@echo "$(RED)Deploying to PRODUCTION$(RESET)"
	@if [ "$(ENVIRONMENT)" != "production" ]; then \
		echo "$(RED)ERROR: ENVIRONMENT must be 'production'$(RESET)"; \
		exit 1; \
	fi
	@read -p "Continue with production deployment? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		make build && make up && make health; \
	else \
		echo "$(YELLOW)Deployment cancelled$(RESET)"; \
	fi
```

Make the Makefile and scripts executable:

```bash
chmod +x scripts/health-check.sh
chmod +x Makefile
```

═══════════════════════════════════════════════════════════════════

## Phase 1 Completion Checklist

- [ ] WSL Ubuntu 22.04 installed and configured
- [ ] Docker Desktop installed with WSL2 integration
- [ ] Claude Code CLI installed and authenticated
- [ ] Project directory structure created
- [ ] .gitignore configured
- [ ] docker-compose.yml created with all infrastructure services
- [ ] .env.example created with all variables documented
- [ ] .env file created with actual values (not committed)
- [ ] Makefile created with all targets
- [ ] Health check script created and executable
- [ ] `make init` executed successfully
- [ ] PostgreSQL container running and accepting connections
- [ ] Redis container running and responding to PING
- [ ] Vault container running (dev mode for now)
- [ ] MinIO container running with console accessible
- [ ] Prometheus container running and scraping targets
- [ ] Loki container running and accepting logs
- [ ] Grafana container running with dashboards accessible
- [ ] Traefik container running with dashboard accessible
- [ ] OTEL Collector running and accepting traces
- [ ] All services passing health checks (`make health` succeeds)
- [ ] Network isolation verified (backend/data networks are internal)
- [ ] Volumes created and data persisting across restarts
- [ ] Grafana accessible at http://localhost:3000
- [ ] Traefik dashboard accessible at http://localhost:8080
- [ ] MinIO console accessible at http://localhost:9001
- [ ] Prometheus accessible at http://localhost:9090
- [ ] Can view logs with `make logs`
- [ ] Can restart individual services
- [ ] Documentation reviewed and understood

═══════════════════════════════════════════════════════════════════

## Next Phase Preview

**Phase 2** will cover:
- Detailed configuration for each infrastructure service
- PostgreSQL schema initialization and migrations
- Redis configuration for job queues
- Vault initialization and policy setup
- Security hardening for all services
- TLS certificate generation and configuration
- Backup and restore procedures
- Advanced monitoring setup

═══════════════════════════════════════════════════════════════════

*End of Phase 1*

**Continue to:** [MNEMOS_Phase_2.md](./MNEMOS_Phase_2.md)
