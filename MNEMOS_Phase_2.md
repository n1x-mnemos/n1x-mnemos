# MNEMOS Phase 2: Infrastructure Configuration & Security

**Version:** 1.0.0  
**Phase:** Infrastructure Configuration & Security Hardening  
**Prerequisites:** Phase 1 Complete  
**Target Environment:** Docker Compose on Ubuntu WSL

═══════════════════════════════════════════════════════════════════

## Table of Contents

1. [Phase Overview](#phase-overview)
2. [PostgreSQL Configuration](#postgresql-configuration)
3. [Redis Configuration](#redis-configuration)
4. [Vault (SOUL) Configuration](#vault-soul-configuration)
5. [MinIO (ENGRAM) Configuration](#minio-engram-configuration)
6. [TLS/SSL Certificate Management](#tlsssl-certificate-management)
7. [Security Hardening](#security-hardening)
8. [Backup & Restore](#backup--restore)
9. [Monitoring Configuration](#monitoring-configuration)
10. [Validation & Testing](#validation--testing)

═══════════════════════════════════════════════════════════════════

## Phase Overview

### Objectives

Phase 2 focuses on configuring and securing all infrastructure services:

- Configure PostgreSQL with optimized settings and schemas
- Set up Redis for high-performance job queuing
- Initialize Vault with policies and secrets
- Configure MinIO for artifact storage
- Generate and configure TLS certificates
- Implement security hardening across all services
- Set up automated backup and restore procedures
- Configure comprehensive monitoring and alerting

### Success Criteria

- [ ] PostgreSQL schemas created and optimized
- [ ] Redis configured for job queues with persistence
- [ ] Vault initialized with all required policies
- [ ] MinIO buckets created with lifecycle policies
- [ ] TLS certificates generated and configured
- [ ] All services hardened according to security best practices
- [ ] Backup automation configured and tested
- [ ] Monitoring dashboards operational
- [ ] All configuration validated

═══════════════════════════════════════════════════════════════════

## PostgreSQL Configuration

### Performance Tuning Configuration

Create `docker/postgres/postgresql.conf`:

```conf
# =============================================================================
# MNEMOS PostgreSQL Configuration
# =============================================================================
# Optimized for: 16GB RAM, 8 CPU cores, SSD storage
# Adjust based on your actual hardware
# =============================================================================

# -----------------------------------------------------------------------------
# CONNECTION SETTINGS
# -----------------------------------------------------------------------------
listen_addresses = '*'
port = 5432
max_connections = 200
superuser_reserved_connections = 3

# -----------------------------------------------------------------------------
# MEMORY SETTINGS
# -----------------------------------------------------------------------------
# Shared buffers: 25% of RAM
shared_buffers = 4GB

# Effective cache: 50-75% of RAM
effective_cache_size = 12GB

# Work memory per operation
work_mem = 64MB
maintenance_work_mem = 512MB

# Temp buffers for sessions
temp_buffers = 32MB

# -----------------------------------------------------------------------------
# WRITE AHEAD LOG (WAL)
# -----------------------------------------------------------------------------
wal_level = replica
wal_buffers = 16MB
min_wal_size = 1GB
max_wal_size = 4GB

# Checkpoints
checkpoint_timeout = 15min
checkpoint_completion_target = 0.9
checkpoint_warning = 30s

# Archive mode (for backups)
archive_mode = on
archive_command = '/usr/local/bin/archive_wal.sh %p %f'

# -----------------------------------------------------------------------------
# QUERY TUNING
# -----------------------------------------------------------------------------
random_page_cost = 1.1  # For SSD
effective_io_concurrency = 200
max_worker_processes = 8
max_parallel_workers_per_gather = 4
max_parallel_workers = 8

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging_collector = on
log_directory = '/var/log/postgresql'
log_filename = 'postgresql-%Y-%m-%d.log'
log_rotation_age = 1d
log_rotation_size = 100MB
log_truncate_on_rotation = on

# What to log
log_min_duration_statement = 1000  # Log queries > 1s
log_connections = on
log_disconnections = on
log_duration = off
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '

# Log statements
log_statement = 'ddl'  # Log DDL statements
log_lock_waits = on
log_temp_files = 0

# -----------------------------------------------------------------------------
# STATISTICS
# -----------------------------------------------------------------------------
shared_preload_libraries = 'pg_stat_statements'
pg_stat_statements.max = 10000
pg_stat_statements.track = all

track_activities = on
track_counts = on
track_io_timing = on
track_functions = all

# -----------------------------------------------------------------------------
# AUTOVACUUM
# -----------------------------------------------------------------------------
autovacuum = on
autovacuum_max_workers = 3
autovacuum_naptime = 30s
autovacuum_vacuum_threshold = 50
autovacuum_analyze_threshold = 50
autovacuum_vacuum_scale_factor = 0.1
autovacuum_analyze_scale_factor = 0.05

# -----------------------------------------------------------------------------
# CLIENT CONNECTION DEFAULTS
# -----------------------------------------------------------------------------
datestyle = 'iso, mdy'
timezone = 'UTC'
lc_messages = 'en_US.UTF-8'
lc_monetary = 'en_US.UTF-8'
lc_numeric = 'en_US.UTF-8'
lc_time = 'en_US.UTF-8'
default_text_search_config = 'pg_catalog.english'

# -----------------------------------------------------------------------------
# SECURITY
# -----------------------------------------------------------------------------
ssl = on
ssl_cert_file = '/etc/ssl/certs/server.crt'
ssl_key_file = '/etc/ssl/private/server.key'
ssl_ca_file = '/etc/ssl/certs/ca.crt'
ssl_ciphers = 'HIGH:!aNULL:!MD5'
ssl_prefer_server_ciphers = on
ssl_min_protocol_version = 'TLSv1.2'

# Password encryption
password_encryption = scram-sha-256

# Statement timeout (prevent runaway queries)
statement_timeout = 60000  # 60 seconds

# Lock timeout
lock_timeout = 30000  # 30 seconds

# Idle in transaction timeout
idle_in_transaction_session_timeout = 300000  # 5 minutes
```

### Database Initialization Script

Create `docker/postgres/init-db.sh`:

```bash
#!/bin/bash
# PostgreSQL initialization script for MNEMOS
# Creates databases, users, schemas, and initial tables

set -e
set -u

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=================================================="
echo "MNEMOS PostgreSQL Initialization"
echo "=================================================="
echo ""

# Function to execute SQL
psql_exec() {
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        $1
EOSQL
}

# Function to create database if not exists
create_database() {
    local db_name=$1
    echo -n "Creating database: $db_name... "
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        SELECT 'CREATE DATABASE $db_name'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$db_name')\gexec
EOSQL
    echo -e "${GREEN}✓${NC}"
}

# Function to create user if not exists
create_user() {
    local username=$1
    local password=$2
    echo -n "Creating user: $username... "
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        DO \$\$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_user WHERE usename = '$username') THEN
                CREATE USER $username WITH PASSWORD '$password';
            END IF;
        END
        \$\$;
EOSQL
    echo -e "${GREEN}✓${NC}"
}

# =============================================================================
# CREATE DATABASES
# =============================================================================

echo "Creating databases..."
create_database "mnemos_cortex"
create_database "mnemos_genome"
create_database "mnemos_synkron"
create_database "mnemos_trace"
echo ""

# =============================================================================
# CREATE USERS
# =============================================================================

echo "Creating users..."
create_user "cortex_user" "${CORTEX_DB_PASSWORD:-cortex_password}"
create_user "genome_user" "${GENOME_DB_PASSWORD:-genome_password}"
create_user "synkron_user" "${SYNKRON_DB_PASSWORD:-synkron_password}"
create_user "trace_user" "${TRACE_DB_PASSWORD:-trace_password}"
echo ""

# =============================================================================
# INSTALL EXTENSIONS
# =============================================================================

echo "Installing extensions..."

for db in mnemos_cortex mnemos_genome mnemos_synkron mnemos_trace; do
    echo -n "  $db: "
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$db" <<-EOSQL
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE EXTENSION IF NOT EXISTS "pg_trgm";
        CREATE EXTENSION IF NOT EXISTS "btree_gin";
        CREATE EXTENSION IF NOT EXISTS "btree_gist";
        CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
EOSQL
    echo -e "${GREEN}✓${NC}"
done
echo ""

# =============================================================================
# CORTEX DATABASE SCHEMA
# =============================================================================

echo "Creating CORTEX schema..."
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "mnemos_cortex" <<-'EOSQL'

-- Schema for jobs
CREATE SCHEMA IF NOT EXISTS jobs;

-- Schema for workers
CREATE SCHEMA IF NOT EXISTS workers;

-- Schema for audit
CREATE SCHEMA IF NOT EXISTS audit;

-- Jobs table
CREATE TABLE IF NOT EXISTS jobs.jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    priority INTEGER NOT NULL DEFAULT 5,
    model_ref VARCHAR(255),
    input_data JSONB NOT NULL,
    output_data JSONB,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    timeout_seconds INTEGER DEFAULT 3600,
    worker_id UUID REFERENCES workers.workers(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    tags TEXT[] DEFAULT ARRAY[]::TEXT[]
);

-- Indexes for jobs
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs.jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs.jobs(priority DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs.jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_worker_id ON jobs.jobs(worker_id);
CREATE INDEX IF NOT EXISTS idx_jobs_model_ref ON jobs.jobs(model_ref);
CREATE INDEX IF NOT EXISTS idx_jobs_tags ON jobs.jobs USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_jobs_metadata ON jobs.jobs USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_jobs_composite ON jobs.jobs(status, priority DESC, created_at);

-- Workers table
CREATE TABLE IF NOT EXISTS workers.workers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    worker_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'idle',
    capabilities JSONB DEFAULT '{}'::jsonb,
    current_load INTEGER DEFAULT 0,
    max_load INTEGER DEFAULT 10,
    last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for workers
CREATE INDEX IF NOT EXISTS idx_workers_status ON workers.workers(status);
CREATE INDEX IF NOT EXISTS idx_workers_type ON workers.workers(worker_type);
CREATE INDEX IF NOT EXISTS idx_workers_heartbeat ON workers.workers(last_heartbeat DESC);
CREATE INDEX IF NOT EXISTS idx_workers_capabilities ON workers.workers USING GIN(capabilities);

-- Job history (for completed/failed jobs)
CREATE TABLE IF NOT EXISTS jobs.job_history (
    id UUID PRIMARY KEY,
    job_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    priority INTEGER NOT NULL,
    model_ref VARCHAR(255),
    input_data JSONB NOT NULL,
    output_data JSONB,
    error_message TEXT,
    retry_count INTEGER,
    worker_id UUID,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    duration_ms INTEGER,
    metadata JSONB,
    archived_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Partition job_history by month
CREATE INDEX IF NOT EXISTS idx_job_history_completed_at ON jobs.job_history(completed_at DESC);
CREATE INDEX IF NOT EXISTS idx_job_history_status ON jobs.job_history(status);

-- Audit log table
CREATE TABLE IF NOT EXISTS audit.audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    entity_id UUID,
    actor VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    changes JSONB,
    metadata JSONB DEFAULT '{}'::jsonb,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit.audit_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit.audit_log(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit.audit_log(actor);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_jobs_updated_at BEFORE UPDATE ON jobs.jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workers_updated_at BEFORE UPDATE ON workers.workers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to archive completed jobs
CREATE OR REPLACE FUNCTION archive_completed_jobs()
RETURNS INTEGER AS $$
DECLARE
    archived_count INTEGER;
BEGIN
    WITH moved_jobs AS (
        DELETE FROM jobs.jobs
        WHERE status IN ('completed', 'failed')
        AND completed_at < CURRENT_TIMESTAMP - INTERVAL '7 days'
        RETURNING *
    )
    INSERT INTO jobs.job_history (
        id, job_type, status, priority, model_ref, input_data, output_data,
        error_message, retry_count, worker_id, created_at, started_at,
        completed_at, duration_ms, metadata
    )
    SELECT 
        id, job_type, status, priority, model_ref, input_data, output_data,
        error_message, retry_count, worker_id, created_at, started_at,
        completed_at,
        EXTRACT(EPOCH FROM (completed_at - started_at)) * 1000,
        metadata
    FROM moved_jobs;
    
    GET DIAGNOSTICS archived_count = ROW_COUNT;
    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT USAGE ON SCHEMA jobs TO cortex_user;
GRANT USAGE ON SCHEMA workers TO cortex_user;
GRANT USAGE ON SCHEMA audit TO cortex_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA jobs TO cortex_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA workers TO cortex_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA audit TO cortex_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA jobs TO cortex_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA workers TO cortex_user;

EOSQL
echo -e "${GREEN}✓ CORTEX schema created${NC}"
echo ""

# =============================================================================
# GENOME DATABASE SCHEMA
# =============================================================================

echo "Creating GENOME schema..."
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "mnemos_genome" <<-'EOSQL'

-- Schema for configurations
CREATE SCHEMA IF NOT EXISTS schemas;

-- Schema for policies
CREATE SCHEMA IF NOT EXISTS policies;

-- Schema for audit
CREATE SCHEMA IF NOT EXISTS audit;

-- Models registry
CREATE TABLE IF NOT EXISTS schemas.models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    kind VARCHAR(50) NOT NULL DEFAULT 'Model',
    version VARCHAR(100) NOT NULL,
    family VARCHAR(100),
    spec JSONB NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(50) DEFAULT 'active',
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, version)
);

CREATE INDEX IF NOT EXISTS idx_models_name ON schemas.models(name);
CREATE INDEX IF NOT EXISTS idx_models_family ON schemas.models(family);
CREATE INDEX IF NOT EXISTS idx_models_status ON schemas.models(status);
CREATE INDEX IF NOT EXISTS idx_models_spec ON schemas.models USING GIN(spec);

-- Pipelines registry
CREATE TABLE IF NOT EXISTS schemas.pipelines (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    kind VARCHAR(50) NOT NULL DEFAULT 'Pipeline',
    spec JSONB NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(50) DEFAULT 'active',
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pipelines_name ON schemas.pipelines(name);
CREATE INDEX IF NOT EXISTS idx_pipelines_status ON schemas.pipelines(status);
CREATE INDEX IF NOT EXISTS idx_pipelines_spec ON schemas.pipelines USING GIN(spec);

-- Policies registry
CREATE TABLE IF NOT EXISTS policies.policies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    kind VARCHAR(50) NOT NULL DEFAULT 'Policy',
    spec JSONB NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(50) DEFAULT 'active',
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_policies_name ON policies.policies(name);
CREATE INDEX IF NOT EXISTS idx_policies_status ON policies.policies(status);
CREATE INDEX IF NOT EXISTS idx_policies_spec ON policies.policies USING GIN(spec);

-- Schema validation history
CREATE TABLE IF NOT EXISTS schemas.validation_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type VARCHAR(100) NOT NULL,
    entity_id UUID NOT NULL,
    validation_result BOOLEAN NOT NULL,
    errors JSONB,
    validated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_validation_history_entity ON schemas.validation_history(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_validation_history_result ON schemas.validation_history(validation_result);

-- Audit log
CREATE TABLE IF NOT EXISTS audit.audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    entity_id UUID,
    actor VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    changes JSONB,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_genome_audit_created_at ON audit.audit_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_genome_audit_entity ON audit.audit_log(entity_type, entity_id);

-- Triggers
CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON schemas.models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_pipelines_updated_at BEFORE UPDATE ON schemas.pipelines
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_policies_updated_at BEFORE UPDATE ON policies.policies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT USAGE ON SCHEMA schemas TO genome_user;
GRANT USAGE ON SCHEMA policies TO genome_user;
GRANT USAGE ON SCHEMA audit TO genome_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA schemas TO genome_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA policies TO genome_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA audit TO genome_user;

EOSQL
echo -e "${GREEN}✓ GENOME schema created${NC}"
echo ""

# =============================================================================
# SYNKRON DATABASE SCHEMA
# =============================================================================

echo "Creating SYNKRON schema..."
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "mnemos_synkron" <<-'EOSQL'

-- Schema for pipelines
CREATE SCHEMA IF NOT EXISTS pipelines;

-- Schema for executions
CREATE SCHEMA IF NOT EXISTS executions;

-- Pipeline definitions
CREATE TABLE IF NOT EXISTS pipelines.definitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    spec JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pipeline_definitions_name ON pipelines.definitions(name);
CREATE INDEX IF NOT EXISTS idx_pipeline_definitions_status ON pipelines.definitions(status);

-- Pipeline executions
CREATE TABLE IF NOT EXISTS executions.runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_id UUID NOT NULL REFERENCES pipelines.definitions(id),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    trigger VARCHAR(100),
    parameters JSONB DEFAULT '{}'::jsonb,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_runs_pipeline_id ON executions.runs(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON executions.runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON executions.runs(created_at DESC);

-- Pipeline steps
CREATE TABLE IF NOT EXISTS executions.steps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES executions.runs(id) ON DELETE CASCADE,
    step_name VARCHAR(255) NOT NULL,
    step_order INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    input_data JSONB,
    output_data JSONB,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_steps_run_id ON executions.steps(run_id);
CREATE INDEX IF NOT EXISTS idx_steps_status ON executions.steps(status);
CREATE INDEX IF NOT EXISTS idx_steps_order ON executions.steps(step_order);

-- Triggers
CREATE TRIGGER update_pipeline_definitions_updated_at BEFORE UPDATE ON pipelines.definitions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_runs_updated_at BEFORE UPDATE ON executions.runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_steps_updated_at BEFORE UPDATE ON executions.steps
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT USAGE ON SCHEMA pipelines TO synkron_user;
GRANT USAGE ON SCHEMA executions TO synkron_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA pipelines TO synkron_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA executions TO synkron_user;

EOSQL
echo -e "${GREEN}✓ SYNKRON schema created${NC}"
echo ""

# =============================================================================
# TRACE DATABASE SCHEMA
# =============================================================================

echo "Creating TRACE schema..."
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "mnemos_trace" <<-'EOSQL'

-- Schema for events
CREATE SCHEMA IF NOT EXISTS events;

-- Event log
CREATE TABLE IF NOT EXISTS events.event_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    source VARCHAR(255) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    message TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    trace_id VARCHAR(255),
    span_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_event_log_created_at ON events.event_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_event_log_type ON events.event_log(event_type);
CREATE INDEX IF NOT EXISTS idx_event_log_source ON events.event_log(source);
CREATE INDEX IF NOT EXISTS idx_event_log_severity ON events.event_log(severity);
CREATE INDEX IF NOT EXISTS idx_event_log_trace_id ON events.event_log(trace_id);

-- Grant permissions
GRANT USAGE ON SCHEMA events TO trace_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA events TO trace_user;

EOSQL
echo -e "${GREEN}✓ TRACE schema created${NC}"
echo ""

# =============================================================================
# FINALIZATION
# =============================================================================

echo "=================================================="
echo "PostgreSQL initialization complete!"
echo "=================================================="
echo ""
echo "Databases created:"
echo "  - mnemos_cortex (job orchestration)"
echo "  - mnemos_genome (schema registry)"
echo "  - mnemos_synkron (pipeline orchestration)"
echo "  - mnemos_trace (event logging)"
echo ""
echo "Users created:"
echo "  - cortex_user"
echo "  - genome_user"
echo "  - synkron_user"
echo "  - trace_user"
echo ""
```

Make the script executable:

```bash
chmod +x docker/postgres/init-db.sh
```

### PostgreSQL Backup Script

Create `scripts/backup/backup-postgres.sh`:

```bash
#!/bin/bash
# PostgreSQL backup script

set -e

BACKUP_DIR="${BACKUP_DIR:-/backups/postgres}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-30}

# Create backup directory
mkdir -p "$BACKUP_DIR"

echo "Starting PostgreSQL backup..."

# Backup all databases
for db in mnemos mnemos_cortex mnemos_genome mnemos_synkron mnemos_trace; do
    echo "Backing up $db..."
    docker exec mnemos-postgres pg_dump -U mnemos -Fc "$db" > \
        "$BACKUP_DIR/${db}_${TIMESTAMP}.dump"
done

# Backup globals (users, roles, etc.)
echo "Backing up globals..."
docker exec mnemos-postgres pg_dumpall -U mnemos --globals-only > \
    "$BACKUP_DIR/globals_${TIMESTAMP}.sql"

# Compress backups
echo "Compressing backups..."
tar -czf "$BACKUP_DIR/postgres_backup_${TIMESTAMP}.tar.gz" \
    -C "$BACKUP_DIR" \
    --remove-files \
    *.dump *.sql

# Clean old backups
echo "Cleaning old backups..."
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete

echo "PostgreSQL backup complete: postgres_backup_${TIMESTAMP}.tar.gz"
```

═══════════════════════════════════════════════════════════════════

## Redis Configuration

### Redis Production Configuration

Create `docker/redis/redis.conf`:

```conf
# =============================================================================
# MNEMOS Redis Configuration
# =============================================================================
# Optimized for job queuing and caching
# =============================================================================

# -----------------------------------------------------------------------------
# NETWORK
# -----------------------------------------------------------------------------
bind 0.0.0.0
protected-mode yes
port 6379
tcp-backlog 511
timeout 0
tcp-keepalive 300

# -----------------------------------------------------------------------------
# SECURITY
# -----------------------------------------------------------------------------
# Password will be set via command line
# requirepass <password>

# Disable dangerous commands
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command KEYS ""
rename-command CONFIG "CONFIG_MNEMOS"

# -----------------------------------------------------------------------------
# MEMORY
# -----------------------------------------------------------------------------
maxmemory 2gb
maxmemory-policy allkeys-lru

# Maximum number of clients
maxclients 10000

# -----------------------------------------------------------------------------
# PERSISTENCE
# -----------------------------------------------------------------------------
# RDB Snapshots
save 900 1      # After 900 sec (15 min) if at least 1 key changed
save 300 10     # After 300 sec (5 min) if at least 10 keys changed
save 60 10000   # After 60 sec if at least 10000 keys changed

stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /data

# AOF (Append Only File) - More durable
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# -----------------------------------------------------------------------------
# REPLICATION
# -----------------------------------------------------------------------------
replica-serve-stale-data yes
replica-read-only yes
repl-diskless-sync no
repl-diskless-sync-delay 5
repl-disable-tcp-nodelay no
replica-priority 100

# -----------------------------------------------------------------------------
# LIMITS
# -----------------------------------------------------------------------------
# Max memory for each request
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit replica 256mb 64mb 60
client-output-buffer-limit pubsub 32mb 8mb 60

# -----------------------------------------------------------------------------
# SLOW LOG
# -----------------------------------------------------------------------------
slowlog-log-slower-than 10000  # microseconds
slowlog-max-len 128

# -----------------------------------------------------------------------------
# LATENCY MONITOR
# -----------------------------------------------------------------------------
latency-monitor-threshold 100  # milliseconds

# -----------------------------------------------------------------------------
# EVENT NOTIFICATION
# -----------------------------------------------------------------------------
notify-keyspace-events ""

# -----------------------------------------------------------------------------
# ADVANCED CONFIG
# -----------------------------------------------------------------------------
hash-max-ziplist-entries 512
hash-max-ziplist-value 64

list-max-ziplist-size -2
list-compress-depth 0

set-max-intset-entries 512

zset-max-ziplist-entries 128
zset-max-ziplist-value 64

hll-sparse-max-bytes 3000

stream-node-max-bytes 4096
stream-node-max-entries 100

activerehashing yes

client-output-buffer-limit normal 0 0 0
client-output-buffer-limit replica 256mb 64mb 60
client-output-buffer-limit pubsub 32mb 8mb 60

hz 10

dynamic-hz yes

aof-rewrite-incremental-fsync yes

rdb-save-incremental-fsync yes

# -----------------------------------------------------------------------------
# TLS/SSL
# -----------------------------------------------------------------------------
# tls-port 6380
# tls-cert-file /path/to/redis.crt
# tls-key-file /path/to/redis.key
# tls-ca-cert-file /path/to/ca.crt
# tls-protocols "TLSv1.2 TLSv1.3"
# tls-ciphers HIGH:!aNULL:!MD5
# tls-prefer-server-ciphers yes
```

### Redis Health Check Script

Create `docker/redis/health-check.sh`:

```bash
#!/bin/sh
# Redis health check

redis-cli --no-auth-warning -a "$REDIS_PASSWORD" ping | grep -q "PONG"
```

═══════════════════════════════════════════════════════════════════

## Vault (SOUL) Configuration

### Vault Configuration File

Create `docker/vault/config/vault.hcl`:

```hcl
# =============================================================================
# MNEMOS Vault Configuration
# =============================================================================

# Storage backend - File storage for Docker environment
storage "file" {
  path = "/vault/data"
}

# TCP listener with TLS
listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = 0
  
  tls_cert_file = "/vault/config/certs/server.crt"
  tls_key_file  = "/vault/config/certs/server.key"
  
  tls_min_version = "tls12"
  tls_cipher_suites = "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"
  
  tls_require_and_verify_client_cert = false
  tls_disable_client_certs = false
}

# API and cluster addresses
api_addr = "https://vault:8200"
cluster_addr = "https://vault:8201"

# UI
ui = true

# Telemetry
telemetry {
  prometheus_retention_time = "30s"
  disable_hostname = false
  
  statsd_address = "localhost:8125"
}

# Audit logging
log_level = "INFO"
log_format = "json"

# Disable mlock in containers (handled by CAP_IPC_LOCK)
disable_mlock = false

# Default lease and token TTL
default_lease_ttl = "168h"   # 7 days
max_lease_ttl = "720h"       # 30 days

# Plugin directory
plugin_directory = "/vault/plugins"
```

### Vault Initialization Script

Create `docker/vault/scripts/init-vault.sh`:

```bash
#!/bin/bash
# Vault initialization and configuration script

set -e

VAULT_ADDR="${VAULT_ADDR:-https://vault:8200}"
VAULT_SKIP_VERIFY="${VAULT_SKIP_VERIFY:-false}"
KEYS_FILE="/vault/data/.vault-keys.json"

echo "=================================================="
echo "MNEMOS Vault Initialization"
echo "=================================================="
echo ""

# Wait for Vault to be ready
echo "Waiting for Vault to be ready..."
until vault status > /dev/null 2>&1 || [ $? -eq 2 ]; do
    echo "  Waiting for Vault..."
    sleep 2
done
echo "✓ Vault is ready"
echo ""

# Check if already initialized
if vault status | grep -q "Initialized.*true"; then
    echo "Vault is already initialized"
    
    # Check if keys file exists
    if [ -f "$KEYS_FILE" ]; then
        echo "Loading unseal keys from $KEYS_FILE"
        
        # Auto-unseal if sealed
        if vault status | grep -q "Sealed.*true"; then
            echo "Vault is sealed, unsealing..."
            
            # Extract keys and unseal
            KEY_1=$(jq -r '.unseal_keys_b64[0]' "$KEYS_FILE")
            KEY_2=$(jq -r '.unseal_keys_b64[1]' "$KEYS_FILE")
            KEY_3=$(jq -r '.unseal_keys_b64[2]' "$KEYS_FILE")
            
            vault operator unseal "$KEY_1"
            vault operator unseal "$KEY_2"
            vault operator unseal "$KEY_3"
            
            echo "✓ Vault unsealed"
        else
            echo "✓ Vault is already unsealed"
        fi
    else
        echo "ERROR: Vault is initialized but keys file not found"
        echo "Manual unseal required"
        exit 1
    fi
else
    echo "Initializing Vault..."
    
    # Initialize Vault (5 shares, threshold 3)
    vault operator init \
        -key-shares=5 \
        -key-threshold=3 \
        -format=json > "$KEYS_FILE"
    
    chmod 600 "$KEYS_FILE"
    echo "✓ Vault initialized"
    echo "⚠️  IMPORTANT: Backup $KEYS_FILE securely!"
    echo ""
    
    # Auto-unseal
    echo "Unsealing Vault..."
    KEY_1=$(jq -r '.unseal_keys_b64[0]' "$KEYS_FILE")
    KEY_2=$(jq -r '.unseal_keys_b64[1]' "$KEYS_FILE")
    KEY_3=$(jq -r '.unseal_keys_b64[2]' "$KEYS_FILE")
    
    vault operator unseal "$KEY_1"
    vault operator unseal "$KEY_2"
    vault operator unseal "$KEY_3"
    
    echo "✓ Vault unsealed"
    echo ""
    
    # Login with root token
    ROOT_TOKEN=$(jq -r '.root_token' "$KEYS_FILE")
    export VAULT_TOKEN="$ROOT_TOKEN"
    
    echo "Configuring Vault..."
    
    # Enable audit logging
    echo "  Enabling audit log..."
    vault audit enable file file_path=/vault/logs/audit.log
    
    # Enable secrets engines
    echo "  Enabling secrets engines..."
    vault secrets enable -path=mnemos/ kv-v2
    vault secrets enable -path=mnemos/transit transit
    vault secrets enable -path=mnemos/pki pki
    
    # Configure PKI
    echo "  Configuring PKI..."
    vault secrets tune -max-lease-ttl=87600h mnemos/pki
    
    vault write mnemos/pki/root/generate/internal \
        common_name="MNEMOS Root CA" \
        ttl=87600h
    
    vault write mnemos/pki/config/urls \
        issuing_certificates="${VAULT_ADDR}/v1/mnemos/pki/ca" \
        crl_distribution_points="${VAULT_ADDR}/v1/mnemos/pki/crl"
    
    vault write mnemos/pki/roles/mnemos-service \
        allowed_domains="mnemos.local,mnemos.svc,localhost" \
        allow_subdomains=true \
        max_ttl="720h"
    
    # Enable authentication methods
    echo "  Enabling auth methods..."
    vault auth enable approle
    vault auth enable jwt
    
    # Create policies
    echo "  Creating policies..."
    
    # CORTEX policy
    vault policy write cortex-policy - <<EOF
path "mnemos/data/cortex/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
path "mnemos/data/database/*" {
  capabilities = ["read"]
}
path "mnemos/pki/issue/mnemos-service" {
  capabilities = ["create", "update"]
}
EOF
    
    # GENOME policy
    vault policy write genome-policy - <<EOF
path "mnemos/data/genome/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
path "mnemos/data/database/*" {
  capabilities = ["read"]
}
EOF
    
    # NEURON policy
    vault policy write neuron-policy - <<EOF
path "mnemos/data/neuron/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
path "mnemos/data/models/*" {
  capabilities = ["read"]
}
EOF
    
    # Store initial secrets
    echo "  Storing initial secrets..."
    
    vault kv put mnemos/database/postgres \
        username="mnemos" \
        password="${POSTGRES_PASSWORD}"
    
    vault kv put mnemos/database/redis \
        password="${REDIS_PASSWORD}"
    
    vault kv put mnemos/object-storage/minio \
        access_key="${MINIO_ROOT_USER}" \
        secret_key="${MINIO_ROOT_PASSWORD}"
    
    vault kv put mnemos/jwt \
        secret_key="${JWT_SECRET_KEY}"
    
    echo ""
    echo "✓ Vault configuration complete"
    echo ""
    echo "Root Token: $ROOT_TOKEN"
    echo "⚠️  Save this token securely!"
fi

echo ""
echo "=================================================="
echo "Vault Status:"
vault status
echo "=================================================="
```

Make scripts executable:

```bash
chmod +x docker/vault/scripts/init-vault.sh
```

═══════════════════════════════════════════════════════════════════

## MinIO (ENGRAM) Configuration

### MinIO Initialization Script

Create `docker/minio/init-minio.sh`:

```bash
#!/bin/bash
# MinIO initialization script

set -e

echo "=================================================="
echo "MNEMOS MinIO Initialization"
echo "=================================================="
echo ""

# Wait for MinIO to be ready
echo "Waiting for MinIO to be ready..."
until mc alias set mnemos http://minio:9000 \
    "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}" > /dev/null 2>&1; do
    echo "  Waiting for MinIO..."
    sleep 2
done
echo "✓ MinIO is ready"
echo ""

# Create buckets
echo "Creating buckets..."

BUCKETS=("mnemos-artifacts" "mnemos-models" "mnemos-backups" "mnemos-logs")

for bucket in "${BUCKETS[@]}"; do
    if mc ls mnemos/$bucket > /dev/null 2>&1; then
        echo "  ✓ Bucket $bucket already exists"
    else
        mc mb mnemos/$bucket
        echo "  ✓ Created bucket: $bucket"
    fi
done

# Set bucket policies
echo ""
echo "Configuring bucket policies..."

# Artifacts bucket - private
mc anonymous set none mnemos/mnemos-artifacts
echo "  ✓ mnemos-artifacts: private"

# Models bucket - private
mc anonymous set none mnemos/mnemos-models
echo "  ✓ mnemos-models: private"

# Backups bucket - private
mc anonymous set none mnemos/mnemos-backups
echo "  ✓ mnemos-backups: private"

# Configure lifecycle policies
echo ""
echo "Configuring lifecycle policies..."

# Artifacts: delete after 90 days
cat > /tmp/artifacts-lifecycle.json <<EOF
{
  "Rules": [
    {
      "ID": "DeleteOldArtifacts",
      "Status": "Enabled",
      "Filter": {
        "Prefix": ""
      },
      "Expiration": {
        "Days": 90
      }
    }
  ]
}
EOF

mc ilm import mnemos/mnemos-artifacts < /tmp/artifacts-lifecycle.json
echo "  ✓ Lifecycle policy set for mnemos-artifacts"

# Logs: delete after 30 days
cat > /tmp/logs-lifecycle.json <<EOF
{
  "Rules": [
    {
      "ID": "DeleteOldLogs",
      "Status": "Enabled",
      "Filter": {
        "Prefix": ""
      },
      "Expiration": {
        "Days": 30
      }
    }
  ]
}
EOF

mc ilm import mnemos/mnemos-logs < /tmp/logs-lifecycle.json
echo "  ✓ Lifecycle policy set for mnemos-logs"

# Enable versioning
echo ""
echo "Enabling versioning..."
mc version enable mnemos/mnemos-artifacts
mc version enable mnemos/mnemos-models
mc version enable mnemos/mnemos-backups
echo "  ✓ Versioning enabled"

# Set quotas
echo ""
echo "Setting quotas..."
mc quota set mnemos/mnemos-artifacts --size 500GB
mc quota set mnemos/mnemos-models --size 1TB
mc quota set mnemos/mnemos-backups --size 200GB
echo "  ✓ Quotas configured"

echo ""
echo "=================================================="
echo "MinIO initialization complete!"
echo "=================================================="
```

═══════════════════════════════════════════════════════════════════

## TLS/SSL Certificate Management

### Certificate Generation Script

Create `scripts/tls/generate-certs.sh`:

```bash
#!/bin/bash
# Generate TLS certificates for MNEMOS services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERTS_DIR="${CERTS_DIR:-$SCRIPT_DIR/../../config/certs}"

mkdir -p "$CERTS_DIR"

echo "=================================================="
echo "MNEMOS Certificate Generation"
echo "=================================================="
echo ""

# Generate CA key and certificate
if [ ! -f "$CERTS_DIR/ca.key" ]; then
    echo "Generating CA certificate..."
    
    # Generate CA private key
    openssl genrsa -out "$CERTS_DIR/ca.key" 4096
    
    # Generate CA certificate
    openssl req -new -x509 -days 3650 -key "$CERTS_DIR/ca.key" \
        -out "$CERTS_DIR/ca.crt" \
        -subj "/C=US/ST=CA/L=San Francisco/O=MNEMOS/OU=Infrastructure/CN=MNEMOS Root CA"
    
    echo "✓ CA certificate generated"
else
    echo "✓ CA certificate already exists"
fi

# Generate server certificates
generate_server_cert() {
    local service=$1
    local cn=$2
    local san=$3
    
    if [ ! -f "$CERTS_DIR/${service}.key" ]; then
        echo "Generating certificate for $service..."
        
        # Generate private key
        openssl genrsa -out "$CERTS_DIR/${service}.key" 2048
        
        # Generate CSR
        openssl req -new -key "$CERTS_DIR/${service}.key" \
            -out "$CERTS_DIR/${service}.csr" \
            -subj "/C=US/ST=CA/L=San Francisco/O=MNEMOS/OU=Services/CN=$cn"
        
        # Create SAN config
        cat > "$CERTS_DIR/${service}.ext" <<EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = $cn
DNS.2 = ${service}.mnemos.local
DNS.3 = ${service}.mnemos.svc
DNS.4 = localhost
IP.1 = 127.0.0.1
$san
EOF
        
        # Sign certificate with CA
        openssl x509 -req -in "$CERTS_DIR/${service}.csr" \
            -CA "$CERTS_DIR/ca.crt" -CAkey "$CERTS_DIR/ca.key" \
            -CAcreateserial -out "$CERTS_DIR/${service}.crt" \
            -days 365 -sha256 -extfile "$CERTS_DIR/${service}.ext"
        
        # Cleanup
        rm "$CERTS_DIR/${service}.csr" "$CERTS_DIR/${service}.ext"
        
        echo "✓ Certificate generated for $service"
    else
        echo "✓ Certificate for $service already exists"
    fi
}

# Generate certificates for each service
echo ""
echo "Generating service certificates..."
generate_server_cert "server" "mnemos.local" "DNS.5 = *.mnemos.local"
generate_server_cert "vault" "vault" ""
generate_server_cert "postgres" "postgres" ""
generate_server_cert "cortex" "cortex" ""
generate_server_cert "genome" "genome" ""
generate_server_cert "neuron" "neuron" ""

# Generate JWT keys (RSA key pair for JWT signing)
if [ ! -f "$CERTS_DIR/jwt-private.pem" ]; then
    echo ""
    echo "Generating JWT signing keys..."
    
    # Generate private key
    openssl genrsa -out "$CERTS_DIR/jwt-private.pem" 2048
    
    # Generate public key
    openssl rsa -in "$CERTS_DIR/jwt-private.pem" \
        -pubout -out "$CERTS_DIR/jwt-public.pem"
    
    echo "✓ JWT keys generated"
else
    echo "✓ JWT keys already exist"
fi

# Set permissions
chmod 600 "$CERTS_DIR"/*.key "$CERTS_DIR"/*.pem
chmod 644 "$CERTS_DIR"/*.crt

echo ""
echo "=================================================="
echo "Certificate generation complete!"
echo "=================================================="
echo ""
echo "Certificates location: $CERTS_DIR"
echo ""
echo "Generated certificates:"
ls -lh "$CERTS_DIR"
echo ""
```

Make executable:

```bash
chmod +x scripts/tls/generate-certs.sh
```

═══════════════════════════════════════════════════════════════════

## Security Hardening

### Docker Security Configuration

Create `config/security/docker-security.yaml`:

```yaml
# =============================================================================
# MNEMOS Docker Security Configuration
# =============================================================================

security_options:
  # Disable new privileges
  no_new_privileges: true
  
  # AppArmor profile
  apparmor: docker-default
  
  # Seccomp profile
  seccomp: /etc/docker/seccomp-profile.json
  
  # Read-only root filesystem where possible
  read_only: true
  
  # Drop all capabilities, add only required ones
  cap_drop:
    - ALL
  
  # Required capabilities per service
  capabilities:
    postgres:
      cap_add:
        - CHOWN
        - DAC_OVERRIDE
        - SETGID
        - SETUID
    
    redis:
      cap_add:
        - SETGID
        - SETUID
    
    vault:
      cap_add:
        - IPC_LOCK  # Required for mlock
    
    minio:
      cap_add:
        - CHOWN
        - SETGID
        - SETUID

# User namespaces
user_namespaces:
  enabled: true
  remap: "mnemos:mnemos"

# Resource limits
resource_limits:
  pids:
    limit: 100
  
  ulimits:
    nofile:
      soft: 65536
      hard: 65536
    nproc:
      soft: 4096
      hard: 4096

# Logging
logging:
  driver: json-file
  options:
    max-size: "10m"
    max-file: "3"
    compress: "true"
    labels: "service,component"
```

### Network Security Rules

Create `config/security/network-rules.yaml`:

```yaml
# =============================================================================
# MNEMOS Network Security Rules
# =============================================================================

network_policies:
  # Frontend network rules
  frontend:
    ingress:
      - from:
          - source: internet
        ports:
          - 80/tcp
          - 443/tcp
      - from:
          - source: backend
        ports:
          - 8080/tcp
          - 3000/tcp
    
    egress:
      - to:
          - destination: backend
        ports:
          - 8080/tcp
          - 9090/tcp
  
  # Backend network rules
  backend:
    ingress:
      - from:
          - source: frontend
        ports:
          - 8080/tcp
          - 9090/tcp
          - 8000/tcp
    
    egress:
      - to:
          - destination: data
        ports:
          - 5432/tcp
          - 6379/tcp
          - 8200/tcp
          - 9000/tcp
  
  # Data network rules
  data:
    ingress:
      - from:
          - source: backend
        ports:
          - 5432/tcp   # PostgreSQL
          - 6379/tcp   # Redis
          - 8200/tcp   # Vault
          - 9000/tcp   # MinIO
          - 9090/tcp   # Prometheus
          - 3100/tcp   # Loki
    
    egress:
      - to:
          - destination: internal
        ports:
          - all

# Firewall rules (UFW)
firewall_rules:
  - rule: allow
    port: 22
    proto: tcp
    comment: "SSH access"
  
  - rule: allow
    port: 80
    proto: tcp
    comment: "HTTP"
  
  - rule: allow
    port: 443
    proto: tcp
    comment: "HTTPS"
  
  - rule: deny
    port: 5432
    proto: tcp
    from: any
    comment: "Block external PostgreSQL"
  
  - rule: deny
    port: 6379
    proto: tcp
    from: any
    comment: "Block external Redis"
  
  - rule: deny
    port: 8200
    proto: tcp
    from: any
    comment: "Block external Vault"
```

### Security Hardening Script

Create `scripts/security/harden.sh`:

```bash
#!/bin/bash
# Security hardening script for MNEMOS deployment

set -e

echo "=================================================="
echo "MNEMOS Security Hardening"
echo "=================================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root or with sudo"
    exit 1
fi

# Update system
echo "Updating system..."
apt update && apt upgrade -y
echo "✓ System updated"
echo ""

# Install security tools
echo "Installing security tools..."
apt install -y \
    ufw \
    fail2ban \
    unattended-upgrades \
    aide \
    rkhunter \
    lynis
echo "✓ Security tools installed"
echo ""

# Configure UFW firewall
echo "Configuring firewall..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp   # SSH
ufw allow 80/tcp   # HTTP
ufw allow 443/tcp  # HTTPS
ufw --force enable
echo "✓ Firewall configured"
echo ""

# Configure fail2ban
echo "Configuring fail2ban..."
cat > /etc/fail2ban/jail.local <<EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5
destemail = admin@mnemos.local
sendername = Fail2Ban
action = %(action_mwl)s

[sshd]
enabled = true
port = ssh
logpath = /var/log/auth.log

[docker-auth]
enabled = true
filter = docker-auth
logpath = /var/log/docker-auth.log
maxretry = 3
EOF

systemctl enable fail2ban
systemctl restart fail2ban
echo "✓ Fail2ban configured"
echo ""

# Configure automatic security updates
echo "Configuring automatic updates..."
cat > /etc/apt/apt.conf.d/50unattended-upgrades <<EOF
Unattended-Upgrade::Allowed-Origins {
    "\${distro_id}:\${distro_codename}-security";
    "\${distro_id}ESMApps:\${distro_codename}-apps-security";
};
Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::MinimalSteps "true";
Unattended-Upgrade::Mail "admin@mnemos.local";
EOF

cat > /etc/apt/apt.conf.d/20auto-upgrades <<EOF
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Download-Upgradeable-Packages "1";
APT::Periodic::AutocleanInterval "7";
APT::Periodic::Unattended-Upgrade "1";
EOF

echo "✓ Automatic updates configured"
echo ""

# Harden SSH
echo "Hardening SSH..."
sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config
systemctl restart sshd
echo "✓ SSH hardened"
echo ""

# Set kernel parameters for security
echo "Setting kernel parameters..."
cat >> /etc/sysctl.conf <<EOF

# MNEMOS Security Parameters
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv4.conf.all.secure_redirects = 0
net.ipv4.conf.default.secure_redirects = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1
net.ipv4.tcp_syncookies = 1
kernel.dmesg_restrict = 1
kernel.kptr_restrict = 2
fs.suid_dumpable = 0
EOF

sysctl -p
echo "✓ Kernel parameters set"
echo ""

# Docker security
echo "Configuring Docker security..."
mkdir -p /etc/docker
cat > /etc/docker/daemon.json <<EOF
{
  "icc": false,
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "live-restore": true,
  "userland-proxy": false,
  "no-new-privileges": true,
  "seccomp-profile": "/etc/docker/seccomp-profile.json"
}
EOF

systemctl restart docker
echo "✓ Docker security configured"
echo ""

echo "=================================================="
echo "Security hardening complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Review firewall rules: sudo ufw status"
echo "  2. Check fail2ban status: sudo fail2ban-client status"
echo "  3. Run security audit: sudo lynis audit system"
echo ""
```

Make executable:

```bash
chmod +x scripts/security/harden.sh
```

═══════════════════════════════════════════════════════════════════

## Backup & Restore

### Comprehensive Backup Script

Create `scripts/backup/backup-all.sh`:

```bash
#!/bin/bash
# Comprehensive backup script for all MNEMOS services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_ROOT="${BACKUP_ROOT:-/backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_ROOT/$TIMESTAMP"
RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-30}

# Create backup directory
mkdir -p "$BACKUP_DIR"

echo "=================================================="
echo "MNEMOS Comprehensive Backup"
echo "Timestamp: $TIMESTAMP"
echo "Backup location: $BACKUP_DIR"
echo "=================================================="
echo ""

# Function to backup with progress
backup_service() {
    local service=$1
    local backup_func=$2
    
    echo "Backing up $service..."
    if $backup_func "$BACKUP_DIR"; then
        echo "✓ $service backup complete"
    else
        echo "✗ $service backup failed"
        return 1
    fi
    echo ""
}

# PostgreSQL backup
backup_postgres() {
    local dest=$1
    local pg_dir="$dest/postgres"
    mkdir -p "$pg_dir"
    
    for db in mnemos mnemos_cortex mnemos_genome mnemos_synkron mnemos_trace; do
        docker exec mnemos-postgres pg_dump -U mnemos -Fc "$db" > \
            "$pg_dir/${db}.dump"
    done
    
    docker exec mnemos-postgres pg_dumpall -U mnemos --globals-only > \
        "$pg_dir/globals.sql"
    
    return 0
}

# Redis backup
backup_redis() {
    local dest=$1
    local redis_dir="$dest/redis"
    mkdir -p "$redis_dir"
    
    docker exec mnemos-redis redis-cli --no-auth-warning -a "$REDIS_PASSWORD" BGSAVE
    sleep 2
    docker cp mnemos-redis:/data/dump.rdb "$redis_dir/"
    docker cp mnemos-redis:/data/appendonly.aof "$redis_dir/" 2>/dev/null || true
    
    return 0
}

# Vault backup
backup_vault() {
    local dest=$1
    local vault_dir="$dest/vault"
    mkdir -p "$vault_dir"
    
    # Backup Vault data
    docker cp mnemos-vault:/vault/data "$vault_dir/"
    
    # Backup keys (encrypted)
    if [ -f "/vault/data/.vault-keys.json" ]; then
        openssl enc -aes-256-cbc -salt -pbkdf2 \
            -in /vault/data/.vault-keys.json \
            -out "$vault_dir/vault-keys.enc" \
            -k "${BACKUP_ENCRYPTION_KEY:?Encryption key required}"
    fi
    
    return 0
}

# MinIO backup
backup_minio() {
    local dest=$1
    local minio_dir="$dest/minio"
    mkdir -p "$minio_dir"
    
    # Backup MinIO data
    docker exec mnemos-minio mc mirror --preserve \
        /data "$minio_dir/"
    
    return 0
}

# Configuration backup
backup_configs() {
    local dest=$1
    local config_dir="$dest/config"
    mkdir -p "$config_dir"
    
    cp -r "$SCRIPT_DIR/../../config" "$config_dir/"
    cp "$SCRIPT_DIR/../../docker-compose.yml" "$config_dir/"
    cp "$SCRIPT_DIR/../../.env" "$config_dir/.env.backup"
    
    return 0
}

# Perform backups
backup_service "PostgreSQL" backup_postgres
backup_service "Redis" backup_redis
backup_service "Vault" backup_vault
backup_service "MinIO" backup_minio
backup_service "Configurations" backup_configs

# Create backup manifest
echo "Creating backup manifest..."
cat > "$BACKUP_DIR/manifest.json" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "version": "1.0.0",
  "services": {
    "postgres": ["mnemos", "mnemos_cortex", "mnemos_genome", "mnemos_synkron", "mnemos_trace"],
    "redis": true,
    "vault": true,
    "minio": true,
    "config": true
  },
  "size_bytes": $(du -sb "$BACKUP_DIR" | cut -f1),
  "backup_dir": "$BACKUP_DIR"
}
EOF
echo "✓ Manifest created"
echo ""

# Compress backup
echo "Compressing backup..."
cd "$BACKUP_ROOT"
tar -czf "mnemos_backup_${TIMESTAMP}.tar.gz" "$TIMESTAMP"
rm -rf "$TIMESTAMP"
echo "✓ Backup compressed"
echo ""

# Calculate checksum
echo "Calculating checksum..."
sha256sum "mnemos_backup_${TIMESTAMP}.tar.gz" > \
    "mnemos_backup_${TIMESTAMP}.tar.gz.sha256"
echo "✓ Checksum created"
echo ""

# Clean old backups
echo "Cleaning old backups..."
find "$BACKUP_ROOT" -name "mnemos_backup_*.tar.gz" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_ROOT" -name "*.sha256" -mtime +$RETENTION_DAYS -delete
echo "✓ Old backups cleaned"
echo ""

BACKUP_SIZE=$(du -h "mnemos_backup_${TIMESTAMP}.tar.gz" | cut -f1)

echo "=================================================="
echo "Backup complete!"
echo "=================================================="
echo "Backup file: mnemos_backup_${TIMESTAMP}.tar.gz"
echo "Backup size: $BACKUP_SIZE"
echo "Location: $BACKUP_ROOT"
echo ""
```

### Restore Script

Create `scripts/backup/restore-all.sh`:

```bash
#!/bin/bash
# Restore script for MNEMOS backups

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <backup-file.tar.gz>"
    exit 1
fi

BACKUP_FILE="$1"
RESTORE_DIR="/tmp/mnemos_restore_$$"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo "=================================================="
echo "MNEMOS Restore"
echo "Backup file: $BACKUP_FILE"
echo "=================================================="
echo ""

# Verify checksum
if [ -f "${BACKUP_FILE}.sha256" ]; then
    echo "Verifying backup integrity..."
    if sha256sum -c "${BACKUP_FILE}.sha256"; then
        echo "✓ Backup integrity verified"
    else
        echo "✗ Backup integrity check failed!"
        exit 1
    fi
    echo ""
fi

# Extract backup
echo "Extracting backup..."
mkdir -p "$RESTORE_DIR"
tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR"
echo "✓ Backup extracted"
echo ""

# Get backup directory name
BACKUP_NAME=$(tar -tzf "$BACKUP_FILE" | head -1 | cut -f1 -d"/")
RESTORE_PATH="$RESTORE_DIR/$BACKUP_NAME"

# Confirm restore
echo "⚠️  WARNING: This will overwrite current data!"
read -p "Continue with restore? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Restore cancelled"
    rm -rf "$RESTORE_DIR"
    exit 0
fi

# Stop services
echo "Stopping services..."
docker compose down
echo "✓ Services stopped"
echo ""

# Restore PostgreSQL
echo "Restoring PostgreSQL..."
docker compose up -d postgres
sleep 10

for dump in "$RESTORE_PATH"/postgres/*.dump; do
    if [ -f "$dump" ]; then
        db=$(basename "$dump" .dump)
        echo "  Restoring $db..."
        docker exec -i mnemos-postgres pg_restore -U mnemos -d "$db" -c < "$dump"
    fi
done
echo "✓ PostgreSQL restored"
echo ""

# Restore Redis
echo "Restoring Redis..."
docker compose down redis
docker cp "$RESTORE_PATH/redis/dump.rdb" mnemos-redis:/data/
docker compose up -d redis
echo "✓ Redis restored"
echo ""

# Restore Vault
echo "Restoring Vault..."
docker compose down vault
docker cp "$RESTORE_PATH/vault/data" mnemos-vault:/vault/
docker compose up -d vault
echo "✓ Vault restored"
echo ""

# Restore MinIO
echo "Restoring MinIO..."
docker compose down minio
docker cp "$RESTORE_PATH/minio/." mnemos-minio:/data/
docker compose up -d minio
echo "✓ MinIO restored"
echo ""

# Cleanup
echo "Cleaning up..."
rm -rf "$RESTORE_DIR"
echo "✓ Cleanup complete"
echo ""

# Start all services
echo "Starting all services..."
docker compose up -d
echo "✓ All services started"
echo ""

echo "=================================================="
echo "Restore complete!"
echo "=================================================="
echo "Please verify all services are running correctly:"
echo "  make health"
echo ""
```

Make scripts executable:

```bash
chmod +x scripts/backup/backup-all.sh
chmod +x scripts/backup/restore-all.sh
```

═══════════════════════════════════════════════════════════════════

## Monitoring Configuration

### Prometheus Configuration

Create `config/prometheus/prometheus.yml`:

```yaml
# =============================================================================
# MNEMOS Prometheus Configuration
# =============================================================================

global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'mnemos'
    environment: 'production'

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

# Load alert rules
rule_files:
  - '/etc/prometheus/alerts/*.yml'

# Scrape configurations
scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  # PostgreSQL
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
  
  # Redis
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
  
  # Vault
  - job_name: 'vault'
    static_configs:
      - targets: ['vault:8200']
    scheme: https
    tls_config:
      insecure_skip_verify: true
  
  # MinIO
  - job_name: 'minio'
    metrics_path: /minio/v2/metrics/cluster
    static_configs:
      - targets: ['minio:9000']
  
  # CORTEX
  - job_name: 'cortex'
    static_configs:
      - targets: ['cortex:8080']
  
  # GENOME
  - job_name: 'genome'
    static_configs:
      - targets: ['genome:8080']
  
  # NEURON
  - job_name: 'neuron'
    static_configs:
      - targets: ['neuron:8000']
  
  # SYNKRON
  - job_name: 'synkron'
    static_configs:
      - targets: ['synkron:8080']
  
  # Docker metrics (cAdvisor)
  - job_name: 'docker'
    static_configs:
      - targets: ['cadvisor:8080']
```

### Alert Rules

Create `config/prometheus/alerts/mnemos-alerts.yml`:

```yaml
# =============================================================================
# MNEMOS Alert Rules
# =============================================================================

groups:
  - name: mnemos_infrastructure
    interval: 30s
    rules:
      # Service down alerts
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"
          description: "{{ $labels.instance }} of job {{ $labels.job }} has been down for more than 1 minute."
      
      # High memory usage
      - alert: HighMemoryUsage
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 90% for more than 5 minutes."
      
      # High CPU usage
      - alert: HighCPUUsage
        expr: 100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% for more than 5 minutes."
      
      # Disk space
      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100 < 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Disk space is running low"
          description: "Available disk space is below 10%."
  
  - name: mnemos_database
    interval: 30s
    rules:
      # PostgreSQL down
      - alert: PostgreSQLDown
        expr: pg_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL is down"
          description: "PostgreSQL database has been down for more than 1 minute."
      
      # High connection count
      - alert: PostgreSQLHighConnections
        expr: sum(pg_stat_activity_count) > 180
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High PostgreSQL connection count"
          description: "PostgreSQL has more than 180 active connections."
      
      # Redis down
      - alert: RedisDown
        expr: redis_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis is down"
          description: "Redis has been down for more than 1 minute."
  
  - name: mnemos_application
    interval: 30s
    rules:
      # High job queue depth
      - alert: HighJobQueueDepth
        expr: cortex_queue_depth > 1000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High job queue depth"
          description: "CORTEX job queue depth is above 1000 for more than 10 minutes."
      
      # Job failures
      - alert: HighJobFailureRate
        expr: rate(cortex_jobs_failed_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High job failure rate"
          description: "Job failure rate is above 10% for the last 5 minutes."
```

### Grafana Datasource Configuration

Create `config/grafana/datasources/datasources.yml`:

```yaml
# =============================================================================
# MNEMOS Grafana Datasources
# =============================================================================

apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    jsonData:
      timeInterval: "15s"
  
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    jsonData:
      maxLines: 1000
  
  - name: PostgreSQL
    type: postgres
    access: proxy
    url: postgres:5432
    database: mnemos
    user: mnemos
    secureJsonData:
      password: ${POSTGRES_PASSWORD}
    jsonData:
      sslmode: "require"
      postgresVersion: 1600
      timescaledb: false
```

═══════════════════════════════════════════════════════════════════

## Validation & Testing

### Infrastructure Validation Script

Create `scripts/validate/validate-infrastructure.sh`:

```bash
#!/bin/bash
# Comprehensive infrastructure validation script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

FAILURES=0

echo "=================================================="
echo "MNEMOS Infrastructure Validation"
echo "=================================================="
echo ""

# Test function
test_service() {
    local test_name=$1
    local test_cmd=$2
    
    echo -n "Testing $test_name... "
    if eval "$test_cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((FAILURES++))
        return 1
    fi
}

# PostgreSQL tests
echo "=== PostgreSQL Tests ==="
test_service "PostgreSQL connection" \
    "docker exec mnemos-postgres pg_isready -U mnemos"

test_service "CORTEX database exists" \
    "docker exec mnemos-postgres psql -U mnemos -lqt | grep -q mnemos_cortex"

test_service "GENOME database exists" \
    "docker exec mnemos-postgres psql -U mnemos -lqt | grep -q mnemos_genome"

test_service "SYNKRON database exists" \
    "docker exec mnemos-postgres psql -U mnemos -lqt | grep -q mnemos_synkron"

test_service "PostgreSQL extensions loaded" \
    "docker exec mnemos-postgres psql -U mnemos -d mnemos_cortex -c '\dx' | grep -q uuid-ossp"

echo ""

# Redis tests
echo "=== Redis Tests ==="
test_service "Redis ping" \
    "docker exec mnemos-redis redis-cli --no-auth-warning -a $REDIS_PASSWORD ping | grep -q PONG"

test_service "Redis persistence enabled" \
    "docker exec mnemos-redis redis-cli --no-auth-warning -a $REDIS_PASSWORD config get appendonly | grep -q yes"

echo ""

# Vault tests
echo "=== Vault Tests ==="
test_service "Vault health check" \
    "curl -sf http://localhost:8200/v1/sys/health"

test_service "Vault unsealed" \
    "vault status | grep -q 'Sealed.*false'"

echo ""

# MinIO tests
echo "=== MinIO Tests ==="
test_service "MinIO health check" \
    "curl -sf http://localhost:9000/minio/health/live"

test_service "MinIO buckets exist" \
    "mc ls mnemos/ | grep -q mnemos-artifacts"

echo ""

# Network tests
echo "=== Network Tests ==="
test_service "Frontend network exists" \
    "docker network inspect mnemos_frontend"

test_service "Backend network exists" \
    "docker network inspect mnemos_backend"

test_service "Data network exists" \
    "docker network inspect mnemos_data"

echo ""

# TLS tests
echo "=== TLS Tests ==="
test_service "CA certificate exists" \
    "[ -f config/certs/ca.crt ]"

test_service "Server certificate exists" \
    "[ -f config/certs/server.crt ]"

test_service "JWT keys exist" \
    "[ -f config/certs/jwt-private.pem ] && [ -f config/certs/jwt-public.pem ]"

echo ""

# Monitoring tests
echo "=== Monitoring Tests ==="
test_service "Prometheus accessible" \
    "curl -sf http://localhost:9090/-/healthy"

test_service "Loki accessible" \
    "curl -sf http://localhost:3100/ready"

test_service "Grafana accessible" \
    "curl -sf http://localhost:3000/api/health"

test_service "OTEL Collector accessible" \
    "curl -sf http://localhost:13133/"

echo ""

echo "=================================================="
if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}$FAILURES test(s) failed${NC}"
    exit 1
fi
```

Make executable:

```bash
chmod +x scripts/validate/validate-infrastructure.sh
```

═══════════════════════════════════════════════════════════════════

## Phase 2 Completion Checklist

- [ ] PostgreSQL performance configuration applied
- [ ] PostgreSQL database schemas created and validated
- [ ] PostgreSQL backup script created and tested
- [ ] Redis production configuration applied
- [ ] Redis persistence enabled and validated
- [ ] Vault configuration file created
- [ ] Vault initialization script created and executed
- [ ] Vault policies created for all services
- [ ] Vault secrets stored securely
- [ ] MinIO buckets created with lifecycle policies
- [ ] MinIO quotas configured
- [ ] TLS certificates generated for all services
- [ ] JWT signing keys generated
- [ ] Docker security options configured
- [ ] Network security rules implemented
- [ ] System security hardening applied
- [ ] UFW firewall configured
- [ ] Fail2ban configured
- [ ] Comprehensive backup script created and tested
- [ ] Restore script created and tested
- [ ] Backup encryption configured
- [ ] Prometheus configuration deployed
- [ ] Alert rules configured
- [ ] Grafana datasources configured
- [ ] Infrastructure validation script created
- [ ] All validation tests passing
- [ ] Documentation reviewed

═══════════════════════════════════════════════════════════════════

## Next Phase Preview

**Phase 3** will cover:
- GENOME service implementation (schema registry)
- CORTEX service implementation (orchestrator)
- ENGRAM service implementation (storage abstraction)
- Data models and Pydantic schemas
- API specifications
- Service-to-service communication
- Authentication and authorization

═══════════════════════════════════════════════════════════════════

*End of Phase 2*

**Continue to:** [MNEMOS_Phase_3.md](./MNEMOS_Phase_3.md)
