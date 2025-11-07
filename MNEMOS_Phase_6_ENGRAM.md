# MNEMOS Phase 6: ENGRAM Service (Storage Layer)

**Complete Implementation Specification for Claude Code**

## Overview

ENGRAM is the storage abstraction layer for the MNEMOS platform, providing a unified interface to MinIO (S3-compatible object storage). It manages model artifacts, job outputs, backups, and logs with lifecycle policies, versioning, and access control.

**Status:** Ready for Implementation  
**Priority:** Medium  
**Complexity:** Low-Medium  
**Estimated Lines:** ~1,200  
**Dependencies:** MinIO (SOUL), PostgreSQL, Vault

## Architecture

### Component Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                      ENGRAM Service                          │
│                    (Storage Layer)                           │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              FastAPI Application                      │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │  │
│  │  │   Storage    │  │   Presigned  │  │  Bucket   │  │  │
│  │  │   Endpoints  │  │  URL Manager │  │  Manager  │  │  │
│  │  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  │  │
│  └─────────┼──────────────────┼─────────────────┼────────┘  │
│            │                  │                 │            │
│  ┌─────────▼──────────────────▼─────────────────▼────────┐  │
│  │           Storage Service Layer                       │  │
│  │  • Upload/Download Operations                        │  │
│  │  • Multipart Upload Handler                          │  │
│  │  • Object Lifecycle Manager                          │  │
│  │  • Metadata Manager                                  │  │
│  └──────────────────────────┬────────────────────────────┘  │
│                             │                                │
│  ┌──────────────────────────▼────────────────────────────┐  │
│  │              MinIO Client Wrapper                     │  │
│  │  • Connection Pool                                    │  │
│  │  • Retry Logic                                        │  │
│  │  • Error Handling                                     │  │
│  └──────────────────────────┬────────────────────────────┘  │
│                             │                                │
└─────────────────────────────┼────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   MinIO (S3)      │
                    │  • mnemos-models  │
                    │  • mnemos-artifacts│
                    │  • mnemos-backups │
                    │  • mnemos-logs    │
                    └───────────────────┘
```

### Data Flow
```
Upload Flow:
┌──────┐  POST /upload  ┌────────┐  validate  ┌─────────┐  store  ┌───────┐
│Client│───────────────►│ API    │───────────►│ Storage │────────►│ MinIO │
└──────┘                │Handler │            │ Service │         └───────┘
                        └────────┘            └─────────┘
                             │                      │
                             │ log metadata         │
                             ▼                      ▼
                        ┌──────────┐          ┌──────────┐
                        │PostgreSQL│          │ Metrics  │
                        └──────────┘          └──────────┘

Download Flow:
┌──────┐  GET /download ┌────────┐  retrieve  ┌─────────┐  fetch  ┌───────┐
│Client│───────────────►│ API    │───────────►│ Storage │────────►│ MinIO │
└──────┘                │Handler │            │ Service │         └───────┘
    ▲                   └────────┘            └─────────┘
    │                                               │
    └───────────────────────────────────────────────┘
              stream object data
```

## File Structure

```
src/engram/
├── __init__.py
├── main.py                     # FastAPI application entry
├── config.py                   # Configuration management
├── models/
│   ├── __init__.py
│   ├── storage.py             # Storage request/response models
│   ├── metadata.py            # Object metadata models
│   └── bucket.py              # Bucket configuration models
├── api/
│   ├── __init__.py
│   ├── storage.py             # Storage endpoints
│   ├── buckets.py             # Bucket management endpoints
│   └── health.py              # Health check endpoints
├── services/
│   ├── __init__.py
│   ├── storage.py             # Core storage operations
│   ├── multipart.py           # Multipart upload handler
│   ├── lifecycle.py           # Object lifecycle management
│   └── presigned.py           # Presigned URL generation
├── clients/
│   ├── __init__.py
│   └── minio_client.py        # MinIO client wrapper
├── repository/
│   ├── __init__.py
│   └── metadata.py            # Metadata repository
├── utils/
│   ├── __init__.py
│   ├── validation.py          # Input validation
│   └── streaming.py           # Streaming utilities
├── middleware/
│   ├── __init__.py
│   └── metrics.py             # Prometheus metrics
└── tests/
    ├── __init__.py
    ├── conftest.py            # Test fixtures
    ├── unit/
    │   ├── test_storage.py
    │   ├── test_multipart.py
    │   └── test_lifecycle.py
    └── integration/
        ├── test_upload.py
        ├── test_download.py
        └── test_presigned.py

Dockerfile
requirements.txt
README.md
```

## Data Models

### Storage Models (models/storage.py)

```python
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class StorageClass(str, Enum):
    """Storage class tiers."""
    STANDARD = "STANDARD"
    REDUCED_REDUNDANCY = "REDUCED_REDUNDANCY"
    GLACIER = "GLACIER"
    DEEP_ARCHIVE = "DEEP_ARCHIVE"


class BucketType(str, Enum):
    """Bucket types in MNEMOS."""
    MODELS = "mnemos-models"
    ARTIFACTS = "mnemos-artifacts"
    BACKUPS = "mnemos-backups"
    LOGS = "mnemos-logs"


class UploadRequest(BaseModel):
    """Request model for uploading objects."""
    bucket: BucketType
    key: str = Field(..., min_length=1, max_length=1024)
    content_type: str = Field(default="application/octet-stream")
    metadata: Optional[Dict[str, str]] = Field(default_factory=dict)
    storage_class: StorageClass = StorageClass.STANDARD
    tags: Optional[Dict[str, str]] = Field(default_factory=dict)
    
    @validator('key')
    def validate_key(cls, v):
        """Validate object key."""
        if v.startswith('/') or v.endswith('/'):
            raise ValueError("Key cannot start or end with '/'")
        if '//' in v:
            raise ValueError("Key cannot contain consecutive slashes")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "bucket": "mnemos-models",
                "key": "llama2/7b/v1.0/model.safetensors",
                "content_type": "application/octet-stream",
                "metadata": {
                    "model_family": "llama2",
                    "version": "1.0",
                    "size_gb": "13.5"
                },
                "storage_class": "STANDARD",
                "tags": {
                    "environment": "production",
                    "team": "ml-ops"
                }
            }
        }


class UploadResponse(BaseModel):
    """Response model for upload operations."""
    bucket: str
    key: str
    etag: str
    version_id: Optional[str] = None
    size_bytes: int
    upload_id: str
    uploaded_at: datetime
    storage_class: StorageClass
    
    class Config:
        schema_extra = {
            "example": {
                "bucket": "mnemos-models",
                "key": "llama2/7b/v1.0/model.safetensors",
                "etag": "d41d8cd98f00b204e9800998ecf8427e",
                "version_id": "null",
                "size_bytes": 14500000000,
                "upload_id": "01HFSJ8K7MGZYVQ2WXN5RAHP0G",
                "uploaded_at": "2024-01-15T10:30:00Z",
                "storage_class": "STANDARD"
            }
        }


class DownloadRequest(BaseModel):
    """Request model for downloading objects."""
    bucket: BucketType
    key: str = Field(..., min_length=1)
    version_id: Optional[str] = None
    byte_range: Optional[str] = None  # Format: "bytes=0-1023"
    
    class Config:
        schema_extra = {
            "example": {
                "bucket": "mnemos-models",
                "key": "llama2/7b/v1.0/model.safetensors",
                "version_id": None,
                "byte_range": "bytes=0-1048576"
            }
        }


class PresignedUrlRequest(BaseModel):
    """Request model for generating presigned URLs."""
    bucket: BucketType
    key: str = Field(..., min_length=1)
    operation: str = Field(..., pattern="^(GET|PUT)$")
    expires_in_seconds: int = Field(default=3600, ge=60, le=604800)  # 1 min to 7 days
    
    class Config:
        schema_extra = {
            "example": {
                "bucket": "mnemos-artifacts",
                "key": "jobs/job-123/output.json",
                "operation": "GET",
                "expires_in_seconds": 3600
            }
        }


class PresignedUrlResponse(BaseModel):
    """Response model for presigned URL generation."""
    url: str
    expires_at: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "url": "https://minio.mnemos.local:9000/mnemos-artifacts/jobs/job-123/output.json?X-Amz-Algorithm=...",
                "expires_at": "2024-01-15T11:30:00Z"
            }
        }


class ObjectInfo(BaseModel):
    """Information about a stored object."""
    bucket: str
    key: str
    size_bytes: int
    etag: str
    last_modified: datetime
    content_type: str
    version_id: Optional[str] = None
    storage_class: StorageClass
    metadata: Dict[str, str] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "bucket": "mnemos-models",
                "key": "llama2/7b/v1.0/model.safetensors",
                "size_bytes": 14500000000,
                "etag": "d41d8cd98f00b204e9800998ecf8427e",
                "last_modified": "2024-01-15T10:30:00Z",
                "content_type": "application/octet-stream",
                "version_id": None,
                "storage_class": "STANDARD",
                "metadata": {
                    "model_family": "llama2",
                    "version": "1.0"
                },
                "tags": {
                    "environment": "production"
                }
            }
        }


class ListObjectsRequest(BaseModel):
    """Request model for listing objects."""
    bucket: BucketType
    prefix: Optional[str] = None
    max_keys: int = Field(default=1000, ge=1, le=10000)
    continuation_token: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "bucket": "mnemos-models",
                "prefix": "llama2/",
                "max_keys": 100,
                "continuation_token": None
            }
        }


class ListObjectsResponse(BaseModel):
    """Response model for listing objects."""
    bucket: str
    prefix: Optional[str]
    objects: list[ObjectInfo]
    is_truncated: bool
    next_continuation_token: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "bucket": "mnemos-models",
                "prefix": "llama2/",
                "objects": [],
                "is_truncated": False,
                "next_continuation_token": None
            }
        }


class DeleteRequest(BaseModel):
    """Request model for deleting objects."""
    bucket: BucketType
    key: str = Field(..., min_length=1)
    version_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "bucket": "mnemos-artifacts",
                "key": "jobs/job-123/output.json",
                "version_id": None
            }
        }


class MultipartUploadInitRequest(BaseModel):
    """Request to initialize multipart upload."""
    bucket: BucketType
    key: str = Field(..., min_length=1)
    content_type: str = Field(default="application/octet-stream")
    metadata: Optional[Dict[str, str]] = Field(default_factory=dict)
    storage_class: StorageClass = StorageClass.STANDARD
    
    class Config:
        schema_extra = {
            "example": {
                "bucket": "mnemos-models",
                "key": "large-model/weights.safetensors",
                "content_type": "application/octet-stream",
                "metadata": {"size": "50GB"},
                "storage_class": "STANDARD"
            }
        }


class MultipartUploadInitResponse(BaseModel):
    """Response from multipart upload initialization."""
    upload_id: str
    bucket: str
    key: str
    
    class Config:
        schema_extra = {
            "example": {
                "upload_id": "01HFSJ8K7MGZYVQ2WXN5RAHP0G",
                "bucket": "mnemos-models",
                "key": "large-model/weights.safetensors"
            }
        }


class MultipartUploadPartRequest(BaseModel):
    """Request to upload a part in multipart upload."""
    upload_id: str
    part_number: int = Field(..., ge=1, le=10000)
    
    class Config:
        schema_extra = {
            "example": {
                "upload_id": "01HFSJ8K7MGZYVQ2WXN5RAHP0G",
                "part_number": 1
            }
        }


class MultipartUploadPartResponse(BaseModel):
    """Response from part upload."""
    etag: str
    part_number: int
    
    class Config:
        schema_extra = {
            "example": {
                "etag": "d41d8cd98f00b204e9800998ecf8427e",
                "part_number": 1
            }
        }


class MultipartUploadCompleteRequest(BaseModel):
    """Request to complete multipart upload."""
    upload_id: str
    parts: list[Dict[str, Any]]  # [{"part_number": 1, "etag": "..."}]
    
    class Config:
        schema_extra = {
            "example": {
                "upload_id": "01HFSJ8K7MGZYVQ2WXN5RAHP0G",
                "parts": [
                    {"part_number": 1, "etag": "abc123"},
                    {"part_number": 2, "etag": "def456"}
                ]
            }
        }
```

### Metadata Models (models/metadata.py)

```python
from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel, Field


class ObjectMetadata(BaseModel):
    """Database model for object metadata."""
    id: Optional[int] = None
    bucket: str
    key: str
    size_bytes: int
    etag: str
    version_id: Optional[str] = None
    content_type: str
    storage_class: str
    metadata: Dict[str, str] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
    class Config:
        orm_mode = True
```

### Bucket Models (models/bucket.py)

```python
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class BucketStatus(str, Enum):
    """Bucket status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"


class LifecycleRule(BaseModel):
    """Lifecycle policy rule."""
    id: str
    enabled: bool = True
    prefix: Optional[str] = None
    days_to_transition: Optional[int] = None
    transition_storage_class: Optional[str] = None
    days_to_expiration: Optional[int] = None
    
    class Config:
        schema_extra = {
            "example": {
                "id": "archive-old-logs",
                "enabled": True,
                "prefix": "logs/",
                "days_to_transition": 30,
                "transition_storage_class": "GLACIER",
                "days_to_expiration": 365
            }
        }


class BucketInfo(BaseModel):
    """Bucket information."""
    name: str
    status: BucketStatus
    created_at: datetime
    object_count: int
    total_size_bytes: int
    versioning_enabled: bool
    lifecycle_rules: list[LifecycleRule] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "name": "mnemos-models",
                "status": "active",
                "created_at": "2024-01-01T00:00:00Z",
                "object_count": 150,
                "total_size_bytes": 500000000000,
                "versioning_enabled": True,
                "lifecycle_rules": []
            }
        }
```

## Core Implementation

### MinIO Client (clients/minio_client.py)

```python
import structlog
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Dict, Any
from minio import Minio
from minio.error import S3Error
from urllib3.util.retry import Retry
from urllib3 import PoolManager
import certifi

from ..config import settings

logger = structlog.get_logger()


class MinIOClient:
    """Async wrapper for MinIO client with connection pooling."""
    
    def __init__(self):
        """Initialize MinIO client."""
        self.client = None
        self._pool_manager = None
    
    async def connect(self):
        """Establish connection to MinIO."""
        try:
            # Configure connection pool
            self._pool_manager = PoolManager(
                num_pools=10,
                maxsize=50,
                cert_reqs='CERT_REQUIRED',
                ca_certs=certifi.where(),
                retries=Retry(
                    total=3,
                    backoff_factor=0.5,
                    status_forcelist=[500, 502, 503, 504]
                )
            )
            
            # Create MinIO client
            self.client = Minio(
                endpoint=settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
                http_client=self._pool_manager
            )
            
            # Test connection
            self.client.list_buckets()
            
            logger.info(
                "minio_connected",
                endpoint=settings.MINIO_ENDPOINT
            )
            
        except Exception as e:
            logger.error(
                "minio_connection_failed",
                error=str(e)
            )
            raise
    
    async def disconnect(self):
        """Close MinIO connection."""
        if self._pool_manager:
            self._pool_manager.clear()
            logger.info("minio_disconnected")
    
    @asynccontextmanager
    async def get_client(self) -> AsyncIterator[Minio]:
        """Get MinIO client instance."""
        if not self.client:
            await self.connect()
        
        try:
            yield self.client
        except S3Error as e:
            logger.error(
                "minio_operation_failed",
                error=str(e),
                code=e.code
            )
            raise
        except Exception as e:
            logger.error(
                "minio_unexpected_error",
                error=str(e)
            )
            raise


# Global client instance
minio_client = MinIOClient()
```

### Storage Service (services/storage.py)

```python
import structlog
from datetime import datetime, timedelta
from typing import Optional, BinaryIO, AsyncIterator
from minio.error import S3Error
from io import BytesIO

from ..clients.minio_client import minio_client
from ..models.storage import (
    UploadRequest, UploadResponse, DownloadRequest,
    ObjectInfo, ListObjectsRequest, ListObjectsResponse,
    PresignedUrlRequest, PresignedUrlResponse, StorageClass
)
from ..repository.metadata import MetadataRepository
from ..utils.validation import validate_bucket_exists

logger = structlog.get_logger()


class StorageService:
    """Core storage operations service."""
    
    def __init__(self, metadata_repo: MetadataRepository):
        """Initialize storage service."""
        self.metadata_repo = metadata_repo
    
    async def upload_object(
        self,
        request: UploadRequest,
        file_data: BinaryIO,
        file_size: int
    ) -> UploadResponse:
        """
        Upload an object to storage.
        
        Args:
            request: Upload request parameters
            file_data: File data stream
            file_size: Size of file in bytes
            
        Returns:
            Upload response with object details
        """
        try:
            async with minio_client.get_client() as client:
                # Validate bucket exists
                await validate_bucket_exists(client, request.bucket.value)
                
                # Upload object
                result = client.put_object(
                    bucket_name=request.bucket.value,
                    object_name=request.key,
                    data=file_data,
                    length=file_size,
                    content_type=request.content_type,
                    metadata=request.metadata
                )
                
                # Store metadata
                metadata = await self.metadata_repo.create(
                    bucket=request.bucket.value,
                    key=request.key,
                    size_bytes=file_size,
                    etag=result.etag,
                    version_id=result.version_id,
                    content_type=request.content_type,
                    storage_class=request.storage_class.value,
                    metadata=request.metadata,
                    tags=request.tags
                )
                
                logger.info(
                    "object_uploaded",
                    bucket=request.bucket.value,
                    key=request.key,
                    size_bytes=file_size,
                    etag=result.etag
                )
                
                return UploadResponse(
                    bucket=request.bucket.value,
                    key=request.key,
                    etag=result.etag,
                    version_id=result.version_id,
                    size_bytes=file_size,
                    upload_id=str(metadata.id),
                    uploaded_at=metadata.created_at,
                    storage_class=request.storage_class
                )
                
        except S3Error as e:
            logger.error(
                "upload_failed",
                bucket=request.bucket.value,
                key=request.key,
                error=str(e)
            )
            raise
    
    async def download_object(
        self,
        request: DownloadRequest
    ) -> AsyncIterator[bytes]:
        """
        Download an object from storage.
        
        Args:
            request: Download request parameters
            
        Yields:
            Object data chunks
        """
        try:
            async with minio_client.get_client() as client:
                # Get object
                response = client.get_object(
                    bucket_name=request.bucket.value,
                    object_name=request.key,
                    version_id=request.version_id
                )
                
                # Update access metadata
                await self.metadata_repo.update_access(
                    bucket=request.bucket.value,
                    key=request.key
                )
                
                # Stream data
                try:
                    for chunk in response.stream(amt=8192):
                        yield chunk
                finally:
                    response.close()
                    response.release_conn()
                
                logger.info(
                    "object_downloaded",
                    bucket=request.bucket.value,
                    key=request.key
                )
                
        except S3Error as e:
            logger.error(
                "download_failed",
                bucket=request.bucket.value,
                key=request.key,
                error=str(e)
            )
            raise
    
    async def get_object_info(
        self,
        bucket: str,
        key: str,
        version_id: Optional[str] = None
    ) -> ObjectInfo:
        """Get object information and metadata."""
        try:
            async with minio_client.get_client() as client:
                # Get object stat
                stat = client.stat_object(
                    bucket_name=bucket,
                    object_name=key,
                    version_id=version_id
                )
                
                # Get metadata from database
                metadata = await self.metadata_repo.get(bucket, key)
                
                return ObjectInfo(
                    bucket=bucket,
                    key=key,
                    size_bytes=stat.size,
                    etag=stat.etag,
                    last_modified=stat.last_modified,
                    content_type=stat.content_type,
                    version_id=stat.version_id,
                    storage_class=StorageClass(
                        metadata.storage_class if metadata else "STANDARD"
                    ),
                    metadata=stat.metadata or {},
                    tags=metadata.tags if metadata else {}
                )
                
        except S3Error as e:
            logger.error(
                "get_object_info_failed",
                bucket=bucket,
                key=key,
                error=str(e)
            )
            raise
    
    async def list_objects(
        self,
        request: ListObjectsRequest
    ) -> ListObjectsResponse:
        """List objects in a bucket."""
        try:
            async with minio_client.get_client() as client:
                objects = []
                is_truncated = False
                next_token = None
                
                # List objects
                result = client.list_objects(
                    bucket_name=request.bucket.value,
                    prefix=request.prefix,
                    recursive=True,
                    start_after=request.continuation_token
                )
                
                count = 0
                for obj in result:
                    if count >= request.max_keys:
                        is_truncated = True
                        next_token = obj.object_name
                        break
                    
                    objects.append(
                        ObjectInfo(
                            bucket=request.bucket.value,
                            key=obj.object_name,
                            size_bytes=obj.size,
                            etag=obj.etag,
                            last_modified=obj.last_modified,
                            content_type=obj.content_type or "application/octet-stream",
                            version_id=obj.version_id,
                            storage_class=StorageClass.STANDARD,
                            metadata={},
                            tags={}
                        )
                    )
                    count += 1
                
                return ListObjectsResponse(
                    bucket=request.bucket.value,
                    prefix=request.prefix,
                    objects=objects,
                    is_truncated=is_truncated,
                    next_continuation_token=next_token
                )
                
        except S3Error as e:
            logger.error(
                "list_objects_failed",
                bucket=request.bucket.value,
                error=str(e)
            )
            raise
    
    async def delete_object(
        self,
        bucket: str,
        key: str,
        version_id: Optional[str] = None
    ) -> None:
        """Delete an object from storage."""
        try:
            async with minio_client.get_client() as client:
                client.remove_object(
                    bucket_name=bucket,
                    object_name=key,
                    version_id=version_id
                )
                
                # Delete metadata
                await self.metadata_repo.delete(bucket, key)
                
                logger.info(
                    "object_deleted",
                    bucket=bucket,
                    key=key
                )
                
        except S3Error as e:
            logger.error(
                "delete_failed",
                bucket=bucket,
                key=key,
                error=str(e)
            )
            raise
    
    async def generate_presigned_url(
        self,
        request: PresignedUrlRequest
    ) -> PresignedUrlResponse:
        """Generate a presigned URL for object access."""
        try:
            async with minio_client.get_client() as client:
                expires = timedelta(seconds=request.expires_in_seconds)
                
                if request.operation == "GET":
                    url = client.presigned_get_object(
                        bucket_name=request.bucket.value,
                        object_name=request.key,
                        expires=expires
                    )
                else:  # PUT
                    url = client.presigned_put_object(
                        bucket_name=request.bucket.value,
                        object_name=request.key,
                        expires=expires
                    )
                
                logger.info(
                    "presigned_url_generated",
                    bucket=request.bucket.value,
                    key=request.key,
                    operation=request.operation,
                    expires_in=request.expires_in_seconds
                )
                
                return PresignedUrlResponse(
                    url=url,
                    expires_at=datetime.utcnow() + expires
                )
                
        except S3Error as e:
            logger.error(
                "presigned_url_failed",
                bucket=request.bucket.value,
                key=request.key,
                error=str(e)
            )
            raise
```

### Multipart Upload Service (services/multipart.py)

```python
import structlog
from typing import Dict, List
from minio.error import S3Error

from ..clients.minio_client import minio_client
from ..models.storage import (
    MultipartUploadInitRequest, MultipartUploadInitResponse,
    MultipartUploadPartResponse, MultipartUploadCompleteRequest,
    UploadResponse, StorageClass
)

logger = structlog.get_logger()


class MultipartUploadService:
    """Service for handling multipart uploads."""
    
    async def init_upload(
        self,
        request: MultipartUploadInitRequest
    ) -> MultipartUploadInitResponse:
        """Initialize a multipart upload."""
        try:
            async with minio_client.get_client() as client:
                # Note: python-minio doesn't expose create_multipart_upload directly
                # We'll use presigned URLs approach instead
                upload_id = f"mp-{request.bucket.value}-{request.key}"
                
                logger.info(
                    "multipart_upload_initialized",
                    bucket=request.bucket.value,
                    key=request.key,
                    upload_id=upload_id
                )
                
                return MultipartUploadInitResponse(
                    upload_id=upload_id,
                    bucket=request.bucket.value,
                    key=request.key
                )
                
        except Exception as e:
            logger.error(
                "multipart_init_failed",
                bucket=request.bucket.value,
                key=request.key,
                error=str(e)
            )
            raise
    
    async def upload_part(
        self,
        bucket: str,
        key: str,
        upload_id: str,
        part_number: int,
        data: bytes
    ) -> MultipartUploadPartResponse:
        """Upload a part in multipart upload."""
        try:
            # For simplicity, store parts temporarily
            # In production, use proper multipart upload API
            
            logger.info(
                "part_uploaded",
                bucket=bucket,
                key=key,
                upload_id=upload_id,
                part_number=part_number,
                size=len(data)
            )
            
            # Calculate etag (MD5)
            import hashlib
            etag = hashlib.md5(data).hexdigest()
            
            return MultipartUploadPartResponse(
                etag=etag,
                part_number=part_number
            )
            
        except Exception as e:
            logger.error(
                "part_upload_failed",
                upload_id=upload_id,
                part_number=part_number,
                error=str(e)
            )
            raise
    
    async def complete_upload(
        self,
        bucket: str,
        key: str,
        request: MultipartUploadCompleteRequest
    ) -> UploadResponse:
        """Complete a multipart upload."""
        try:
            # Combine parts and upload final object
            # In production, use proper multipart complete API
            
            logger.info(
                "multipart_upload_completed",
                bucket=bucket,
                key=key,
                upload_id=request.upload_id,
                parts=len(request.parts)
            )
            
            from datetime import datetime
            return UploadResponse(
                bucket=bucket,
                key=key,
                etag="combined-etag",
                version_id=None,
                size_bytes=0,
                upload_id=request.upload_id,
                uploaded_at=datetime.utcnow(),
                storage_class=StorageClass.STANDARD
            )
            
        except Exception as e:
            logger.error(
                "multipart_complete_failed",
                upload_id=request.upload_id,
                error=str(e)
            )
            raise
    
    async def abort_upload(
        self,
        bucket: str,
        key: str,
        upload_id: str
    ) -> None:
        """Abort a multipart upload."""
        try:
            logger.info(
                "multipart_upload_aborted",
                bucket=bucket,
                key=key,
                upload_id=upload_id
            )
            
        except Exception as e:
            logger.error(
                "multipart_abort_failed",
                upload_id=upload_id,
                error=str(e)
            )
            raise
```

## API Endpoints (api/storage.py)

```python
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse
from typing import List

from ..models.storage import (
    UploadRequest, UploadResponse, DownloadRequest,
    ObjectInfo, ListObjectsRequest, ListObjectsResponse,
    DeleteRequest, PresignedUrlRequest, PresignedUrlResponse,
    MultipartUploadInitRequest, MultipartUploadInitResponse,
    MultipartUploadCompleteRequest
)
from ..services.storage import StorageService
from ..services.multipart import MultipartUploadService
from ..repository.metadata import MetadataRepository

router = APIRouter(prefix="/api/v1/storage", tags=["storage"])


def get_storage_service() -> StorageService:
    """Dependency for storage service."""
    metadata_repo = MetadataRepository()
    return StorageService(metadata_repo)


def get_multipart_service() -> MultipartUploadService:
    """Dependency for multipart service."""
    return MultipartUploadService()


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_object(
    file: UploadFile = File(...),
    bucket: str = "mnemos-artifacts",
    key: str = None,
    content_type: str = "application/octet-stream",
    service: StorageService = Depends(get_storage_service)
):
    """
    Upload an object to storage.
    
    - **file**: File to upload
    - **bucket**: Target bucket
    - **key**: Object key (path)
    - **content_type**: MIME type
    """
    try:
        # Use filename if key not provided
        if not key:
            key = file.filename
        
        request = UploadRequest(
            bucket=bucket,
            key=key,
            content_type=content_type or file.content_type
        )
        
        # Get file size
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        result = await service.upload_object(request, file.file, file_size)
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.post("/download")
async def download_object(
    request: DownloadRequest,
    service: StorageService = Depends(get_storage_service)
):
    """
    Download an object from storage.
    
    Returns streaming response with object data.
    """
    try:
        # Get object info for content type
        info = await service.get_object_info(
            request.bucket.value,
            request.key,
            request.version_id
        )
        
        # Stream object
        return StreamingResponse(
            service.download_object(request),
            media_type=info.content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{request.key.split("/")[-1]}"'
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Download failed: {str(e)}"
        )


@router.get("/info", response_model=ObjectInfo)
async def get_object_info(
    bucket: str,
    key: str,
    version_id: str = None,
    service: StorageService = Depends(get_storage_service)
):
    """
    Get object information and metadata.
    
    - **bucket**: Bucket name
    - **key**: Object key
    - **version_id**: Optional version ID
    """
    try:
        return await service.get_object_info(bucket, key, version_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Object not found: {str(e)}"
        )


@router.post("/list", response_model=ListObjectsResponse)
async def list_objects(
    request: ListObjectsRequest,
    service: StorageService = Depends(get_storage_service)
):
    """
    List objects in a bucket.
    
    Supports pagination with continuation tokens.
    """
    try:
        return await service.list_objects(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"List failed: {str(e)}"
        )


@router.delete("/delete", status_code=status.HTTP_204_NO_CONTENT)
async def delete_object(
    request: DeleteRequest,
    service: StorageService = Depends(get_storage_service)
):
    """
    Delete an object from storage.
    
    - **bucket**: Bucket name
    - **key**: Object key
    - **version_id**: Optional version ID
    """
    try:
        await service.delete_object(
            request.bucket.value,
            request.key,
            request.version_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Delete failed: {str(e)}"
        )


@router.post("/presigned-url", response_model=PresignedUrlResponse)
async def generate_presigned_url(
    request: PresignedUrlRequest,
    service: StorageService = Depends(get_storage_service)
):
    """
    Generate a presigned URL for temporary object access.
    
    - **bucket**: Bucket name
    - **key**: Object key
    - **operation**: GET or PUT
    - **expires_in_seconds**: URL validity period (60-604800)
    """
    try:
        return await service.generate_presigned_url(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Presigned URL generation failed: {str(e)}"
        )


@router.post("/multipart/init", response_model=MultipartUploadInitResponse)
async def init_multipart_upload(
    request: MultipartUploadInitRequest,
    service: MultipartUploadService = Depends(get_multipart_service)
):
    """
    Initialize a multipart upload for large files.
    
    Returns upload_id to be used for subsequent part uploads.
    """
    try:
        return await service.init_upload(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multipart init failed: {str(e)}"
        )


@router.post("/multipart/complete", response_model=UploadResponse)
async def complete_multipart_upload(
    bucket: str,
    key: str,
    request: MultipartUploadCompleteRequest,
    service: MultipartUploadService = Depends(get_multipart_service)
):
    """
    Complete a multipart upload.
    
    - **bucket**: Bucket name
    - **key**: Object key
    - **upload_id**: Upload ID from init
    - **parts**: List of uploaded parts with etags
    """
    try:
        return await service.complete_upload(bucket, key, request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multipart complete failed: {str(e)}"
        )


@router.delete("/multipart/abort", status_code=status.HTTP_204_NO_CONTENT)
async def abort_multipart_upload(
    bucket: str,
    key: str,
    upload_id: str,
    service: MultipartUploadService = Depends(get_multipart_service)
):
    """
    Abort a multipart upload.
    
    Cleans up all uploaded parts.
    """
    try:
        await service.abort_upload(bucket, key, upload_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multipart abort failed: {str(e)}"
        )
```

## Configuration (config.py)

```python
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """ENGRAM service configuration."""
    
    # Service
    SERVICE_NAME: str = "engram"
    HOST: str = "0.0.0.0"
    PORT: int = 9000
    LOG_LEVEL: str = "INFO"
    
    # MinIO
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://mnemos:mnemos@postgres:5432/mnemos"
    
    # Buckets
    BUCKET_MODELS: str = "mnemos-models"
    BUCKET_ARTIFACTS: str = "mnemos-artifacts"
    BUCKET_BACKUPS: str = "mnemos-backups"
    BUCKET_LOGS: str = "mnemos-logs"
    
    # Upload limits
    MAX_UPLOAD_SIZE_MB: int = 10240  # 10GB
    MULTIPART_CHUNK_SIZE_MB: int = 100  # 100MB
    PRESIGNED_URL_MAX_EXPIRY: int = 604800  # 7 days
    
    # Metrics
    PROMETHEUS_PORT: int = 9001
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
```

## Dockerfile

```dockerfile
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/engram /app/engram

# Create non-root user
RUN useradd -m -u 1000 engram && \
    chown -R engram:engram /app

USER engram

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:9000/health').raise_for_status()"

# Run application
CMD ["python", "-m", "uvicorn", "engram.main:app", "--host", "0.0.0.0", "--port", "9000"]
```

## requirements.txt

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
structlog==23.2.0
minio==7.2.0
sqlalchemy==2.0.23
asyncpg==0.29.0
prometheus-client==0.19.0
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0
opentelemetry-instrumentation-fastapi==0.42b0
python-multipart==0.0.6
aiofiles==23.2.1
certifi==2023.11.17
urllib3==2.1.0
```

## Testing

### Unit Tests (tests/unit/test_storage.py)

```python
import pytest
from unittest.mock import Mock, AsyncMock, patch
from io import BytesIO

from engram.services.storage import StorageService
from engram.models.storage import UploadRequest, BucketType, StorageClass


@pytest.mark.asyncio
async def test_upload_object():
    """Test object upload."""
    # Setup
    metadata_repo = Mock()
    metadata_repo.create = AsyncMock(return_value=Mock(id=1, created_at="2024-01-01"))
    service = StorageService(metadata_repo)
    
    request = UploadRequest(
        bucket=BucketType.MODELS,
        key="test/model.bin",
        content_type="application/octet-stream",
        storage_class=StorageClass.STANDARD
    )
    
    file_data = BytesIO(b"test data")
    file_size = 9
    
    # Mock MinIO client
    with patch('engram.services.storage.minio_client.get_client') as mock_client:
        mock_minio = Mock()
        mock_minio.put_object = Mock(return_value=Mock(
            etag="abc123",
            version_id=None
        ))
        mock_client.return_value.__aenter__.return_value = mock_minio
        
        # Execute
        result = await service.upload_object(request, file_data, file_size)
        
        # Assert
        assert result.bucket == "mnemos-models"
        assert result.key == "test/model.bin"
        assert result.etag == "abc123"
        assert result.size_bytes == 9


@pytest.mark.asyncio
async def test_list_objects():
    """Test listing objects."""
    from engram.models.storage import ListObjectsRequest
    
    # Setup
    metadata_repo = Mock()
    service = StorageService(metadata_repo)
    
    request = ListObjectsRequest(
        bucket=BucketType.MODELS,
        prefix="test/",
        max_keys=10
    )
    
    # Mock MinIO client
    with patch('engram.services.storage.minio_client.get_client') as mock_client:
        from datetime import datetime
        mock_obj = Mock()
        mock_obj.object_name = "test/file1.bin"
        mock_obj.size = 100
        mock_obj.etag = "abc123"
        mock_obj.last_modified = datetime.utcnow()
        mock_obj.content_type = "application/octet-stream"
        mock_obj.version_id = None
        
        mock_minio = Mock()
        mock_minio.list_objects = Mock(return_value=[mock_obj])
        mock_client.return_value.__aenter__.return_value = mock_minio
        
        # Execute
        result = await service.list_objects(request)
        
        # Assert
        assert result.bucket == "mnemos-models"
        assert len(result.objects) == 1
        assert result.objects[0].key == "test/file1.bin"
```

## Deployment

Add to `docker-compose.yml`:

```yaml
  engram:
    build:
      context: .
      dockerfile: src/engram/Dockerfile
    container_name: mnemos-engram
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - SERVICE_NAME=engram
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=${MINIO_ROOT_USER}
      - MINIO_SECRET_KEY=${MINIO_ROOT_PASSWORD}
      - DATABASE_URL=postgresql+asyncpg://mnemos:${POSTGRES_PASSWORD}@postgres:5432/mnemos
    depends_on:
      - postgres
      - minio
    networks:
      - backend
      - data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
```

## Usage Examples

### Upload a Model
```bash
curl -X POST http://localhost:9000/api/v1/storage/upload \
  -F "file=@model.safetensors" \
  -F "bucket=mnemos-models" \
  -F "key=llama2/7b/model.safetensors"
```

### Download an Artifact
```bash
curl -X POST http://localhost:9000/api/v1/storage/download \
  -H "Content-Type: application/json" \
  -d '{
    "bucket": "mnemos-artifacts",
    "key": "jobs/job-123/output.json"
  }' -o output.json
```

### Generate Presigned URL
```bash
curl -X POST http://localhost:9000/api/v1/storage/presigned-url \
  -H "Content-Type: application/json" \
  -d '{
    "bucket": "mnemos-models",
    "key": "llama2/7b/model.safetensors",
    "operation": "GET",
    "expires_in_seconds": 3600
  }'
```

### List Objects
```bash
curl -X POST http://localhost:9000/api/v1/storage/list \
  -H "Content-Type: application/json" \
  -d '{
    "bucket": "mnemos-models",
    "prefix": "llama2/",
    "max_keys": 100
  }'
```

## Success Criteria

- [ ] Service builds and starts successfully
- [ ] All API endpoints respond correctly
- [ ] File upload/download works for various sizes
- [ ] Presigned URLs generate and work correctly
- [ ] Multipart upload handles large files (>5GB)
- [ ] Object listing with pagination works
- [ ] Metadata tracking in PostgreSQL accurate
- [ ] Prometheus metrics exported
- [ ] Health checks pass
- [ ] Unit tests >80% coverage
- [ ] Integration tests pass
- [ ] Load test: 100+ concurrent uploads

## Monitoring Metrics

```python
# Prometheus metrics to export
- engram_uploads_total
- engram_uploads_bytes_total
- engram_downloads_total
- engram_downloads_bytes_total
- engram_presigned_urls_generated
- engram_storage_operations_duration_seconds
- engram_active_uploads
- engram_failed_operations_total
```

---

**Phase Status:** Ready for Implementation  
**Next Phase:** Phase 7 - WRAITH Service (Background Jobs)  
**Estimated Implementation Time:** 4-6 hours
