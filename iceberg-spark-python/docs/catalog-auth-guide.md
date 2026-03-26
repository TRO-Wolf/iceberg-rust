# Catalog Authentication Guide

This guide covers how to configure `iceberg_spark` to connect to different Iceberg catalog backends with authentication.

## Multi-Catalog Setup

You can configure multiple catalogs in a single session:

```python
from iceberg_spark import IcebergSession

session = (
    IcebergSession.builder()
    .catalog("rest", name="prod", uri="http://rest-catalog:8181")
    .catalog("sql", name="local", uri="sqlite:///catalog.db", warehouse="/data")
    .catalog("glue", name="aws", warehouse="s3://my-bucket/warehouse")
    .defaultCatalog("prod")
    .build()
)

# Switch catalogs
session.sql("USE CATALOG local")
session.sql("SHOW CATALOGS")

# Python API
session.catalog.currentCatalog()       # "local"
session.catalog.setCurrentCatalog("prod")
session.catalog.listCatalogs()         # ["prod", "local", "aws"]
```

## REST Catalog

### No authentication (open catalog)

```python
session = IcebergSession.builder().catalog(
    "rest", name="my_rest",
    uri="http://rest-catalog:8181/",
).build()
```

### Bearer token

```python
session = IcebergSession.builder().catalog(
    "rest", name="my_rest",
    uri="http://rest-catalog:8181/",
    token="eyJhbGciOiJSUzI1NiIs...",
    warehouse="production",
).build()
```

### OAuth2 client credentials

```python
session = IcebergSession.builder().catalog(
    "rest", name="my_rest",
    uri="http://rest-catalog:8181/",
    credential="my_client_id:my_client_secret",
    warehouse="production",
    **{"oauth2-server-uri": "https://auth.example.com/oauth/token"},
).build()
```

### Custom HTTP headers

```python
session = IcebergSession.builder().catalog(
    "rest", name="my_rest",
    uri="http://rest-catalog:8181/",
    **{
        "header.X-Api-Key": "abc123",
        "header.X-Team": "analytics",
    },
).build()
```

## AWS Glue Catalog

### Default credentials (environment / IAM role)

```bash
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
export AWS_REGION=us-west-2
```

```python
session = IcebergSession.builder().catalog(
    "glue", name="my_glue",
    warehouse="s3://my-bucket/warehouse",
).build()
```

### Explicit credentials

```python
session = IcebergSession.builder().catalog(
    "glue", name="my_glue",
    warehouse="s3://my-bucket/warehouse",
    **{
        "glue.access-key-id": "AKIAIOSFODNN7EXAMPLE",
        "glue.secret-access-key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "glue.region": "us-west-2",
    },
).build()
```

### Local testing with moto mock

```python
session = IcebergSession.builder().catalog(
    "glue", name="local_glue",
    warehouse="s3://warehouse/hive",
    **{
        "glue.endpoint": "http://localhost:5000",
        "glue.access-key-id": "testing",
        "glue.secret-access-key": "testing",
        "glue.region": "us-east-1",
        "s3.endpoint": "http://localhost:9000",
        "s3.access-key-id": "admin",
        "s3.secret-access-key": "password",
        "s3.region": "us-east-1",
    },
).build()
```

## S3-Compatible Storage (MinIO)

When using any catalog type with MinIO or S3-compatible storage, pass these properties:

```python
session = IcebergSession.builder().catalog(
    "rest", name="minio_rest",
    uri="http://rest-catalog:8181/",
    warehouse="s3://icebergdata/demo",
    **{
        "s3.endpoint": "http://localhost:9000",
        "s3.access-key-id": "admin",
        "s3.secret-access-key": "password",
        "s3.region": "us-east-1",
    },
).build()
```

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `NoCredentialError` | Missing AWS credentials | Set `AWS_*` env vars or pass explicit credentials |
| `Connection refused` | Wrong endpoint URI | Verify `uri` / `glue.endpoint` / `s3.endpoint` |
| `Bucket not found` | Warehouse path doesn't match | Ensure S3 bucket exists and warehouse path is correct |
| `SSL certificate verify failed` | Self-signed cert | Set `s3.ssl-enabled=false` for local MinIO |
