"""
Job store with three persistence layers (fastest → most durable):
  1. In-memory dict        — instant reads, lost on restart
  2. PostgreSQL            — survives restarts, shared across instances
  3. Cloudflare R2         — result blobs + fallback if DB unavailable

On startup: loads all jobs from DB into memory.
If DATABASE_URL is not set: falls back to disk file + R2 (original behaviour).
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_results: dict[str, dict] = {}
_candidates: dict[str, str] = {}

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
_JOBS_FILE = CACHE_DIR / "jobs.json"

# ── R2 client ─────────────────────────────────────────────────────────────────

_R2_ENDPOINT = os.environ.get("R2_ENDPOINT")
_R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY_ID")
_R2_SECRET_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
_R2_BUCKET = os.environ.get("R2_BUCKET_NAME")

_r2: Any = None
if _R2_ENDPOINT and _R2_ACCESS_KEY and _R2_SECRET_KEY and _R2_BUCKET:
    _r2 = boto3.client(
        "s3",
        endpoint_url=_R2_ENDPOINT,
        aws_access_key_id=_R2_ACCESS_KEY,
        aws_secret_access_key=_R2_SECRET_KEY,
        region_name="auto",
    )


def _r2_put(key: str, body: str, content_type: str = "application/json") -> None:
    if not _r2:
        return
    try:
        _r2.put_object(Bucket=_R2_BUCKET, Key=key, Body=body.encode(), ContentType=content_type)
    except Exception as e:
        print(f"[cache] R2 put failed for {key}: {e}")


def _r2_get(key: str) -> str | None:
    if not _r2:
        return None
    try:
        resp = _r2.get_object(Bucket=_R2_BUCKET, Key=key)
        return resp["Body"].read().decode()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        print(f"[cache] R2 get failed for {key}: {e}")
        return None
    except Exception as e:
        print(f"[cache] R2 get failed for {key}: {e}")
        return None


def _r2_delete(key: str) -> None:
    if not _r2:
        return
    try:
        _r2.delete_object(Bucket=_R2_BUCKET, Key=key)
    except Exception as e:
        print(f"[cache] R2 delete failed for {key}: {e}")


# ── DB ↔ dict conversion ──────────────────────────────────────────────────────

_MAIN_COLS = {
    "id", "status", "pipeline", "file_name", "file_size",
    "pdf_path", "result_r2_key", "progress", "created_at",
    "completed_at", "total_items", "total_value", "error",
}

# camelCase job dict key → DB column name
_CAMEL_TO_COL = {
    "id": "id",
    "status": "status",
    "pipeline": "pipeline",
    "fileName": "file_name",
    "fileSize": "file_size",
    "pdfPath": "pdf_path",
    "resultR2Key": "result_r2_key",
    "progress": "progress",
    "createdAt": "created_at",
    "completedAt": "completed_at",
    "totalItems": "total_items",
    "totalValue": "total_value",
    "error": "error",
}


def _job_dict_to_row(job: dict) -> dict:
    """Split a job dict into main column values + extra JSON blob."""
    row: dict = {}
    extra: dict = {}
    for k, v in job.items():
        col = _CAMEL_TO_COL.get(k)
        if col:
            # Parse ISO strings → datetime for timestamp columns
            if col in ("created_at", "completed_at") and isinstance(v, str):
                try:
                    v = datetime.fromisoformat(v)
                except ValueError:
                    pass
            row[col] = v
        else:
            extra[k] = v
    row["extra"] = extra or None
    return row


def _row_to_job_dict(row) -> dict:
    """Convert a ScrapperJob ORM row back to a camelCase job dict."""
    col_to_camel = {v: k for k, v in _CAMEL_TO_COL.items()}
    job: dict = {}
    for col in _CAMEL_TO_COL.values():
        val = getattr(row, col, None)
        if val is None:
            continue
        camel = col_to_camel[col]
        # Convert datetimes back to ISO strings
        if isinstance(val, datetime):
            val = val.isoformat()
        job[camel] = val
    # Merge extra fields back in
    if row.extra:
        job.update(row.extra)
    return job


# ── DB helpers ────────────────────────────────────────────────────────────────

def _db_upsert(job: dict) -> None:
    """Write/update a job row in PostgreSQL (no-op if DB unavailable)."""
    try:
        from db.database import get_session, db_available
        from db.models import ScrapperJob
        if not db_available():
            return
        row_data = _job_dict_to_row(job)
        with get_session() as session:
            if session is None:
                return
            existing = session.get(ScrapperJob, job["id"])
            if existing:
                for k, v in row_data.items():
                    setattr(existing, k, v)
            else:
                session.add(ScrapperJob(**row_data))
    except Exception as e:
        print(f"[cache] DB upsert failed for {job.get('id')}: {e}")


def _db_delete(job_id: str) -> None:
    try:
        from db.database import get_session, db_available
        from db.models import ScrapperJob
        if not db_available():
            return
        with get_session() as session:
            if session is None:
                return
            row = session.get(ScrapperJob, job_id)
            if row:
                session.delete(row)
    except Exception as e:
        print(f"[cache] DB delete failed for {job_id}: {e}")


def _load_from_db() -> None:
    """Load all jobs from PostgreSQL into memory on startup."""
    try:
        from db.database import get_session, db_available
        from db.models import ScrapperJob
        if not db_available():
            return
        with get_session() as session:
            if session is None:
                return
            rows = session.query(ScrapperJob).all()
            for row in rows:
                job = _row_to_job_dict(row)
                _jobs[job["id"]] = job
        print(f"[cache] Loaded {len(_jobs)} job(s) from PostgreSQL")
    except Exception as e:
        print(f"[cache] DB load failed — falling back to disk: {e}")
        _load_from_disk()


def _load_from_disk() -> None:
    """Fallback: load jobs from disk + R2 when DB is unavailable."""
    if _JOBS_FILE.exists():
        try:
            _jobs.update(json.loads(_JOBS_FILE.read_text()))
            return
        except Exception:
            pass
    raw = _r2_get("scrapper/jobs.json")
    if raw:
        try:
            _jobs.update(json.loads(raw))
            _JOBS_FILE.write_text(raw)
        except Exception:
            pass


def _flush_disk() -> None:
    """Write jobs to disk + R2 (used as fallback when DB unavailable)."""
    try:
        from db.database import db_available
        if db_available():
            return  # DB is primary — skip disk flush
    except Exception:
        pass
    try:
        serialised = json.dumps(_jobs, ensure_ascii=False)
        _JOBS_FILE.write_text(serialised)
        _r2_put("scrapper/jobs.json", serialised)
    except Exception:
        pass


# Initialise on import
_load_from_db()


# ── Job store (public API — unchanged signatures) ─────────────────────────────

def create(job: dict) -> dict:
    with _lock:
        _jobs[job["id"]] = job
        _flush_disk()
    _db_upsert(job)
    return job


def get(job_id: str) -> dict | None:
    with _lock:
        return _jobs.get(job_id)


def update(job_id: str, fields: dict) -> None:
    with _lock:
        if job_id not in _jobs:
            return
        _jobs[job_id].update(fields)
        job = dict(_jobs[job_id])
        _flush_disk()
    _db_upsert(job)


def list_jobs() -> list[dict]:
    with _lock:
        return list(_jobs.values())


def delete(job_id: str) -> bool:
    with _lock:
        if job_id not in _jobs:
            return False
        del _jobs[job_id]
        _results.pop(job_id, None)
        _candidates.pop(job_id, None)
        _flush_disk()
    _db_delete(job_id)
    _result_path(job_id).unlink(missing_ok=True)
    _candidates_path(job_id).unlink(missing_ok=True)
    _r2_delete(f"scrapper/{job_id}_result.json")
    _r2_delete(f"scrapper/{job_id}_candidates.txt")
    return True


# ── Result persistence ────────────────────────────────────────────────────────

def _result_path(job_id: str) -> Path:
    return CACHE_DIR / f"{job_id}_result.json"


def _candidates_path(job_id: str) -> Path:
    return CACHE_DIR / f"{job_id}_candidates.txt"


def save_result(job_id: str, result: Any) -> None:
    serialised = json.dumps(result, ensure_ascii=False)
    r2_key = f"scrapper/{job_id}_result.json"
    with _lock:
        _results[job_id] = result
    _result_path(job_id).write_text(serialised)
    _r2_put(r2_key, serialised)
    # Store R2 key in DB so we can find the result after a restart
    update(job_id, {"resultR2Key": r2_key})


def get_result(job_id: str) -> Any | None:
    with _lock:
        if job_id in _results:
            return _results[job_id]
    path = _result_path(job_id)
    if path.exists():
        data = json.loads(path.read_text())
        with _lock:
            _results[job_id] = data
        return data
    # Determine R2 key — prefer DB-stored key, fall back to convention
    job = get(job_id)
    r2_key = (job or {}).get("resultR2Key") or f"scrapper/{job_id}_result.json"
    raw = _r2_get(r2_key)
    if raw:
        data = json.loads(raw)
        with _lock:
            _results[job_id] = data
        _result_path(job_id).write_text(raw)
        return data
    return None


def save_candidates(job_id: str, text: str) -> None:
    with _lock:
        _candidates[job_id] = text
    _candidates_path(job_id).write_text(text, encoding="utf-8")
    _r2_put(f"scrapper/{job_id}_candidates.txt", text, content_type="text/plain")


def get_candidates(job_id: str) -> str | None:
    with _lock:
        if job_id in _candidates:
            return _candidates[job_id]
    path = _candidates_path(job_id)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        with _lock:
            _candidates[job_id] = text
        return text
    text = _r2_get(f"scrapper/{job_id}_candidates.txt")
    if text:
        with _lock:
            _candidates[job_id] = text
        _candidates_path(job_id).write_text(text, encoding="utf-8")
        return text
    return None
