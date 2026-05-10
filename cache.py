"""
Job store with disk persistence + R2 backup.
Jobs survive server restarts; results and candidates are file-backed locally
and also synced to Cloudflare R2 so data survives Railway redeploys.
"""

import json
import os
import threading
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


# ── Persistence helpers ───────────────────────────────────────────────────────

def _load_jobs() -> None:
    """Load jobs from disk into memory on startup, falling back to R2."""
    if _JOBS_FILE.exists():
        try:
            data = json.loads(_JOBS_FILE.read_text())
            _jobs.update(data)
            return
        except Exception:
            pass  # corrupt file — try R2

    # Fallback: restore from R2 (survives Railway redeploys)
    raw = _r2_get("scrapper/jobs.json")
    if raw:
        try:
            data = json.loads(raw)
            _jobs.update(data)
            _JOBS_FILE.write_text(raw)  # warm the local cache
        except Exception:
            pass


def _flush_jobs() -> None:
    """Write current jobs dict to disk and R2 (called inside _lock)."""
    try:
        serialised = json.dumps(_jobs, ensure_ascii=False)
        _JOBS_FILE.write_text(serialised)
        _r2_put("scrapper/jobs.json", serialised)
    except Exception:
        pass


# Load persisted jobs immediately when module is imported
_load_jobs()


# ── Job store ─────────────────────────────────────────────────────────────────

def create(job: dict) -> dict:
    with _lock:
        _jobs[job["id"]] = job
        _flush_jobs()
    return job


def get(job_id: str) -> dict | None:
    with _lock:
        return _jobs.get(job_id)


def update(job_id: str, fields: dict) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)
            _flush_jobs()


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
        _flush_jobs()
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
    with _lock:
        _results[job_id] = result
    _result_path(job_id).write_text(serialised)
    _r2_put(f"scrapper/{job_id}_result.json", serialised)


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
    # Fallback to R2
    raw = _r2_get(f"scrapper/{job_id}_result.json")
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
    # Fallback to R2
    text = _r2_get(f"scrapper/{job_id}_candidates.txt")
    if text:
        with _lock:
            _candidates[job_id] = text
        _candidates_path(job_id).write_text(text, encoding="utf-8")
        return text
    return None
