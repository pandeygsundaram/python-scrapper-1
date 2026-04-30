"""
In-memory job store. Survives the process lifetime only.
For persistence across restarts, swap _jobs / _results to a file or DB.
"""

import json
import threading
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_results: dict[str, dict] = {}
_candidates: dict[str, str] = {}

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


# ── Job store ─────────────────────────────────────────────────────────────────

def create(job: dict) -> dict:
    with _lock:
        _jobs[job["id"]] = job
    return job


def get(job_id: str) -> dict | None:
    with _lock:
        return _jobs.get(job_id)


def update(job_id: str, fields: dict) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


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
    _result_path(job_id).unlink(missing_ok=True)
    _candidates_path(job_id).unlink(missing_ok=True)
    return True


# ── Result persistence ────────────────────────────────────────────────────────

def _result_path(job_id: str) -> Path:
    return CACHE_DIR / f"{job_id}_result.json"


def _candidates_path(job_id: str) -> Path:
    return CACHE_DIR / f"{job_id}_candidates.txt"


def save_result(job_id: str, result: Any) -> None:
    with _lock:
        _results[job_id] = result
    _result_path(job_id).write_text(json.dumps(result, ensure_ascii=False))


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
    return None


def save_candidates(job_id: str, text: str) -> None:
    with _lock:
        _candidates[job_id] = text
    _candidates_path(job_id).write_text(text, encoding="utf-8")


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
    return None
