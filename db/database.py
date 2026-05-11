"""
Sync SQLAlchemy engine for PostgreSQL.
Falls back gracefully if DATABASE_URL is not set (file-only mode).
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from db.models import Base

_DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = None
_Session = None


def _normalize_url(url: str) -> tuple[str, dict]:
    """
    Convert URL to psycopg2 sync format.
    Returns (normalized_url, connect_args) — SSL params are stripped from
    the URL and passed as connect_args since psycopg2 doesn't accept them inline.
    """
    # Strip ?ssl=require — handle it via connect_args
    ssl_required = "ssl=require" in url
    url = url.split("?")[0] if "?" in url else url

    url = url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    if url.startswith("postgres://"):
        url = "postgresql+psycopg2://" + url[len("postgres://"):]
    elif url.startswith("postgresql://") and "+psycopg2" not in url:
        url = "postgresql+psycopg2://" + url[len("postgresql://"):]

    connect_args = {"sslmode": "require"} if ssl_required else {}
    return url, connect_args


if _DATABASE_URL:
    _url, _connect_args = _normalize_url(_DATABASE_URL)
    engine = create_engine(
        _url,
        connect_args=_connect_args,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )
    _Session = sessionmaker(bind=engine)


def init_db() -> None:
    """Create scrapper_jobs table if it doesn't exist. Safe to call on every startup."""
    if engine:
        Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Session:
    """Context manager that yields a DB session and handles commit/rollback."""
    if _Session is None:
        yield None
        return
    session = _Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def db_available() -> bool:
    return engine is not None
