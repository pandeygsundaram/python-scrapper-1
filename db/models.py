from datetime import datetime
from typing import Optional
from sqlalchemy import String, Integer, Float, DateTime, Text, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ScrapperJob(Base):
    """
    Persists scrapper job metadata to PostgreSQL.
    Shared DB with call-summariser — uses scrapper_jobs table to avoid conflicts.
    Large result blobs stay in R2; result_r2_key points to them.
    """

    __tablename__ = "scrapper_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="pending")
    pipeline: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    file_name: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    pdf_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    result_r2_key: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    progress: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    total_items: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Stores any extra fields (docType, confidence, reprocessedFrom, etc.)
    extra: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
