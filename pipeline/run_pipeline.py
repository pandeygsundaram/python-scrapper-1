"""
run_pipeline — main orchestrator for the Multi-Agent Multi-Block pipeline.

Usage:
    from pipeline.run_pipeline import run_pipeline
    result = asyncio.run(run_pipeline("path/to/file.pdf", api_key="..."))

Stages:
    1  Text extraction         (pdfplumber, deterministic)
    2  Table segment detection (Agent 1 — Gemini)
    3  Segment grouping        (deterministic)
    4  Table reconstruction    (pdfplumber geometry, deterministic)
    5  Block segmentation      (deterministic)
    6  Structuring             (Agent 2 — Gemini)
    7  Output                  (JSON + optional debug CSV)
"""

import asyncio
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from pipeline.agents.structuring_agent import StructuringAgent
from pipeline.agents.table_segment_detector import TableSegmentDetector
from pipeline.core.block_segmentation import segment_into_blocks
from pipeline.core.segment_processor import build_segments
from pipeline.parsers.table_reconstructor import extract_rows_from_segment
from pipeline.parsers.text_extractor import extract_text_with_markers
from pipeline.utils.logger import get_logger

logger = get_logger("RunPipeline")


# ── Confidence heuristic ──────────────────────────────────────────────────────

def _confidence(segments: list[dict], all_blocks: list[dict]) -> float:
    data_rows = [b for b in all_blocks if b["type"] == "data"]
    if not data_rows or not segments:
        return 0.0
    # More segments detected + denser data = higher confidence
    seg_score = min(1.0, len(segments) / 5)
    row_score = min(1.0, len(data_rows) / 50)
    return round((seg_score + row_score) / 2, 2)


# ── Debug helpers ─────────────────────────────────────────────────────────────

def _save_debug_csv(all_blocks: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "page", "cells"])
        for b in all_blocks:
            writer.writerow([b["type"], b["page"], " | ".join(b["cells"])])
    logger.info(f"Debug CSV saved: {path}")


def _print_segment_table(rows: list[dict], segment_label: str) -> None:
    logger.debug(f"── {segment_label} ──")
    for row in rows:
        logger.debug(f"  p{row['page']} | {' | '.join(row['cells'])}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_pipeline(
    pdf_path: str,
    api_key: str | None = None,
    model_name: str = "gemini-2.5-pro",
    output_dir: str | None = None,
    debug: bool = False,
) -> dict:
    """
    Run the full multi-agent extraction pipeline on a PDF.

    Args:
        pdf_path   : path to the PDF file
        api_key    : Gemini API key (falls back to GEMINI_API_KEY env var)
        model_name : Gemini model to use for both agents
        output_dir : directory to write JSON/CSV output (default: pipeline_output/)
        debug      : if True, log intermediate rows and save debug CSV

    Returns:
        Full output dict with metadata + structured data.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("Gemini API key required (pass api_key= or set GEMINI_API_KEY)")

    pdf_path = str(pdf_path)
    out_dir = Path(output_dir) if output_dir else Path("pipeline_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf_path).stem

    logger.info(f"{'='*60}")
    logger.info(f"Pipeline start: {pdf_path}")
    logger.info(f"{'='*60}")

    # ── Step 1: Text extraction ───────────────────────────────────────────────
    logger.info("Step 1 ▶ Text extraction")
    full_text, pages = extract_text_with_markers(pdf_path)
    total_pages = len(pages)

    # ── Step 2: Agent 1 — segment detection ──────────────────────────────────
    logger.info("Step 2 ▶ Agent 1 — table segment detection")
    detector = TableSegmentDetector(api_key, model_name)
    table_segments = await detector.detect(full_text, total_pages)

    # ── Step 3: Build segment groups ──────────────────────────────────────────
    logger.info("Step 3 ▶ Segment grouping")
    segments = build_segments(pages, table_segments)

    # ── Steps 4+5: Reconstruct + block-segment each segment ───────────────────
    all_blocks: list[dict] = []

    for i, seg in enumerate(segments, 1):
        label = f"Segment {i}: pages {seg['start_page']}-{seg['end_page']}"
        logger.info(f"Step 4.{i} ▶ Table reconstruction — {label}")

        rows = extract_rows_from_segment(seg, pdf_path)

        if debug:
            _print_segment_table(rows, label)

        logger.info(f"Step 5.{i} ▶ Block segmentation — {label}")
        blocks = segment_into_blocks(rows)
        seg["blocks"] = blocks
        all_blocks.extend(blocks)

    logger.info(
        f"Total blocks: {len(all_blocks)} "
        f"({sum(1 for b in all_blocks if b['type'] == 'data')} data rows, "
        f"{sum(1 for b in all_blocks if b['type'] == 'header')} headers)"
    )

    # ── Step 6: Agent 2 — structuring ─────────────────────────────────────────
    logger.info("Step 6 ▶ Agent 2 — structuring and normalization")
    structurer = StructuringAgent(api_key, model_name)
    structured = await structurer.structure(all_blocks)

    # ── Step 7: Output ────────────────────────────────────────────────────────
    confidence = _confidence(segments, all_blocks)
    source_name = Path(pdf_path).name

    output = {
        "source": source_name,
        "page_range": f"1-{total_pages}",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "v2",
        "segments_detected": len(segments),
        "total_blocks": len(all_blocks),
        "confidence": confidence,
        "data": structured,
    }

    out_json = out_dir / f"{stem}_result.json"
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info(f"JSON saved: {out_json}")

    if debug:
        _save_debug_csv(all_blocks, out_dir / f"{stem}_debug.csv")

    logger.info(f"{'='*60}")
    logger.info(
        f"Pipeline complete — {len(segments)} segment(s), "
        f"confidence={confidence}"
    )
    logger.info(f"{'='*60}")
    return output


# ── Standalone usage ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.run_pipeline <pdf_path> [debug]")
        sys.exit(1)

    _debug = len(sys.argv) > 2 and sys.argv[2] == "debug"
    result = asyncio.run(run_pipeline(sys.argv[1], debug=_debug))
    print(json.dumps(result, indent=2))
