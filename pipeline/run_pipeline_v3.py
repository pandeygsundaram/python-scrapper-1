"""
run_pipeline_v3 — V3 pipeline orchestrator.

Same extraction as V2 (pdfplumber rows + block segmentation) for all segments.
For line_items: uses structure_subsection_v3 — lean prompt returning only
item_number + description (no financial columns).
All other segment types (room_recap, trade_recap, etc.) use full V2 Agent 2 prompt unchanged.
All Agent 1 output (doc_type, column_schema, sub_sections) is preserved.
"""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

from pipeline.agents.structuring_agent import StructuringAgent
from pipeline.agents.table_segment_detector import TableSegmentDetector
from pipeline.core.block_segmentation import segment_into_blocks
from pipeline.core.segment_processor import build_segments
from pipeline.parsers.table_reconstructor import extract_rows_from_segment
from pipeline.parsers.text_extractor import extract_text_with_markers
from pipeline.run_pipeline import _confidence, _merge_structured, _save_stage_files
from pipeline.utils.logger import get_logger

logger = get_logger("RunPipelineV3")


async def run_pipeline_v3(
    pdf_path: str,
    api_key: str | None = None,
    model_name: str = "gemini-2.5-pro",
    output_dir: str | None = None,
    debug: bool = False,
) -> dict:
    """
    Run the V3 multi-agent extraction pipeline on a PDF.

    Returns the complete output dict (also saved as _FINAL.json).
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("Gemini API key required")

    pdf_path = str(pdf_path)
    out_dir = Path(output_dir) if output_dir else Path("pipeline_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf_path).stem
    source_name = Path(pdf_path).name

    logger.info("=" * 60)
    logger.info(f"V3 Pipeline start: {pdf_path}")
    logger.info("=" * 60)

    # ── Step 1: Text extraction ───────────────────────────────────────────────
    logger.info("Step 1 ▶ Text extraction")
    full_text, pages = extract_text_with_markers(pdf_path)
    total_pages = len(pages)

    # ── Step 2: Agent 1 — segment detection ──────────────────────────────────
    logger.info("Step 2 ▶ Agent 1 — table segment detection")
    detector = TableSegmentDetector(api_key, model_name)
    agent1_result = await detector.detect(full_text, total_pages)
    doc_type = agent1_result["doc_type"]
    table_segment_list = agent1_result["table_segments"]

    # ── Step 3: Build segment groups ──────────────────────────────────────────
    logger.info("Step 3 ▶ Segment grouping")
    segments = build_segments(pages, table_segment_list)

    # ── Step 4: Table reconstruction (V2 path for all segments) ───────────────
    all_rows = []
    all_blocks = []
    extraction_items = []

    for seg in segments:
        logger.info(
            f"Step 4 ▶ Table reconstruction — "
            f"[{seg['table_type']}] p{seg['start_page']}-{seg['end_page']}"
        )
        rows = extract_rows_from_segment(seg, pdf_path)
        all_rows.extend(rows)

        logger.info("Step 5 ▶ Block segmentation")
        blocks = segment_into_blocks(rows)
        seg["blocks"] = blocks
        all_blocks.extend(blocks)

    # ── Step 6: Build tasks per sub-section ───────────────────────────────────
    structurer = StructuringAgent(api_key, model_name)
    tasks = []
    task_meta = []

    BOUNDARY_ROWS = 30

    for seg in segments:
        is_line_items = seg.get("table_type") == "line_items"
        blocks = seg.get("blocks", [])
        subsections = seg.get("sub_sections", [])

        for sub in subsections:
            sub_start = sub["start_page"]
            sub_end = sub["end_page"]

            sub_blocks = [b for b in blocks if sub_start <= b["page"] <= sub_end]
            before_blocks = [b for b in blocks if b["page"] < sub_start]
            context_before = before_blocks[-BOUNDARY_ROWS:]
            after_blocks = [b for b in blocks if b["page"] > sub_end]
            context_after = after_blocks[:BOUNDARY_ROWS]

            lines = []
            for b in sub_blocks:
                cells = " | ".join(b.get("cells", []))
                prefix = "[HEADER]" if b.get("type") == "header" else f"[ROW p{b.get('page','?')}]"
                lines.append(f"{prefix} {cells}")

            extraction_items.append({
                "subSection": sub["name"],
                "tableType": seg.get("table_type"),
                "schema": seg.get("column_schema", []),
                "pages": f"{sub_start}-{sub_end}",
                "startPage": sub_start,
                "pipeline": "v3" if is_line_items else "v2",
                "text": "\n".join(lines),
            })

            if is_line_items:
                tasks.append(
                    structurer.structure_subsection_v3(
                        sub_blocks,
                        seg.get("column_schema", []),
                        sub["name"],
                        number_range=sub.get("number_range"),
                        context_before=context_before,
                        context_after=context_after,
                    )
                )
            else:
                tasks.append(
                    structurer.structure_subsection(
                        sub_blocks,
                        seg.get("table_type", "line_items"),
                        seg.get("column_schema", []),
                        sub["name"],
                        number_range=sub.get("number_range"),
                        context_before=context_before,
                        context_after=context_after,
                    )
                )
            task_meta.append((seg, sub))

    # ── Step 7: Run all Agent 2 calls in parallel ─────────────────────────────
    logger.info("Step 7 ▶ Agent 2 — structuring (parallel)")
    structured_results = await asyncio.gather(*tasks)
    sub_sections_count = len(tasks)

    # ── Step 8: Merge ─────────────────────────────────────────────────────────
    logger.info("Step 8 ▶ Merging results")
    merged = _merge_structured(list(structured_results), task_meta)

    confidence = _confidence(segments, all_blocks) if all_blocks else 1.0

    def _to_float(v):
        try:
            return float(v) if v not in (None, "", "0") else 0.0
        except (ValueError, TypeError):
            return 0.0

    total_value_raw = sum(_to_float(item.get("total")) for item in merged["line_items"])

    output = {
        "source": source_name,
        "page_range": f"1-{total_pages}",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "v3",
        "doc_type": doc_type,
        "segments_detected": len(segments),
        "sub_sections_processed": sub_sections_count,
        "total_line_items": len(merged["line_items"]),
        "total_value": total_value_raw,
        "confidence": confidence,
        "data": {
            "sections": merged["data_sections"],
            "items": [
                item
                for section in merged["data_sections"]
                for item in section["items"]
            ],
        },
        "lineItems": merged["line_items"],
        "roomRecap": merged["room_recap"],
        "tradeRecap": merged["trade_recap"],
        "costBreakdown": {
            "materials": merged["materials"],
            "labor": merged["labor"],
            "equipment": merged["equipment"],
        },
        "summary": {
            "sourceFile": source_name,
            "extractedAt": datetime.now(timezone.utc).isoformat(),
            "pipelineVersion": "v3",
            "docType": doc_type,
            "confidence": confidence,
            "pageRange": f"1-{total_pages}",
            "stats": {
                "segments": len(segments),
                "subSections": sub_sections_count,
                "lineItems": len(merged["line_items"]),
            },
            "totalValue": f"${total_value_raw:,.2f}",
        },
    }

    _save_stage_files(
        stem=stem,
        out_dir=out_dir,
        pages=pages,
        doc_type=doc_type,
        segments=segments,
        all_rows=all_rows,
        all_blocks=all_blocks,
        extraction_items=extraction_items,
        stage5_items=merged["stage5_items"],
        output=output,
    )

    logger.info("=" * 60)
    logger.info(
        f"V3 Pipeline complete — {len(segments)} segment(s), "
        f"{sub_sections_count} sub-section(s), "
        f"{len(merged['line_items'])} line items, "
        f"confidence={confidence}"
    )
    logger.info("=" * 60)

    return output
