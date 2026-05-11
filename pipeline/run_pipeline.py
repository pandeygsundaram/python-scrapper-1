"""
run_pipeline — Multi-Agent Multi-Block pipeline orchestrator.

Stages:
  1  Text extraction         (pdfplumber, deterministic)
  2  Table segment detection (Agent 1 — Gemini)
  3  Segment grouping        (deterministic, preserves Agent 1 metadata)
  4  Table reconstruction    (pdfplumber geometry, per sub-section)
  5  Block segmentation      (deterministic header/data split)
  6  Structuring             (Agent 2 — Gemini, parallel per sub-section)
  7  Merge + save stage files
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from pipeline.agents.structuring_agent import StructuringAgent
from pipeline.agents.table_segment_detector import TableSegmentDetector
from pipeline.core.block_segmentation import segment_into_blocks
from pipeline.core.segment_processor import build_segments
from pipeline.parsers.item_detector import (
    build_pre_structured,
    detect_item_anchors,
    detect_subtotals,
    split_by_subtotals,
)
from pipeline.parsers.table_reconstructor import extract_rows_from_segment
from pipeline.parsers.text_extractor import extract_text_with_markers
from pipeline.utils.logger import get_logger

logger = get_logger("RunPipeline")


def _confidence(segments: list[dict], all_blocks: list[dict]) -> float:
    data_rows = [b for b in all_blocks if b["type"] == "data"]
    if not data_rows or not segments:
        return 0.0
    seg_score = min(1.0, len(segments) / 5)
    row_score = min(1.0, len(data_rows) / 50)
    return round((seg_score + row_score) / 2, 2)


def _safe_num(val) -> str:
    """Format a number or return empty string."""
    if val is None:
        return ""
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val)


def _merge_structured(
    structured_results: list[dict],
    task_meta: list[tuple],
) -> dict:
    """
    Merge per-sub-section Agent 2 outputs into the final shape expected by
    the Stage 6 viewer and run_v2_extraction in main.py.
    """
    line_items: list[dict] = []
    room_recap: list[dict] = []
    trade_recap: list[dict] = []
    materials: list[dict] = []
    labor: list[dict] = []
    equipment: list[dict] = []
    stage5_items: list[dict] = []
    data_sections: list[dict] = []
    item_id = 1

    for result, (seg, sub) in zip(structured_results, task_meta):
        table_type = result.get("table_type", seg.get("table_type", "line_items"))
        section_name = result.get("section", sub["name"])
        items = result.get("items", [])
        sub_page_range = f"{sub['start_page']}-{sub['end_page']}"

        # ── Stage 5 viewer (generic display) ──────────────────────────────────
        stage5_items.append({
            "subSection": section_name,
            "type": table_type,
            "pages": sub_page_range,
            "count": len(items),
            "data": [
                {
                    "description": (
                        i.get("description") or i.get("room") or
                        i.get("name") or i.get("type") or ""
                    ),
                    "qty": _safe_num(
                        i.get("quantity") or i.get("hours") or i.get("qty")
                    ),
                    "unit": i.get("unit") or "",
                    "price": _safe_num(i.get("unit_price") or i.get("rate")),
                    "total": _safe_num(i.get("total")),
                }
                for i in items
            ],
        })

        # ── data.sections (for run_v2_extraction compat) ───────────────────────
        data_sections.append({
            "name": section_name,
            "table_type": table_type,
            "items": [
                {
                    "description": (
                        i.get("description") or i.get("room") or
                        i.get("name") or i.get("type") or ""
                    ),
                    "quantity": i.get("quantity") or i.get("hours"),
                    "unit": i.get("unit"),
                    "unit_price": i.get("unit_price") or i.get("rate"),
                    "total": i.get("total"),
                }
                for i in items
            ],
        })

        # ── Type-specific collections for Stage 6 ─────────────────────────────
        if table_type == "line_items":
            for i in items:
                line_items.append({
                    "id": item_id,
                    "item_number": i.get("item_number"),
                    "section": section_name,
                    "description": i.get("description", ""),
                    "qty": _safe_num(i.get("quantity")),
                    "unit": i.get("unit") or "",
                    "price": _safe_num(i.get("unit_price")),
                    "total": _safe_num(i.get("total")),
                    "page": sub["start_page"],
                })
                item_id += 1

        elif table_type == "room_recap":
            for i in items:
                room_recap.append({
                    "room": i.get("room", ""),
                    "total": _safe_num(i.get("total")),
                    "percent": i.get("percent") or "0%",
                })

        elif table_type == "trade_recap":
            for i in items:
                trade_recap.append({
                    "code": i.get("code", ""),
                    "name": i.get("name", ""),
                    "total": _safe_num(i.get("total")),
                })

        elif table_type == "materials_breakdown":
            for i in items:
                materials.append({
                    "desc": i.get("description", ""),
                    "qty": _safe_num(i.get("quantity")),
                    "unit": i.get("unit") or "",
                    "total": _safe_num(i.get("total")),
                })

        elif table_type == "labor_breakdown":
            for i in items:
                labor.append({
                    "type": i.get("type", ""),
                    "hours": _safe_num(i.get("hours")),
                    "rate": _safe_num(i.get("rate")),
                    "total": _safe_num(i.get("total")),
                })

        elif table_type == "equipment_breakdown":
            for i in items:
                equipment.append({
                    "desc": i.get("description", ""),
                    "qty": _safe_num(i.get("quantity")),
                    "total": _safe_num(i.get("total")),
                })

    return {
        "data_sections": data_sections,
        "stage5_items": stage5_items,
        "line_items": line_items,
        "room_recap": room_recap,
        "trade_recap": trade_recap,
        "materials": materials,
        "labor": labor,
        "equipment": equipment,
    }


def _save_stage_files(
    stem: str,
    out_dir: Path,
    pages: list[dict],
    doc_type: str,
    segments: list[dict],
    all_rows: list[dict],
    all_blocks: list[dict],
    extraction_items: list[dict],
    stage5_items: list[dict],
    output: dict,
) -> None:
    """Save all stage files for the pipeline viewer."""

    # Stage 1 — raw text per page
    stage1 = {
        "totalPages": len(pages),
        "totalChars": sum(len(p["text"]) for p in pages),
        "pages": [
            {
                "page": p["page_number"],
                "text": p["text"],
                "chars": len(p["text"]),
                "words": len(p["text"].split()),
            }
            for p in pages
        ],
    }
    (out_dir / f"{stem}_01_text.json").write_text(
        json.dumps(stage1, indent=2, ensure_ascii=False)
    )

    # Stage 2 — Agent 1 segments
    stage2 = {
        "docType": doc_type,
        "total_segments": len(segments),
        "items": [
            {
                "type": seg.get("table_type", "line_items"),
                "name": seg.get("description") or f"Segment {i + 1}",
                "pages": f"{seg['start_page']}-{seg['end_page']}",
                "schema": seg.get("column_schema", []),
                "subSections": [
                    {
                        "name": sub["name"],
                        "pages": f"{sub['start_page']}-{sub['end_page']}",
                        "start_page": sub["start_page"],
                    }
                    for sub in seg.get("sub_sections", [])
                ],
            }
            for i, seg in enumerate(segments)
        ],
    }
    (out_dir / f"{stem}_02_segments.json").write_text(
        json.dumps(stage2, indent=2, ensure_ascii=False)
    )

    # Stage 3 — geometric rows
    rule_count = sum(1 for r in all_rows if r.get("has_rule"))
    stage3 = {
        "total": len(all_rows),
        "rule_based": rule_count,
        "position_based": len(all_rows) - rule_count,
        "items": [
            {
                "id": i + 1,
                "page": r["page"],
                "cells": r["cells"],
                "has_rule": r.get("has_rule", False),
            }
            for i, r in enumerate(all_rows)
        ],
    }
    (out_dir / f"{stem}_03_rows.json").write_text(
        json.dumps(stage3, indent=2, ensure_ascii=False)
    )

    # Stage 4 — blocks (header/data split)
    header_count = sum(1 for b in all_blocks if b["type"] == "header")
    stage4 = {
        "total": len(all_blocks),
        "headers": header_count,
        "data": len(all_blocks) - header_count,
        "items": [
            {
                "id": i + 1,
                "type": b["type"],
                "page": b["page"],
                "content": " | ".join(b["cells"]),
            }
            for i, b in enumerate(all_blocks)
        ],
    }
    (out_dir / f"{stem}_04_blocks.json").write_text(
        json.dumps(stage4, indent=2, ensure_ascii=False)
    )

    # Stage 3b — layout-preserved extraction sent to Agent 2
    (out_dir / f"{stem}_03b_extraction.json").write_text(
        json.dumps(
            {"totalSubSections": len(extraction_items), "items": extraction_items},
            indent=2,
            ensure_ascii=False,
        )
    )

    # Stage 5 — Agent 2 structured per sub-section
    (out_dir / f"{stem}_05_structured.json").write_text(
        json.dumps({"items": stage5_items}, indent=2, ensure_ascii=False)
    )

    # Final — complete merged output
    (out_dir / f"{stem}_FINAL.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False)
    )

    logger.info(f"Stage files saved: {stem}_01..05 + _FINAL")


async def run_pipeline(
    pdf_path: str,
    api_key: str | None = None,
    model_name: str = "gemini-2.5-pro",
    output_dir: str | None = None,
    debug: bool = False,
) -> dict:
    """
    Run the full multi-agent extraction pipeline on a PDF.

    Returns the complete output dict (also saved as _FINAL.json).
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("Gemini API key required (pass api_key= or set GEMINI_API_KEY)")

    pdf_path = str(pdf_path)
    out_dir = Path(output_dir) if output_dir else Path("pipeline_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf_path).stem
    source_name = Path(pdf_path).name

    logger.info("=" * 60)
    logger.info(f"Pipeline start: {pdf_path}")
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

    # ── Step 3: Build segment groups (preserve metadata) ──────────────────────
    logger.info("Step 3 ▶ Segment grouping")
    segments = build_segments(pages, table_segment_list)

    # ── Steps 4+5: Reconstruct rows + block segmentation per segment ──────────
    all_rows: list[dict] = []
    all_blocks: list[dict] = []

    for i, seg in enumerate(segments, 1):
        label = f"Segment {i}: [{seg['table_type']}] p{seg['start_page']}-{seg['end_page']}"
        logger.info(f"Step 4.{i} ▶ Table reconstruction — {label}")

        rows = extract_rows_from_segment(seg, pdf_path)
        all_rows.extend(rows)

        logger.info(f"Step 5.{i} ▶ Block segmentation — {label}")
        blocks = segment_into_blocks(rows)
        seg["blocks"] = blocks
        all_blocks.extend(blocks)

    logger.info(
        f"Total: {len(all_rows)} rows → "
        f"{len(all_blocks)} blocks "
        f"({sum(1 for b in all_blocks if b['type'] == 'data')} data, "
        f"{sum(1 for b in all_blocks if b['type'] == 'header')} headers)"
    )

    # ── Step 6: Agent 2 — parallel structuring per sub-section ───────────────
    logger.info("Step 6 ▶ Agent 2 — structuring (parallel per sub-section)")
    structurer = StructuringAgent(api_key, model_name)

    tasks = []
    task_meta: list[tuple] = []  # (segment, sub_section) for each task
    extraction_items: list[dict] = []  # what gets sent to Agent 2, for the viewer

    BOUNDARY_ROWS = 30  # rows of context before/after each sub-section

    for seg in segments:
        seg_blocks = seg["blocks"]
        is_line_items = seg.get("table_type") == "line_items"
        seg_start = seg["start_page"]
        seg_end = seg["end_page"]

        # ── Full-segment anchor + subtotal scan (line_items only) ─────────────
        # Scan once across the whole segment, split at subtotal lines so each
        # group maps cleanly to one sub-section without boundary-page overlap.
        anchor_groups: list[list[dict]] = []
        if is_line_items:
            all_seg_anchors = detect_item_anchors(pdf_path, seg_start, seg_end)
            subtotals = detect_subtotals(pdf_path, seg_start, seg_end)
            anchor_groups = split_by_subtotals(all_seg_anchors, subtotals)
            logger.info(
                f"  Segment p{seg_start}-{seg_end}: "
                f"{len(all_seg_anchors)} anchors, "
                f"{len(subtotals)} subtotal(s) → "
                f"{len(anchor_groups)} section group(s)"
            )

        subsections = seg.get("sub_sections", [])

        for i, sub in enumerate(subsections):
            sub_start = sub["start_page"]
            sub_end = sub["end_page"]

            sub_blocks = [
                b for b in seg_blocks
                if sub_start <= b["page"] <= sub_end
            ]

            # Boundary context rows
            before_blocks = [b for b in seg_blocks if b["page"] < sub_start]
            context_before = before_blocks[-BOUNDARY_ROWS:] if before_blocks else []
            after_blocks = [b for b in seg_blocks if b["page"] > sub_end]
            context_after = after_blocks[:BOUNDARY_ROWS] if after_blocks else []

            # ── Assign this sub-section's anchor group ────────────────────────
            anchors: list[dict] = []
            pre_structured: list[dict] = []
            number_range = sub.get("number_range")

            if is_line_items:
                if i < len(anchor_groups):
                    anchors = anchor_groups[i]
                else:
                    # Fallback: scan just this sub-section's pages
                    anchors = detect_item_anchors(pdf_path, sub_start, sub_end)

                if anchors:
                    pre_structured = build_pre_structured(
                        anchors, all_rows, seg_start, seg_end
                    )
                    number_range = {
                        "start": anchors[0]["item_number"],
                        "end": anchors[-1]["item_number"],
                    }
                    logger.info(
                        f"  [{sub['name']}] {len(anchors)} anchors → "
                        f"{len(pre_structured)} items "
                        f"({number_range['start']}→{number_range['end']})"
                    )

            tasks.append(
                structurer.structure_subsection(
                    sub_blocks,
                    seg.get("table_type", "line_items"),
                    seg.get("column_schema", []),
                    sub["name"],
                    number_range=number_range,
                    context_before=context_before,
                    context_after=context_after,
                    pre_structured=pre_structured,
                )
            )
            task_meta.append((seg, sub))

            # Build viewer text
            lines = []
            for b in context_before:
                lines.append(f"[CONTEXT_BEFORE p{b.get('page','?')}] {' | '.join(b.get('cells', []))}")
            for b in sub_blocks:
                cells = " | ".join(b.get("cells", []))
                page = b.get("page", "?")
                prefix = "[HEADER]" if b.get("type") == "header" else f"[ROW p{page}]"
                lines.append(f"{prefix} {cells}")
            for b in context_after:
                lines.append(f"[CONTEXT_AFTER p{b.get('page','?')}] {' | '.join(b.get('cells', []))}")

            extraction_items.append({
                "subSection": sub["name"],
                "tableType": seg.get("table_type", "line_items"),
                "schema": seg.get("column_schema", []),
                "pages": f"{sub_start}-{sub_end}",
                "startPage": sub_start,
                "numberRange": number_range,
                "anchorCount": len(anchors),
                "totalBlocks": len(sub_blocks),
                "headerCount": sum(1 for b in sub_blocks if b["type"] == "header"),
                "dataCount": sum(1 for b in sub_blocks if b["type"] == "data"),
                "contextBeforeCount": len(context_before),
                "contextAfterCount": len(context_after),
                "text": "\n".join(lines),
            })

    structured_results = await asyncio.gather(*tasks)
    sub_sections_count = len(tasks)

    # ── Step 7: Merge + build output ──────────────────────────────────────────
    logger.info("Step 7 ▶ Merging results")
    merged = _merge_structured(list(structured_results), task_meta)

    confidence = _confidence(segments, all_blocks)
    # Only sum line items — not recap/breakdown tables
    def _to_float(v):
        try:
            return float(v) if v not in (None, "", "0") else 0.0
        except (ValueError, TypeError):
            return 0.0

    total_value_raw = sum(_to_float(item.get("total")) for item in merged["line_items"])

    output = {
        # Metadata (used by list_documents and run_v2_extraction in main.py)
        "source": source_name,
        "page_range": f"1-{total_pages}",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "v2",
        "doc_type": doc_type,
        "segments_detected": len(segments),
        "sub_sections_processed": sub_sections_count,
        "total_line_items": len(merged["line_items"]),
        "total_value": total_value_raw,
        "confidence": confidence,
        # data.sections used by run_v2_extraction
        "data": {
            "sections": merged["data_sections"],
            "items": [
                item
                for section in merged["data_sections"]
                for item in section["items"]
            ],
        },
        # Stage 6 viewer tabs
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
            "pipelineVersion": "v2",
            "docType": doc_type,
            "confidence": confidence,
            "pageRange": f"1-{total_pages}",
            "stats": {
                "segments": len(segments),
                "subSections": sub_sections_count,
                "totalBlocks": len(all_blocks),
                "lineItems": len(merged["line_items"]),
            },
            "totalValue": f"${total_value_raw:,.2f}",
        },
    }

    # ── Save all stage files ──────────────────────────────────────────────────
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
        f"Pipeline complete — {len(segments)} segment(s), "
        f"{sub_sections_count} sub-section(s), "
        f"{len(merged['line_items'])} line items, "
        f"confidence={confidence}"
    )
    logger.info("=" * 60)

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
