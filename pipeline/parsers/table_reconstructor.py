"""
Deterministic table reconstruction â sub-section aware.

Row detection strategy (in priority order):
  1. Horizontal rule lines â HARD row boundaries (authoritative)
     - If a page has rule lines, pairs of consecutive rules define each row band
     - Words whose vertical centre falls in a band â belong to that row
  2. Y-position grouping â fallback when no rule lines present
     - Words with similar top-coordinate (within ROW_Y_TOL) â same row

Key improvement over v1:
  - Processes ONE sub-section at a time (small, focused page range)
  - Horizontal rules are the authoritative signal â no guessing across rules
  - Continuation merging only applies in the y-position fallback path
  - Narrative/non-table text filtered by checking for numeric content
    when a column schema is provided
"""

import re
import pdfplumber

from pipeline.utils.logger import get_logger

logger = get_logger("TableReconstructor")

MIN_RULE_WIDTH = 80   # pts â minimum horizontal extent to count as a table rule
ROW_Y_TOL = 3         # pts â y-tolerance for grouping words into same row (fallback)
COL_GAP = 12          # pts â horizontal gap that signals a new column cell

_NUMERIC_RE = re.compile(r"\d")


# ââ Geometry ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def _horizontal_rule_ys(page) -> list[float]:
    """Sorted y-positions (top-of-page coords) of significant horizontal rules."""
    ys: set[float] = set()
    for src in (page.lines or [], page.edges or []):
        for obj in src:
            top = obj.get("top", 0)
            bottom = obj.get("bottom", 0)
            x0 = obj.get("x0", 0)
            x1 = obj.get("x1", 0)
            if abs(top - bottom) < 2 and abs(x1 - x0) >= MIN_RULE_WIDTH:
                ys.add(round(top, 1))
    return sorted(ys)


def _words_in_band(words: list[dict], y_top: float, y_bottom: float) -> list[dict]:
    result = []
    for w in words:
        cy = (w["top"] + w["bottom"]) / 2
        if y_top <= cy <= y_bottom:
            result.append(w)
    return sorted(result, key=lambda w: w["x0"])


def _words_to_cells(words: list[dict]) -> list[str]:
    if not words:
        return []
    cells: list[str] = []
    buf = [words[0]["text"]]
    prev_x1 = words[0]["x1"]
    for w in words[1:]:
        if w["x0"] - prev_x1 > COL_GAP:
            cells.append(" ".join(buf))
            buf = [w["text"]]
        else:
            buf.append(w["text"])
        prev_x1 = w["x1"]
    if buf:
        cells.append(" ".join(buf))
    return cells


# ââ Header detection ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def _header_signature(cells: list[str]) -> frozenset[str]:
    return frozenset(c.strip().lower() for c in cells if c.strip())


def _is_repeated_header(cells: list[str], sig: frozenset[str]) -> bool:
    if not sig or not cells:
        return False
    row_sig = frozenset(c.strip().lower() for c in cells if c.strip())
    return len(row_sig & sig) >= min(2, len(sig))


# ââ Row extraction per page âââââââââââââââââââââââââââââââââââââââââââââââââââ

def _rows_by_rules(words: list[dict], rule_ys: list[float], page_num: int) -> list[dict]:
    """Use horizontal rule pairs as hard row boundaries."""
    rows = []
    for i in range(len(rule_ys) - 1):
        band_words = _words_in_band(words, rule_ys[i], rule_ys[i + 1])
        if band_words:
            cells = _words_to_cells(band_words)
            if any(c.strip() for c in cells):
                rows.append({
                    "page": page_num,
                    "cells": cells,
                    "y_top": rule_ys[i],
                    "y_bottom": rule_ys[i + 1],
                    "has_rule": True,
                })
    return rows


def _rows_by_word_positions(words: list[dict], page_num: int) -> list[dict]:
    """Fallback: group by y-position when no rule lines present."""
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: w["top"])
    rows: list[dict] = []
    bucket: list[dict] = [sorted_words[0]]
    ref_y = sorted_words[0]["top"]

    for w in sorted_words[1:]:
        if abs(w["top"] - ref_y) <= ROW_Y_TOL:
            bucket.append(w)
        else:
            ordered = sorted(bucket, key=lambda x: x["x0"])
            cells = _words_to_cells(ordered)
            if any(c.strip() for c in cells):
                rows.append({
                    "page": page_num,
                    "cells": cells,
                    "y_top": bucket[0]["top"],
                    "y_bottom": bucket[0]["bottom"],
                    "has_rule": False,
                })
            bucket = [w]
            ref_y = w["top"]

    if bucket:
        ordered = sorted(bucket, key=lambda x: x["x0"])
        cells = _words_to_cells(ordered)
        if any(c.strip() for c in cells):
            rows.append({
                "page": page_num,
                "cells": cells,
                "y_top": bucket[0]["top"],
                "y_bottom": bucket[0]["bottom"],
                "has_rule": False,
            })
    return rows


# ââ Continuation merging (fallback path only) âââââââââââââââââââââââââââââââââ

def _merge_continuation_rows(rows: list[dict]) -> list[dict]:
    """
    In the y-position fallback path, a continuation row has exactly one
    non-empty cell and contains NO numeric characters.
    Only merge when the row was NOT bounded by rule lines.
    """
    merged: list[dict] = []
    for row in rows:
        if row.get("has_rule"):
            merged.append(row)
            continue
        non_empty = [c for c in row["cells"] if c.strip()]
        is_continuation = (
            len(non_empty) == 1
            and not _NUMERIC_RE.search(non_empty[0])
        )
        if is_continuation and merged:
            prev = merged[-1]
            if prev["cells"]:
                prev["cells"][0] = prev["cells"][0] + " " + non_empty[0]
            continue
        merged.append(row)
    return merged


# ââ Public API ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def extract_rows_from_subsection(
    subsection: dict,
    pdf_path: str,
    column_schema: list[str] | None = None,
) -> list[dict]:
    """
    Extract rows for a single sub-section (a named block within a segment).

    Args:
        subsection     : {name, start_page, end_page}
        pdf_path       : path to the PDF
        column_schema  : column headers from Agent 1 (used for logging/debug)

    Returns:
        List of row dicts: {page, cells, y_top, y_bottom, has_rule}
    """
    start_page = subsection["start_page"]
    end_page = subsection["end_page"]
    name = subsection.get("name", "Unknown")
    all_rows: list[dict] = []
    header_sig: frozenset[str] = frozenset()

    logger.debug(
        f"Extracting '{name}' p{start_page}-{end_page} "
        f"| schema={column_schema}"
    )

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        used_rules = False

        for page_num in range(start_page, end_page + 1):
            idx = page_num - 1
            if idx >= total:
                logger.warning(f"Page {page_num} out of range, skipping")
                continue

            page = pdf.pages[idx]
            words = page.extract_words(x_tolerance=3, y_tolerance=3) or []
            rule_ys = _horizontal_rule_ys(page)

            if len(rule_ys) >= 2:
                rows = _rows_by_rules(words, rule_ys, page_num)
                used_rules = True
                logger.debug(
                    f"  p{page_num}: {len(words)} words, "
                    f"{len(rule_ys)} rules â {len(rows)} rows (rule-based)"
                )
            else:
                rows = _rows_by_word_positions(words, page_num)
                logger.debug(
                    f"  p{page_num}: {len(words)} words, "
                    f"0 rules â {len(rows)} rows (position-based)"
                )

            # Learn and strip repeated column header rows
            if page_num == start_page and rows:
                header_sig = _header_signature(rows[0]["cells"])
                logger.debug(f"  Header sig for '{name}': {header_sig}")
            elif header_sig and rows:
                if _is_repeated_header(rows[0]["cells"], header_sig):
                    logger.debug(f"  p{page_num}: stripped repeated header")
                    rows = rows[1:]

            all_rows.extend(rows)
            logger.info(f"  p{page_num} [{name}]: {len(rows)} rows")

    # Only merge continuations in the fallback path
    if not used_rules:
        pre = len(all_rows)
        all_rows = _merge_continuation_rows(all_rows)
        if pre != len(all_rows):
            logger.info(f"'{name}': merged {pre - len(all_rows)} continuation row(s)")

    logger.info(
        f"Sub-section '{name}' p{start_page}-{end_page}: "
        f"{len(all_rows)} rows extracted"
    )
    return all_rows
