"""
Deterministic line-item number detector using PyMuPDF.

Strategy (in priority order):
  1. Blue spans  — text is an integer AND color == #0000FF
  2. Left-anchor — text is an integer AND x0 is within LEFT_ANCHOR_MAX pts of the
                   page's leftmost text x (catches black-numbered docs)

After detecting item-number anchors we:
  - Sort them by (page, y)
  - Group pdfplumber rows into per-item buckets (rows between anchor[i].y and anchor[i+1].y)
  - Return a list of pre-structured items with item_number + merged cells

Section splitting:
  - detect_subtotals() finds "subtotal" lines in the PDF (section-end markers)
  - split_by_subtotals() divides anchor list at those positions → one group per section
"""

import re
import fitz  # PyMuPDF

from pipeline.utils.logger import get_logger

logger = get_logger("ItemDetector")

_INT_RE = re.compile(r"^\d+$")
_INT_DOT_STANDALONE = re.compile(r"^(\d+)\.$")   # span is exactly "N."
_INT_DOT_PREFIX = re.compile(r"^(\d+)\.\s+\S")   # span starts with "N. description"

BLUE_COLOR = 0x0000FF
LEFT_ANCHOR_MAX = 45   # pts from leftmost text x for black bare-integer docs
MIN_ITEM_NUMBER = 1
MAX_ITEM_NUMBER = 9999

_SUBTOTAL_RE = re.compile(r"sub[\s\-]?total", re.IGNORECASE)


def detect_subtotals(pdf_path: str, start_page: int, end_page: int) -> list[dict]:
    """
    Find lines containing "subtotal" (case-insensitive) — these mark section ends.
    Returns [{page, y, text}] sorted by (page, y).
    """
    markers: list[dict] = []
    doc = fitz.open(pdf_path)
    total = len(doc)

    for page_num in range(start_page, end_page + 1):
        idx = page_num - 1
        if idx >= total:
            continue
        page = doc[idx]
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            for line in b.get("lines", []):
                text = " ".join(s["text"] for s in line.get("spans", []))
                if _SUBTOTAL_RE.search(text):
                    markers.append({
                        "page": page_num,
                        "y": round(line["bbox"][1], 2),
                        "text": text.strip(),
                    })

    doc.close()
    result = sorted(markers, key=lambda x: (x["page"], x["y"]))
    if result:
        logger.info(
            f"p{start_page}-{end_page}: {len(result)} subtotal(s) detected"
        )
    return result


def split_by_subtotals(
    anchors: list[dict],
    subtotals: list[dict],
) -> list[list[dict]]:
    """
    Partition anchors into groups using subtotal y-positions as section-end markers.

    Items at or before a subtotal line belong to that section's group.
    Leftover anchors after the last subtotal form the final group.
    """
    if not anchors:
        return []
    if not subtotals:
        return [list(anchors)]

    groups: list[list[dict]] = []
    remaining = list(anchors)

    for st in subtotals:
        sp, sy = st["page"], st["y"]
        before = [a for a in remaining if (a["page"], a["y"]) < (sp, sy)]
        remaining = [a for a in remaining if not (a["page"], a["y"]) < (sp, sy)]
        if before:
            groups.append(before)

    if remaining:
        groups.append(remaining)

    return groups


def _parse_item_number(text: str) -> int | None:
    """
    Return the integer item number if text looks like an item anchor, else None.
    Handles three formats:
      1. Bare integer:         "47"       (blue or left-anchored)
      2. Standalone dot:       "47."      (blue or left-anchored)
      3. Dot-prefix line:      "47. FRM…" (the N. itself IS the signal regardless of position)
    """
    text = text.strip()
    # Format 3 — "N. description" — the pattern alone is distinctive, no position needed
    m = _INT_DOT_PREFIX.match(text)
    if m:
        n = int(m.group(1))
        return n if MIN_ITEM_NUMBER <= n <= MAX_ITEM_NUMBER else None
    # Formats 1 & 2 — bare integer or standalone "N." — need color/position check
    m = _INT_RE.match(text) or _INT_DOT_STANDALONE.match(text)
    if m:
        n = int(m.group(0).rstrip("."))
        return n if MIN_ITEM_NUMBER <= n <= MAX_ITEM_NUMBER else None
    return None


def _needs_position_check(text: str) -> bool:
    """True if this text needs color/position verification (not self-evident)."""
    text = text.strip()
    return bool(_INT_RE.match(text) or _INT_DOT_STANDALONE.match(text))


def _is_item_number(text: str, x0: float, color: int, left_margin: float) -> bool:
    n = _parse_item_number(text)
    if n is None:
        return False
    if not _needs_position_check(text):
        return True   # "N. description" format — self-evident
    if color == BLUE_COLOR:
        return True
    if x0 - left_margin <= LEFT_ANCHOR_MAX:
        return True
    return False


def detect_item_anchors(pdf_path: str, start_page: int, end_page: int) -> list[dict]:
    """
    Return a list of item-number anchors found on pages start_page..end_page.
    Each anchor: {item_number, page, y, x0, x1}
    Sorted by (page, y).
    """
    anchors: list[dict] = []
    doc = fitz.open(pdf_path)
    total = len(doc)

    for page_num in range(start_page, end_page + 1):
        idx = page_num - 1
        if idx >= total:
            continue
        page = doc[idx]
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        # Find leftmost text x on this page to calibrate left_margin
        xs = [
            span["bbox"][0]
            for b in blocks if b.get("type") == 0
            for line in b.get("lines", [])
            for span in line.get("spans", [])
            if span["text"].strip()
        ]
        left_margin = min(xs) if xs else 0.0

        for b in blocks:
            if b.get("type") != 0:
                continue
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    color = span.get("color", 0)
                    bbox = span["bbox"]
                    x0 = bbox[0]
                    y = bbox[1]  # top of span
                    parsed_n = _parse_item_number(text)
                    if parsed_n and _is_item_number(text, x0, color, left_margin):
                        anchors.append({
                            "item_number": parsed_n,
                            "page": page_num,
                            "y": round(y, 2),
                            "x0": round(x0, 2),
                            "x1": round(bbox[2], 2),
                        })

    doc.close()

    # De-duplicate: keep first occurrence of each item number (by page, y)
    seen: dict[int, dict] = {}
    for a in sorted(anchors, key=lambda x: (x["page"], x["y"])):
        n = a["item_number"]
        if n not in seen:
            seen[n] = a
    anchors = sorted(seen.values(), key=lambda x: (x["page"], x["y"]))

    # Within a sub-section the numbers must be monotonically increasing.
    # Filter items that jump backward by more than a small tolerance.
    # (Only effective when scanning a narrow page range — not a full document.)
    page_span = end_page - start_page + 1
    if page_span <= 10:  # subsection-level scan: apply monotonic filter
        filtered: list[dict] = []
        prev_n = 0
        for a in anchors:
            n = a["item_number"]
            if n >= prev_n - 2:
                filtered.append(a)
                prev_n = max(prev_n, n)
            else:
                logger.debug(f"  Dropping out-of-sequence anchor #{n} (prev={prev_n})")
        anchors = filtered

    if anchors:
        logger.info(
            f"p{start_page}-{end_page}: {len(anchors)} item anchors "
            f"({anchors[0]['item_number']}→{anchors[-1]['item_number']})"
        )
    else:
        logger.info(f"p{start_page}-{end_page}: no item anchors found")

    return anchors


def build_pre_structured(
    anchors: list[dict],
    rows: list[dict],
    start_page: int,
    end_page: int,
) -> list[dict]:
    """
    Group pdfplumber rows into per-item buckets using anchor y-positions.

    For each anchor[i], its rows are all rows on the same page with
    y_top >= anchor[i].y  AND  y_top < anchor[i+1].y  (or end of page).
    Rows on later pages before the next anchor also belong to item[i].

    Returns list of:
      {item_number, page, cells: [first_row_cells], extra_rows: [[cells], ...]}
    """
    if not anchors:
        return []

    # Filter rows to the sub-section page range
    sub_rows = [r for r in rows if start_page <= r["page"] <= end_page]
    sub_rows.sort(key=lambda r: (r["page"], r.get("y_top", 0)))

    items: list[dict] = []

    for i, anchor in enumerate(anchors):
        next_anchor = anchors[i + 1] if i + 1 < len(anchors) else None

        item_rows = []
        for row in sub_rows:
            rp = row["page"]
            ry = row.get("y_top", 0)

            # Row must be at or below this anchor
            if rp < anchor["page"]:
                continue
            if rp == anchor["page"] and ry < anchor["y"]:
                continue

            # Row must be before next anchor
            if next_anchor:
                if rp > next_anchor["page"]:
                    break
                if rp == next_anchor["page"] and ry >= next_anchor["y"]:
                    break

            item_rows.append(row)

        if not item_rows:
            # No rows found — still create the item with empty cells
            items.append({
                "item_number": anchor["item_number"],
                "page": anchor["page"],
                "cells": [],
                "extra_rows": [],
            })
            continue

        first = item_rows[0]
        items.append({
            "item_number": anchor["item_number"],
            "page": anchor["page"],
            "cells": first.get("cells", []),
            "extra_rows": [r.get("cells", []) for r in item_rows[1:]],
        })

    return items
