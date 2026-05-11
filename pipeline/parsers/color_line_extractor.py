"""
Color-anchor line extractor for V3 pipeline.

For each sub-section in a line_items segment:
  1. Detect item-number anchors using the same color/position logic as item_detector.py
  2. For each anchor, collect all text spans between anchor[i].y and anchor[i+1].y
     whose x0 >= anchor.x1 (i.e. to the RIGHT of the item number column)
  3. Concatenate those spans into a single raw_text string — handles multiline items
  4. Output: [{item_number, page, section_name, raw_text}] per sub-section

This gives Agent 2 a lean, clean payload with no table geometry noise.
"""

import re
import fitz  # PyMuPDF

from pipeline.utils.logger import get_logger

logger = get_logger("ColorLineExtractor")

_INT_RE = re.compile(r"^\d+$")
_INT_DOT_STANDALONE = re.compile(r"^(\d+)\.$")
_INT_DOT_PREFIX = re.compile(r"^(\d+)\.\s+\S")

BLUE_COLOR = 0x0000FF
LEFT_ANCHOR_MAX = 45
MIN_ITEM_NUMBER = 1
MAX_ITEM_NUMBER = 9999
LINE_Y_TOL = 3.0        # pts — tolerance to group spans onto the same line
NUMBER_COL_WIDTH = 50   # pts — assumed width of the number column (right-of-anchor buffer)


# ── Item number parsing (mirrors item_detector.py) ────────────────────────────

def _parse_item_number(text: str):
    text = text.strip()
    m = _INT_DOT_PREFIX.match(text)
    if m:
        n = int(m.group(1))
        return n if MIN_ITEM_NUMBER <= n <= MAX_ITEM_NUMBER else None
    m = _INT_RE.match(text) or _INT_DOT_STANDALONE.match(text)
    if m:
        n = int(m.group(0).rstrip("."))
        return n if MIN_ITEM_NUMBER <= n <= MAX_ITEM_NUMBER else None
    return None


def _is_item_number(text: str, x0: float, color: int, left_margin: float) -> bool:
    text = text.strip()
    n = _parse_item_number(text)
    if n is None:
        return False
    # "N. description" format is self-evident
    if _INT_DOT_PREFIX.match(text):
        return True
    if color == BLUE_COLOR:
        return True
    if x0 - left_margin <= LEFT_ANCHOR_MAX:
        return True
    return False


# ── Core extractor ────────────────────────────────────────────────────────────

def extract_lines_for_subsection(
    pdf_path: str,
    start_page: int,
    end_page: int,
    section_name: str,
) -> list[dict]:
    """
    Extract color-anchored line items for a single sub-section.

    Returns:
        [
          {
            "item_number": int,
            "page": int,
            "section_name": str,
            "raw_text": str,   # all text to the right of the item number, including continuations
          },
          ...
        ]
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    # ── Pass 1: collect all anchors and all spans across the page range ───────
    anchors: list[dict] = []   # {item_number, page, y, x1}
    all_spans: list[dict] = [] # {page, y, x0, text}

    for page_num in range(start_page, end_page + 1):
        idx = page_num - 1
        if idx >= total_pages:
            continue
        page = doc[idx]
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        # Calibrate left margin for position-based detection
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
                    if not text:
                        continue
                    bbox = span["bbox"]
                    x0, y, x1 = bbox[0], bbox[1], bbox[2]
                    color = span.get("color", 0)

                    # Check if this span is an item number anchor
                    n = _parse_item_number(text)
                    if n and _is_item_number(text, x0, color, left_margin):
                        anchors.append({
                            "item_number": n,
                            "page": page_num,
                            "y": round(y, 2),
                            "x1": round(x1, 2),
                        })
                    else:
                        all_spans.append({
                            "page": page_num,
                            "y": round(y, 2),
                            "x0": round(x0, 2),
                            "text": text,
                        })

    doc.close()

    if not anchors:
        logger.info(f"[{section_name}] No anchors found — returning empty")
        return []

    # ── De-duplicate anchors (keep first occurrence per item number) ──────────
    seen: dict[int, dict] = {}
    for a in sorted(anchors, key=lambda x: (x["page"], x["y"])):
        if a["item_number"] not in seen:
            seen[a["item_number"]] = a
    anchors = sorted(seen.values(), key=lambda x: (x["page"], x["y"]))

    logger.info(
        f"[{section_name}] {len(anchors)} anchors "
        f"({anchors[0]['item_number']}→{anchors[-1]['item_number']})"
    )

    # ── Pass 2: for each anchor, collect text to the right within y-range ─────
    items: list[dict] = []

    for i, anchor in enumerate(anchors):
        next_anchor = anchors[i + 1] if i + 1 < len(anchors) else None
        anchor_page = anchor["page"]
        anchor_y = anchor["y"]
        # Number column right edge — any span starting past this x is content
        content_x_min = anchor["x1"]

        collected: list[dict] = []

        for span in all_spans:
            sp = span["page"]
            sy = span["y"]
            sx0 = span["x0"]

            # Must be at or below the anchor
            if sp < anchor_page:
                continue
            if sp == anchor_page and sy < anchor_y - LINE_Y_TOL:
                continue

            # Must be before the next anchor
            if next_anchor:
                na_page = next_anchor["page"]
                na_y = next_anchor["y"]
                if sp > na_page:
                    break
                if sp == na_page and sy >= na_y - LINE_Y_TOL:
                    break

            # Must be to the right of the number column
            if sx0 < content_x_min:
                continue

            collected.append(span)

        # Sort by (page, y, x0) and join into raw_text
        collected.sort(key=lambda s: (s["page"], s["y"], s["x0"]))

        # Group into lines, join lines with space, separate line groups with \n
        lines_grouped: list[list[str]] = []
        current_line: list[str] = []
        current_y: float | None = None

        for span in collected:
            if current_y is None or abs(span["y"] - current_y) <= LINE_Y_TOL:
                current_line.append(span["text"])
                current_y = span["y"]
            else:
                if current_line:
                    lines_grouped.append(current_line)
                current_line = [span["text"]]
                current_y = span["y"]
        if current_line:
            lines_grouped.append(current_line)

        raw_text = "\n".join(" ".join(line) for line in lines_grouped).strip()

        items.append({
            "item_number": anchor["item_number"],
            "page": anchor_page,
            "section_name": section_name,
            "raw_text": raw_text,
        })

    logger.info(f"[{section_name}] {len(items)} items extracted")
    return items
