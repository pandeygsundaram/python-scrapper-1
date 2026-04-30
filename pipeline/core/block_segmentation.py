"""
Block segmentation: split raw rows into section headers vs. data rows.

A row is treated as a section header when it:
  - Has 1â3 non-empty cells
  - Contains no currency / decimal values
  - Is short enough to plausibly be a room / section label
"""

import re
from pipeline.utils.logger import get_logger

logger = get_logger("BlockSegmentation")

_NUMERIC_RE = re.compile(r"\$[\d,]+|\d{1,3}(?:,\d{3})*\.\d{2}")
_MAX_HEADER_LEN = 60
_MAX_HEADER_CELLS = 3


def _is_section_header(cells: list[str]) -> bool:
    non_empty = [c for c in cells if c.strip()]
    if not non_empty or len(non_empty) > _MAX_HEADER_CELLS:
        return False
    combined = " ".join(non_empty)
    if len(combined) > _MAX_HEADER_LEN:
        return False
    return not _NUMERIC_RE.search(combined)


def segment_into_blocks(rows: list[dict]) -> list[dict]:
    """
    Tag each row as 'header' or 'data'.

    Returns list of block dicts:
      { type: "header" | "data", cells: [...], page: int }
    """
    blocks: list[dict] = []
    header_count = 0
    data_count = 0

    for row in rows:
        # Skip completely empty rows
        if not any(c.strip() for c in row["cells"]):
            continue

        block_type = "header" if _is_section_header(row["cells"]) else "data"
        if block_type == "header":
            header_count += 1
        else:
            data_count += 1

        blocks.append({
            "type": block_type,
            "cells": row["cells"],
            "page": row["page"],
        })

    logger.info(
        f"Block segmentation complete: {header_count} header(s), {data_count} data row(s)"
    )
    return blocks
