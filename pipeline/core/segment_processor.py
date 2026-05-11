from pipeline.utils.logger import get_logger

logger = get_logger("SegmentProcessor")


def _build_subsections(raw_subs: list | None, seg_start: int, seg_end: int) -> list[dict]:
    """
    Normalise Agent 1 sub-section list.
    - Fills in missing end_page by using the next sub's start_page - 1
      (last sub gets the segment's end_page).
    - Normalises number_range from list → dict if needed.
    """
    if not raw_subs:
        return [{"name": "Main", "start_page": seg_start, "end_page": seg_end, "number_range": None}]

    result = []
    for i, sub in enumerate(raw_subs):
        sub_start = sub.get("start_page", seg_start)

        # end_page: use explicit value, else infer from next sub's start - 1, else seg_end
        if sub.get("end_page"):
            sub_end = sub["end_page"]
        elif i + 1 < len(raw_subs) and raw_subs[i + 1].get("start_page"):
            sub_end = raw_subs[i + 1]["start_page"] - 1
        else:
            sub_end = seg_end

        nr = sub.get("number_range")
        if isinstance(nr, list) and len(nr) == 2:
            nr = {"start": nr[0], "end": nr[1]}

        result.append({
            "name": sub.get("name", "Main"),
            "start_page": sub_start,
            "end_page": sub_end,
            "number_range": nr,
        })

    return result


def build_segments(pages: list[dict], table_segments: list[dict]) -> list[dict]:
    """
    Map Agent 1 output (page ranges + metadata) onto the extracted page data.

    Returns a list of segment dicts, each containing:
      start_page, end_page, pages (list of page dicts),
      table_type, description, column_schema, sub_sections
    """
    page_map = {p["page_number"]: p for p in pages}
    segments: list[dict] = []

    for seg in table_segments:
        start = seg["start_page"]
        end = seg["end_page"]
        seg_pages = [
            page_map[n] for n in range(start, end + 1) if n in page_map
        ]

        if not seg_pages:
            logger.warning(f"Segment {start}-{end}: no matching pages found, skipping")
            continue

        logger.info(f"Segment {start}-{end}: {len(seg_pages)} page(s) mapped")
        segments.append({
            "start_page": start,
            "end_page": end,
            "pages": seg_pages,
            "table_type": seg.get("table_type", "line_items"),
            "description": seg.get("description", ""),
            "column_schema": seg.get("column_schema", []),
            "sub_sections": _build_subsections(
                seg.get("sub_sections"), start, end
            ),
        })

    if not segments:
        logger.warning("No segments built — falling back to full document")
        all_nums = sorted(page_map.keys())
        if all_nums:
            segments.append({
                "start_page": all_nums[0],
                "end_page": all_nums[-1],
                "pages": list(page_map.values()),
                "table_type": "line_items",
                "description": "Full document fallback",
                "column_schema": [],
                "sub_sections": [
                    {"name": "Main", "start_page": all_nums[0], "end_page": all_nums[-1]}
                ],
            })

    return segments
