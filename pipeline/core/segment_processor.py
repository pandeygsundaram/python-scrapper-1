from pipeline.utils.logger import get_logger

logger = get_logger("SegmentProcessor")


def build_segments(pages: list[dict], table_segments: list[dict]) -> list[dict]:
    """
    Map Agent 1 output (page ranges) onto the extracted page data.

    Returns a list of segment dicts, each containing:
      start_page, end_page, pages (list of page dicts)
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
        })

    if not segments:
        logger.warning("No segments built â falling back to full document")
        all_nums = sorted(page_map.keys())
        if all_nums:
            segments.append({
                "start_page": all_nums[0],
                "end_page": all_nums[-1],
                "pages": list(page_map.values()),
            })

    return segments
