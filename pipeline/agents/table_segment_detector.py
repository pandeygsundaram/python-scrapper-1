import asyncio
import json
from pathlib import Path

import google.generativeai as genai

from pipeline.utils.logger import get_logger

logger = get_logger("TableSegmentDetector")

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "table_segment_detector.txt"
_MAX_RETRIES = 3
_BASE_DELAY = 2.0

VALID_TABLE_TYPES = {
    "line_items", "room_recap", "trade_recap",
    "materials_breakdown", "labor_breakdown", "equipment_breakdown",
}
VALID_DOC_TYPES = {"detailed", "recap", "mixed"}


def _validate_subsections(sub_sections: list, start: int, end: int) -> list[dict]:
    """Validate and clamp sub-section page ranges within the parent segment."""
    valid = []
    for s in sub_sections:
        sp = max(start, s.get("start_page", start))
        ep = min(end, s.get("end_page", end))
        if sp > ep:
            continue
        valid.append({
            "name": s.get("name", "Unknown"),
            "start_page": sp,
            "end_page": ep,
        })
    # Fallback: if no valid sub-sections, one covering the full segment
    if not valid:
        valid.append({"name": "Main", "start_page": start, "end_page": end})
    return valid


class TableSegmentDetector:
    """
    Agent 1 — classifies document type, detects all table segments with:
      - table_type
      - column_schema (actual headers from the doc)
      - sub_sections (internal named divisions within each segment)
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name,
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
        )
        self._system_prompt = _PROMPT_PATH.read_text()
        logger.info(f"TableSegmentDetector ready (model={model_name})")

    async def detect(self, full_text: str, total_pages: int) -> dict:
        """
        Returns:
          {
            doc_type: str,
            table_segments: [
              {
                start_page, end_page, table_type, description,
                column_schema: [str, ...],
                sub_sections: [{name, start_page, end_page}, ...]
              }
            ]
          }
        Falls back to a single full-doc line_items segment on failure.
        """
        prompt = f"{self._system_prompt}\n\nDOCUMENT:\n{full_text}"
        logger.info(f"Agent 1 — classifying {total_pages}-page document...")

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await asyncio.to_thread(
                    self.model.generate_content, prompt
                )
                raw = response.candidates[0].content.parts[0].text.strip()
                logger.debug(f"Agent 1 raw:\n{raw[:800]}")

                parsed = json.loads(raw)

                doc_type = parsed.get("doc_type", "detailed")
                if doc_type not in VALID_DOC_TYPES:
                    doc_type = "detailed"

                segments = []
                for s in parsed.get("table_segments", []):
                    if "start_page" not in s or "end_page" not in s:
                        logger.warning(f"Malformed segment skipped: {s}")
                        continue

                    table_type = s.get("table_type", "line_items")
                    if table_type not in VALID_TABLE_TYPES:
                        table_type = "line_items"

                    start = s["start_page"]
                    end = s["end_page"]
                    column_schema = s.get("column_schema", [])
                    sub_sections = _validate_subsections(
                        s.get("sub_sections", []), start, end
                    )

                    segments.append({
                        "start_page": start,
                        "end_page": end,
                        "table_type": table_type,
                        "description": s.get("description", ""),
                        "column_schema": column_schema,
                        "sub_sections": sub_sections,
                    })

                logger.info(
                    f"doc_type={doc_type} | {len(segments)} segment(s) detected:"
                )
                for seg in segments:
                    subs = seg["sub_sections"]
                    logger.info(
                        f"  [{seg['table_type']}] p{seg['start_page']}-{seg['end_page']} "
                        f"| schema={seg['column_schema']} "
                        f"| {len(subs)} sub-section(s)"
                    )
                    for sub in subs:
                        logger.info(
                            f"    └ '{sub['name']}' p{sub['start_page']}-{sub['end_page']}"
                        )

                return {"doc_type": doc_type, "table_segments": segments}

            except Exception as e:
                delay = _BASE_DELAY * (2 ** (attempt - 1))
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        f"Attempt {attempt}/{_MAX_RETRIES} failed: {e} "
                        f"— retrying in {delay:.0f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Agent 1 gave up: {e}")

        fallback = {
            "doc_type": "detailed",
            "table_segments": [{
                "start_page": 1,
                "end_page": total_pages,
                "table_type": "line_items",
                "description": "Fallback — full document",
                "column_schema": ["Description", "Quantity", "Unit Price", "Per", "RC", "Depreciation", "ACV"],
                "sub_sections": [{"name": "Main", "start_page": 1, "end_page": total_pages}],
            }],
        }
        logger.warning("Using fallback detection")
        return fallback
