import asyncio
import json
from pathlib import Path

import google.generativeai as genai

from pipeline.utils.logger import get_logger

logger = get_logger("StructuringAgent")

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "agent2_dynamic.txt"
_MAX_RETRIES = 3
_BASE_DELAY = 2.0


class StructuringAgent:
    """
    Agent 2 â type-aware, schema-aware, per-sub-section structuring.

    One LLM call per sub-section. Receives:
      - raw rows from the deterministic parser
      - table_type (from Agent 1)
      - column_schema (actual headers detected by Agent 1)
      - section_name (the sub-section name)

    Never extracts new data â only cleans and structures.
    All sub-sections run in parallel via asyncio.gather in run_pipeline.
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
        self._prompt_template = _PROMPT_PATH.read_text()
        logger.info(f"StructuringAgent ready (model={model_name})")

    async def structure_subsection(
        self,
        blocks: list[dict],
        table_type: str,
        column_schema: list[str],
        section_name: str,
    ) -> dict:
        """
        Structure one sub-section. One LLM call.

        Args:
            blocks        : output of block_segmentation for this sub-section
            table_type    : e.g. "line_items", "trade_recap"
            column_schema : column headers from Agent 1
            section_name  : sub-section name, e.g. "Kitchen 1"
        """
        if not blocks:
            logger.warning(f"[{table_type}/{section_name}] No blocks to structure")
            return {"table_type": table_type, "section": section_name, "items": []}

        raw_text = self._blocks_to_text(blocks)
        schema_str = " | ".join(column_schema) if column_schema else "unknown"

        prompt = (
            self._prompt_template
            .replace("{table_type_here}", table_type)
            .replace("{section_name_here}", section_name)
            .replace("{column_schema_here}", schema_str)
            .replace("{raw_rows_here}", raw_text)
        )

        logger.info(
            f"[{