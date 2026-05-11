import asyncio
import json
import re
from pathlib import Path

from json_repair import repair_json

import google.generativeai as genai

from pipeline.utils.logger import get_logger

logger = get_logger("StructuringAgent")

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_MAX_RETRIES = 3
_BASE_DELAY = 2.0
_NUMERIC_RE = re.compile(r"[\$,]")

_PROMPT_FILES = {
    "line_items":            "line_items_agent.txt",
    "room_recap":            "room_recap_agent.txt",
    "trade_recap":           "trade_recap_agent.txt",
    "materials_breakdown":   "materials_breakdown_agent.txt",
    "labor_breakdown":       "labor_breakdown_agent.txt",
    "equipment_breakdown":   "equipment_breakdown_agent.txt",
    # fallback
    "default":               "line_items_agent.txt",
}

_V3_LINE_ITEMS_PROMPT = "line_items_v3_agent.txt"


class StructuringAgent:
    """
    Agent 2 — type-aware, schema-aware, per-sub-section structuring.

    One LLM call per sub-section, all sub-sections run in parallel via
    asyncio.gather in run_pipeline.
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
        # Load all specialized prompts at init
        self._prompts: dict[str, str] = {}
        for table_type, fname in _PROMPT_FILES.items():
            path = _PROMPTS_DIR / fname
            self._prompts[table_type] = path.read_text()
        self._v3_line_items_prompt = (_PROMPTS_DIR / _V3_LINE_ITEMS_PROMPT).read_text()
        logger.info(f"StructuringAgent ready (model={model_name}, {len(self._prompts)} prompt(s))")

    def _get_prompt_template(self, table_type: str) -> str:
        return self._prompts.get(table_type) or self._prompts["default"]

    def _pre_structured_to_text(
        self,
        pre_structured: list[dict],
        blocks: list[dict],
        context_before: list[dict] | None,
        context_after: list[dict] | None,
    ) -> str:
        """
        Build the prompt data section when we have deterministically detected items.
        Format:
          PRE-DETECTED ITEMS (item number confirmed by color/position in PDF):
          [ITEM 47 p12] Description text | 2 | EA | $513 | $616
            [CONT] continuation row cells if any
          ...
          RAW BLOCKS (for reference/verification):
          [HEADER] ...
          [ROW p12] ...
        """
        lines = ["PRE-DETECTED ITEMS (item boundaries confirmed by PDF geometry):"]
        for item in pre_structured:
            n = item["item_number"]
            pg = item["page"]
            cells = " | ".join(str(c) for c in item.get("cells", []))
            lines.append(f"[ITEM {n} p{pg}] {cells}")
            for extra in item.get("extra_rows", []):
                cont = " | ".join(str(c) for c in extra)
                lines.append(f"  [CONT] {cont}")

        lines.append("")
        lines.append("RAW BLOCKS (use for verification and to fill missing values):")
        if context_before:
            lines.append(self._blocks_to_text(context_before, tag="CONTEXT_BEFORE"))
        lines.append(self._blocks_to_text(blocks))
        if context_after:
            lines.append(self._blocks_to_text(context_after, tag="CONTEXT_AFTER"))

        return "\n".join(lines)

    def _blocks_to_text(self, blocks: list[dict], tag: str = "") -> str:
        lines = []
        for b in blocks:
            cells = " | ".join(b.get("cells", []))
            page = b.get("page", "?")
            if tag:
                prefix = f"[{tag} p{page}]"
            elif b.get("type") == "header":
                prefix = "[HEADER]"
            else:
                prefix = f"[ROW p{page}]"
            lines.append(f"{prefix} {cells}")
        return "\n".join(lines)

    async def structure_subsection(
        self,
        blocks: list[dict],
        table_type: str,
        column_schema: list[str],
        section_name: str,
        number_range: dict | None = None,
        context_before: list[dict] | None = None,
        context_after: list[dict] | None = None,
        pre_structured: list[dict] | None = None,
    ) -> dict:
        """
        Structure one sub-section. One LLM call.

        Args:
            blocks         : blocks for this sub-section
            table_type     : e.g. "line_items"
            column_schema  : column headers from Agent 1
            section_name   : sub-section name, e.g. "Kitchen 1"
            number_range   : {"start": N, "end": M} from Agent 1, or None
            context_before : boundary blocks from the page just before this sub-section
            context_after  : boundary blocks from the page just after this sub-section
        """
        if not blocks and not pre_structured:
            logger.warning(f"[{table_type}/{section_name}] No blocks — returning empty")
            return {"table_type": table_type, "section": section_name, "items": []}

        # If we have deterministically detected items, build an enhanced prompt
        use_pre_structured = bool(pre_structured and table_type == "line_items")

        if use_pre_structured:
            raw_text = self._pre_structured_to_text(pre_structured, blocks, context_before, context_after)
        else:
            parts = []
            if context_before:
                parts.append(self._blocks_to_text(context_before, tag="CONTEXT_BEFORE"))
            parts.append(self._blocks_to_text(blocks))
            if context_after:
                parts.append(self._blocks_to_text(context_after, tag="CONTEXT_AFTER"))
            raw_text = "\n".join(parts)

        schema_str = " | ".join(column_schema) if column_schema else "unknown"

        # Normalize number_range — Agent 1 may return a list [start, end] or a dict
        if isinstance(number_range, list) and len(number_range) == 2:
            number_range = {"start": number_range[0], "end": number_range[1]}

        # Build number_range context block (line_items only)
        if table_type == "line_items" and number_range and isinstance(number_range, dict):
            nr_start = number_range.get("start")
            nr_end = number_range.get("end")
            nr_count = nr_end - nr_start + 1
            if use_pre_structured:
                # FORMAT A: item boundaries are already confirmed by PDF geometry
                number_range_block = (
                    f"ITEM COUNT: This section contains exactly {nr_count} items "
                    f"(item numbers {nr_start} through {nr_end}), confirmed by PDF geometry.\n"
                    f"The [ITEM N] tags in the data are HARD BOUNDARIES — each one is a separate item.\n"
                    f"Your output MUST contain exactly {nr_count} objects. "
                    f"If you return fewer, you are incorrectly merging items."
                )
            else:
                # FORMAT B: guide the model to find item boundaries from text
                number_range_block = (
                    f"ITEM COUNT GUIDANCE: This section is expected to contain {nr_count} items "
                    f"(numbers {nr_start} through {nr_end}).\n"
                    f"Item numbers may be blue (#0000FF) or black — treat both as item boundaries.\n"
                    f"Each new integer in range {nr_start}–{nr_end} at the start of a row = new item.\n"
                    f"Set item_number on every item. Never merge two different item numbers."
                )
            line_items_rules = (
                f"6. Item boundary: each new integer in range {nr_start}–{nr_end} at row start = new item. "
                f"Blue OR black — both are valid item numbers. Never merge two numbered items.\n"
                f"7. Expected count: {nr_count} items (numbers {nr_start}–{nr_end}). "
                f"Fewer means over-merging — split them."
            )
        else:
            number_range_block = ""
            line_items_rules = ""

        template = self._get_prompt_template(table_type)
        prompt = (
            template
            .replace("{section_name_here}", section_name)
            .replace("{column_schema_here}", schema_str)
            .replace("{number_range_block}", number_range_block)
            .replace("{line_items_rules}", line_items_rules)
            .replace("{raw_rows_here}", raw_text)
        )

        logger.info(
            f"[{table_type}/{section_name}] Structuring {len(blocks)} blocks "
            f"({sum(1 for b in blocks if b['type'] == 'data')} data rows)..."
        )

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await asyncio.to_thread(
                    self.model.generate_content, prompt
                )
                # Use candidates[0] directly — response.text blows up when the
                # model returns a bare JSON array because the SDK calls .get() on
                # the parsed list internally.
                if not response.candidates or not response.candidates[0].content.parts:
                    raise ValueError("Empty response from model")
                raw = response.candidates[0].content.parts[0].text.strip()
                try:
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    result = json.loads(repair_json(raw))
                if isinstance(result, list):
                    result = {"items": result}
                items = result.get("items", [])
                result["table_type"] = table_type
                result["section"] = section_name
                logger.info(
                    f"[{table_type}/{section_name}] Done — {len(items)} item(s)"
                )
                return result
            except Exception as e:
                delay = _BASE_DELAY * (2 ** (attempt - 1))
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        f"[{table_type}/{section_name}] Attempt {attempt}/{_MAX_RETRIES} "
                        f"failed: {e} — retrying in {delay:.0f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"[{table_type}/{section_name}] gave up: {e}")

        return {"table_type": table_type, "section": section_name, "items": []}

    async def structure_subsection_v3(
        self,
        blocks: list[dict],
        column_schema: list[str],
        section_name: str,
        number_range: dict | None = None,
        context_before: list[dict] | None = None,
        context_after: list[dict] | None = None,
    ) -> dict:
        """
        V3 line-items structuring — V2 block extraction, lean prompt returning
        only item_number + description.

        Args:
            blocks         : V2 blocks for this sub-section
            column_schema  : column headers from Agent 1
            section_name   : sub-section label
            number_range   : {"start": N, "end": M} from Agent 1
            context_before : boundary blocks before this sub-section
            context_after  : boundary blocks after this sub-section
        """
        if not blocks:
            logger.warning(f"[v3/{section_name}] No blocks — returning empty")
            return {"table_type": "line_items", "section": section_name, "items": []}

        parts = []
        if context_before:
            parts.append(self._blocks_to_text(context_before, tag="CONTEXT_BEFORE"))
        parts.append(self._blocks_to_text(blocks))
        if context_after:
            parts.append(self._blocks_to_text(context_after, tag="CONTEXT_AFTER"))
        raw_text = "\n".join(parts)

        schema_str = " | ".join(column_schema) if column_schema else "unknown"

        if isinstance(number_range, list) and len(number_range) == 2:
            number_range = {"start": number_range[0], "end": number_range[1]}

        if number_range and isinstance(number_range, dict):
            nr_start = number_range.get("start")
            nr_end = number_range.get("end")
            nr_count = nr_end - nr_start + 1
            number_range_block = (
                f"ITEM COUNT GUIDANCE: This section is expected to contain {nr_count} items "
                f"(numbers {nr_start} through {nr_end}).\n"
                f"Each new integer in range {nr_start}–{nr_end} at the start of a row = new item."
            )
            line_items_rules = (
                f"7. Expected item numbers: {nr_start}–{nr_end}. "
                f"Each integer in that range at row start = one item."
            )
        else:
            number_range_block = ""
            line_items_rules = ""

        prompt = (
            self._v3_line_items_prompt
            .replace("{section_name_here}", section_name)
            .replace("{column_schema_here}", schema_str)
            .replace("{number_range_block}", number_range_block)
            .replace("{line_items_rules}", line_items_rules)
            .replace("{raw_rows_here}", raw_text)
        )

        logger.info(
            f"[v3/{section_name}] Structuring {len(blocks)} blocks "
            f"({sum(1 for b in blocks if b['type'] == 'data')} data rows)..."
        )

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await asyncio.to_thread(
                    self.model.generate_content, prompt
                )
                if not response.candidates or not response.candidates[0].content.parts:
                    raise ValueError("Empty response from model")
                raw = response.candidates[0].content.parts[0].text.strip()
                try:
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    result = json.loads(repair_json(raw))
                if isinstance(result, list):
                    result = {"items": result}
                items = result.get("items", [])
                result["table_type"] = "line_items"
                result["section"] = section_name
                logger.info(f"[v3/{section_name}] Done — {len(items)} item(s)")
                return result
            except Exception as e:
                delay = _BASE_DELAY * (2 ** (attempt - 1))
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        f"[v3/{section_name}] Attempt {attempt}/{_MAX_RETRIES} "
                        f"failed: {e} — retrying in {delay:.0f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"[v3/{section_name}] gave up: {e}")

        return {"table_type": "line_items", "section": section_name, "items": []}
