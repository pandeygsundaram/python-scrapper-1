import asyncio
import json
import google.generativeai as genai
from typing import Literal

DocType = Literal["detailed", "recap"]

CHUNK_SIZE = 8
CONCURRENCY = 3

DETAILED_AGENT_SYSTEM = """You are an expert data extraction agent specializing in Xactimate insurance estimate documents.

These documents contain numbered line items where each row has:
  ItemNumber + Description | Quantity | Unit Price | Unit | RC (Replacement Cost) | Depreciation | ACV

Your job is to extract EVERY numbered line item precisely and completely — including items with $0.00 value. A numbered item is a line item regardless of its price.

CRITICAL RULE — totalPrice must be the RC (Replacement Cost) column:
- RC is the dollar amount that appears BEFORE the Depreciation column
- ACV (last column) = RC minus Depreciation — never use ACV as totalPrice
- Example: "26 Sand Floor | 179.67 | $11.44 SF | $2,055.42 | $69.63 | $1,985.79"
  → totalPrice = 2055.42  (RC)  NOT 1985.79 (ACV)
- Example: "5 Drywall | 389.82 | $1.08 SF | $421.00 | $0.00 | $421.00"
  → totalPrice = 421.00 (RC = ACV here because depreciation is $0.00)"""

RECAP_AGENT_SYSTEM = """You are an expert data extraction agent specializing in Xactimate insurance estimate recap/summary documents.

These documents show costs broken down by trade category. Two patterns exist:

PATTERN A — Category with sub-items (bold header + non-bold rows below it):
  **APP - Appliances** | **$2,098.87** | **0.98%**   ← bold subtotal → SKIP (sub-items below will cover it)
  APP - Dishwashers    | $810.05       | 0.38%        ← non-bold sub-item → EXTRACT
  APP - Other          | $317.77       | 0.15%        ← non-bold sub-item → EXTRACT

PATTERN B — Category with NO sub-items (only one bold row, nothing below with the same prefix):
  **DMO - General Demolition** | **$15,892.00** | **7.44%**   ← only entry for DMO → EXTRACT (no sub-items to double-count)
  **EQP - Equipment Rentals**  | **$3,279.00**  | **1.54%**   ← only entry for EQP → EXTRACT

Rule: A bold row is a double-count risk ONLY when non-bold rows with the same trade prefix follow it.
If a bold row is the ONLY row for its trade prefix, extract it — it IS the line item."""


def _build_annotated_text(pages: list[dict]) -> str:
    return "\n\n".join(f"=== PAGE {p['pageNumber']} ===\n{p['text']}" for p in pages)


def _parse_json_array(raw: str) -> list:
    try:
        parsed = json.loads(raw.strip())
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _get_model(api_key: str, model_name: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name,
        generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
    )


async def _with_retry(fn, retries=3, base_delay=2.0, label="request"):
    last_err = None
    for attempt in range(retries):
        try:
            return await fn()
        except Exception as err:
            last_err = err
            if attempt < retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"   🔁 {label} failed (attempt {attempt+1}/{retries}): {err}")
                await asyncio.sleep(delay)
            else:
                print(f"   ❌ {label} gave up after {retries} attempts: {err}")
    raise last_err


async def _run_chunked_extraction(relevant_pages: list[dict], extract_fn, label: str) -> list[dict]:
    chunks = [relevant_pages[i:i + CHUNK_SIZE] for i in range(0, len(relevant_pages), CHUNK_SIZE)]
    print(f"\n🏗  {label} ({len(chunks)} chunk{'s' if len(chunks) > 1 else ''} of ~{CHUNK_SIZE} pages)...")

    all_items = []
    for i in range(0, len(chunks), CONCURRENCY):
        batch = chunks[i:i + CONCURRENCY]

        async def process_chunk(chunk, idx):
            chunk_text = _build_annotated_text(chunk)
            chunk_label = f"Chunk {idx + 1}"
            try:
                return await _with_retry(lambda: extract_fn(chunk_text), 3, 2.0, chunk_label)
            except Exception as err:
                print(f"   ⚠ {chunk_label} giving up: {err}")
                return []

        results = await asyncio.gather(*[process_chunk(chunk, i + bi) for bi, chunk in enumerate(batch)])

        for bi, items in enumerate(results):
            print(f"     Chunk {i + bi + 1}: {len(items)} items")
            all_items.extend(items)

        done = min(i + CONCURRENCY, len(chunks))
        print(f"   ✓ Chunks {i+1}-{done}/{len(chunks)} done → {len(all_items)} total so far")

    return all_items


# ── Classifier ────────────────────────────────────────────────────────────────

async def classify_document(full_text: str, api_key: str, model_name: str) -> DocType:
    model = _get_model(api_key, model_name)
    prompt = f"""You are classifying an insurance/contractor estimate PDF.

Read the document below and decide which type it is:

"detailed" — Contains numbered line items with columns: Description | Quantity | Unit Price | RC | Depreciation | ACV
  Example row: "5 Drywall | 389.82 | $1.08 SF | $421.00 | $0.00 | $421.00"

"recap" — Contains only a category summary with columns: Category | Total Cost | Percentage
  Example row: "APP - Dishwashers | $810.05 | 0.38%"
  No individual numbered items with quantities.

Return ONLY this JSON (no markdown):
{{ "type": "detailed" | "recap" }}

Document:
{full_text}"""

    try:
        response = await asyncio.to_thread(model.generate_content, prompt)
        parsed = json.loads(response.text.strip())
        doc_type: DocType = "recap" if parsed.get("type") == "recap" else "detailed"
        print(f"   📄 Document classified as: {doc_type}")
        return doc_type
    except Exception:
        print("   ⚠ Classification failed, defaulting to detailed")
        return "detailed"


# ── Detailed Agent ────────────────────────────────────────────────────────────

async def detailed_find_pages(full_text: str, api_key: str, model_name: str, total_pages: int) -> list[dict]:
    model = _get_model(api_key, model_name)
    prompt = f"""{DETAILED_AGENT_SYSTEM}

TASK: Identify which pages contain numbered line items (actual billable work items with quantities and prices).

Pages are labeled "=== PAGE N ===".

INCLUDE: pages with numbered rows like "5 Drywall | 389.82 | $1.08 SF | $421.00 | $0.00 | $421.00"
EXCLUDE: cover page, policy info, floor plans, photo pages, recap/summary tables, materials breakdown, labor breakdown, equipment breakdown, depreciation schedules, signature pages

There are usually many room/area sections — include ALL of them.

Return ONLY a JSON array (no markdown):
[{{ "start": <page>, "end": <page>, "description": "<room or section name>" }}]

Document:
{full_text}"""

    try:
        response = await asyncio.to_thread(model.generate_content, prompt)
        parsed = json.loads(response.text.strip())
        return parsed if isinstance(parsed, list) else []
    except Exception:
        print("   ⚠ Detailed agent page identification failed, using all pages")
        return [{"start": 1, "end": total_pages, "description": "All pages"}]


async def detailed_extract_chunk(chunk_text: str, api_key: str, model_name: str) -> list[dict]:
    model = _get_model(api_key, model_name)
    prompt = f"""{DETAILED_AGENT_SYSTEM}

TASK: Extract every numbered line item from the text below.

Pages are marked "=== PAGE N ===". Columns are pipe-separated ( | ).
Column order: ItemNum+Description | Quantity | UnitPrice+Unit | RC | Depreciation | ACV

For each numbered item return:
{{
  "name": "clean description — strip the leading item number and trade codes (INS/FRM/ELE/DMO/etc)",
  "quantity": <number or null>,
  "unit": "<SF|EA|HR|LF|LS|SY|PR|etc or null>",
  "unitPrice": <number or null>,
  "totalPrice": <RC value — the dollar amount BEFORE depreciation — required>,
  "page": <page number>
}}

Rules:
- Extract EVERY numbered item without exception — missing any item is the worst possible outcome
- Items with $0.00 RC MUST still be extracted with totalPrice: 0 — do NOT skip zero-value items
- Continuation/note lines below an item (no item number) are NOT separate items, skip them
- Skip ONLY: un-numbered room/section headers, subtotal rows, page headers/footers, photo labels
- Return ONLY a JSON array, no markdown

Text:
{chunk_text}"""

    response = await asyncio.to_thread(model.generate_content, prompt)
    return _parse_json_array(response.text)


# ── Recap Agent ───────────────────────────────────────────────────────────────

async def recap_find_pages(full_text: str, api_key: str, model_name: str, total_pages: int) -> list[dict]:
    model = _get_model(api_key, model_name)
    prompt = f"""{RECAP_AGENT_SYSTEM}

TASK: Identify which pages contain the category breakdown table (rows with trade categories, dollar amounts, and percentages).

Pages are labeled "=== PAGE N ===".

INCLUDE: pages with rows like "APP - Dishwashers | $810.05 | 0.38%"
EXCLUDE: cover page, floor plans, photo pages, grand total only pages, signature pages

Return ONLY a JSON array (no markdown):
[{{ "start": <page>, "end": <page>, "description": "<section name>" }}]

Document:
{full_text}"""

    try:
        response = await asyncio.to_thread(model.generate_content, prompt)
        parsed = json.loads(response.text.strip())
        return parsed if isinstance(parsed, list) else []
    except Exception:
        print("   ⚠ Recap agent page identification failed, using all pages")
        return [{"start": 1, "end": total_pages, "description": "All pages"}]


async def recap_extract_chunk(chunk_text: str, api_key: str, model_name: str) -> list[dict]:
    model = _get_model(api_key, model_name)
    prompt = f"""{RECAP_AGENT_SYSTEM}

TASK: Extract all non-bold sub-item rows from the category recap below.

Pages are marked "=== PAGE N ===". Columns are pipe-separated ( | ).
Column order: Description | TotalCost | Percentage

Remember: bold rows (**like this**) are category subtotals — SKIP THEM.
Only extract plain (non-bold) rows.

For each non-bold sub-item return:
{{
  "name": "description as-is — keep the trade prefix (APP, CAB, FRM, etc)",
  "quantity": null,
  "unit": null,
  "unitPrice": null,
  "totalPrice": <dollar amount — required>,
  "page": <page number>
}}

Rules:
- Extract every non-bold (plain) row that has a dollar amount and a percentage
- For bold rows: SKIP if non-bold rows with the same trade prefix appear below it. EXTRACT if it is the only row for its trade prefix
- Skip: "Total, all categories" grand total row, tax rows, overhead/profit rows, page headers/footers
- Return ONLY a JSON array, no markdown

Text:
{chunk_text}"""

    response = await asyncio.to_thread(model.generate_content, prompt)
    return _parse_json_array(response.text)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

async def run_smart_extraction(pages: list[dict], api_key: str, model_name: str) -> dict:
    full_text = _build_annotated_text(pages)
    print(f"\n🧠 Smart Extraction Pipeline")
    print(f"   Total pages: {len(pages)} | Text size: {len(full_text)} chars")

    # Step 0: Classify
    print("\n🔍 Step 0: Classifying document...")
    try:
        doc_type = await _with_retry(
            lambda: classify_document(full_text, api_key, model_name),
            3, 2.0, "Classifier"
        )
    except Exception:
        doc_type = "detailed"

    # Step 1: Find relevant pages
    print("\n📋 Step 1: Identifying relevant pages...")
    try:
        if doc_type == "recap":
            relevant_ranges = await _with_retry(
                lambda: recap_find_pages(full_text, api_key, model_name, len(pages)),
                3, 2.0, "Recap agent page identification"
            )
        else:
            relevant_ranges = await _with_retry(
                lambda: detailed_find_pages(full_text, api_key, model_name, len(pages)),
                3, 2.0, "Detailed agent page identification"
            )
    except Exception:
        relevant_ranges = [{"start": 1, "end": len(pages), "description": "All pages"}]

    if not relevant_ranges:
        relevant_ranges = [{"start": 1, "end": len(pages), "description": "All pages"}]

    print(f"   ✓ Found {len(relevant_ranges)} relevant section(s):")
    for r in relevant_ranges:
        print(f"     • Pages {r['start']}-{r['end']}: {r['description']}")

    relevant_page_nums = set()
    for r in relevant_ranges:
        for p in range(r["start"], r["end"] + 1):
            relevant_page_nums.add(p)

    relevant_pages = [p for p in pages if p["pageNumber"] in relevant_page_nums]
    candidates_text = _build_annotated_text(relevant_pages)
    print(f"   ✓ {len(relevant_pages)} pages selected for extraction")

    # Step 2: Extract
    if doc_type == "recap":
        all_items = await _run_chunked_extraction(
            relevant_pages,
            lambda chunk: recap_extract_chunk(chunk, api_key, model_name),
            "Recap Agent — Pass 2"
        )
    else:
        all_items = await _run_chunked_extraction(
            relevant_pages,
            lambda chunk: detailed_extract_chunk(chunk, api_key, model_name),
            "Detailed Agent — Pass 2"
        )

    print(f"\n✅ Done — {len(all_items)} total items extracted")

    return {
        "relevantPages": relevant_ranges,
        "candidatesText": candidates_text,
        "items": all_items,
        "docType": doc_type,
    }
