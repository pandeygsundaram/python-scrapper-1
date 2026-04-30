# PDF Extraction Service — Pipeline Journal

**Project:** Xactimate Insurance Estimate PDF Extraction System
**Service:** `pdf-service` (Python/FastAPI backend)
**Stack:** FastAPI · PyMuPDF · pdfplumber · Gemini 2.5 Pro
**Last updated:** 2026-04-30

---

## What This Project Does

Extracts structured line-item data from Xactimate insurance/contractor estimate PDFs. These are digitally generated (not scanned) PDFs from estimating software, containing tables with columns like:

```
Description | Quantity | Unit Price | Per | RC | Depreciation | ACV
```

The goal: take a PDF, return clean JSON with every line item — description, quantity, unit, unit price, total.

Two document types exist:
- **Detailed** — numbered line items grouped by room/section (most common)
- **Recap** — category summary only (trade codes + dollar amounts + percentages)

---

## Architecture Overview

```
CLIENT
  └─→ FastAPI (port 8000)
        ├─→ V1: smart_extractor.py   (two-pass, Gemini-heavy)
        └─→ V2: pipeline/            (multi-agent, deterministic core)
```

---

## Evolution: How We Got Here

### Stage 1 — Node.js + pdfjs-dist (First attempt)

Started in Node.js using `pdfjs-dist` for flat text extraction, feeding everything into a single Gemini call.

**Problems:**
- Flat text lost all column alignment
- Single Gemini call on full document hit token limits
- Only extracted ~50 items from documents with 900+
- No retry logic — one API error = total failure

**Result:** Abandoned Node.js. Moved to Python where PDF libraries are significantly better.

---

### Stage 2 — Two-Pass Pipeline (`smart_extractor.py`)

Introduced the two-pass approach still used in V1:

**Pass 1 (Page Identification):** Feed full document text to Gemini → get back which pages contain tables → filter down to relevant pages only.

**Pass 2 (Chunked Extraction):** Split relevant pages into 8-page chunks → extract each chunk in parallel (3 concurrent) → merge results.

**Config:**
| Parameter | Value |
|-----------|-------|
| Chunk size | 8 pages |
| Concurrency | 3 chunks |
| Model | gemini-2.5-pro |

**Why chunking:** Full documents can be 40–80 pages. Gemini context isn't the issue — accuracy degrades over long inputs. Smaller chunks = higher extraction accuracy.

---

### Stage 3 — Retry Logic (The 50→927 Items Fix)

Adding exponential backoff retry was the single biggest quality improvement.

**The bug:** On large documents, 1–2 chunks would silently fail with API errors. The rest succeeded. Result: 50 items extracted from a document with 927. The missing ~880 were in the failed chunks — no error was raised, just empty results merged in.

**Fix:** 3 retry attempts per chunk with 2s/4s/8s backoff. Now all chunks complete.

```python
retries = 3
base_delay = 2.0  # seconds, doubles each attempt
```

**Result:** 50 → 927 items. This fix alone made the system production-viable.

---

### Stage 4 — Python FastAPI + PyMuPDF Bold Detection (`pdf_parser.py`)

Replaced Node.js text extraction with PyMuPDF (`fitz`). Key improvement: **bold text detection**.

Xactimate PDFs use bold formatting to indicate summary/subtotal rows. Detecting bold lets us:
- Mark bold rows in the extracted text (`**text**` markdown)
- Tell the AI which rows are subtotals to skip in recap documents

**Extraction logic:**
- Y_TOLERANCE = 4pt — groups text spans into rows
- COL_GAP = 10pt — splits row into columns
- Bold check: `flags & 2**4` OR `"bold" in font.lower()`

**Output per page:** `{ pageNumber, text, lines }` where text is pipe-separated columns.

---

### Stage 5 — Document Type Routing (Classifier)

Added a pre-pass classifier that determines whether a document is `detailed` or `recap` before extraction. This was critical because:

- Detailed documents: extract every numbered line item, totalPrice = RC column (NOT ACV)
- Recap documents: extract non-bold category rows, skip bold subtotals unless they're the only row for that trade prefix

**The RC vs ACV distinction** is the most important extraction rule. RC (Replacement Cost) is the column before Depreciation. ACV = RC - Depreciation. We always want RC.

```
"26 Sand Floor | 179.67 | $11.44 SF | $2,055.42 | $69.63 | $1,985.79"
→ totalPrice = 2055.42  ← RC (correct)
→ NOT 1985.79           ← ACV (wrong)
```

---

### Stage 6 — Vision Pipeline (Explored, Not Deployed)

Explored rendering PDF pages to images and passing them directly to Gemini's vision API. Added `/render` endpoint in `pdf_parser.py` that returns base64 PNG per page.

**Decision:** Not deployed. Text extraction is more reliable, faster, and cheaper for digitally-generated PDFs. Vision is a fallback for scanned/handwritten documents — not needed here.

The `/render` endpoint remains in the code for future use.

---

### Stage 7 — V2: Multi-Agent Multi-Block Pipeline (`pipeline/`)

The current architecture. Motivation: V1's single-pass chunking loses table structure — it doesn't know where tables start/end, doesn't handle multi-page tables, and loses sub-section grouping (room names, categories).

**Core principle: LLM for structure detection only, pdfplumber for actual extraction.**

```
Stage 1  Text extraction          pdfplumber — deterministic
Stage 2  Agent 1 (Gemini)         Detects table segments + sub-sections
Stage 3  Segment grouping         Deterministic — maps pages to segments
Stage 4  Table reconstruction     pdfplumber geometry — horizontal lines + X/Y coords
Stage 5  Block segmentation       Deterministic — tags rows as header vs data
Stage 6  Agent 2 (Gemini)         Structures raw rows per sub-section (parallel)
Stage 7  Output                   JSON + optional debug CSV
```

---

## V2 Deep Dive

### Agent 1 — `TableSegmentDetector`

**Input:** Full document text with `=== PAGE N ===` markers

**Output:**
```json
{
  "doc_type": "detailed",
  "table_segments": [
    {
      "start_page": 3,
      "end_page": 18,
      "table_type": "line_items",
      "column_schema": ["Description", "Quantity", "Unit Price", "Per", "RC", "Depreciation", "ACV"],
      "sub_sections": [
        { "name": "Kitchen 1", "start_page": 3, "end_page": 7 },
        { "name": "Master Bedroom", "start_page": 8, "end_page": 12 }
      ]
    }
  ]
}
```

**Table types:** `line_items`, `room_recap`, `trade_recap`, `materials_breakdown`, `labor_breakdown`, `equipment_breakdown`

**Why sub-sections matter:** A 40-page document might have 15 rooms. Agent 2 runs one LLM call per sub-section (not per full segment). This gives the AI a focused, small context → higher accuracy, no cross-room confusion.

**Retry:** 3 attempts, 2s/4s/8s. Falls back to full-doc single segment on total failure.

---

### Table Reconstructor — `pipeline/parsers/table_reconstructor.py`

The deterministic core. No LLM.

**Row detection strategy (priority order):**
1. **Horizontal rule lines** (authoritative) — pdfplumber detects horizontal lines on the page. Pairs of consecutive lines define row bands. Words whose vertical center falls in a band → that row.
2. **Y-position grouping** (fallback) — when no rule lines, words within ROW_Y_TOL=3pt of each other → same row.

**Column reconstruction:** Sort words by X coordinate. Gap > COL_GAP=12pt → new column cell.

**Handles:**
- Multi-page tables (processes sub-section page range as a unit)
- Repeated headers (detected and tagged)
- Multi-line descriptions (merged in block segmentation step)

---

### Agent 2 — `StructuringAgent`

**One LLM call per sub-section, all run in `asyncio.gather` (parallel).**

Takes raw rows from the deterministic parser, cleans and normalizes them into structured JSON. It never extracts new data — only organizes what's already there.

Output schema by table type:
- `line_items` → `{ items: [{ description, quantity, unit, unit_price, total }] }`
- `room_recap` → `{ rooms: [...] }`
- `trade_recap` → `{ trades: [...] }`

Prompt is templated with `{table_type_here}`, `{section_name_here}`, `{column_schema_here}`, `{raw_rows_here}` so every sub-section gets a precisely targeted prompt.

---

## API Endpoints

### V1
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/extract` | Upload PDF, start V1 job |
| GET | `/api/jobs` | List all jobs |
| GET | `/api/jobs/{id}` | Status + progress |
| GET | `/api/jobs/{id}/result` | Final result |
| GET | `/api/jobs/{id}/candidates` | Raw text used for extraction |
| GET | `/api/jobs/{id}/pdf` | Serve original PDF |
| DELETE | `/api/jobs/{id}` | Delete job |

### V2
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/v2/extract` | Upload PDF, start V2 job |
| GET | `/api/v2/documents` | All docs with version history |
| GET | `/api/v2/processing` | Active jobs |
| POST | `/api/v2/documents/{id}/reprocess` | Re-run → new version |
| POST | `/api/v2/reprocess-all` | Re-run all docs |
| GET | `/api/v2/documents/{id}/stage/text` | Stage 1 raw text |
| GET | `/api/v2/documents/{id}/stage/segments` | Stage 2 Agent 1 output |
| GET | `/api/v2/documents/{id}/stage/rows` | Stage 3 geometric rows |
| GET | `/api/v2/documents/{id}/stage/blocks` | Stage 4 block segmentation |
| GET | `/api/v2/documents/{id}/stage/structured` | Stage 5 Agent 2 output |
| GET | `/api/v2/documents/{id}/stage/final` | Final JSON |
| GET | `/api/v2/documents/{id}/stage/all` | All stages in one call |

---

## Pipeline Output Files

Every V2 job saves intermediate files to `pipeline_output/`:

```
{job_id}-{filename}_01_text.json       ← raw text per page
{job_id}-{filename}_02_segments.json   ← Agent 1 output
{job_id}-{filename}_03_rows.json       ← geometric rows per sub-section
{job_id}-{filename}_04_blocks.json     ← blocks (header/data tagged)
{job_id}-{filename}_05_structured.json ← Agent 2 output per sub-section
{job_id}-{filename}_FINAL.json         ← complete merged output
```

These feed the frontend's stage viewer — each stage endpoint in the API reads the corresponding file.

---

## Versioning

Every time a document is re-processed (`/reprocess` or `/reprocess-all`), a new job is created. The `/api/v2/documents` endpoint groups all jobs by original filename and returns them sorted newest-first with version numbers (v1 = oldest, v3 = newest).

The frontend uses this for the version toggle in the sidebar.

---

## Key Config

```bash
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-pro   # default, overridable via env
```

| Parameter | Value | Where |
|-----------|-------|-------|
| Chunk size (V1) | 8 pages | `smart_extractor.py` |
| Concurrency (V1) | 3 | `smart_extractor.py` |
| Retry attempts | 3 | Both V1 and V2 agents |
| Retry base delay | 2s (2/4/8s) | Both V1 and V2 agents |
| Y tolerance (pdf_parser) | 4pt | `pdf_parser.py` |
| Col gap (pdf_parser) | 10pt | `pdf_parser.py` |
| Min rule width (V2) | 80pt | `table_reconstructor.py` |
| Row Y tolerance (V2) | 3pt | `table_reconstructor.py` |
| Col gap (V2) | 12pt | `table_reconstructor.py` |

---

## Folder Structure

```
pdf-service/
├── main.py                                  ← FastAPI app (all routes)
├── smart_extractor.py                       ← V1 two-pass extraction pipeline
├── pdf_parser.py                            ← PyMuPDF page extractor + bold detection
├── cache.py                                 ← In-memory job store
├── .env                                     ← GEMINI_API_KEY, GEMINI_MODEL
├── requirements.txt
├── uploads/                                 ← Uploaded PDFs (uuid-prefixed)
├── pipeline_output/                         ← Per-job stage files
├── test_pdfs/                               ← 10 Xactimate sample PDFs
└── pipeline/
    ├── run_pipeline.py                      ← V2 orchestrator (7 stages)
    ├── agents/
    │   ├── table_segment_detector.py        ← Agent 1 (Gemini)
    │   └── structuring_agent.py             ← Agent 2 (Gemini, per sub-section)
    ├── parsers/
    │   ├── text_extractor.py                ← pdfplumber text + PAGE markers
    │   └── table_reconstructor.py           ← Geometric row extraction
    ├── core/
    │   ├── segment_processor.py             ← Agent 1 output → page groups
    │   └── block_segmentation.py            ← Tag rows as header vs data
    ├── utils/
    │   └── logger.py                        ← Named logger wrapper
    └── prompts/
        ├── table_segment_detector.txt       ← Agent 1 system prompt
        └── structuring_agent.txt            ← Agent 2 system prompt (templated)
```

---

## Current Status & What's Next

**Working:**
- V1 pipeline fully operational — classifier + two-pass + retry
- V2 pipeline complete — all 7 stages implemented
- All stage inspection API endpoints live
- Versioning system implemented
- Reprocess-all endpoint for batch re-runs

**Pending integration:**
- `pdf-viewer` React frontend (in `python-pdf-folder/pdf-viewer/`) needs to be wired to the V2 stage endpoints
- Frontend has: `api.ts`, `App.tsx`, `store.ts`, `Navbar.tsx`, `PDFViewer.tsx`, `Sidebar.tsx`, `Stage1.tsx` through `Stage6.tsx`
- Each stage component should call the corresponding `/api/v2/documents/{id}/stage/*` endpoint

**Known issues with V1 (why V2 was built):**
- Loses table structure across page boundaries
- No sub-section grouping (all items in one flat list)
- Can't handle multiple table types in one document
- Chunk boundaries can split a single logical row
