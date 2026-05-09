import os
import uuid
import json
import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Security, Depends
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import dotenv

import cache
import pdf_parser
from smart_extractor import run_smart_extraction
from pipeline.run_pipeline import run_pipeline

dotenv.load_dotenv()

PYTHON_SERVICE_API_KEY = os.environ.get("PYTHON_SERVICE_API_KEY", "")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def require_api_key(key: str = Security(_api_key_header)):
    if not PYTHON_SERVICE_API_KEY:
        raise HTTPException(status_code=500, detail="Service API key not configured")
    if key != PYTHON_SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR = Path(os.getcwd()) / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

TEST_PDFS_DIR = Path(os.getcwd()) / "test-pdfs"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")


# ── Background extraction ─────────────────────────────────────────────────────

async def run_extraction(job_id: str, pdf_path: str):
    try:
        cache.update(job_id, {
            "status": "processing",
            "progress": {"stage": "Loading PDF", "detail": "Reading and parsing document..."}
        })

        pages = await asyncio.to_thread(pdf_parser.extract_pages, pdf_path)

        cache.update(job_id, {
            "progress": {"stage": "Pass 1", "detail": f"{len(pages)} pages — identifying relevant sections..."}
        })

        result = await run_smart_extraction(pages, GEMINI_API_KEY, GEMINI_MODEL)

        cache.update(job_id, {
            "progress": {"stage": "Pass 2", "detail": f"Structuring {len(result['items'])} items..."}
        })

        total_value = sum(item.get("totalPrice", 0) or 0 for item in result["items"])
        source_name = Path(pdf_path).name
        # strip uuid prefix from filename
        if "-" in source_name:
            parts = source_name.split("-", 5)
            if len(parts) == 6:
                source_name = parts[5]

        output = {
            "source": source_name,
            "pageRange": f"1-{len(pages)}",
            "extractedAt": datetime.now(timezone.utc).isoformat(),
            "extractionMode": "smart",
            "docType": result["docType"],
            "relevantSections": result["relevantPages"],
            "items": [
                {
                    "name": item.get("name"),
                    "price": item.get("totalPrice"),
                    "quantity": item.get("quantity"),
                    "unit": item.get("unit"),
                    "unitPrice": item.get("unitPrice"),
                    "page": item.get("page"),
                    "extractionMethod": "ai",
                    "confidence": 0.95,
                }
                for item in result["items"]
            ],
            "totalItems": len(result["items"]),
            "totalValue": total_value,
            "stats": {
                "tableExtracted": 0,
                "aiExtracted": len(result["items"]),
                "duplicatesRemoved": 0,
            },
        }

        cache.save_result(job_id, output)
        cache.save_candidates(job_id, result["candidatesText"])
        cache.update(job_id, {
            "status": "done",
            "completedAt": datetime.now(timezone.utc).isoformat(),
            "totalItems": len(result["items"]),
            "totalValue": total_value,
            "docType": result["docType"],
            "relevantSections": result["relevantPages"],
            "progress": {"stage": "Done", "detail": f"{len(result['items'])} items extracted"},
        })

    except Exception as err:
        cache.update(job_id, {
            "status": "error",
            "error": str(err),
            "progress": {"stage": "Error", "detail": str(err)},
        })


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/extract", dependencies=[Depends(require_api_key)])
async def extract(background_tasks: BackgroundTasks, pdf: UploadFile = File(...)):
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    job_id = str(uuid.uuid4())
    filename = f"{job_id}-{pdf.filename}"
    pdf_path = str(UPLOADS_DIR / filename)

    contents = await pdf.read()
    with open(pdf_path, "wb") as f:
        f.write(contents)

    job = cache.create({
        "id": job_id,
        "status": "pending",
        "fileName": pdf.filename,
        "fileSize": len(contents),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "pdfPath": pdf_path,
        "progress": {"stage": "Queued", "detail": "Waiting to start..."},
    })

    background_tasks.add_task(run_extraction, job_id, pdf_path)

    return {"jobId": job_id, "status": job["status"], "fileName": job["fileName"]}


@app.get("/api/jobs", dependencies=[Depends(require_api_key)])
def list_jobs():
    return cache.list_jobs()


@app.get("/api/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def get_job(job_id: str):
    job = cache.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs/{job_id}/result", dependencies=[Depends(require_api_key)])
def get_result(job_id: str):
    job = cache.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job not complete: {job['status']}")
    result = cache.get_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result


@app.get("/api/jobs/{job_id}/candidates")
def get_candidates(job_id: str):
    job = cache.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job not complete: {job['status']}")
    text = cache.get_candidates(job_id)
    if text is None:
        raise HTTPException(status_code=404, detail="Candidates not found")
    return PlainTextResponse(text)


@app.get("/api/jobs/{job_id}/pdf")
def get_pdf(job_id: str):
    job = cache.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    pdf_path = job.get("pdfPath")
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found")
    return FileResponse(pdf_path, media_type="application/pdf", filename=job["fileName"])


@app.delete("/api/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def delete_job(job_id: str):
    if not cache.delete(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"success": True}


# ── V2: Multi-Agent Multi-Block Pipeline ─────────────────────────────────────

async def run_v2_extraction(job_id: str, pdf_path: str):
    try:
        cache.update(job_id, {
            "status": "processing",
            "progress": {"stage": "Loading PDF", "detail": "Starting v2 pipeline..."},
        })

        output = await run_pipeline(
            pdf_path=pdf_path,
            api_key=GEMINI_API_KEY,
            model_name=GEMINI_MODEL,
            output_dir="pipeline_output",
            debug=True,
        )

        data = output.get("data", {})
        sections = data.get("sections", [])
        items = data.get("items", [])
        all_items = [i for s in sections for i in s.get("items", [])] if sections else items
        total_value = sum(i.get("total") or 0 for i in all_items)

        source_name = Path(pdf_path).name
        if "-" in source_name:
            parts = source_name.split("-", 5)
            if len(parts) == 6:
                source_name = parts[5]

        result = {
            "source": source_name,
            "pageRange": output.get("page_range", ""),
            "extractedAt": output.get("extracted_at"),
            "extractionMode": "v2-multi-agent",
            "pipelineVersion": "v2",
            "segmentsDetected": output.get("segments_detected", 0),
            "confidence": output.get("confidence", 0),
            "data": data,
            "items": [
                {
                    "name": i.get("description"),
                    "price": i.get("total"),
                    "quantity": i.get("quantity"),
                    "unit": i.get("unit"),
                    "unitPrice": i.get("unit_price"),
                    "extractionMethod": "v2-pipeline",
                    "confidence": output.get("confidence", 0),
                }
                for i in all_items
            ],
            "totalItems": len(all_items),
            "totalValue": total_value,
        }

        cache.save_result(job_id, result)
        cache.update(job_id, {
            "status": "done",
            "completedAt": datetime.now(timezone.utc).isoformat(),
            "totalItems": len(all_items),
            "totalValue": total_value,
            "progress": {"stage": "Done", "detail": f"{len(all_items)} items extracted (v2)"},
        })

    except Exception as err:
        cache.update(job_id, {
            "status": "error",
            "error": str(err),
            "progress": {"stage": "Error", "detail": str(err)},
        })


@app.post("/api/v2/extract", dependencies=[Depends(require_api_key)])
async def v2_extract(background_tasks: BackgroundTasks, pdf: UploadFile = File(...)):
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    job_id = str(uuid.uuid4())
    filename = f"{job_id}-{pdf.filename}"
    pdf_path = str(UPLOADS_DIR / filename)

    contents = await pdf.read()
    with open(pdf_path, "wb") as f:
        f.write(contents)

    job = cache.create({
        "id": job_id,
        "status": "pending",
        "fileName": pdf.filename,
        "fileSize": len(contents),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "pdfPath": pdf_path,
        "pipeline": "v2",
        "progress": {"stage": "Queued", "detail": "Waiting to start..."},
    })

    background_tasks.add_task(run_v2_extraction, job_id, pdf_path)
    return {"jobId": job_id, "status": job["status"], "fileName": job["fileName"], "pipeline": "v2"}


# ── V2: Pipeline Inspection Endpoints ────────────────────────────────────────
# These serve per-stage intermediate files for the frontend pipeline viewer.
# All files live in pipeline_output/ named {job_id}-{filename}_{stage}.json

PIPELINE_OUTPUT_DIR = Path(os.getcwd()) / "pipeline_output"


def _find_stage_file(job_id: str, stage_suffix: str) -> Path | None:
    """Find a pipeline intermediate file by job_id and stage suffix."""
    matches = list(PIPELINE_OUTPUT_DIR.glob(f"{job_id}-*{stage_suffix}"))
    return matches[0] if matches else None


def _load_stage(job_id: str, suffix: str) -> dict:
    f = _find_stage_file(job_id, suffix)
    if not f:
        raise HTTPException(status_code=404, detail=f"Stage file not found: {suffix}")
    return json.loads(f.read_text())


@app.get("/api/v2/documents")
def list_documents():
    """
    List all v2 documents grouped by original filename.
    Each document has a list of versions (runs), newest first.
    """
    jobs = cache.list_jobs()
    v2_jobs = [j for j in jobs if j.get("pipeline") == "v2"]

    # Group by original filename
    groups: dict[str, list] = {}
    for job in v2_jobs:
        name = job.get("fileName", "unknown")
        groups.setdefault(name, [])

        final = _find_stage_file(job["id"], "_FINAL.json")
        meta = {}
        if final:
            try:
                data = json.loads(final.read_text())
                meta = {
                    "docType": data.get("doc_type"),
                    "totalLineItems": data.get("total_line_items", 0),
                    "totalValue": data.get("total_value", 0),
                    "segmentsDetected": data.get("segments_detected", 0),
                    "subSectionsProcessed": data.get("sub_sections_processed", 0),
                    "confidence": data.get("confidence", 0),
                    "pageRange": data.get("page_range"),
                }
            except Exception:
                pass
        groups[name].append({**job, **meta})

    # Build versioned response
    result = []
    for filename, versions in groups.items():
        versions_sorted = sorted(versions, key=lambda j: j["createdAt"], reverse=True)
        for i, v in enumerate(versions_sorted):
            v["version"] = len(versions_sorted) - i  # v3, v2, v1 oldest=1
        result.append({
            "fileName": filename,
            "latestJobId": versions_sorted[0]["id"],
            "latestStatus": versions_sorted[0]["status"],
            "versionCount": len(versions_sorted),
            "versions": versions_sorted,
        })

    # Sort by latest activity
    result.sort(key=lambda d: d["versions"][0]["createdAt"], reverse=True)
    return result


@app.get("/api/v2/processing")
def get_processing_status():
    """Return all currently active (pending/processing) jobs with progress."""
    jobs = cache.list_jobs()
    active = [
        j for j in jobs
        if j.get("pipeline") == "v2" and j.get("status") in ("pending", "processing")
    ]
    return {
        "activeCount": len(active),
        "jobs": active,
    }


@app.post("/api/v2/documents/{job_id}/reprocess")
async def reprocess_document(background_tasks: BackgroundTasks, job_id: str):
    """Rerun the pipeline on the same PDF — creates a new version."""
    original = cache.get(job_id)
    if not original:
        raise HTTPException(status_code=404, detail="Job not found")

    pdf_path = original.get("pdfPath")
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Original PDF no longer on disk")

    new_job_id = str(uuid.uuid4())
    # Copy the PDF with the new job id
    original_filename = original.get("fileName", "document.pdf")
    new_filename = f"{new_job_id}-{original_filename}"
    new_pdf_path = str(UPLOADS_DIR / new_filename)
    import shutil
    shutil.copy2(pdf_path, new_pdf_path)

    job = cache.create({
        "id": new_job_id,
        "status": "pending",
        "fileName": original_filename,
        "fileSize": original.get("fileSize", 0),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "pdfPath": new_pdf_path,
        "pipeline": "v2",
        "reprocessedFrom": job_id,
        "progress": {"stage": "Queued", "detail": "Reprocess queued..."},
    })

    background_tasks.add_task(run_v2_extraction, new_job_id, new_pdf_path)
    return {"jobId": new_job_id, "status": "pending", "fileName": original_filename, "version": "new"}


@app.post("/api/v2/reprocess-all")
async def reprocess_all(background_tasks: BackgroundTasks):
    """Rerun the pipeline on every unique document (latest version of each)."""
    jobs = cache.list_jobs()
    v2_jobs = [j for j in jobs if j.get("pipeline") == "v2"]

    # Get latest job per filename
    latest: dict[str, dict] = {}
    for job in v2_jobs:
        name = job.get("fileName", "")
        if name not in latest or job["createdAt"] > latest[name]["createdAt"]:
            latest[name] = job

    queued = []
    import shutil
    for name, job in latest.items():
        pdf_path = job.get("pdfPath")
        if not pdf_path or not os.path.exists(pdf_path):
            continue
        new_job_id = str(uuid.uuid4())
        new_pdf_path = str(UPLOADS_DIR / f"{new_job_id}-{name}")
        shutil.copy2(pdf_path, new_pdf_path)
        cache.create({
            "id": new_job_id,
            "status": "pending",
            "fileName": name,
            "fileSize": job.get("fileSize", 0),
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "pdfPath": new_pdf_path,
            "pipeline": "v2",
            "reprocessedFrom": job["id"],
            "progress": {"stage": "Queued", "detail": "Batch reprocess queued..."},
        })
        background_tasks.add_task(run_v2_extraction, new_job_id, new_pdf_path)
        queued.append({"jobId": new_job_id, "fileName": name})

    return {"queued": len(queued), "jobs": queued}


@app.get("/api/v2/documents/{job_id}/pdf")
def get_v2_pdf(job_id: str):
    """Serve the original PDF file."""
    job = cache.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    pdf_path = job.get("pdfPath")
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(pdf_path, media_type="application/pdf", filename=job["fileName"])


@app.get("/api/v2/documents/{job_id}/stage/text")
def get_stage_text(job_id: str):
    """Stage 1 — raw text extraction per page."""
    return _load_stage(job_id, "_01_text.json")


@app.get("/api/v2/documents/{job_id}/stage/segments")
def get_stage_segments(job_id: str):
    """Stage 2 — Agent 1 output: doc_type, segments, column_schema, sub_sections."""
    return _load_stage(job_id, "_02_segments.json")


@app.get("/api/v2/documents/{job_id}/stage/rows")
def get_stage_rows(job_id: str):
    """Stage 3 — deterministic geometric row extraction per sub-section."""
    return _load_stage(job_id, "_03_rows.json")


@app.get("/api/v2/documents/{job_id}/stage/blocks")
def get_stage_blocks(job_id: str):
    """Stage 4 — block segmentation output (headers vs data rows)."""
    return _load_stage(job_id, "_04_blocks.json")


@app.get("/api/v2/documents/{job_id}/stage/structured")
def get_stage_structured(job_id: str):
    """Stage 5 — Agent 2 output per sub-section."""
    return _load_stage(job_id, "_05_structured.json")


@app.get("/api/v2/documents/{job_id}/stage/final")
def get_stage_final(job_id: str):
    """Final — complete structured output with all table types."""
    return _load_stage(job_id, "_FINAL.json")


@app.get("/api/v2/documents/{job_id}/stage/all")
def get_all_stages(job_id: str):
    """Return all available stages in one call (for initial page load)."""
    stages = {}
    for key, suffix in [
        ("text", "_01_text.json"),
        ("segments", "_02_segments.json"),
        ("rows", "_03_rows.json"),
        ("extraction", "_03b_extraction.json"),
        ("blocks", "_04_blocks.json"),
        ("structured", "_05_structured.json"),
        ("final", "_FINAL.json"),
    ]:
        f = _find_stage_file(job_id, suffix)
        if f:
            try:
                stages[key] = json.loads(f.read_text())
            except Exception:
                stages[key] = None
        else:
            stages[key] = None
    return stages


# ── Test PDFs ─────────────────────────────────────────────────────────────────

@app.get("/api/test-pdfs")
def list_test_pdfs():
    """List all PDFs available in the test-pdfs directory."""
    if not TEST_PDFS_DIR.exists():
        return []
    files = sorted(
        f.name for f in TEST_PDFS_DIR.iterdir()
        if f.suffix.lower() == ".pdf"
    )
    return [{"name": f, "path": str(TEST_PDFS_DIR / f)} for f in files]


@app.post("/api/v2/run-test-pdf")
async def run_test_pdf(background_tasks: BackgroundTasks, body: dict):
    """Kick off the v2 pipeline on a named file from test-pdfs/."""
    filename = body.get("filename", "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="filename required")

    src = TEST_PDFS_DIR / filename
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    job_id = str(uuid.uuid4())
    dest = UPLOADS_DIR / f"{job_id}-{filename}"
    import shutil
    shutil.copy2(src, dest)

    job = cache.create({
        "id": job_id,
        "status": "pending",
        "fileName": filename,
        "fileSize": src.stat().st_size,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "pdfPath": str(dest),
        "pipeline": "v2",
        "progress": {"stage": "Queued", "detail": "Waiting to start..."},
    })

    background_tasks.add_task(run_v2_extraction, job_id, str(dest))
    return {"jobId": job_id, "status": "pending", "fileName": filename, "pipeline": "v2"}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}
