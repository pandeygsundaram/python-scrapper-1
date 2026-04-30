"""
Text extraction with page markers for Agent 1 input.

Returns:
  full_text : single string with === PAGE N === markers between pages
  pages     : list of { page_number, text } dicts
"""

import pdfplumber

from pipeline.utils.logger import get_logger

logger = get_logger("TextExtractor")


def extract_text_with_markers(pdf_path: str) -> tuple[str, list[dict]]:
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            pages.append({
                "page_number": page_num,
                "text": text.strip(),
            })

    logger.info(f"Extracted {len(pages)} pages from {pdf_path}")

    full_text = "\n\n".join(
        f"=== PAGE {p['page_number']} ===\n{p['text']}"
        for p in pages
    )

    return full_text, pages
