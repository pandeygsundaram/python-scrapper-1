import fitz
import base64

PDF_RENDER_DPI = 150
Y_TOLERANCE = 4
COL_GAP = 10


def is_bold(span: dict) -> bool:
    flags = span.get("flags", 0)
    font = span.get("font", "")
    return bool(flags & 2**4) or "bold" in font.lower()


def extract_pages(file_path: str) -> list[dict]:
    doc = fitz.open(file_path)
    pages = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        all_spans = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    bbox = span["bbox"]
                    all_spans.append({
                        "text": text,
                        "x0": bbox[0],
                        "x1": bbox[2],
                        "y0": bbox[1],
                        "bold": is_bold(span),
                    })

        rows = []
        for span in sorted(all_spans, key=lambda s: s["y0"]):
            placed = False
            for row in rows:
                if abs(span["y0"] - row["y"]) <= Y_TOLERANCE:
                    row["spans"].append(span)
                    placed = True
                    break
            if not placed:
                rows.append({"y": span["y0"], "spans": [span]})

        lines_out = []
        for row in rows:
            sorted_spans = sorted(row["spans"], key=lambda s: s["x0"])
            columns = []
            current_col = []
            prev_x1 = None

            for span in sorted_spans:
                txt = f"**{span['text']}**" if span["bold"] else span["text"]
                if prev_x1 is not None and (span["x0"] - prev_x1) > COL_GAP:
                    if current_col:
                        columns.append(" ".join(current_col))
                    current_col = [txt]
                else:
                    current_col.append(txt)
                prev_x1 = span["x1"]

            if current_col:
                columns.append(" ".join(current_col))
            if columns:
                lines_out.append(" | ".join(columns) if len(columns) > 1 else columns[0])

        pages.append({
            "pageNumber": page_num,
            "text": "\n".join(lines_out),
            "lines": lines_out,
        })

    doc.close()
    return pages


def render_pages(file_path: str, page_numbers: list[int]) -> list[dict]:
    zoom = PDF_RENDER_DPI / 72
    matrix = fitz.Matrix(zoom, zoom)
    doc = fitz.open(file_path)
    total = len(doc)
    result = []

    for page_num in page_numbers:
        if page_num < 1 or page_num > total:
            continue
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=matrix)
        img_bytes = pix.tobytes("png")
        result.append({
            "pageNumber": page_num,
            "image": base64.b64encode(img_bytes).decode("utf-8"),
            "mimeType": "image/png",
            "width": pix.width,
            "height": pix.height,
        })

    doc.close()
    return result
