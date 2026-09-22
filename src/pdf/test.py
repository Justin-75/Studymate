from __future__ import annotations

from typing import List, Tuple

from src.pdf.extractor import extract_pages_text_pymupdf
from src.pdf.ocr_fallback import needs_ocr, extract_pages_text_ocr
from src.pdf.cleaner import clean_page_text
from src.pdf.chunker import chunk_pages


# =========================
# CONFIG: Put your 2 PDFs here
# =========================
PDF_NO_OCR = "data/Raw PDF/C Problem set.pdf"       # born-digital PDF
PDF_NEEDS_OCR = "data/Raw PDF/OCR.pdf"          # scanned PDF 

MAX_PAGES_TO_SHOW = 5
PREVIEW_CHARS = 350


def _preview_pages(label: str, pages: List[Tuple[int, str]], max_pages: int = 5) -> None:
    print(f"\n==============================")
    print(f"[{label}] FIRST {min(max_pages, len(pages))} PAGES PREVIEW")
    print(f"==============================")

    for page_no, text in pages[:max_pages]:
        print(f"\n--- Page {page_no} (len={len(text)}) ---")
        print(text[:PREVIEW_CHARS].strip() or "[EMPTY]")


def _process_pdf(file_path: str, label: str) -> None:
    print(f"\n\n########################################")
    print(f"Processing: {label}")
    print(f"File: {file_path}")
    print(f"########################################")

    # 1) Normal extraction
    pages_raw = extract_pages_text_pymupdf(file_path)
    print(f"Total pages extracted (raw): {len(pages_raw)}")

    # 2) OCR decision
    ocr_needed = needs_ocr(pages_raw, min_chars_per_page=30, min_ratio_empty=0.6)
    print(f"OCR needed? {ocr_needed}")

    # 3) OCR fallback if needed
    if ocr_needed: #language = chi_sim + eng psm = 3 
        pages_raw = extract_pages_text_ocr(file_path, dpi=250, lang="chi_sim", psm=6, use_preprocess=True)
        print(f"Total pages extracted (OCR): {len(pages_raw)}")

    # Only take first 5 pages for preview + chunking in this test
    pages_first = pages_raw[:MAX_PAGES_TO_SHOW]

    # 4) Cleaning
    pages_clean = [(pno, clean_page_text(txt)) for pno, txt in pages_first]

    # 5) Preview cleaned pages
    _preview_pages(f"{label} (CLEANED)", pages_clean, max_pages=MAX_PAGES_TO_SHOW)

    # 6) Chunking (first 5 pages only)
    doc_id = f"test-{label.lower().replace(' ', '-')}"
    chunks = chunk_pages(doc_id, pages_clean, chunk_size=800, overlap=120)

    print(f"\n[{label}] Total chunks (first {MAX_PAGES_TO_SHOW} pages): {len(chunks)}")
    if chunks:
        print("\nFirst 3 chunks preview:")
        for c in chunks[:3]:
            print(f"- page={c.page_no}, chunk_index={c.chunk_index}, chunk_id={c.chunk_id}, len={len(c.text)}")
            print(f"  text: {c.text[:180].strip()}\n")


def main():
    # PDF that should NOT need OCR
    _process_pdf(PDF_NO_OCR, "NO_OCR_PDF")

    # PDF that SHOULD need OCR (scanned)
    _process_pdf(PDF_NEEDS_OCR, "OCR_PDF")


if __name__ == "__main__":
    main()
