# src/pdf/extractor.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


def extract_pages_text_pymupdf(file_path: str) -> List[Tuple[int, str]]:
    """
    Extract PDF text page-by-page using PyMuPDF (fitz).

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of (page_no, text_raw) tuples where page_no is 1-based.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If PyMuPDF is not installed.
        ValueError: If the PDF is encrypted (password not handled here).
        RuntimeError: If the PDF cannot be opened for any other reason.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is not installed. Install it with: pip install pymupdf"
        ) from e

    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    pages: List[Tuple[int, str]] = []

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {file_path}. Reason: {e}") from e

    try:
        # Encrypted PDFs require a password; we keep it simple for Phase 1.
        if getattr(doc, "is_encrypted", False):
            raise ValueError("PDF is encrypted and cannot be processed without a password.")

        for idx in range(doc.page_count):
            page = doc.load_page(idx)

            # 'sort=True' improves reading order in many PDFs, but not all versions support it.
            try:
                text = page.get_text("text", sort=True)
            except TypeError:
                text = page.get_text("text")

            pages.append((idx + 1, text or ""))
    finally:
        try:
            doc.close()
        except Exception:
            pass

    return pages

"""
Simple explanation (what this code does):

1. Validates the file path = Checks the PDF exists and is a file. If not, it raises FileNotFoundError.
2. Uses PyMuPDF to open the PDF
Imports fitz (PyMuPDF) inside the function.
If PyMuPDF isn’t installed, it raises a clear ImportError.

3. Rejects encrypted PDFs
If the PDF is encrypted, it raises ValueError.
Extracts text per page

4. Loops over all pages using doc.page_count.

Extracts text with page.get_text("text", sort=True) when possible:
sort=True can improve text reading order.
If the installed PyMuPDF version doesn’t support it, it falls back to page.get_text("text").

5. Returns [(page_no, text_raw), ...]
Page numbers are 1-based so citations match human page numbering.
Ensures text is always a string ("" if PyMuPDF returns None).
"""