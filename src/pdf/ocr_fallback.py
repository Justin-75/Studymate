# src/pdf/ocr_fallback.py
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import re



def needs_ocr(
    pages_text: List[Tuple[int, str]],
    min_chars_per_page: int = 30,
    min_ratio_empty: float = 0.6,
) -> bool:
    """
    Decide whether OCR is needed based on extracted text quality.

    Args:
        pages_text: [(page_no, text_raw)] from normal extraction
        min_chars_per_page: pages with fewer characters than this are considered "empty"
        min_ratio_empty: if >= this fraction of pages are "empty", OCR is likely needed

    Returns:
        True if OCR is likely required, otherwise False.
    """
    if not pages_text:
        return True

    empty = 0
    for _, text in pages_text:
        if len((text or "").strip()) < min_chars_per_page:
            empty += 1

    ratio = empty / max(1, len(pages_text))
    return ratio >= min_ratio_empty


def extract_pages_text_ocr(
    file_path: str,
    dpi: int = 300,
    lang: str = "chi_sim",
    psm: int = 3,
    use_preprocess: bool = False,
    pages: Optional[List[int]] = None,
) -> List[Tuple[int, str]]:
    """
    OCR a PDF page-by-page. pages: 1-based page numbers to OCR (None = every page).

    Implementation:
      - Use PyMuPDF (fitz) to render each page into an image at specified DPI
      - Use pytesseract to OCR the image into text
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError("PyMuPDF is required for OCR rendering. Install: pip install pymupdf") from e

    try:
        import pytesseract
        from pytesseract import TesseractNotFoundError
    except ImportError as e:
        raise ImportError("pytesseract is required for OCR. Install: pip install pytesseract") from e
    _use_env_tesseract(pytesseract)

    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Pillow is required for OCR image handling. Install: pip install Pillow") from e

    # Avoid mixed-language OCR by default.
    lang = "chi_sim"

    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    cv2 = None
    np = None
    if use_preprocess:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            cv2 = None
            np = None

    pages_out: List[Tuple[int, str]] = []
    zoom = dpi / 72.0

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF for OCR: {file_path}. Reason: {e}") from e

    try:
        if getattr(doc, "is_encrypted", False):
            raise ValueError("PDF is encrypted and cannot be OCR-processed without a password.")

        for idx in (range(doc.page_count) if pages is None else [p - 1 for p in pages]):
            page = doc.load_page(idx)

            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            if cv2 is not None and np is not None:
                img = _preprocess_for_ocr(img, cv2, np)

            config = f"--psm {psm}"
            try:
                text = pytesseract.image_to_string(img, lang=lang, config=config)
            except TesseractNotFoundError as e:
                raise RuntimeError(
                    "Tesseract OCR is not installed or not on PATH. "
                    "Install Tesseract and ensure it's accessible, then retry."
                ) from e

            pages_out.append((idx + 1, text or ""))

    finally:
        try:
            doc.close()
        except Exception:
            pass

    return pages_out


def _use_env_tesseract(pytesseract) -> None:
    """
    Use the env's own Tesseract and language data. On Windows conda puts tesseract.exe in Library/bin
    (on PATH only after `conda activate`) and the data in share/tessdata, which tesseract can't find alone.
    """
    prefix = Path(sys.prefix)
    exe = shutil.which("tesseract", path=os.pathsep.join(str(prefix / d) for d in ("Library/bin", "bin")))
    if exe:
        pytesseract.pytesseract.tesseract_cmd = exe
    if (prefix / "share" / "tessdata").is_dir():
        os.environ.setdefault("TESSDATA_PREFIX", str(prefix / "share" / "tessdata"))


def _preprocess_for_ocr(pil_img, cv2, np):
    """
    Light OCR preprocessing:
      - grayscale
      - denoise
      - adaptive threshold
    """
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    return pil_img.__class__.fromarray(th)
