# src/pdf/ocr_fallback.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import re


def normalize_ocr_lang(lang: str | None) -> str:
    """Normalize OCR language selection.

    Mixed-language OCR (e.g., "eng+chi_sim") often harms Chinese recognition quality.
    This project therefore avoids mixed-language runs by default.

    Behavior:
      - If lang is empty/None -> "chi_sim"
      - If multiple languages are provided with '+' or ',' ->
          * choose "chi_sim" if present
          * otherwise choose the first language
      - Returns a single Tesseract language code.
    """
    raw = (lang or "").strip()
    if not raw:
        return "chi_sim"

    parts = [p.strip() for p in re.split(r"[+,]", raw) if p.strip()]
    if not parts:
        return "chi_sim"

    # Prefer Simplified Chinese if present.
    if any(p == "chi_sim" for p in parts):
        return "chi_sim"
    return parts[0]


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
) -> List[Tuple[int, str]]:
    """
    OCR a PDF page-by-page.

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

    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Pillow is required for OCR image handling. Install: pip install Pillow") from e

    # Avoid mixed-language OCR by default.
    lang = normalize_ocr_lang(lang)

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

        for idx in range(doc.page_count):
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
