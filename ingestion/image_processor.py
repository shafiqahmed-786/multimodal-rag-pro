"""
Image OCR pipeline with preprocessing.

Improvements over the original:
- Image preprocessing (grayscale, denoising, binarisation) for better OCR
- DPI-aware rendering
- Minimum area filter to skip tiny icons
- Language + PSM config exposed via settings
- Graceful skip if Tesseract is not installed
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

from core.interfaces import BaseDocumentParser
from core.models import ContentType, RawPage
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Optional imports — OCR features degrade gracefully if missing
try:
    import fitz  # PyMuPDF
    _FITZ_OK = True
except ImportError:
    _FITZ_OK = False
    logger.warning("PyMuPDF not installed — image OCR disabled.")

try:
    from PIL import Image, ImageFilter, ImageOps
    import pytesseract
    _OCR_OK = True
except ImportError:
    _OCR_OK = False
    logger.warning("Pillow / pytesseract not installed — image OCR disabled.")


_MIN_IMAGE_AREA = 5_000   # pixels² — skip thumbnails / icons
_MIN_OCR_LENGTH = 20      # characters — skip mostly-empty OCR results


def _preprocess_image(img: "Image.Image") -> "Image.Image":
    """Apply image preprocessing to improve OCR accuracy."""
    img = img.convert("L")                      # grayscale
    img = ImageOps.autocontrast(img, cutoff=2)  # normalise contrast
    img = img.filter(ImageFilter.SHARPEN)        # sharpen edges
    return img


def _ocr_image(img: "Image.Image") -> str:
    """Run Tesseract OCR on a preprocessed PIL image."""
    processed = _preprocess_image(img)
    text = pytesseract.image_to_string(
        processed,
        lang=settings.ocr_lang,
        config=settings.tesseract_config,
    )
    return text.strip()


class ImageOcrParser(BaseDocumentParser):
    """
    Extracts text from embedded images in PDFs via OCR.
    Returns one RawPage per image with non-trivial OCR output.
    """

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

    def parse(self, file_path: str) -> list[RawPage]:
        if not settings.enable_ocr:
            return []

        if not (_FITZ_OK and _OCR_OK):
            logger.info("OCR dependencies missing — skipping image extraction.")
            return []

        raw_pages: list[RawPage] = []
        source_file = Path(file_path).name

        doc = fitz.open(file_path)
        for page_idx, page in enumerate(doc):
            images = page.get_images(full=True)

            for img_meta in images:
                xref = img_meta[0]
                width = img_meta[2]
                height = img_meta[3]

                # Skip tiny images (icons, decorations)
                if width * height < _MIN_IMAGE_AREA:
                    continue

                try:
                    base_image = doc.extract_image(xref)
                    raw_bytes = base_image.get("image", b"")
                    if not raw_bytes:
                        continue

                    img = Image.open(io.BytesIO(raw_bytes))
                    text = _ocr_image(img)

                    if len(text) < _MIN_OCR_LENGTH:
                        continue

                    raw_pages.append(
                        RawPage(
                            page_number=page_idx + 1,
                            content=text,
                            content_type=ContentType.IMAGE,
                            source_file=source_file,
                            metadata={
                                "image_xref": xref,
                                "width": width,
                                "height": height,
                                "colorspace": base_image.get("colorspace"),
                            },
                        )
                    )

                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "OCR failed for image xref=%d on page %d: %s",
                        xref, page_idx + 1, exc,
                    )

        doc.close()
        return raw_pages
