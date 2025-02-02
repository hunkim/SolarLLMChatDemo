import os
import re
from enum import Enum


class PDFType(Enum):
    SCANNED = "Scanned PDF (lacks embedded fonts and text objects)"
    DIGITAL = "Digital PDF (text-based and font information present)"
    UNCERTAIN = "Uncertain (No strong indicators found)"


def is_ocr_pdf(pdf_path):
    """
    Determine if a PDF file is digital–born (text-based) or scanned (mostly images)
    without relying on any external PDF libraries.

    The function reads the raw PDF file content and decodes it to a string.
    It then uses basic heuristics by searching for:
      - Embedded fonts (via the '/Font' keyword)
      - Text drawing commands (via the 'BT' operator; PDFs typically use 'BT' ... 'ET'
        to delimit text blocks)
      - Image objects (via the '/Subtype /Image' declaration)
      - Keywords (like 'scan', 'ocr', or 'adobe acrobat') which sometimes appear in
        scanned PDFs

    If the PDF contains both font definitions and text commands, and the (roughly)
    extracted text (from between BT and ET markers) sums up to a significant length,
    it is assumed to be digital–born. If the PDF appears dominated by images or has
    keywords suggesting scanning, it is classified as scanned. In other cases, the PDF
    is marked as uncertain.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        PDFType: DIGITAL if the PDF appears text–based; SCANNED if it appears to be a scanned image; 
                 UNCERTAIN if the evidence isn't strong enough.
    """
    try:
        with open(pdf_path, "rb") as f:
            content = f.read().decode("latin-1", errors="replace")
    except Exception:
        # If the file cannot be read or decoded, we return UNCERTAIN.
        return PDFType.UNCERTAIN

    # Count indicators of digital PDFs.
    font_count = len(re.findall(r'/Font\b', content))
    text_command_count = len(re.findall(r'\bBT\b', content))
    
    # Try to extract content from text objects (between 'BT' and 'ET').
    text_segments = re.findall(r'BT\s*(.*?)\s*ET', content, re.DOTALL)
    extracted_text = " ".join(text_segments)
    extracted_text_length = len(extracted_text.strip())

    # Count indications of images that might suggest a scan.
    image_count = len(re.findall(r'/Subtype\s*/Image', content))
    # Look for scanned–related keywords.
    scanned_keyword = bool(re.search(r'(?i)\b(?:scan(?:ned)?|ocr|adobe\s+acrobat)\b', content))

    # Heuristics:
    # 1. A digital PDF will usually have embedded fonts and multiple text drawing commands,
    #    resulting in a reasonable amount of extracted text.
    # 2. A scanned PDF may have few (or zero) font markers and often many image objects.
    if font_count > 0 and text_command_count > 0 and extracted_text_length > 100:
        return PDFType.DIGITAL
    elif image_count > font_count or scanned_keyword:
        return PDFType.SCANNED
    else:
        return PDFType.UNCERTAIN
