import pytesseract
from pdf2image import convert_from_path

def ocr_pdf(pdf_path: str) -> str:
    """
    Uses OCR to extract text from scanned PDFs.
    Returns all text as a single string.
    """
    pages = convert_from_path(pdf_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)
    return text
