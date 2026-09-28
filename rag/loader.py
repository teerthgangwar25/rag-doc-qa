from pypdf import PdfReader

def load_pdf(file) -> list[dict]:
    """Return one dict per page: {"page": 1, "text": "..."}"""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i, "text": text})
    if not pages:
        raise ValueError("No extractable text found. Is this a scanned PDF?")
    return pages