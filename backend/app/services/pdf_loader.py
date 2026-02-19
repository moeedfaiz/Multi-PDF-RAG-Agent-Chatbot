from pathlib import Path
from typing import List, Tuple
from pypdf import PdfReader

def extract_pdf_text_by_page(pdf_path: Path) -> List[Tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        pages.append((i, text))
    return pages
