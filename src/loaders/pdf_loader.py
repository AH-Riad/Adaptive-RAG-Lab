from pathlib import Path
from pypdf import PdfReader

from src.core import Document
from .base_loader import BaseLoader


class PDFLoader(BaseLoader):

    def load(self, path: str):
        reader = PdfReader(path)

        pages = []

        for page in reader.pages:
            pages.append(page.extract_text())

        text = "\n".join(pages)

        document = Document(
            id=Path(path).stem,
            source=Path(path).name,
            text=text,
            metadata={
                "num_pages": len(reader.pages)
            }
        )

        return [document]