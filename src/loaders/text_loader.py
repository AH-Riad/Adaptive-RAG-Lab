from pathlib import Path

from src.core import Document
from .base_loader import BaseLoader


class TextLoader(BaseLoader):

    def load(self, path: str):
        file_path = Path(path)

        text = file_path.read_text(encoding="utf-8")

        document = Document(
            id=file_path.stem,
            source=file_path.name,
            text=text
        )

        return [document]