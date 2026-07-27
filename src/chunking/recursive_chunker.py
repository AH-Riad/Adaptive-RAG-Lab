from langchain_text_splitters import RecursiveCharacterTextSplitter

from core import Document, Chunk
from .base_chunker import BaseChunker


class RecursiveChunker(BaseChunker):
    """
    Splits documents using LangChain's RecursiveCharacterTextSplitter
    and converts them into our custom Chunk objects.
    """

    def __init__(self,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def split(self, documents):
        """
        Split Document objects into Chunk objects.
        """

        all_chunks = []

        for document in documents:

            texts = self.splitter.split_text(document.text)

            for index, text in enumerate(texts):

                chunk = Chunk(
                    id=f"{document.id}_CHUNK_{index + 1:03}",
                    document_id=document.id,
                    text=text,
                    metadata={
                        "source": document.source,
                        "chunk_index": index + 1,
                        "chunk_size": len(text)
                    }
                )

                all_chunks.append(chunk)

        return all_chunks