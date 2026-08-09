from src.loaders.text_loader import TextLoader
from src.chunking.recursive_chunker import RecursiveChunker
from src.retrievers.bm25_retriever import BM25Retriever


def main():

    print("=" * 60)
    print("BM25 RETRIEVER TEST")
    print("=" * 60)

    # 1. Load document

    loader = TextLoader(
    )

    documents = loader.load("datasets\\sample2.txt")


    print(
        f"Loaded Documents: {len(documents)}"
    )

    # 2. Chunk document

    chunker = RecursiveChunker(
        chunk_size=150,
        chunk_overlap=30
    )

    chunks = chunker.split(
        documents
    )

    print(
        f"Created Chunks: {len(chunks)}"
    )

    # 3. Initialize BM25

    retriever = BM25Retriever(
        chunks=chunks,
        top_k=3
    )

    # 4. Query

    query = "self-attention"

    print(
        f"\nQuery: '{query}'"
    )

    result = retriever.retrieve(
        query
    )

    # 5. Display results

    print("\n" + "=" * 60)
    print("BM25 RESULTS")
    print("=" * 60)

    for index, chunk in enumerate(
        result.retrieved_chunks,
        start=1
    ):

        print(
            f"\n--- Result {index} ---"
        )

        print(
            "ID:",
            chunk.chunk_id
        )

        print(
            "Normalized Score:",
            round(chunk.score, 4)
        )

        print(
            "Raw BM25 Score:",
            round(
                chunk.metadata[
                    "raw_bm25_score"
                ],
                4
            )
        )

        print("-" * 30)

        print(chunk.text)

    print("\n" + "=" * 60)
    print("BM25 TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()