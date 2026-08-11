from src.loaders.text_loader import TextLoader
from src.chunking.recursive_chunker import RecursiveChunker


def main():

    print("=" * 60)
    print("CORPUS CHUNK INSPECTION")
    print("=" * 60)

    loader = TextLoader()

    documents = loader.load(
        "datasets/sample_retrieval_corpus.txt"
    )

    chunker = RecursiveChunker(
        chunk_size=150,
        chunk_overlap=30
    )

    chunks = chunker.split(documents)

    print(
        f"\nTotal Chunks: {len(chunks)}"
    )

    for chunk in chunks:

        print("\n" + "-" * 60)

        print(
            "ID:",
            chunk.id
        )

        print(
            "Text:"
        )

        print(
            chunk.text
        )


if __name__ == "__main__":
    main()