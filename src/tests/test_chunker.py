from src.loaders.text_loader import TextLoader
from src.chunking.recursive_chunker import RecursiveChunker


loader = TextLoader()

documents = loader.load("datasets\\sample2.txt")

chunker = RecursiveChunker(
    chunk_size=150,
    chunk_overlap=30
)

chunks = chunker.split(documents)

print("=" * 60)

print(f"Total Chunks : {len(chunks)}")

print("=" * 60)

for chunk in chunks:

    print()

    print(chunk.id)

    print("-" * 40)

    print(chunk.text)

    print()

    print(chunk.metadata)

    print("=" * 60)