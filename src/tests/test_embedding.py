from src.loaders.text_loader import TextLoader
from src.chunking.recursive_chunker import RecursiveChunker
from src.embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding

loader = TextLoader()

documents = loader.load("datasets/sample2.txt")

chunker = RecursiveChunker(
    chunk_size=150,
    chunk_overlap=30
)

chunks = chunker.split(documents)

embedding_model = SentenceTransformerEmbedding()

results = embedding_model.encode(chunks)

print("=" * 60)

print(f"Total Embeddings : {len(results)}")

print("=" * 60)

for result in results:

    print()

    print(result.chunk.id)

    print(result.model_name)

    print(len(result.embedding))

    print(result.embedding[:10])

    print("=" * 60)