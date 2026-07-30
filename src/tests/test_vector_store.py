from src.loaders.text_loader import TextLoader
from src.chunking.recursive_chunker import RecursiveChunker
from src.embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from src.vectorstore.chroma_store import ChromaVectorStore

loader = TextLoader()

documents = loader.load("datasets/sample2.txt")

chunker = RecursiveChunker(
    chunk_size=150,
    chunk_overlap=30
)
chunks = chunker.split(documents)

embedding_model = SentenceTransformerEmbedding()

embeddings = embedding_model.encode(chunks)

store = ChromaVectorStore()

store.reset()

store.add(embeddings)
print(store.count())
# Assuming you have a list of chunks ready to embed
print(f"Total chunks created by chunker: {len(chunks)}")

# ... then later ...
print(f"Total chunks in vector store: {store.count()}")
print(len(documents))
print(len(chunks))
print(len(embeddings))
print(store.collection.get()["ids"])