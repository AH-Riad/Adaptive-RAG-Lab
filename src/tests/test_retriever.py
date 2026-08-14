from src.core import chunk
from src.loaders.text_loader import TextLoader
from src.chunking.recursive_chunker import RecursiveChunker
from src.embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from src.vectorstore.chroma_store import ChromaVectorStore
from src.retrievers.dense_retriever import DenseRetriever

def main():
    print("1. Loading Document...")
    loader = TextLoader()
    documents = loader.load("datasets/sample2.txt") 

    print("2. Chunking Document...")
    chunker = RecursiveChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.split(documents)

    print("3. Initializing Embedding Model...")
    embedding_model = SentenceTransformerEmbedding()

    print("4. Encoding Chunks...")
    embeddings = embedding_model.encode(chunks)

    print("5. Initializing & Populating Vector Store...")
    vector_store = ChromaVectorStore()
    vector_store.reset() # Clean slate for the test
    vector_store.add(embeddings)
    
    print(f"   -> Successfully stored {vector_store.count()} chunks in Chroma.")

    print("\n6. Initializing Dense Retriever...")
    retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=3
    )

    # The actual retrieval test!
    query = "Explain self-attention"
    print(f"\n7. Running Query: '{query}'")
    
    result = retriever.retrieve(query)

    print("\n" + "=" * 60)
    print(f"Top {len(result.retrieved_chunks)} Retrieved Chunks")
    print("=" * 60)

    for i, chunk in enumerate(result.retrieved_chunks):
        print(f"\n--- Result {i + 1} ---")
        print(f"ID: {chunk.chunk_id}")
        print(f"Score (Relevance): {chunk.score}")
        print("-" * 30)
        print(f"{chunk.text}")
        print("=" * 60)

if __name__ == "__main__":
    main()