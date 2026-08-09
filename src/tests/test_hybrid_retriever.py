from src.loaders.text_loader import TextLoader
from src.chunking.recursive_chunker import RecursiveChunker
from src.embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding
)
from src.vectorstore.chroma_store import ChromaVectorStore
from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.hybrid_retriever import HybridRetriever

def print_results(title, result):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for index, chunk in enumerate(
        result.retrieved_chunks,
        start=1
    ):

        print(f"\n--- Result {index} ---")

        print("ID:", chunk.chunk_id)
        print("Score:", round(chunk.score, 4))

        if "dense_score" in chunk.metadata:
            print(
                "Dense Score:",
                round(
                    chunk.metadata["dense_score"],
                    4
                )
            )

        if "bm25_score" in chunk.metadata:
            print(
                "BM25 Score:",
                round(
                    chunk.metadata["bm25_score"],
                    4
                )
            )

        if "hybrid_score" in chunk.metadata:
            print(
                "Hybrid Score:",
                round(
                    chunk.metadata["hybrid_score"],
                    4
                )
            )

        print("-" * 30)
        print(chunk.text)

def main():

    print("=" * 60)
    print("HYBRID RETRIEVER TEST")
    print("=" * 60)

    # Load document
    loader = TextLoader()
    documents = loader.load("datasets/sample2.txt") 

    print(f"Loaded Documents: {len(documents)}")

    # Chunk document
    chunker = RecursiveChunker(
        chunk_size=150,
        chunk_overlap=30
    )

    chunks = chunker.split(documents)

    print(f"Created Chunks: {len(chunks)}")

    # Initialize embedding model
    print("\nInitializing embedding model...")
    embedding_model = SentenceTransformerEmbedding()

    # Create embeddings
    print("Creating embeddings...")
    embeddings = embedding_model.encode(chunks)

    # Initialize vector store 
    print("Initializing & Populating vector store...")
    vector_store = ChromaVectorStore()
    vector_store.reset()  # Clean slate for the test
    vector_store.add(embeddings)

    # Initialize retrievers
    dense_retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=3
    )

    bm25_retriever = BM25Retriever(
        chunks=chunks,
        top_k=3
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        top_k=3,
        alpha=0.7
    )

    # Query
    query = "self-attention"
    print(f"\nQuery: '{query}'")

    # Run retrieval
    dense_result = dense_retriever.retrieve(query)
    bm25_result = bm25_retriever.retrieve(query)
    hybrid_result = hybrid_retriever.retrieve(query)

    # Display results
    print_results("DENSE RETRIEVAL", dense_result)
    print_results("BM25 RETRIEVAL", bm25_result)
    print_results("HYBRID RETRIEVAL", hybrid_result)

    print("\n" + "=" * 60)
    print("HYBRID RETRIEVER TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()