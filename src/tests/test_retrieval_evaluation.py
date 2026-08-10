from src.loaders.text_loader import TextLoader
from src.chunking.recursive_chunker import RecursiveChunker
from src.embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding
)
from src.vectorstore.chroma_store import ChromaVectorStore

from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.hybrid_retriever import HybridRetriever

from src.evaluation.retrieval_evaluator import (
    RetrievalEvaluator
)


def main():

    print("=" * 60)
    print("MULTI-QUERY RETRIEVAL EVALUATION")
    print("=" * 60)

    # Load corpus

    loader = TextLoader()

    documents = loader.load(
        "datasets/sample_retrieval_corpus.txt"
    )

    print(
        f"Loaded Documents: {len(documents)}"
    )

    # Chunk corpus

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

    # Initialize embedding model

    print("\nInitializing embedding model...")

    embedding_model = (
        SentenceTransformerEmbedding()
    )

    # Create embeddings

    print("Creating embeddings...")

    embeddings = embedding_model.encode(
        chunks
    )

    # Initialize vector store

    print(
        "Initializing vector store..."
    )

    vector_store = ChromaVectorStore()

    vector_store.reset()

    vector_store.add(
        embeddings
    )

    # Initialize retrievers

    dense_retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=5
    )

    bm25_retriever = BM25Retriever(
        chunks=chunks,
        top_k=5
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        top_k=5,
        alpha=0.7
    )

    # Create evaluator

    evaluator = RetrievalEvaluator(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        hybrid_retriever=hybrid_retriever
    )

    # Evaluation queries

    queries = [

        {
            "type": "lexical",
            "query": "self-attention"
        },

        {
            "type": "semantic",
            "query": "How does the Transformer process information?"
        },

        {
            "type": "comparison",
            "query": "How are Transformers different from recurrent neural networks?"
        },

        {
            "type": "technical",
            "query": "query key value representations"
        },

        {
            "type": "ambiguous",
            "query": "attention"
        }

    ]

    # Run evaluation

    results = evaluator.evaluate(
        queries
    )

    # Display results

    for result in results:

        print("\n" + "=" * 60)

        print(
            f"Query Type: {result.query_type}"
        )

        print(
            f"Query: {result.query}"
        )

        print(
            "\nDense:"
        )

        print(
            result.dense_ids
        )

        print(
            "\nBM25:"
        )

        print(
            result.bm25_ids
        )

        print(
            "\nHybrid:"
        )

        print(
            result.hybrid_ids
        )

    print("\n" + "=" * 60)
    print("MULTI-QUERY EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()