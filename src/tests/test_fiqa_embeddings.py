import numpy as np
from src.evaluation.beir_loader import BEIRDataset
from src.evaluation.benchmark_corpus import BenchmarkCorpus
from src.evaluation.benchmark_embeddings import BenchmarkEmbeddingStore
from src.embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding

def main():

    print("=" * 60)
    print("FIQA BATCH EMBEDDING TEST")
    print("=" * 60)

    dataset = BEIRDataset(
        name="fiqa"
    )

    corpus, queries, qrels = (
        dataset.load(
            split="test"
        )
    )

    benchmark_corpus = BenchmarkCorpus(
        dataset_name="fiqa",
        corpus=corpus
    )

    documents = (
        benchmark_corpus.to_documents()
    )

    documents = documents[:500]

    document_ids = [
        doc.id
        for doc in documents
    ]

    print(
        "Documents:",
        len(documents)
    )

    print(
        "\nInitializing embedding model..."
    )

    embedding_model = (
        SentenceTransformerEmbedding()
    )

    print(
        "Encoding benchmark documents..."
    )

    results = (
        embedding_model.encode(
            documents
        )
    )

    # Extract the raw embedding vectors from the EmbeddingResult objects
    raw_embeddings = [
        result.embedding
        for result in results
    ]

    # Convert the raw vectors into a NumPy array
    embeddings = np.array(raw_embeddings)

    print(
        "Embedding shape:",
        embeddings.shape
    )

    store = BenchmarkEmbeddingStore(
        dataset_name="fiqa_test500",
        model_name="all-MiniLM-L6-v2"
    )

    store.save(
        embeddings=embeddings,
        document_ids=document_ids
    )

    print(
        "\nSaved embeddings:"
    )

    print(
        store.path
    )

    print(
        "\nReloading from disk..."
    )

    loaded_embeddings, metadata = (
        store.load()
    )

    print(
        "Loaded shape:",
        loaded_embeddings.shape
    )

    print(
        "Stored document count:",
        metadata["count"]
    )

    print(
        "Embedding dimension:",
        metadata[
            "embedding_dimension"
        ]
    )

    assert loaded_embeddings.shape == (
        500,
        embeddings.shape[1]
    )

    assert (
        metadata["count"]
        ==
        500
    )

    assert (
        metadata["document_ids"][0]
        ==
        document_ids[0]
    )

    print("\n" + "=" * 60)
    print(
        "FIQA BATCH EMBEDDING TEST PASSED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()