import numpy as np

from src.evaluation.beir_loader import BEIRDataset
from src.evaluation.benchmark_corpus import BenchmarkCorpus
from src.evaluation.benchmark_embeddings import (
    BenchmarkEmbeddingStore
)
from src.embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding
)


def main():

    print("=" * 60)
    print("FIQA FULL EMBEDDING BUILD")
    print("=" * 60)

    dataset = BEIRDataset(
        name="fiqa"
    )

    corpus, _, _ = dataset.load(
        split="test"
    )

    benchmark_corpus = BenchmarkCorpus(
        dataset_name="fiqa",
        corpus=corpus
    )

    documents = benchmark_corpus.to_documents()

    document_ids = [
        document.id
        for document in documents
    ]

    total_documents = len(
        documents
    )

    print(
        "Total Documents:",
        total_documents
    )

    print(
        "\nInitializing embedding model..."
    )

    embedding_model = (
        SentenceTransformerEmbedding()
    )

    batch_size = 500

    all_embeddings = []

    total_batches = (
        (total_documents + batch_size - 1)
        // batch_size
    )

    for batch_number, start in enumerate(
        range(
            0,
            total_documents,
            batch_size
        ),
        start=1
    ):

        end = min(
            start + batch_size,
            total_documents
        )

        batch_documents = documents[
            start:end
        ]

        print(
            f"Encoding batch "
            f"{batch_number}/{total_batches} "
            f"({start + 1}-{end})"
        )

        results = embedding_model.encode(
            batch_documents
        )

        batch_embeddings = np.asarray(
            [
                result.embedding
                for result in results
            ],
            dtype=np.float32
        )

        all_embeddings.append(
            batch_embeddings
        )

    embeddings = np.vstack(
        all_embeddings
    )

    print(
        "\nFinal embedding shape:",
        embeddings.shape
    )

    store = BenchmarkEmbeddingStore(
        dataset_name="fiqa",
        model_name="all-MiniLM-L6-v2"
    )

    store.save(
        embeddings=embeddings,
        document_ids=document_ids
    )

    print(
        "\nEmbeddings saved:"
    )

    print(
        store.path
    )

    print(
        "\nMetadata saved:"
    )

    print(
        store.metadata_path
    )

    assert (
        embeddings.shape[0]
        ==
        total_documents
    )

    assert (
        embeddings.shape[1]
        == 384
    )

    print("\n" + "=" * 60)
    print(
        "FIQA FULL EMBEDDING BUILD PASSED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()