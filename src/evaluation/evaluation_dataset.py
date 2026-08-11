from src.evaluation.ground_truth import GroundTruthDataset


def build_ground_truth():

    dataset = GroundTruthDataset()

    # Exact lexical concept

    dataset.add(
        query="self-attention",
        query_type="lexical",
        relevant_chunks=[
            "sample_retrieval_corpus_CHUNK_003",
            "sample_retrieval_corpus_CHUNK_004",
            "sample_retrieval_corpus_CHUNK_007",
            "sample_retrieval_corpus_CHUNK_011",
            "sample_retrieval_corpus_CHUNK_013",
            "sample_retrieval_corpus_CHUNK_015"
        ]
    )

    # Semantic Transformer processing query

    dataset.add(
        query="How does the Transformer process information?",
        query_type="semantic",
        relevant_chunks=[
            "sample_retrieval_corpus_CHUNK_002",
            "sample_retrieval_corpus_CHUNK_003",
            "sample_retrieval_corpus_CHUNK_007",
            "sample_retrieval_corpus_CHUNK_013",
            "sample_retrieval_corpus_CHUNK_014"
        ]
    )

    # Transformer versus RNN comparison

    dataset.add(
        query="How are Transformers different from recurrent neural networks?",
        query_type="comparison",
        relevant_chunks=[
            "sample_retrieval_corpus_CHUNK_009",
            "sample_retrieval_corpus_CHUNK_010"
        ]
    )

    # Technical query about QKV

    dataset.add(
        query="query key value representations",
        query_type="technical",
        relevant_chunks=[
            "sample_retrieval_corpus_CHUNK_004"
        ]
    )

    # Attention query

    dataset.add(
        query="attention",
        query_type="ambiguous",
        relevant_chunks=[
            "sample_retrieval_corpus_CHUNK_003",
            "sample_retrieval_corpus_CHUNK_004",
            "sample_retrieval_corpus_CHUNK_007",
            "sample_retrieval_corpus_CHUNK_008",
            "sample_retrieval_corpus_CHUNK_011",
            "sample_retrieval_corpus_CHUNK_013",
            "sample_retrieval_corpus_CHUNK_015"
        ]
    )

    return dataset