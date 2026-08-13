from src.analyzer.query_analysis import QueryAnalysis

class QueryAnalyzer:

    COMPARISON_WORDS = {
        "compare",
        "comparison",
        "difference",
        "different",
        "versus",
        "vs",
        "better",
        "contrast",
        "contrasting",
        "compared",
        "distinguish",
        "distinction"
    }

    QUESTION_WORDS = {
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "which"
    }

    def analyze(self, query: str) -> dict:

        tokens = query.lower().split()

        token_count = len(tokens)

        contains_comparison = any(
            word in tokens
            for word in self.COMPARISON_WORDS
        )

        contains_question = any(
            word in tokens
            for word in self.QUESTION_WORDS
        )

        contains_numbers = any(
            char.isdigit()
            for char in query
        )

        contains_hyphenated_term = any(
            "-" in token
            for token in tokens
        )

        # Short non-question queries are usually lexical.
        # Hyphenated terms and identifiers are also strong
        # lexical signals.

        lexical_signal = (
            contains_numbers
            or contains_hyphenated_term
            or (
                2 <= token_count <= 5
                and not contains_question
            )
        )

        if contains_comparison:

            query_type = "comparison"

            type_confidence = 0.94

        elif lexical_signal:

            query_type = "lexical"

            type_confidence = 0.90

        elif contains_question:

            query_type = "semantic"

            type_confidence = 0.85

        else:

            query_type = "ambiguous"

            type_confidence = 0.42

        if token_count > 15:

            complexity = "high"
            complexity_confidence = 0.88

        elif token_count < 5:

            complexity = "low"
            complexity_confidence = 0.95

        else:

            complexity = "medium"
            complexity_confidence = 0.80

        analysis = QueryAnalysis(
            query=query,
            token_count=token_count,
            character_count=len(query),
            query_type=query_type,
            contains_numbers=contains_numbers,
            contains_comparison=contains_comparison,
            contains_question_word=contains_question,
            suggested_top_k=5,
            query_type_confidence=type_confidence,
            complexity=complexity,
            complexity_confidence=complexity_confidence
        )

        return analysis.to_dict()