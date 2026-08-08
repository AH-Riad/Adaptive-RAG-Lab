class RetrieverRegistry:
    """
    Registry for all retrieval backends.
    """

    def __init__(self):
        self._retrievers = {}

    def register(self, strategy, retriever):
        self._retrievers[strategy] = retriever

    def get(self, strategy):
        if strategy not in self._retrievers:
            raise ValueError(f"No retriever registered for {strategy}")
        return self._retrievers[strategy]