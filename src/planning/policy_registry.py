from planning.policies.confidence_policy import ConfidencePolicy
from src.planning.policies.chunk_policy import ChunkPolicy
from src.planning.policies.retrieval_policy import RetrievalPolicy
from src.planning.policies.topk_policy import TopKPolicy


class PolicyRegistry:

    @staticmethod
    def build():

        return [

            RetrievalPolicy(),

            TopKPolicy(),

            ChunkPolicy(),
            ConfidencePolicy()


        ]
