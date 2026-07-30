from sentence_transformers import SentenceTransformer
from .base_embedding import BaseEmbedding
from .embedding_result import EmbeddingResult

class SentenceTransformerEmbedding(BaseEmbedding):

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2"
    ):

        print(f"Loading {model_name}...")

        self.model = SentenceTransformer(model_name)

        self.model_name = model_name
        
    def encode(self, chunks):

        texts = [chunk.text for chunk in chunks]

        vectors = self.model.encode(texts)

        results = []

        for chunk, vector in zip(chunks, vectors):

            results.append(

                EmbeddingResult(

                    chunk=chunk,

                    embedding=vector.tolist(),

                    model_name=self.model_name

                )

            )

        return results
    
    def encode_query(self, query: str):

        return self.model.encode(query).tolist()