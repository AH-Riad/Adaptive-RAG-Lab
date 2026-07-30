import chromadb
from typing import List
from src.vectorstore.base_vector_store import BaseVectorStore

class ChromaVectorStore(BaseVectorStore):

    def __init__(
        self,
        collection_name="adaptive_rag"
    ):

        self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
        
    def add(self, embedding_results):

        ids = []

        embeddings = []

        documents = []

        metadatas = []

        for result in embedding_results:

            ids.append(result.chunk.id)

            embeddings.append(result.embedding)

            documents.append(result.chunk.text)

            metadatas.append(result.chunk.metadata)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def search(self, query_embedding, top_k=5):
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return results

    def count(self):
        
        return self.collection.count()

    def reset(self):
        
        current_name = self.collection.name
        
        self.client.delete_collection(current_name)

        self.collection = self.client.create_collection(
            current_name
        )