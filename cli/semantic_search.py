import os

from sentence_transformers import SentenceTransformer
import numpy as np


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = dict()

    def verify_model(self):
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")

    def generate_embedding(self, text):
        if not text or text.isspace():
            raise ValueError("text is empty or contains only whitespace")
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents
        list_of_movie_strings = []
        for document in documents:
            self.document_map[document["id"]] = document
            list_of_movie_strings.append(
                f"{document['title']}: {document['description']}"
            )
        self.embeddings = self.model.encode(
            list_of_movie_strings, show_progress_bar=True
        )
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        query_embedding = self.generate_embedding(query)
        score = []
        for doc_id, embedding in zip(self.document_map.keys(), self.embeddings):
            score.append(
                (
                    cosine_similarity(query_embedding, embedding),
                    self.document_map[doc_id],
                )
            )
        sorted_list = sorted(score, key=lambda item: item[0], reverse=True)
        ret = []
        for i, elem in enumerate(sorted_list[:limit]):
            number = i + 1
            title = f"{elem[1]['title']}"
            desc = f"{elem[1]['description']}"
            score = f"({elem[0]:.4f})"
            ret.append(f"{number}. {title} {score}\n {desc}")
        return "\n\n-----------------------------------\n\n".join(ret)
