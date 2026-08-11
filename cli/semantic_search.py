import os

from sentence_transformers import SentenceTransformer
import numpy as np


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
