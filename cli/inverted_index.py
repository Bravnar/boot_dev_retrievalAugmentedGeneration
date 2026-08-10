import math
import os
from collections import Counter

from loaders import load_movies, load_stopwords
from preprocessing import preprocess_string

import pickle

# CONSTANTS

BM25_K1 = 1.5
BM25_B = 0.75
CACHE_DIR = "cache/"


class InvertedIndex:
    def __init__(self):
        self.index = dict()
        self.docmap = dict()
        self.term_frequencies: dict[int, Counter[str]] = dict()
        self.doc_lengths = dict()
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id, text):
        tokens = preprocess_string(text, load_stopwords("data/stopwords.txt"))
        length_tokens = len(tokens)
        self.doc_lengths[doc_id] = length_tokens
        self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            if self.index.get(token):
                self.index[token].add(doc_id)
            else:
                self.index[token] = set([doc_id])
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        total_sum = 0
        for doc in self.doc_lengths:
            total_sum += self.doc_lengths[doc]
        return total_sum / len(self.doc_lengths)

    def get_tf(self, doc_id, term):
        return self.term_frequencies[doc_id][term]

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.get_documents(term))
        bm25_idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return bm25_idf

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        length_norm = (
            1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        )
        raw_tf = self.get_tf(doc_id, term)
        saturated = (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)
        return saturated

    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit):
        tokens = preprocess_string(query, load_stopwords("data/stopwords.txt"))
        scores_dict = dict()
        for doc in self.docmap:
            for token in tokens:
                scores_dict[doc] = scores_dict.get(doc, 0) + self.bm25(doc, token)
        return sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)[
            :limit
        ]

    def get_documents(self, term) -> list[int]:
        index = self.index.get(term)
        if not index:
            return []
        index_list = list(index)
        index_list.sort()
        return index_list

    def build(self):
        movies = load_movies("data/movies.json")
        for m in movies["movies"]:
            self.docmap[m["id"]] = {
                "id": m["id"],
                "title": m["title"],
                "description": m["description"],
            }
            self.__add_document(m["id"], f"{m['title']} {m['description']}")

    def save(self):
        cwd = os.path.abspath(".")
        if not os.path.exists(cwd + "/cache/"):
            os.mkdir(cwd + "/cache/")
        with open(cwd + "/cache/index.pkl", "wb") as index_file:
            pickle.dump(self.index, index_file)
        with open(cwd + "/cache/docmap.pkl", "wb") as docmap_file:
            pickle.dump(self.docmap, docmap_file)
        with open(cwd + "/cache/term_frequencies.pkl", "wb") as tf_file:
            pickle.dump(self.term_frequencies, tf_file)
        with open(self.doc_lengths_path, "wb") as dl_file:
            pickle.dump(self.doc_lengths, dl_file)

    def load(self):
        cwd = os.path.abspath(".")
        if not os.path.exists(cwd + "/cache/"):
            raise Exception("/cache/ path does not exist")
        with open(cwd + "/cache/index.pkl", "rb") as i_f:
            self.index = pickle.load(i_f)
        with open(cwd + "/cache/docmap.pkl", "rb") as d_f:
            self.docmap = pickle.load(d_f)
        with open(cwd + "/cache/term_frequencies.pkl", "rb") as tf_f:
            self.term_frequencies = pickle.load(tf_f)
        with open(self.doc_lengths_path, "rb") as dl_file:
            self.doc_lengths = pickle.load(dl_file)
