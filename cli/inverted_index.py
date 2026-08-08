import os
from collections import Counter

from loaders import load_movies, load_stopwords
from preprocessing import preprocess_string

import pickle


class InvertedIndex:
    def __init__(self):
        self.index = dict()
        self.docmap = dict()
        self.term_frequencies: dict[int, Counter[str]] = dict()

    def __add_document(self, doc_id, text):
        tokens = preprocess_string(text, load_stopwords("data/stopwords.txt"))
        self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            if self.index.get(token):
                self.index[token].add(doc_id)
            else:
                self.index[token] = set([doc_id])
            self.term_frequencies[doc_id][token] += 1

    def get_tf(self, doc_id, term):
        return self.term_frequencies[doc_id][term]

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
