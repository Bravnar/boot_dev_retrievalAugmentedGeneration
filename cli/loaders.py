from preprocessing import preprocess_string
import json


def load_movies(file_path):
    with open(file_path) as f:
        movie_dict = json.load(f)
    return movie_dict


def load_stopwords(file_path):
    stopwords = set()

    with open(file_path) as f:
        for line in f:
            stopwords.update(preprocess_string(line))
    return stopwords
