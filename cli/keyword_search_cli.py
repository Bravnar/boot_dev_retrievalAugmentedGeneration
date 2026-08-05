import argparse
import json
import string
from nltk.stem import PorterStemmer


def print_search_result(search_result):
    for result in search_result[:5]:
        print(f"{result[0]}. {result[1]}")


def preprocess_string(token_string, stopwords=None):
    tokens = (
        token_string.lower()
        .translate(str.maketrans("", "", string.punctuation))
        .split()
    )
    if stopwords is not None:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens if t not in stopwords]

    return tokens


def search_movies(file_path, query):
    movie_dict = load_dictionary(file_path)
    search_result = []
    stopwords = load_stopwords("data/stopwords.txt")
    clean_query = preprocess_string(query, " ".join(stopwords))
    print(clean_query)
    for movie in movie_dict["movies"]:
        title_translated = (
            movie["title"].lower().translate(str.maketrans("", "", string.punctuation))
        )
        if any(token in title_translated for token in clean_query):
            search_result.append((movie["id"], movie["title"]))
    return search_result


def load_dictionary(file_path):
    with open(file_path) as f:
        movie_dict = json.load(f)
    return movie_dict


def load_stopwords(file_path):
    stopwords = set()

    with open(file_path) as f:
        for line in f:
            stopwords.update(preprocess_string(line))
    return stopwords


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_movies("data/movies.json", args.query)
            print_search_result(results)
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
