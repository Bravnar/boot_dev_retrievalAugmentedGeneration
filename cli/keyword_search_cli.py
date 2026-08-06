import argparse
import string

from loaders import load_movies, load_stopwords
from preprocessing import preprocess_string
from inverted_index import InvertedIndex


def print_search_result(search_result):
    for result in search_result[:5]:
        print(f"{result[0]}. {result[1]}")


def search_movies(file_path, query):
    movie_dict = load_movies(file_path)
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


def build_command():
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()
    print(inverted_index.get_documents("merida")[0])
    return inverted_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser(
        "build", help="Builds the inverted index and saves it to disk"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_movies("data/movies.json", args.query)
            print_search_result(results)
        case "build":
            inverted_index = build_command()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
