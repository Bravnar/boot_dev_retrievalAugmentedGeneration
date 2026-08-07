import argparse

from loaders import load_stopwords
from preprocessing import preprocess_string
from inverted_index import InvertedIndex


def print_search_result(search_result):
    for result in search_result:
        print(f"{result['id']}. {result['title']}")


def search_movies(query):
    inverted_index = InvertedIndex()
    stopwords = load_stopwords("data/stopwords.txt")
    clean_query = preprocess_string(query, " ".join(stopwords))
    try:
        inverted_index.load()
    except Exception:
        print("cache not found, try 'build' first")
        return
    results = []
    for token in clean_query:
        documents = inverted_index.get_documents(token)
        if not documents:
            continue
        for index in documents:
            results.append(inverted_index.docmap[index])
            if len(results) == 5:
                break
        if len(results) == 5:
            break
    return results


def build_command():
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()


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
            results = search_movies(args.query)
            print_search_result(results)
        case "build":
            build_command()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
