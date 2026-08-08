import argparse
import math

from loaders import load_stopwords
from preprocessing import preprocess_string
from inverted_index import InvertedIndex


def load_inverted_index():
    ii = InvertedIndex()
    try:
        ii.load()
    except Exception:
        print("cache not found, try 'build' first")
        exit(1)
    return ii


def print_search_result(search_result):
    for result in search_result:
        print(f"{result['id']}. {result['title']}")


def search_movies(query):
    inverted_index = load_inverted_index()
    stopwords = load_stopwords("data/stopwords.txt")
    clean_query = preprocess_string(query, " ".join(stopwords))
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


def idf_command(term):
    inverted_index = load_inverted_index()
    term_token = tokenize_term(term)
    idf = math.log(
        (len(inverted_index.docmap) + 1)
        / (len(inverted_index.get_documents(term_token)) + 1)
    )
    return idf


def tf_command(doc_id, term):
    inverted_index = load_inverted_index()
    term_token = tokenize_term(term)
    return inverted_index.get_tf(doc_id, term_token)


def tfidf_command(doc_id, term):
    return tf_command(doc_id, term) * idf_command(term)


def tokenize_term(term: str) -> str:
    token = preprocess_string(term, load_stopwords("data/stopwords.txt"))
    if len(token) != 1:
        raise Exception("term can only be one word")
    return token[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # search command + query argument
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    # build command
    subparsers.add_parser(
        "build", help="Builds the inverted index and saves it to disk"
    )

    # tf (token frequency) command
    tf_parser = subparsers.add_parser(
        "tf", help="returns the frequency the given term appears"
    )
    tf_parser.add_argument("doc_id", type=str, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to look up")

    # idf command

    idf_parser = subparsers.add_parser(
        "idf", help="returns the inverse document frequency"
    )
    idf_parser.add_argument("term", type=str, help="Term to look up")

    # tfidf command

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="returns the TF-IDF of the given term and the document"
    )
    tfidf_parser.add_argument("doc_id", type=str, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to look up")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_movies(args.query)
            print_search_result(results)
        case "build":
            build_command()

        case "tf":
            print(f"Looking for {args.term} in document {args.doc_id}")
            print(tf_command(int(args.doc_id), args.term))
        case "idf":
            print(f"idf command ran with '{args.term}' as term")
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tfidf = tfidf_command(int(args.doc_id), args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}"
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
