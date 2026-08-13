import argparse

from loaders import load_movies
from semantic_search import SemanticSearch


def verify_embeddings_command():
    semantic_search = SemanticSearch()
    movies = load_movies("data/movies.json")
    embeddings = semantic_search.load_or_create_embeddings(movies["movies"])
    print(f"Number of docs:  {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def verify_command():
    semantic_search = SemanticSearch()
    semantic_search.verify_model()


def embed_text_command(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text_command(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify model version")

    embed_parser = subparsers.add_parser("embed_text", help="Embeds given text")
    embed_parser.add_argument("text", type=str, help="text to embed")

    subparsers.add_parser("verify_embeddings", help="verifies if embeddings are good")

    embed_query_parser = subparsers.add_parser(
        "embed_query", help="Embeds a given query"
    )
    embed_query_parser.add_argument("query", type=str, help="query to embed")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_command()
        case "embed_text":
            embed_text_command(args.text)
        case "verify_embeddings":
            verify_embeddings_command()
        case "embed_query":
            embed_query_text_command(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
