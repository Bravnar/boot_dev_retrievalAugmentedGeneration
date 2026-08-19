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


def search_command(query, limit):
    semantic_search = SemanticSearch()
    movies = load_movies("data/movies.json")
    semantic_search.load_or_create_embeddings(movies["movies"])
    print(semantic_search.search(query, limit))


def chunk_command(text, chunk_size, overlap):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")

    # def chunk_command(text, chunk_size):
    #     print(chunk_size)
    #     split_text = text.split()
    #     chars = len(text)
    #     length_split_text = len(split_text)
    #     chunk_list = []
    #     while length_split_text > 0:
    #         chunk = []
    #         end = chunk_size if length_split_text >= chunk_size else None
    #         print(end)
    #         for word in split_text[:end]:
    #             chunk.append(word)
    #         split_text = split_text[end:] if end else []
    #         length_split_text = len(split_text)
    #         chunk_list.append(" ".join(chunk))

    # print(f"Chunking {chars} characters")
    # for i, c in enumerate(chunk_list):
    #     print(f"{i + 1}. {c}")


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

    search_parser = subparsers.add_parser(
        "search", help="semantically searches for a given query"
    )
    search_parser.add_argument("query", type=str, help="query to search")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="maximum number of results to return"
    )

    chunk_parser = subparsers.add_parser("chunk", help="chunks the given text")
    chunk_parser.add_argument("text", type=str, help="text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="size of chunks needed"
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="number of words shared between consecutive chunks",
    )

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
        case "search":
            search_command(args.query, args.limit)
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
