from __future__ import annotations

import argparse
import logging

from backend.app.core.config import get_settings
from backend.app.rag.vector_store import QdrantCollectionManager


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or verify Medical-AI Bot Qdrant collection."
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the Qdrant collection.",
    )

    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    settings = get_settings()

    logging.info("Setting up Qdrant collection for Medical-AI Bot.")
    logging.info("Qdrant mode: %s", settings.qdrant_mode)
    logging.info("Collection name: %s", settings.qdrant_collection_name)

    manager = QdrantCollectionManager(settings=settings)

    try:
        manager.create_collection(recreate=args.recreate)
        collection_info = manager.get_collection_info()

        print("\nQDRANT COLLECTION SETUP SUMMARY")
        print("-" * 50)
        print(f"Qdrant mode: {collection_info['qdrant_mode']}")
        print(f"Collection name: {collection_info['collection_name']}")
        print(
            f"Dense vector: {collection_info['dense_vector_name']} "
            f"(size={collection_info['dense_vector_size']}, distance=cosine)"
        )
        print(
            f"Sparse vector: {collection_info['sparse_vector_name']} "
            "(modifier=IDF)"
        )
        print("Payload indexes prepared for:")
        print("  - collection")
        print("  - access_roles")
        print("  - source_document")
        print("  - chunk_type")
        print(f"Points count: {collection_info['points_count']}")
        print(f"Indexed vectors count: {collection_info['indexed_vectors_count']}")
        print(f"Status: {collection_info['status']}")

        logging.info("Qdrant collection setup completed successfully.")
    finally:
        manager.close()


if __name__ == "__main__":
    main()