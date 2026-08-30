from __future__ import annotations

import argparse
import logging
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.rag.embeddings import EmbeddingService
from backend.app.rag.vector_indexer import QdrantChunkIndexer, load_document_chunks
from backend.app.rag.vector_store import QdrantCollectionManager


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate embeddings and index Medical-AI Bot chunks into Qdrant."
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the Qdrant collection before indexing.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of chunks to index for testing. 0 means all chunks.",
    )

    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    settings = get_settings()
    chunks_path = Path(settings.processed_data_dir) / "document_chunks.json"

    logging.info("Starting Qdrant indexing for Medical-AI Bot.")
    logging.info("Chunks path: %s", chunks_path)
    logging.info("Dense model: %s", settings.dense_embedding_model_name)
    logging.info("Sparse model: %s", settings.sparse_embedding_model_name)
    logging.info("Qdrant collection: %s", settings.qdrant_collection_name)

    chunks = load_document_chunks(chunks_path)

    if args.limit > 0:
        chunks = chunks[: args.limit]
        logging.info("Indexing limited chunk count: %s", len(chunks))
    else:
        logging.info("Indexing all chunks: %s", len(chunks))

    collection_manager = QdrantCollectionManager(settings=settings)
    embedding_service = EmbeddingService(settings=settings)
    indexer = QdrantChunkIndexer(
        settings=settings,
        collection_manager=collection_manager,
        embedding_service=embedding_service,
    )

    try:
        result = indexer.index_chunks(
            chunks=chunks,
            recreate_collection=args.recreate,
        )

        print("\nQDRANT INDEXING SUMMARY")
        print("-" * 50)
        print(f"Chunks loaded: {result.chunks_loaded}")
        print(f"Points indexed in this run: {result.points_indexed}")
        print(f"Collection points count: {result.collection_points_count}")
        print(f"Dense vector name: {settings.dense_vector_name}")
        print(f"Sparse vector name: {settings.sparse_vector_name}")
        print(f"Dense embedding model: {settings.dense_embedding_model_name}")
        print(f"Sparse embedding model: {settings.sparse_embedding_model_name}")

        logging.info("Qdrant indexing completed successfully.")
    finally:
        collection_manager.close()


if __name__ == "__main__":
    main()