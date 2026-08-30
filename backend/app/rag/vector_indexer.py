from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import models

from backend.app.core.config import AppSettings
from backend.app.core.exceptions import DatasetNotFoundError, RetrievalError
from backend.app.rag.document_chunker import DocumentChunk
from backend.app.rag.embeddings import EmbeddingService, batch_items
from backend.app.rag.vector_store import QdrantCollectionManager


@dataclass(frozen=True)
class IndexingResult:
    """Summary of Qdrant indexing results."""

    chunks_loaded: int
    points_indexed: int
    collection_points_count: int | None


def load_document_chunks(chunks_path: Path) -> list[DocumentChunk]:
    """Load parsed chunks generated in Phase 5."""

    if not chunks_path.exists():
        raise DatasetNotFoundError(
            f"Document chunks file not found: {chunks_path}. "
            "Please build document chunks first: python -m scripts.build_document_chunks"
        )

    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks_payload = payload.get("chunks", [])

    if not chunks_payload:
        raise DatasetNotFoundError(f"No chunks found inside: {chunks_path}")

    return [DocumentChunk(**chunk) for chunk in chunks_payload]


def create_qdrant_point_id(chunk_id: str) -> str:
    """
    Create deterministic UUID for Qdrant point ID.

    Qdrant supports integer IDs and UUID strings. Our original chunk_id is kept
    in payload, while UUID is used as the actual point ID.
    """

    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def build_point_payload(chunk: DocumentChunk) -> dict[str, Any]:
    """
    Build Qdrant payload.

    RBAC filtering depends on access_roles being stored as payload metadata.
    Source citation also depends on source_document, section_title, and collection.
    """

    metadata = chunk.metadata

    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "chunk_index": chunk.chunk_index,
        "char_count": chunk.char_count,
        "source_path": chunk.source_path,
        "source_document": metadata["source_document"],
        "collection": metadata["collection"],
        "access_roles": metadata["access_roles"],
        "section_title": metadata["section_title"],
        "chunk_type": metadata["chunk_type"],
    }


def build_qdrant_points(
    chunks: list[DocumentChunk],
    dense_vectors: list[list[float]],
    sparse_vectors: list[models.SparseVector],
    settings: AppSettings,
) -> list[models.PointStruct]:
    """Build Qdrant point structures with dense + sparse vectors + payload."""

    if not (len(chunks) == len(dense_vectors) == len(sparse_vectors)):
        raise RetrievalError(
            "Chunk count, dense vector count, and sparse vector count do not match."
        )

    points: list[models.PointStruct] = []

    for chunk, dense_vector, sparse_vector in zip(
        chunks,
        dense_vectors,
        sparse_vectors,
        strict=True,
    ):
        if len(dense_vector) != settings.dense_vector_size:
            raise RetrievalError(
                f"Dense vector size mismatch for chunk {chunk.chunk_id}. "
                f"Expected {settings.dense_vector_size}, got {len(dense_vector)}."
            )

        points.append(
            models.PointStruct(
                id=create_qdrant_point_id(chunk.chunk_id),
                vector={
                    settings.dense_vector_name: dense_vector,
                    settings.sparse_vector_name: sparse_vector,
                },
                payload=build_point_payload(chunk),
            )
        )

    return points


class QdrantChunkIndexer:
    """Index parsed document chunks into Qdrant."""

    def __init__(
        self,
        settings: AppSettings,
        collection_manager: QdrantCollectionManager,
        embedding_service: EmbeddingService,
    ) -> None:
        self._settings = settings
        self._collection_manager = collection_manager
        self._embedding_service = embedding_service

    def index_chunks(
        self,
        chunks: list[DocumentChunk],
        recreate_collection: bool = False,
    ) -> IndexingResult:
        """Generate vectors and upsert chunks into Qdrant."""

        if not chunks:
            raise RetrievalError("No chunks provided for indexing.")

        self._collection_manager.create_collection(recreate=recreate_collection)

        total_indexed = 0

        for chunk_batch in batch_items(chunks, self._settings.embedding_batch_size):
            texts = [chunk.text for chunk in chunk_batch]

            dense_vectors = self._embedding_service.embed_dense(texts)
            sparse_vectors = self._embedding_service.embed_sparse(texts)

            points = build_qdrant_points(
                chunks=chunk_batch,
                dense_vectors=dense_vectors,
                sparse_vectors=sparse_vectors,
                settings=self._settings,
            )

            for point_batch in batch_items(
                points,
                self._settings.vector_upload_batch_size,
            ):
                self._collection_manager.client.upsert(
                    collection_name=self._settings.qdrant_collection_name,
                    points=point_batch,
                )
                total_indexed += len(point_batch)

        collection_info = self._collection_manager.get_collection_info()

        return IndexingResult(
            chunks_loaded=len(chunks),
            points_indexed=total_indexed,
            collection_points_count=collection_info["points_count"],
        )