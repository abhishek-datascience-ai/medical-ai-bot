from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import models

from backend.app.core.config import AppSettings
from backend.app.core.exceptions import RetrievalError


class EmbeddingService:
    """Generate dense and sparse embeddings for Qdrant indexing/search."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._dense_model = TextEmbedding(
            model_name=settings.dense_embedding_model_name,
        )
        self._sparse_model = SparseTextEmbedding(
            model_name=settings.sparse_embedding_model_name,
        )

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """Generate dense semantic vectors."""

        if not texts:
            return []

        try:
            embeddings = self._dense_model.embed(texts)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as exc:
            raise RetrievalError(f"Failed to generate dense embeddings: {exc}") from exc

    def embed_sparse(self, texts: list[str]) -> list[models.SparseVector]:
        """Generate sparse BM25-style vectors."""

        if not texts:
            return []

        try:
            sparse_embeddings = self._sparse_model.embed(texts)
            return [
                self._to_qdrant_sparse_vector(sparse_embedding)
                for sparse_embedding in sparse_embeddings
            ]
        except Exception as exc:
            raise RetrievalError(f"Failed to generate sparse embeddings: {exc}") from exc

    @staticmethod
    def _to_qdrant_sparse_vector(sparse_embedding: Any) -> models.SparseVector:
        """Convert FastEmbed sparse vector into Qdrant SparseVector."""

        return models.SparseVector(
            indices=[int(index) for index in sparse_embedding.indices.tolist()],
            values=[float(value) for value in sparse_embedding.values.tolist()],
        )


def batch_items(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    """Yield items in fixed-size batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    for start_index in range(0, len(items), batch_size):
        yield items[start_index : start_index + batch_size]