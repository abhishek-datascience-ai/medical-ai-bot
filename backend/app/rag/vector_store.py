from __future__ import annotations

from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from backend.app.core.config import AppSettings
from backend.app.core.exceptions import ConfigurationError, RetrievalError


class QdrantCollectionManager:
    """Manage Qdrant client connection and collection setup."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._client = self._create_client(settings)

    @property
    def client(self) -> QdrantClient:
        """Return the initialized Qdrant client."""

        return self._client

    @staticmethod
    def _create_client(settings: AppSettings) -> QdrantClient:
        """Create Qdrant client based on configured connection mode."""

        if settings.qdrant_mode == "local":
            Path(settings.qdrant_local_path).mkdir(parents=True, exist_ok=True)
            return QdrantClient(path=str(settings.qdrant_local_path))

        if settings.qdrant_mode == "server":
            return QdrantClient(url=settings.qdrant_url)

        if settings.qdrant_mode == "cloud":
            return QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )

        raise ConfigurationError(f"Unsupported Qdrant mode: {settings.qdrant_mode}")

    def collection_exists(self) -> bool:
        """Check whether the configured Qdrant collection already exists."""

        collection_name = self._settings.qdrant_collection_name

        try:
            self._client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create Qdrant collection with dense and sparse vector slots.

        Dense vector slot is used for semantic retrieval.
        Sparse vector slot is prepared for BM25-style lexical retrieval.
        """

        collection_name = self._settings.qdrant_collection_name

        if recreate and self.collection_exists():
            self._client.delete_collection(collection_name=collection_name)

        if self.collection_exists():
            return

        try:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    self._settings.dense_vector_name: models.VectorParams(
                        size=self._settings.dense_vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    self._settings.sparse_vector_name: models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                },
            )
        except Exception as exc:
            raise RetrievalError(
                f"Failed to create Qdrant collection '{collection_name}': {exc}"
            ) from exc

        self.create_payload_indexes()

    def create_payload_indexes(self) -> None:
        """
        Create payload indexes for fields used in RBAC and source filtering.

        access_roles is the critical field for retrieval-layer RBAC.
        """

        collection_name = self._settings.qdrant_collection_name

        fields_to_index = {
            "collection": models.PayloadSchemaType.KEYWORD,
            "access_roles": models.PayloadSchemaType.KEYWORD,
            "source_document": models.PayloadSchemaType.KEYWORD,
            "chunk_type": models.PayloadSchemaType.KEYWORD,
        }

        for field_name, field_schema in fields_to_index.items():
            try:
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception:
                # Qdrant can raise if an index already exists.
                # This is safe to ignore during repeated local setup.
                continue

    def get_collection_info(self) -> dict[str, Any]:
        """Return compact collection information for verification output."""

        collection_name = self._settings.qdrant_collection_name

        try:
            collection_info = self._client.get_collection(
                collection_name=collection_name,
            )
        except Exception as exc:
            raise RetrievalError(
                f"Could not read Qdrant collection '{collection_name}': {exc}"
            ) from exc

        return {
            "collection_name": collection_name,
            "qdrant_mode": self._settings.qdrant_mode,
            "dense_vector_name": self._settings.dense_vector_name,
            "dense_vector_size": self._settings.dense_vector_size,
            "sparse_vector_name": self._settings.sparse_vector_name,
            "points_count": getattr(collection_info, "points_count", None),
            "indexed_vectors_count": getattr(
                collection_info,
                "indexed_vectors_count",
                None,
            ),
            "status": str(getattr(collection_info, "status", "unknown")),
        }

    def close(self) -> None:
        """Close Qdrant client connection when supported by the client."""

        close_method = getattr(self._client, "close", None)

        if callable(close_method):
            close_method()