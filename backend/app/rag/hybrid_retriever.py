from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import models

from backend.app.core.config import AppSettings
from backend.app.core.exceptions import RetrievalError, RoleAccessError
from backend.app.core.roles import UserRole, get_accessible_collections
from backend.app.rag.embeddings import EmbeddingService
from backend.app.rag.vector_store import QdrantCollectionManager


@dataclass(frozen=True)
class RetrievedChunk:
    """Represents one RBAC-filtered retrieval result from Qdrant."""

    chunk_id: str
    text: str
    score: float
    source_document: str
    section_title: str
    collection: str
    chunk_type: str
    access_roles: list[str]


class HybridRetriever:
    """
    Perform RBAC-filtered hybrid retrieval from Qdrant.

    Retrieval uses:
    - dense vector search for semantic similarity
    - sparse BM25-style search for keyword/term matching
    - Qdrant fusion to combine both result sets
    - access_roles payload filter to enforce RBAC before results are returned
    """

    def __init__(
        self,
        settings: AppSettings,
        collection_manager: QdrantCollectionManager,
        embedding_service: EmbeddingService,
    ) -> None:
        self._settings = settings
        self._collection_manager = collection_manager
        self._embedding_service = embedding_service

    def retrieve(
        self,
        question: str,
        role: UserRole,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve role-authorized chunks for a user question."""

        cleaned_question = question.strip()

        if not cleaned_question:
            raise RetrievalError("Question cannot be empty.")

        final_limit = limit or self._settings.dense_top_k

        if final_limit <= 0:
            raise RetrievalError("Retrieval limit must be greater than zero.")

        dense_query_vector = self._embedding_service.embed_dense([cleaned_question])[0]
        sparse_query_vector = self._embedding_service.embed_sparse([cleaned_question])[0]

        rbac_filter = self._build_role_filter(role)

        try:
            response = self._collection_manager.client.query_points(
                collection_name=self._settings.qdrant_collection_name,
                prefetch=[
                    models.Prefetch(
                        query=sparse_query_vector,
                        using=self._settings.sparse_vector_name,
                        filter=rbac_filter,
                        limit=self._settings.dense_top_k,
                    ),
                    models.Prefetch(
                        query=dense_query_vector,
                        using=self._settings.dense_vector_name,
                        filter=rbac_filter,
                        limit=self._settings.dense_top_k,
                    ),
                ],
                query=self._build_rrf_query(),
                query_filter=rbac_filter,
                limit=final_limit,
                with_payload=True,
            )
        except Exception as exc:
            raise RetrievalError(f"Hybrid retrieval failed: {exc}") from exc

        points = self._extract_points(response)
        retrieved_chunks = [self._point_to_retrieved_chunk(point) for point in points]

        self._validate_results_against_role(
            retrieved_chunks=retrieved_chunks,
            role=role,
        )

        return retrieved_chunks

    @staticmethod
    def _build_role_filter(role: UserRole) -> models.Filter:
        """
        Build Qdrant payload filter for retrieval-layer RBAC.

        This filter is intentionally strict:
        1. access_roles must contain the authenticated role
        2. collection must be one of the collections allowed for that role

        Both conditions are applied inside Qdrant before retrieval results are returned.
        """

        allowed_collections = [
            collection.value for collection in get_accessible_collections(role)
        ]

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="access_roles",
                    match=models.MatchValue(value=role.value),
                ),
                models.FieldCondition(
                    key="collection",
                    match=models.MatchAny(any=allowed_collections),
                ),
            ]
        )

    @staticmethod
    def _build_rrf_query() -> Any:
        """
        Build Reciprocal Rank Fusion query.

        Qdrant client versions may expose RRF through RrfQuery or FusionQuery.
        This compatibility method keeps the code stable across minor versions.
        """

        rrf_query = getattr(models, "RrfQuery", None)
        rrf = getattr(models, "Rrf", None)

        if rrf_query is not None and rrf is not None:
            return models.RrfQuery(rrf=models.Rrf())

        return models.FusionQuery(fusion=models.Fusion.RRF)

    @staticmethod
    def _extract_points(response: Any) -> list[Any]:
        """Extract scored points from Qdrant query response."""

        points = getattr(response, "points", None)

        if points is None:
            if isinstance(response, list):
                return response

            raise RetrievalError("Qdrant response did not contain points.")

        return list(points)

    @staticmethod
    def _point_to_retrieved_chunk(point: Any) -> RetrievedChunk:
        """Convert Qdrant scored point into internal retrieval result."""

        payload = point.payload or {}

        required_payload_fields = [
            "chunk_id",
            "text",
            "source_document",
            "section_title",
            "collection",
            "chunk_type",
            "access_roles",
        ]

        missing_fields = [
            field_name
            for field_name in required_payload_fields
            if field_name not in payload
        ]

        if missing_fields:
            raise RetrievalError(
                f"Retrieved point is missing payload fields: {missing_fields}"
            )

        return RetrievedChunk(
            chunk_id=str(payload["chunk_id"]),
            text=str(payload["text"]),
            score=float(point.score),
            source_document=str(payload["source_document"]),
            section_title=str(payload["section_title"]),
            collection=str(payload["collection"]),
            chunk_type=str(payload["chunk_type"]),
            access_roles=list(payload["access_roles"]),
        )

    @staticmethod
    def _validate_results_against_role(
        retrieved_chunks: list[RetrievedChunk],
        role: UserRole,
    ) -> None:
        """
        Defense-in-depth validation after retrieval.

        This is not the primary RBAC control. The primary control is the Qdrant
        query_filter. This check only catches accidental configuration mistakes.
        """

        unauthorized_chunks = [
            chunk
            for chunk in retrieved_chunks
            if role.value not in chunk.access_roles
        ]

        if unauthorized_chunks:
            leaked_sources = [
                f"{chunk.source_document}::{chunk.collection}"
                for chunk in unauthorized_chunks
            ]

            raise RoleAccessError(
                "RBAC validation failed. Qdrant returned unauthorized chunks: "
                f"{leaked_sources}"
            )