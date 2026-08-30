from __future__ import annotations

from dataclasses import dataclass

from fastembed.rerank.cross_encoder import TextCrossEncoder

from backend.app.core.config import AppSettings
from backend.app.core.exceptions import RetrievalError, RoleAccessError
from backend.app.core.roles import UserRole
from backend.app.rag.hybrid_retriever import RetrievedChunk


@dataclass(frozen=True)
class RerankedChunk:
    """Represents one retrieved chunk after cross-encoder reranking."""

    chunk_id: str
    text: str
    original_rank: int
    original_score: float
    rerank_score: float
    source_document: str
    section_title: str
    collection: str
    chunk_type: str
    access_roles: list[str]


class CrossEncoderReranker:
    """
    Rerank retrieved chunks using a FastEmbed cross-encoder.

    The reranker receives a small candidate set from Qdrant hybrid retrieval.
    It scores each question/chunk pair jointly and returns only the strongest
    candidates for LLM context building.
    """

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._model = TextCrossEncoder(model_name=settings.reranker_model_name)

    def rerank(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
        role: UserRole,
        top_k: int | None = None,
    ) -> list[RerankedChunk]:
        """Rerank retrieved chunks and return the top candidates."""

        cleaned_question = question.strip()

        if not cleaned_question:
            raise RetrievalError("Question cannot be empty for reranking.")

        if not retrieved_chunks:
            return []

        self._validate_chunks_against_role(
            retrieved_chunks=retrieved_chunks,
            role=role,
        )

        final_top_k = top_k or self._settings.rerank_top_k

        if final_top_k <= 0:
            raise RetrievalError("Rerank top_k must be greater than zero.")

        candidate_texts = [chunk.text for chunk in retrieved_chunks]

        try:
            scores = list(self._model.rerank(cleaned_question, candidate_texts))
        except Exception as exc:
            raise RetrievalError(f"Cross-encoder reranking failed: {exc}") from exc

        if len(scores) != len(retrieved_chunks):
            raise RetrievalError(
                "Reranker score count does not match retrieved chunk count."
            )

        reranked_chunks = [
            RerankedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                original_rank=index,
                original_score=chunk.score,
                rerank_score=float(score),
                source_document=chunk.source_document,
                section_title=chunk.section_title,
                collection=chunk.collection,
                chunk_type=chunk.chunk_type,
                access_roles=chunk.access_roles,
            )
            for index, (chunk, score) in enumerate(
                zip(retrieved_chunks, scores, strict=True),
                start=1,
            )
        ]

        reranked_chunks.sort(key=lambda chunk: chunk.rerank_score, reverse=True)

        top_chunks = reranked_chunks[:final_top_k]

        self._validate_reranked_chunks_against_role(
            reranked_chunks=top_chunks,
            role=role,
        )

        return top_chunks

    @staticmethod
    def _validate_chunks_against_role(
        retrieved_chunks: list[RetrievedChunk],
        role: UserRole,
    ) -> None:
        """Ensure retrieved candidates are still authorized before reranking."""

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
                "Unauthorized chunks were passed into reranking: "
                f"{leaked_sources}"
            )

    @staticmethod
    def _validate_reranked_chunks_against_role(
        reranked_chunks: list[RerankedChunk],
        role: UserRole,
    ) -> None:
        """Ensure reranking output still respects RBAC."""

        unauthorized_chunks = [
            chunk
            for chunk in reranked_chunks
            if role.value not in chunk.access_roles
        ]

        if unauthorized_chunks:
            leaked_sources = [
                f"{chunk.source_document}::{chunk.collection}"
                for chunk in unauthorized_chunks
            ]

            raise RoleAccessError(
                "Unauthorized chunks appeared after reranking: "
                f"{leaked_sources}"
            )