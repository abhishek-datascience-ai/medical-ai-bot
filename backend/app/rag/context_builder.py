from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.core.config import AppSettings
from backend.app.core.exceptions import RetrievalError, RoleAccessError
from backend.app.core.roles import UserRole
from backend.app.rag.reranker import RerankedChunk


@dataclass(frozen=True)
class SourceCitation:
    """Source metadata shown with a final answer."""

    source_document: str
    section_title: str
    collection: str
    chunk_id: str


@dataclass(frozen=True)
class AnswerContext:
    """LLM-ready context and source citations."""

    context_text: str
    sources: list[SourceCitation]
    chunks_used: int
    total_chars: int


class AnswerContextBuilder:
    """Build LLM-ready context from reranked chunks."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    def build_context(
        self,
        question: str,
        reranked_chunks: list[RerankedChunk],
        role: UserRole,
    ) -> AnswerContext:
        """Build context text and source citations for final answer generation."""

        cleaned_question = question.strip()

        if not cleaned_question:
            raise RetrievalError("Question cannot be empty for context building.")

        self._validate_chunks_against_role(
            reranked_chunks=reranked_chunks,
            role=role,
        )

        context_blocks: list[str] = []
        sources: list[SourceCitation] = []
        used_chars = 0

        for index, chunk in enumerate(reranked_chunks, start=1):
            block = self._format_context_block(index=index, chunk=chunk)
            projected_chars = used_chars + len(block)

            if projected_chars > self._settings.context_max_chars:
                break

            context_blocks.append(block)
            used_chars = projected_chars

            sources.append(
                SourceCitation(
                    source_document=chunk.source_document,
                    section_title=chunk.section_title,
                    collection=chunk.collection,
                    chunk_id=chunk.chunk_id,
                )
            )

        if not context_blocks:
            raise RetrievalError(
                "No context could be built from reranked chunks. "
                "Check context_max_chars or retrieval output."
            )

        context_text = "\n\n".join(context_blocks)

        return AnswerContext(
            context_text=context_text,
            sources=sources,
            chunks_used=len(context_blocks),
            total_chars=len(context_text),
        )

    @staticmethod
    def _format_context_block(index: int, chunk: RerankedChunk) -> str:
        """Format one chunk as a source-tagged context block."""

        return (
            f"[SOURCE {index}]\n"
            f"Document: {chunk.source_document}\n"
            f"Section: {chunk.section_title}\n"
            f"Collection: {chunk.collection}\n"
            f"Chunk Type: {chunk.chunk_type}\n"
            f"Rerank Score: {chunk.rerank_score:.6f}\n"
            f"Content:\n{chunk.text.strip()}"
        )

    @staticmethod
    def _validate_chunks_against_role(
        reranked_chunks: list[RerankedChunk],
        role: UserRole,
    ) -> None:
        """Ensure only role-authorized chunks are used in LLM context."""

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
                "Unauthorized chunks were passed into context builder: "
                f"{leaked_sources}"
            )


def citations_to_response_payload(
    sources: list[SourceCitation],
) -> list[dict[str, Any]]:
    """Convert source citations into API response-compatible dictionaries."""

    return [
        {
            "source_document": source.source_document,
            "section_title": source.section_title,
            "collection": source.collection,
        }
        for source in sources
    ]