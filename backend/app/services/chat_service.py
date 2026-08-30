from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.core.config import AppSettings
from backend.app.core.exceptions import RoleAccessError
from backend.app.core.roles import (
    UserRole,
    get_accessible_collections,
)
from backend.app.rag.context_builder import (
    AnswerContextBuilder,
    citations_to_response_payload,
)
from backend.app.rag.document_answer import DocumentAnswerGenerator
from backend.app.rag.embeddings import EmbeddingService
from backend.app.rag.hybrid_retriever import HybridRetriever
from backend.app.rag.reranker import CrossEncoderReranker
from backend.app.rag.sql_rag import run_sql_rag_for_role
from backend.app.rag.vector_store import QdrantCollectionManager


@dataclass(frozen=True)
class ChatServiceResponse:
    """Internal chat response returned by the chat service."""

    answer: str
    sources: list[dict[str, str]]
    retrieval_type: str
    role: str


class ChatService:
    """Route chat questions to SQL analytics or document RAG."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    def answer_question(self, question: str, role: UserRole) -> ChatServiceResponse:
        """Answer a user question with role-aware routing."""

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        if self._is_analytical_question(cleaned_question):
            return self._answer_sql_question(
                question=cleaned_question,
                role=role,
            )

        return self._answer_document_question(
            question=cleaned_question,
            role=role,
        )

    def _answer_sql_question(
        self,
        question: str,
        role: UserRole,
    ) -> ChatServiceResponse:
        """Answer analytical questions using SQL RAG."""

        try:
            sql_response = run_sql_rag_for_role(
                question=question,
                role=role,
            )
        except RoleAccessError:
            accessible_collections = [
                collection.value for collection in get_accessible_collections(role)
            ]
            return ChatServiceResponse(
                answer=(
                    f"As a {role.value}, you do not have access to SQL analytics. "
                    f"I can answer document questions from these collections: "
                    f"{', '.join(accessible_collections)}."
                ),
                sources=[],
                retrieval_type="blocked",
                role=role.value,
            )

        return ChatServiceResponse(
            answer=sql_response.answer,
            sources=[],
            retrieval_type="sql_rag",
            role=role.value,
        )

    def _answer_document_question(
        self,
        question: str,
        role: UserRole,
    ) -> ChatServiceResponse:
        """Answer document questions using hybrid retrieval, reranking, and LLM generation."""

        collection_manager = QdrantCollectionManager(settings=self._settings)

        try:
            embedding_service = EmbeddingService(settings=self._settings)
            retriever = HybridRetriever(
                settings=self._settings,
                collection_manager=collection_manager,
                embedding_service=embedding_service,
            )
            reranker = CrossEncoderReranker(settings=self._settings)
            context_builder = AnswerContextBuilder(settings=self._settings)

            candidates = retriever.retrieve(
                question=question,
                role=role,
                limit=self._settings.dense_top_k,
            )

            reranked_chunks = reranker.rerank(
                question=question,
                retrieved_chunks=candidates,
                role=role,
                top_k=self._settings.rerank_top_k,
            )

            answer_context = context_builder.build_context(
                question=question,
                reranked_chunks=reranked_chunks,
                role=role,
            )

            if self._settings.llm_mode == "cloud":
                answer_generator = DocumentAnswerGenerator(settings=self._settings)
                answer = answer_generator.generate_answer(
                    question=question,
                    answer_context=answer_context,
                )
            else:
                answer = self._build_local_document_answer(answer_context.context_text)

            return ChatServiceResponse(
                answer=answer,
                sources=citations_to_response_payload(answer_context.sources),
                retrieval_type="hybrid_rag",
                role=role.value,
            )
        finally:
            collection_manager.close()

    @staticmethod
    def _is_analytical_question(question: str) -> bool:
        """Detect whether a question should be routed to SQL analytics."""

        normalized_question = question.lower()

        analytical_patterns = (
            r"\bhow many\b",
            r"\bcount\b",
            r"\bnumber of\b",
            r"\btotal\b",
            r"\bhighest\b",
            r"\bmost\b",
            r"\bdistribution\b",
            r"\bstatus\b",
            r"\bamount\b",
            r"\bclaims?\b",
            r"\btickets?\b",
            r"\bfault codes?\b",
            r"\bcampus\b",
            r"\bcategory\b",
        )

        return any(
            re.search(pattern, normalized_question)
            for pattern in analytical_patterns
        )

    @staticmethod
    def _build_local_document_answer(context_text: str) -> str:
        """Fallback answer mode when cloud LLM calls are disabled."""

        preview = context_text.strip()

        if len(preview) > 1200:
            preview = f"{preview[:1200]}..."

        return (
            "Based on the retrieved authorized documents, here is the most relevant "
            f"context:\n\n{preview}"
        )