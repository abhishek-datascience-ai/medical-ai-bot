from __future__ import annotations

from backend.app.core.config import AppSettings
from backend.app.core.exceptions import LLMServiceError
from backend.app.rag.context_builder import AnswerContext
from backend.app.services.llm_service import GeminiLLMService


class DocumentAnswerGenerator:
    """Generate final document-grounded answers from retrieved context."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._llm_service = GeminiLLMService(settings=settings)

    def generate_answer(
        self,
        question: str,
        answer_context: AnswerContext,
    ) -> str:
        cleaned_question = question.strip()

        if not cleaned_question:
            raise LLMServiceError("Question cannot be empty.")

        system_instruction = (
            "You are Medical-AI Bot, a role-aware healthcare knowledge assistant. "
            "Answer only from the provided context. "
            "If the context does not contain enough information, say that the available documents do not provide enough information. "
            "Do not invent medical, billing, or equipment details. "
            "Keep the answer concise and cite sources using [SOURCE 1], [SOURCE 2], etc."
        )

        prompt = f"""
User question:
{cleaned_question}

Retrieved context:
{answer_context.context_text}

Write the final answer with source references.
"""

        response = self._llm_service.generate_text(
            prompt=prompt,
            system_instruction=system_instruction,
        )

        return response.text