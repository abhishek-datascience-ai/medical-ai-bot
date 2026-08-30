from __future__ import annotations

from dataclasses import dataclass

from google import genai
from google.genai import types

from backend.app.core.config import AppSettings
from backend.app.core.exceptions import ConfigurationError, LLMServiceError


@dataclass(frozen=True)
class LLMResponse:
    """Represents a cloud LLM text response."""

    text: str
    model: str
    provider: str


class GeminiLLMService:
    """Cloud LLM service backed by Gemini API."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

        if not settings.gemini_api_key:
            raise ConfigurationError("GEMINI_API_KEY is required for Gemini calls.")

        self._client = genai.Client(api_key=settings.gemini_api_key)

    def generate_text(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> LLMResponse:
        """Generate text using Gemini."""

        cleaned_prompt = prompt.strip()

        if not cleaned_prompt:
            raise LLMServiceError("Prompt cannot be empty.")

        try:
            config = types.GenerateContentConfig(
                temperature=self._settings.gemini_temperature,
                max_output_tokens=self._settings.gemini_max_output_tokens,
                system_instruction=system_instruction,
            )

            response = self._client.models.generate_content(
                model=self._settings.gemini_model,
                contents=cleaned_prompt,
                config=config,
            )
        except Exception as exc:
            raise LLMServiceError(f"Gemini generation failed: {exc}") from exc

        response_text = getattr(response, "text", "") or ""

        if not response_text.strip():
            raise LLMServiceError("Gemini returned an empty response.")

        return LLMResponse(
            text=response_text.strip(),
            model=self._settings.gemini_model,
            provider="gemini",
        )