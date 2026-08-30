from __future__ import annotations

import logging

from backend.app.core.config import get_settings
from backend.app.services.llm_service import GeminiLLMService


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def main() -> None:
    configure_logging()

    settings = get_settings()
    llm_service = GeminiLLMService(settings=settings)

    response = llm_service.generate_text(
        prompt="Reply with exactly: Gemini connection successful.",
        system_instruction="Follow the user instruction exactly.",
    )

    print("\nGEMINI CONNECTION CHECK")
    print("-" * 50)
    print(response.text)
    print(f"Model: {response.model}")
    print(f"Provider: {response.provider}")


if __name__ == "__main__":
    main()