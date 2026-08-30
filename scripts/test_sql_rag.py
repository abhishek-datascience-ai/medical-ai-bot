from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any

from backend.app.core.exceptions import RoleAccessError
from backend.app.core.roles import UserRole, parse_user_role
from backend.app.rag.sql_rag import run_sql_rag_for_role
from backend.app.core.config import get_settings


@dataclass(frozen=True)
class SQLRAGScenario:
    """Represents one SQL RAG test scenario."""

    role: UserRole
    question: str
    should_be_allowed: bool


DEFAULT_SQL_RAG_SCENARIOS: tuple[SQLRAGScenario, ...] = (
    SQLRAGScenario(
        role=UserRole.BILLING_EXECUTIVE,
        question="How many claims are escalated?",
        should_be_allowed=True,
    ),
    SQLRAGScenario(
        role=UserRole.BILLING_EXECUTIVE,
        question="What is the total approved amount by insurer?",
        should_be_allowed=True,
    ),
    SQLRAGScenario(
        role=UserRole.ADMIN,
        question="Which equipment category has the most open maintenance tickets?",
        should_be_allowed=True,
    ),
    SQLRAGScenario(
        role=UserRole.ADMIN,
        question="Which campus has the highest number of unresolved tickets?",
        should_be_allowed=True,
    ),
    SQLRAGScenario(
        role=UserRole.ADMIN,
        question="What are the most common fault codes?",
        should_be_allowed=True,
    ),
    SQLRAGScenario(
        role=UserRole.NURSE,
        question="How many claims are escalated?",
        should_be_allowed=False,
    ),
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test local SQL RAG over the MediAssist SQLite database."
    )

    parser.add_argument(
        "--role",
        type=str,
        default="",
        help="Optional role for a custom SQL RAG question.",
    )

    parser.add_argument(
        "--question",
        type=str,
        default="",
        help="Optional custom SQL RAG question.",
    )

    return parser.parse_args()


def build_scenarios(args: argparse.Namespace) -> list[SQLRAGScenario]:
    if args.role or args.question:
        if not args.role or not args.question:
            raise ValueError("Both --role and --question are required.")

        return [
            SQLRAGScenario(
                role=parse_user_role(args.role),
                question=args.question,
                should_be_allowed=True,
            )
        ]

    return list(DEFAULT_SQL_RAG_SCENARIOS)


def print_success_result(role: UserRole, question: str, sql: str, answer: str) -> None:
    print("\n" + "=" * 80)
    print(f"ROLE: {role.value}")
    print(f"QUESTION: {question}")
    print("-" * 80)
    print("SQL:")
    print(sql)
    print("-" * 80)
    print("ANSWER:")
    print(answer)


def print_blocked_result(role: UserRole, question: str, error_message: str) -> None:
    print("\n" + "=" * 80)
    print(f"ROLE: {role.value}")
    print(f"QUESTION: {question}")
    print("-" * 80)
    print("ACCESS BLOCKED:")
    print(error_message)


def serialize_response(response: object) -> dict[str, Any]:
    """Convert a SQL RAG response to a dictionary without assuming its schema."""

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        return model_dump()

    legacy_dict = getattr(response, "dict", None)
    if callable(legacy_dict):
        return legacy_dict()

    if isinstance(response, dict):
        return response

    try:
        return vars(response)
    except TypeError as exc:
        raise TypeError(
            f"Unsupported SQL RAG response type: {type(response).__name__}"
        ) from exc


def first_response_value(
    response_data: dict[str, Any],
    *field_names: str,
    default: str = "Not returned",
) -> Any:
    """Return the first populated response field from the supplied names."""

    for field_name in field_names:
        value = response_data.get(field_name)
        if value is not None:
            return value

    return default


def main() -> None:
    configure_logging()
    args = parse_args()

    settings = get_settings()
    role = parse_user_role(args.role)
    question = args.question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    logging.info("Starting SQL RAG tests.")

    scenarios = build_scenarios(args)

    try:
        response = run_sql_rag_for_role(
            question=question,
            role=role,
        )
    except RoleAccessError as exc:
        print("\nSQL ANALYTICS ACCESS CHECK")
        print("-" * 50)
        print(f"Role: {role.value}")
        print(f"Question: {question}")
        print("Access allowed: False")
        print(f"Message: {exc}")
        return

    print("\nSQL RAG RESULT")
    print("-" * 50)
    print(f"Role: {role.value}")
    print(f"Question: {question}")
    response_data = serialize_response(response)

    sql_query = first_response_value(
        response_data,
        "sql_query",
        "generated_sql",
        "generated_query",
        "sql",
        "query",
    )
    answer = first_response_value(
        response_data,
        "answer",
        "response",
        "result",
        "summary",
        "natural_language_answer",
    )

    print(f"SQL Query: {sql_query}")
    print(f"Answer: {answer}")

    if sql_query == "Not returned" or answer == "Not returned":
        available_fields = ", ".join(response_data) or "none"
        logging.warning("Available response fields: %s", available_fields)

    logging.info("SQL RAG tests completed successfully.")


if __name__ == "__main__":
    main()
