from __future__ import annotations
import logging
from typing import Any
from backend.app.services.llm_service import GeminiLLMService

import re
from dataclasses import dataclass
from typing import Any

from backend.app.core.config import get_settings
from backend.app.core.exceptions import RoleAccessError, SQLRAGError
from backend.app.core.roles import UserRole, can_use_sql_rag
from backend.app.db.sqlite_client import SQLQueryResult, SQLiteClient


@dataclass(frozen=True)
class SQLRAGResponse:
    """Detailed SQL RAG response used by scripts and API handlers."""

    question: str
    sql: str
    answer: str
    columns: list[str]
    rows: list[dict[str, Any]]


class LocalSQLTranslator:
    """
    Deterministic local translator for analytical questions.

    This avoids paid LLM calls during development while keeping the same
    SQL-generation, SQL-cleaning, execution, and answer-building workflow.
    """

    def translate(self, question: str) -> str:
        cleaned_question = question.strip()

        if not cleaned_question:
            raise SQLRAGError("Question cannot be empty.")

        normalized_question = self._normalize(cleaned_question)

        if self._is_claims_question(normalized_question):
            return self._translate_claims_question(normalized_question)

        if self._is_maintenance_question(normalized_question):
            return self._translate_maintenance_question(normalized_question)

        raise SQLRAGError(
            "This local SQL mode supports analytical questions about claims "
            "and maintenance tickets only."
        )

    @staticmethod
    def _normalize(question: str) -> str:
        return re.sub(r"\s+", " ", question.lower()).strip()

    @staticmethod
    def _is_claims_question(question: str) -> bool:
        claim_terms = {
            "claim",
            "claims",
            "billing",
            "insurer",
            "insurance",
            "approved amount",
            "claimed amount",
        }
        return any(term in question for term in claim_terms)

    @staticmethod
    def _is_maintenance_question(question: str) -> bool:
        maintenance_terms = {
            "maintenance",
            "ticket",
            "tickets",
            "equipment",
            "fault",
            "campus",
            "issue",
        }
        return any(term in question for term in maintenance_terms)

    def _translate_claims_question(self, question: str) -> str:
        status = self._extract_known_status(
            question=question,
            allowed_statuses=("escalated", "rejected", "approved", "pending", "submitted"),
        )

        year = self._extract_year(question)

        if status and self._asks_for_count(question):
            where_clause = f"WHERE LOWER(status) = '{status}'"

            if year:
                where_clause += f" AND strftime('%Y', submitted_date) = '{year}'"

            return f"""
                SELECT COUNT(*) AS claim_count
                FROM claims
                {where_clause}
            """

        if "last month" in question and "escalated" in question:
            return """
                SELECT COUNT(*) AS escalated_claim_count
                FROM claims
                WHERE LOWER(status) = 'escalated'
                  AND submitted_date >= date(
                        (SELECT MAX(submitted_date) FROM claims),
                        'start of month',
                        '-1 month'
                  )
                  AND submitted_date < date(
                        (SELECT MAX(submitted_date) FROM claims),
                        'start of month'
                  )
            """

        if "total approved" in question and "insurer" in question:
            return """
                SELECT insurer, ROUND(SUM(approved_amount), 2) AS total_approved_amount
                FROM claims
                GROUP BY insurer
                ORDER BY total_approved_amount DESC
            """

        if "total claimed" in question and "department" in question:
            return """
                SELECT department, ROUND(SUM(claimed_amount), 2) AS total_claimed_amount
                FROM claims
                GROUP BY department
                ORDER BY total_claimed_amount DESC
            """

        if (
            "highest" in question
            and "claimed" in question
            and "department" in question
        ):
            return """
                SELECT department, ROUND(SUM(claimed_amount), 2) AS total_claimed_amount
                FROM claims
                GROUP BY department
                ORDER BY total_claimed_amount DESC
                LIMIT 1
            """

        if "by insurer" in question:
            return """
                SELECT
                    insurer,
                    COUNT(*) AS claim_count,
                    ROUND(SUM(claimed_amount), 2) AS total_claimed_amount,
                    ROUND(SUM(approved_amount), 2) AS total_approved_amount
                FROM claims
                GROUP BY insurer
                ORDER BY claim_count DESC
            """

        if "status" in question or "distribution" in question:
            return """
                SELECT status, COUNT(*) AS claim_count
                FROM claims
                GROUP BY status
                ORDER BY claim_count DESC
            """

        if self._asks_for_count(question):
            return """
                SELECT COUNT(*) AS claim_count
                FROM claims
            """

        raise SQLRAGError(
            "Unsupported claims question for local SQL mode. Try asking about "
            "claim counts, claim status, total approved amount by insurer, or "
            "highest claimed amount by department."
        )

    def _translate_maintenance_question(self, question: str) -> str:
        status = self._extract_known_status(
            question=question,
            allowed_statuses=("open", "closed", "resolved", "escalated", "pending"),
        )

        if (
            "category" in question
            and ("most open" in question or "highest" in question)
        ):
            return """
                SELECT category, COUNT(*) AS open_ticket_count
                FROM maintenance_tickets
                WHERE LOWER(status) = 'open'
                GROUP BY category
                ORDER BY open_ticket_count DESC
                LIMIT 1
            """

        if status and self._asks_for_count(question):
            return f"""
                SELECT COUNT(*) AS maintenance_ticket_count
                FROM maintenance_tickets
                WHERE LOWER(status) = '{status}'
            """

        if "unresolved" in question and "campus" in question:
            return """
                SELECT campus, COUNT(*) AS unresolved_ticket_count
                FROM maintenance_tickets
                WHERE resolved_date IS NULL OR LOWER(status) != 'resolved'
                GROUP BY campus
                ORDER BY unresolved_ticket_count DESC
                LIMIT 1
            """

        if "fault" in question:
            return """
                SELECT fault_code, COUNT(*) AS ticket_count
                FROM maintenance_tickets
                WHERE fault_code IS NOT NULL AND TRIM(fault_code) != ''
                GROUP BY fault_code
                ORDER BY ticket_count DESC
                LIMIT 10
            """

        if "category" in question:
            return """
                SELECT category, COUNT(*) AS ticket_count
                FROM maintenance_tickets
                GROUP BY category
                ORDER BY ticket_count DESC
            """

        if "campus" in question:
            return """
                SELECT campus, COUNT(*) AS ticket_count
                FROM maintenance_tickets
                GROUP BY campus
                ORDER BY ticket_count DESC
            """

        if "status" in question or "distribution" in question:
            return """
                SELECT status, COUNT(*) AS ticket_count
                FROM maintenance_tickets
                GROUP BY status
                ORDER BY ticket_count DESC
            """

        if self._asks_for_count(question):
            return """
                SELECT COUNT(*) AS maintenance_ticket_count
                FROM maintenance_tickets
            """

        raise SQLRAGError(
            "Unsupported maintenance question for local SQL mode. Try asking "
            "about open tickets by category, unresolved tickets by campus, "
            "fault codes, or ticket status distribution."
        )

    @staticmethod
    def _asks_for_count(question: str) -> bool:
        count_terms = ("how many", "count", "number of", "total number")
        return any(term in question for term in count_terms)

    @staticmethod
    def _extract_known_status(
        question: str,
        allowed_statuses: tuple[str, ...],
    ) -> str | None:
        for status in allowed_statuses:
            if status in question:
                return status

        return None

    @staticmethod
    def _extract_year(question: str) -> str | None:
        match = re.search(r"\b(20\d{2})\b", question)
        return match.group(1) if match else None


class LocalSQLAnswerBuilder:
    """Build natural-language answers from SQL query results."""

    def build_answer(
        self,
        question: str,
        sql: str,
        result: SQLQueryResult,
    ) -> str:
        if result.row_count == 0:
            return "No matching records were found for this question."

        if result.row_count == 1:
            return self._build_single_row_answer(result.rows[0])

        return self._build_multi_row_answer(result.rows)

    @staticmethod
    def _build_single_row_answer(row: dict[str, Any]) -> str:
        if len(row) == 1:
            column_name, value = next(iter(row.items()))
            return f"The {column_name.replace('_', ' ')} is {value}."

        parts = [
            f"{column_name.replace('_', ' ')} is {value}"
            for column_name, value in row.items()
        ]

        return "The top result is: " + "; ".join(parts) + "."

    @staticmethod
    def _build_multi_row_answer(rows: list[dict[str, Any]]) -> str:
        formatted_rows: list[str] = []

        for row in rows[:10]:
            row_text = ", ".join(
                f"{column_name.replace('_', ' ')}: {value}"
                for column_name, value in row.items()
            )
            formatted_rows.append(row_text)

        return "Here are the results: " + " | ".join(formatted_rows) + "."


def clean_sql_statement(raw_sql: str) -> str:
    """Extract a single safe SQL statement from raw generated text."""

    cleaned_sql = raw_sql.strip()

    cleaned_sql = re.sub(
        r"^```(?:sql)?\s*",
        "",
        cleaned_sql,
        flags=re.IGNORECASE,
    )
    cleaned_sql = re.sub(r"\s*```$", "", cleaned_sql)

    select_match = re.search(
        r"\bselect\b[\s\S]*",
        cleaned_sql,
        flags=re.IGNORECASE,
    )

    if not select_match:
        raise SQLRAGError("No SELECT statement found in generated SQL.")

    sql = select_match.group(0).strip().rstrip(";")

    if not sql.lower().startswith("select"):
        raise SQLRAGError("Only SELECT statements are allowed.")

    return sql


def run_sql_rag(question: str) -> SQLRAGResponse:
    """
    Run SQL RAG workflow.

    Steps:
    1. Translate the question into SQL.
    2. Clean the generated SQL.
    3. Execute SQL.
    4. Build a natural-language answer.
    """

    settings = get_settings()
    sqlite_client = SQLiteClient(settings=settings)

    if settings.llm_mode == "cloud" and settings.llm_provider == "gemini":
        llm_service = GeminiLLMService(settings=settings)
        translator = GeminiSQLTranslator(
            llm_service=llm_service,
            sqlite_client=sqlite_client,
        )
        answer_builder = GeminiSQLAnswerBuilder(llm_service=llm_service)
    else:
        translator = LocalSQLTranslator()
        answer_builder = LocalSQLAnswerBuilder()

    raw_sql = translator.translate(question)
    cleaned_sql = clean_sql_statement(raw_sql)
    result = sqlite_client.execute_read_query(cleaned_sql)
    answer = answer_builder.build_answer(
        question=question,
        sql=cleaned_sql,
        result=result,
    )

    return SQLRAGResponse(
        question=question,
        sql=cleaned_sql,
        answer=answer,
        columns=result.columns,
        rows=result.rows,
    )


def run_sql_rag_for_role(question: str, role: UserRole) -> SQLRAGResponse:
    """Run SQL RAG only for roles allowed to access analytical data."""

    if not can_use_sql_rag(role):
        raise RoleAccessError(
            f"Role '{role.value}' is not allowed to use SQL analytics."
        )

    return run_sql_rag(question)


def sql_rag_chain(question: str) -> str:
    """
    Plain function required by the assignment-style interface.

    Role checks should be applied before this function is called.
    """

    return run_sql_rag(question).answer

class GeminiSQLTranslator:
    """Translate analytical questions into SQLite SELECT queries using Gemini."""

    def __init__(self, llm_service: GeminiLLMService, sqlite_client: SQLiteClient) -> None:
        self._llm_service = llm_service
        self._sqlite_client = sqlite_client

    def translate(self, question: str) -> str:
        schema_text = self._sqlite_client.get_schema_text()

        system_instruction = (
            "You are a careful SQLite query generator. "
            "Return only one SELECT statement. "
            "Do not include markdown fences, explanation, comments, INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA, or multiple statements. "
            "For text filtering, always use LOWER(TRIM(column_name)) for case-insensitive matching. "
            "Always use clear aliases for aggregate columns."
        )

        prompt = f"""
Database schema:
{schema_text}

User question:
{question}

Generate one safe SQLite SELECT query.
"""

        response = self._llm_service.generate_text(
            prompt=prompt,
            system_instruction=system_instruction,
        )

        return response.text

class GeminiSQLAnswerBuilder:
    """Build a natural-language SQL answer using Gemini."""

    def __init__(self, llm_service: GeminiLLMService) -> None:
        self._llm_service = llm_service

    def build_answer(
        self,
        question: str,
        sql: str,
        result: SQLQueryResult,
    ) -> str:
        system_instruction = (
            "You are a concise healthcare operations analytics assistant. "
            "Answer using only the SQL result provided. "
            "Do not invent numbers."
        )

        prompt = f"""
Question:
{question}

SQL query:
{sql}

SQL result rows:
{result.rows}

Write a concise natural-language answer.
"""

        response = self._llm_service.generate_text(
            prompt=prompt,
            system_instruction=system_instruction,
        )

        return response.text