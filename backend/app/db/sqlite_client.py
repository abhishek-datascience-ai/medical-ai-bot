from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.core.config import AppSettings
from backend.app.core.exceptions import DatasetNotFoundError, SQLRAGError


@dataclass(frozen=True)
class SQLColumn:
    """Represents one SQLite table column."""

    name: str
    data_type: str
    not_null: bool
    primary_key: bool


@dataclass(frozen=True)
class SQLQueryResult:
    """Represents a read-only SQL query result."""

    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int


class SQLiteClient:
    """Small read-only SQLite client for MediAssist analytical queries."""

    def __init__(self, settings: AppSettings) -> None:
        self._db_path = Path(settings.sqlite_db_path)
        self._validate_database_path()

    def _validate_database_path(self) -> None:
        if not self._db_path.exists():
            raise DatasetNotFoundError(f"SQLite database not found: {self._db_path}")

        if not self._db_path.is_file():
            raise DatasetNotFoundError(
                f"SQLite database path is not a file: {self._db_path}"
            )

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(self._db_path)
            connection.row_factory = sqlite3.Row
            return connection
        except sqlite3.Error as exc:
            raise SQLRAGError(f"Could not connect to SQLite database: {exc}") from exc

    def get_table_names(self) -> list[str]:
        query = """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name;
        """

        with self._connect() as connection:
            rows = connection.execute(query).fetchall()

        return [str(row["name"]) for row in rows]

    def get_table_schema(self, table_name: str) -> list[SQLColumn]:
        self._validate_identifier(table_name)

        with self._connect() as connection:
            rows = connection.execute(f"PRAGMA table_info({table_name});").fetchall()

        return [
            SQLColumn(
                name=str(row["name"]),
                data_type=str(row["type"]),
                not_null=bool(row["notnull"]),
                primary_key=bool(row["pk"]),
            )
            for row in rows
        ]

    def get_schema_text(self) -> str:
        """Return compact schema text useful for debugging and future LLM prompts."""

        schema_lines: list[str] = []

        for table_name in self.get_table_names():
            columns = self.get_table_schema(table_name)
            column_text = ", ".join(
                f"{column.name} {column.data_type}".strip()
                for column in columns
            )
            schema_lines.append(f"{table_name}({column_text})")

        return "\n".join(schema_lines)

    def execute_read_query(
        self,
        sql: str,
        parameters: tuple[Any, ...] = (),
    ) -> SQLQueryResult:
        """Execute a safe read-only SQL query."""

        cleaned_sql = sql.strip()
        self._validate_read_only_sql(cleaned_sql)

        try:
            with self._connect() as connection:
                cursor = connection.execute(cleaned_sql, parameters)
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description or []]
        except sqlite3.Error as exc:
            raise SQLRAGError(f"SQL execution failed: {exc}") from exc

        result_rows = [dict(row) for row in rows]

        return SQLQueryResult(
            columns=columns,
            rows=result_rows,
            row_count=len(result_rows),
        )

    @staticmethod
    def _validate_identifier(identifier: str) -> None:
        if not identifier.replace("_", "").isalnum():
            raise SQLRAGError(f"Invalid SQL identifier: {identifier}")

    @staticmethod
    def _validate_read_only_sql(sql: str) -> None:
        normalized_sql = sql.strip().lower()

        if not normalized_sql.startswith("select"):
            raise SQLRAGError("Only SELECT queries are allowed.")

        blocked_keywords = {
            "insert",
            "update",
            "delete",
            "drop",
            "alter",
            "create",
            "replace",
            "truncate",
            "attach",
            "detach",
            "pragma",
        }

        tokens = set(normalized_sql.replace(";", " ").split())
        blocked_matches = sorted(tokens.intersection(blocked_keywords))

        if blocked_matches:
            raise SQLRAGError(
                f"Unsafe SQL keyword detected: {', '.join(blocked_matches)}"
            )

        if ";" in normalized_sql.rstrip(";"):
            raise SQLRAGError("Multiple SQL statements are not allowed.")