from __future__ import annotations

import json
import logging
import sqlite3
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

DATASET_ZIP_PATH = RAW_DATA_DIR / "mediassist_data.zip"
EXTRACTED_DATASET_DIR = RAW_DATA_DIR / "mediassist_data"
SQLITE_DB_PATH = EXTRACTED_DATASET_DIR / "db" / "mediassist.db"

INSPECTION_REPORT_PATH = PROCESSED_DATA_DIR / "dataset_inspection_report.json"


EXPECTED_COLLECTIONS = {
    "general": {
        "expected_extensions": {".pdf"},
        "access_roles": ["doctor", "nurse", "billing_executive", "technician", "admin"],
    },
    "clinical": {
        "expected_extensions": {".pdf"},
        "access_roles": ["doctor", "admin"],
    },
    "nursing": {
        "expected_extensions": {".pdf"},
        "access_roles": ["nurse", "doctor", "admin"],
    },
    "billing": {
        "expected_extensions": {".pdf", ".md"},
        "access_roles": ["billing_executive", "admin"],
    },
    "equipment": {
        "expected_extensions": {".pdf"},
        "access_roles": ["technician", "admin"],
    },
}

EXPECTED_DB_TABLES = {"claims", "maintenance_tickets"}


@dataclass(frozen=True)
class TableInspectionResult:
    """SQLite table metadata used later by the SQL RAG module."""

    table_name: str
    row_count: int
    columns: list[dict[str, Any]]
    sample_rows: list[dict[str, Any]]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def ensure_required_paths() -> None:
    """Validate required project folders and dataset ZIP before extraction."""

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

    if not DATASET_ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Dataset ZIP not found: {DATASET_ZIP_PATH}\n"
            "Please place mediassist_data.zip inside data/raw/."
        )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def extract_dataset() -> None:
    """Extract the dataset ZIP only when the extracted folder is missing."""

    if EXTRACTED_DATASET_DIR.exists():
        logging.info("Dataset already extracted: %s", EXTRACTED_DATASET_DIR)
        return

    logging.info("Extracting dataset ZIP: %s", DATASET_ZIP_PATH)

    try:
        with zipfile.ZipFile(DATASET_ZIP_PATH, "r") as zip_file:
            zip_file.extractall(RAW_DATA_DIR)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid or corrupted ZIP file: {DATASET_ZIP_PATH}") from exc

    if not EXTRACTED_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset extraction completed, but expected folder was not found: "
            f"{EXTRACTED_DATASET_DIR}"
        )

    logging.info("Dataset extracted successfully.")


def inspect_collections() -> dict[str, Any]:
    """Inspect document collection folders and files."""

    collection_report: dict[str, Any] = {}

    for collection_name, rules in EXPECTED_COLLECTIONS.items():
        collection_dir = EXTRACTED_DATASET_DIR / collection_name

        if not collection_dir.exists():
            collection_report[collection_name] = {
                "exists": False,
                "file_count": 0,
                "files": [],
                "access_roles": rules["access_roles"],
                "issues": [f"Missing collection directory: {collection_dir}"],
            }
            continue

        files = sorted(
            file_path
            for file_path in collection_dir.iterdir()
            if file_path.is_file()
        )

        issues: list[str] = []
        file_details: list[dict[str, Any]] = []

        for file_path in files:
            extension = file_path.suffix.lower()

            if extension not in rules["expected_extensions"]:
                issues.append(
                    f"Unexpected file extension in {collection_name}: {file_path.name}"
                )

            file_details.append(
                {
                    "file_name": file_path.name,
                    "extension": extension,
                    "size_bytes": file_path.stat().st_size,
                    "collection": collection_name,
                    "access_roles": rules["access_roles"],
                }
            )

        collection_report[collection_name] = {
            "exists": True,
            "file_count": len(files),
            "files": file_details,
            "access_roles": rules["access_roles"],
            "issues": issues,
        }

    return collection_report


def get_sqlite_tables(connection: sqlite3.Connection) -> set[str]:
    query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name;
    """
    rows = connection.execute(query).fetchall()
    return {row[0] for row in rows}


def inspect_table(
    connection: sqlite3.Connection,
    table_name: str,
    sample_limit: int = 5,
) -> TableInspectionResult:
    """Inspect one SQLite table's columns, row count, and sample records."""

    columns_query = f"PRAGMA table_info({table_name});"
    column_rows = connection.execute(columns_query).fetchall()

    columns = [
        {
            "cid": row[0],
            "name": row[1],
            "type": row[2],
            "not_null": bool(row[3]),
            "default_value": row[4],
            "primary_key": bool(row[5]),
        }
        for row in column_rows
    ]

    row_count_query = f"SELECT COUNT(*) FROM {table_name};"
    row_count = int(connection.execute(row_count_query).fetchone()[0])

    sample_query = f"SELECT * FROM {table_name} LIMIT ?;"
    cursor = connection.execute(sample_query, (sample_limit,))
    column_names = [description[0] for description in cursor.description]
    sample_rows = [
        dict(zip(column_names, row, strict=False))
        for row in cursor.fetchall()
    ]

    return TableInspectionResult(
        table_name=table_name,
        row_count=row_count,
        columns=columns,
        sample_rows=sample_rows,
    )


def inspect_sqlite_database() -> dict[str, Any]:
    """Inspect mediassist.db and validate expected tables."""

    if not SQLITE_DB_PATH.exists():
        raise FileNotFoundError(f"SQLite database not found: {SQLITE_DB_PATH}")

    try:
        connection = sqlite3.connect(SQLITE_DB_PATH)
    except sqlite3.Error as exc:
        raise ConnectionError(f"Could not connect to SQLite DB: {SQLITE_DB_PATH}") from exc

    try:
        actual_tables = get_sqlite_tables(connection)
        missing_tables = sorted(EXPECTED_DB_TABLES - actual_tables)

        table_results = {
            table_name: inspect_table(connection, table_name).__dict__
            for table_name in sorted(actual_tables)
            if table_name in EXPECTED_DB_TABLES
        }

        return {
            "database_path": str(SQLITE_DB_PATH),
            "exists": True,
            "actual_tables": sorted(actual_tables),
            "expected_tables": sorted(EXPECTED_DB_TABLES),
            "missing_tables": missing_tables,
            "tables": table_results,
        }
    finally:
        connection.close()


def save_report(report: dict[str, Any]) -> None:
    INSPECTION_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logging.info("Inspection report saved: %s", INSPECTION_REPORT_PATH)


def main() -> None:
    configure_logging()

    logging.info("Starting Medical-AI Bot dataset inspection.")

    ensure_required_paths()
    extract_dataset()

    collection_report = inspect_collections()
    database_report = inspect_sqlite_database()

    final_report = {
        "project_name": "Medical-AI Bot",
        "dataset_zip_path": str(DATASET_ZIP_PATH),
        "extracted_dataset_dir": str(EXTRACTED_DATASET_DIR),
        "collections": collection_report,
        "database": database_report,
    }

    save_report(final_report)

    logging.info("Dataset inspection completed successfully.")

    print("\nDATASET INSPECTION SUMMARY")
    print("-" * 40)

    for collection_name, details in collection_report.items():
        print(
            f"{collection_name}: "
            f"exists={details['exists']}, "
            f"files={details['file_count']}, "
            f"roles={details['access_roles']}"
        )

    print("\nSQLITE DATABASE SUMMARY")
    print("-" * 40)
    print(f"database: {database_report['database_path']}")
    print(f"tables: {database_report['actual_tables']}")

    for table_name, table_details in database_report["tables"].items():
        column_names = [column["name"] for column in table_details["columns"]]
        print(f"{table_name}: rows={table_details['row_count']}")
        print(f"columns={column_names}")


if __name__ == "__main__":
    main()