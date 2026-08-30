from __future__ import annotations

from backend.app.core.config import get_settings
from backend.app.core.roles import (
    DocumentCollection,
    UserRole,
    can_access_collection,
    can_use_sql_rag,
    get_accessible_collections,
)


def main() -> None:
    settings = get_settings()

    print("\nCORE CONFIGURATION")
    print("-" * 50)
    print(f"App name: {settings.app_name}")
    print(f"Environment: {settings.app_env}")
    print(f"Debug: {settings.app_debug}")
    print(f"Host: {settings.app_host}")
    print(f"Port: {settings.app_port}")
    print(f"Raw data dir: {settings.raw_data_dir}")
    print(f"SQLite DB path: {settings.sqlite_db_path}")
    print(f"Qdrant collection: {settings.qdrant_collection_name}")
    print(f"Dense top K: {settings.dense_top_k}")
    print(f"Rerank top K: {settings.rerank_top_k}")

    print("\nROLE ACCESS MATRIX")
    print("-" * 50)

    for role in UserRole:
        collections = get_accessible_collections(role)
        collection_values = [collection.value for collection in collections]

        print(f"{role.value}: {collection_values}")

    print("\nACCESS CHECK EXAMPLES")
    print("-" * 50)
    print(
        "Can nurse access billing?",
        can_access_collection(UserRole.NURSE, DocumentCollection.BILLING),
    )
    print(
        "Can doctor access clinical?",
        can_access_collection(UserRole.DOCTOR, DocumentCollection.CLINICAL),
    )
    print(
        "Can technician access equipment?",
        can_access_collection(UserRole.TECHNICIAN, DocumentCollection.EQUIPMENT),
    )
    print(
        "Can billing executive use SQL RAG?",
        can_use_sql_rag(UserRole.BILLING_EXECUTIVE),
    )
    print(
        "Can nurse use SQL RAG?",
        can_use_sql_rag(UserRole.NURSE),
    )


if __name__ == "__main__":
    main()