from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backend.app.core.config import AppSettings
from backend.app.core.exceptions import DatasetNotFoundError
from backend.app.core.roles import (
    DocumentCollection,
    get_access_roles_for_collection,
)


SUPPORTED_DOCUMENT_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".md"})


@dataclass(frozen=True)
class DocumentInventoryItem:
    """Represents one source document prepared for ingestion."""

    file_name: str
    file_path: str
    relative_path: str
    collection: str
    file_extension: str
    size_bytes: int
    access_roles: list[str]


@dataclass(frozen=True)
class ChunkMetadataTemplate:
    """
    Represents required metadata fields for future vector-store chunks.

    Actual section_title and chunk_type values will be populated during
    Docling parsing and chunk creation.
    """

    source_document: str
    collection: str
    access_roles: list[str]
    section_title: str
    chunk_type: str


def validate_raw_data_dir(raw_data_dir: Path) -> None:
    """Ensure the extracted dataset directory exists before inventory building."""

    if not raw_data_dir.exists():
        raise DatasetNotFoundError(
            f"Raw data directory not found: {raw_data_dir}. "
            "Please run Phase 2 dataset extraction first."
        )

    if not raw_data_dir.is_dir():
        raise DatasetNotFoundError(
            f"Raw data path exists but is not a directory: {raw_data_dir}"
        )


def get_collection_directory(
    raw_data_dir: Path,
    collection: DocumentCollection,
) -> Path:
    """Return the directory path for a document collection."""

    return raw_data_dir / collection.value


def validate_collection_directory(collection_dir: Path) -> None:
    """Ensure a collection directory exists."""

    if not collection_dir.exists():
        raise DatasetNotFoundError(f"Collection directory not found: {collection_dir}")

    if not collection_dir.is_dir():
        raise DatasetNotFoundError(
            f"Collection path exists but is not a directory: {collection_dir}"
        )


def build_document_inventory(settings: AppSettings) -> list[DocumentInventoryItem]:
    """
    Scan dataset folders and build document inventory.

    This does not parse document content. It only prepares metadata needed
    before ingestion and vector indexing.
    """

    raw_data_dir = settings.raw_data_dir
    validate_raw_data_dir(raw_data_dir)

    inventory_items: list[DocumentInventoryItem] = []

    for collection in DocumentCollection:
        collection_dir = get_collection_directory(raw_data_dir, collection)
        validate_collection_directory(collection_dir)

        access_roles = [
            role.value for role in get_access_roles_for_collection(collection)
        ]

        document_paths = sorted(
            file_path
            for file_path in collection_dir.iterdir()
            if file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
        )

        for document_path in document_paths:
            inventory_items.append(
                DocumentInventoryItem(
                    file_name=document_path.name,
                    file_path=str(document_path),
                    relative_path=str(document_path.relative_to(raw_data_dir)),
                    collection=collection.value,
                    file_extension=document_path.suffix.lower(),
                    size_bytes=document_path.stat().st_size,
                    access_roles=access_roles,
                )
            )

    return inventory_items


def build_chunk_metadata_template(
    inventory_item: DocumentInventoryItem,
) -> ChunkMetadataTemplate:
    """
    Prepare required chunk metadata shape for a document.

    section_title and chunk_type will be finalized after parsing.
    """

    return ChunkMetadataTemplate(
        source_document=inventory_item.file_name,
        collection=inventory_item.collection,
        access_roles=inventory_item.access_roles,
        section_title="UNPARSED_SECTION",
        chunk_type="unparsed",
    )


def summarize_inventory(
    inventory_items: list[DocumentInventoryItem],
) -> dict[str, Any]:
    """Create a compact summary grouped by collection."""

    summary: dict[str, Any] = {
        "total_documents": len(inventory_items),
        "collections": {},
    }

    for item in inventory_items:
        collection_summary = summary["collections"].setdefault(
            item.collection,
            {
                "document_count": 0,
                "total_size_bytes": 0,
                "access_roles": item.access_roles,
                "documents": [],
            },
        )

        collection_summary["document_count"] += 1
        collection_summary["total_size_bytes"] += item.size_bytes
        collection_summary["documents"].append(item.file_name)

    return summary


def save_document_inventory(
    inventory_items: list[DocumentInventoryItem],
    output_path: Path,
) -> None:
    """Save document inventory as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "documents": [asdict(item) for item in inventory_items],
        "chunk_metadata_templates": [
            asdict(build_chunk_metadata_template(item)) for item in inventory_items
        ],
        "summary": summarize_inventory(inventory_items),
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )