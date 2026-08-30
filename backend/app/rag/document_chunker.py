from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

from backend.app.core.exceptions import DatasetNotFoundError, DocumentParsingError
from backend.app.rag.document_inventory import DocumentInventoryItem


VALID_CHUNK_TYPES: frozenset[str] = frozenset({"text", "table", "heading", "code"})


@dataclass(frozen=True)
class DocumentChunk:
    """A RAG-ready chunk with required vector-store metadata."""

    chunk_id: str
    text: str
    chunk_index: int
    char_count: int
    source_path: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DocumentParsingFailure:
    """Represents a document that failed during conversion or chunking."""

    file_name: str
    file_path: str
    error_message: str


def load_document_inventory(inventory_path: Path) -> list[DocumentInventoryItem]:
    """Load document inventory generated in Phase 4."""

    if not inventory_path.exists():
        raise DatasetNotFoundError(
            f"Document inventory not found: {inventory_path}. "
            "Please build the document inventory first: python -m scripts.build_document_inventory"
        )

    payload = json.loads(inventory_path.read_text(encoding="utf-8"))

    documents = payload.get("documents", [])
    if not documents:
        raise DatasetNotFoundError(
            f"No documents found inside inventory file: {inventory_path}"
        )

    return [DocumentInventoryItem(**document) for document in documents]


def create_safe_chunk_id(
    collection: str,
    source_document: str,
    chunk_index: int,
) -> str:
    """Create a deterministic chunk ID that is safe for JSON and vector DB usage."""

    document_stem = Path(source_document).stem.lower()
    safe_document_stem = re.sub(r"[^a-z0-9]+", "_", document_stem).strip("_")

    return f"{collection}_{safe_document_stem}_{chunk_index:04d}"


def infer_section_title(chunk: Any) -> str:
    """
    Extract the most specific available section heading from a Docling chunk.

    Docling chunks can carry metadata such as headings. We use the deepest
    heading available because it is usually the most useful citation label.
    """

    metadata = getattr(chunk, "meta", None)
    headings = getattr(metadata, "headings", None)

    if headings:
        cleaned_headings = [
            str(heading).strip()
            for heading in headings
            if str(heading).strip()
        ]

        if cleaned_headings:
            return cleaned_headings[-1]

    return "General"


def infer_chunk_type(chunk: Any, contextualized_text: str) -> str:
    """
    Infer assignment-compatible chunk type.

    Required values are: text, table, heading, code.
    """

    normalized_text = contextualized_text.strip()

    if "```" in normalized_text:
        return "code"

    if normalized_text.startswith("|") or "\n|" in normalized_text:
        return "table"

    metadata = getattr(chunk, "meta", None)
    doc_items = getattr(metadata, "doc_items", []) or []

    for item in doc_items:
        item_type = type(item).__name__.lower()

        if "table" in item_type:
            return "table"

        if "code" in item_type:
            return "code"

        if "heading" in item_type or "title" in item_type:
            return "heading"

    return "text"


def build_chunk_metadata(
    inventory_item: DocumentInventoryItem,
    section_title: str,
    chunk_type: str,
) -> dict[str, Any]:
    """Build the required vector-store metadata for one chunk."""

    if chunk_type not in VALID_CHUNK_TYPES:
        raise DocumentParsingError(
            f"Invalid chunk_type '{chunk_type}' for {inventory_item.file_name}. "
            f"Valid values are: {sorted(VALID_CHUNK_TYPES)}"
        )

    return {
        "source_document": inventory_item.file_name,
        "collection": inventory_item.collection,
        "access_roles": inventory_item.access_roles,
        "section_title": section_title,
        "chunk_type": chunk_type,
    }


class DoclingDocumentChunker:
    """Converts source documents into structured chunks using Docling."""

    def __init__(self) -> None:
        self._converter = DocumentConverter()
        self._chunker = HybridChunker()

    def parse_document(self, inventory_item: DocumentInventoryItem) -> list[DocumentChunk]:
        """Parse and chunk a single source document."""

        source_path = Path(inventory_item.file_path)

        if not source_path.exists():
            raise DatasetNotFoundError(f"Source document not found: {source_path}")

        try:
            conversion_result = self._converter.convert(source=str(source_path))
            docling_document = conversion_result.document
            docling_chunks = list(self._chunker.chunk(dl_doc=docling_document))
        except Exception as exc:
            raise DocumentParsingError(
                f"Failed to parse document {inventory_item.file_name}: {exc}"
            ) from exc

        parsed_chunks: list[DocumentChunk] = []

        for chunk_index, chunk in enumerate(docling_chunks, start=1):
            contextualized_text = self._chunker.contextualize(chunk=chunk).strip()

            if not contextualized_text:
                continue

            section_title = infer_section_title(chunk)
            chunk_type = infer_chunk_type(chunk, contextualized_text)

            metadata = build_chunk_metadata(
                inventory_item=inventory_item,
                section_title=section_title,
                chunk_type=chunk_type,
            )

            parsed_chunks.append(
                DocumentChunk(
                    chunk_id=create_safe_chunk_id(
                        collection=inventory_item.collection,
                        source_document=inventory_item.file_name,
                        chunk_index=chunk_index,
                    ),
                    text=contextualized_text,
                    chunk_index=chunk_index,
                    char_count=len(contextualized_text),
                    source_path=inventory_item.file_path,
                    metadata=metadata,
                )
            )

        if not parsed_chunks:
            raise DocumentParsingError(
                f"No chunks were created for document: {inventory_item.file_name}"
            )

        return parsed_chunks


def parse_inventory_documents(
    inventory_items: list[DocumentInventoryItem],
) -> tuple[list[DocumentChunk], list[DocumentParsingFailure]]:
    """Parse all inventory documents and collect failures safely."""

    chunker = DoclingDocumentChunker()

    all_chunks: list[DocumentChunk] = []
    failures: list[DocumentParsingFailure] = []

    for inventory_item in inventory_items:
        try:
            document_chunks = chunker.parse_document(inventory_item)
            all_chunks.extend(document_chunks)
        except Exception as exc:
            failures.append(
                DocumentParsingFailure(
                    file_name=inventory_item.file_name,
                    file_path=inventory_item.file_path,
                    error_message=str(exc),
                )
            )

    return all_chunks, failures


def summarize_chunks(chunks: list[DocumentChunk]) -> dict[str, Any]:
    """Summarize parsed chunks by collection, source document, and chunk type."""

    summary: dict[str, Any] = {
        "total_chunks": len(chunks),
        "collections": {},
        "chunk_types": {},
    }

    for chunk in chunks:
        collection = chunk.metadata["collection"]
        source_document = chunk.metadata["source_document"]
        chunk_type = chunk.metadata["chunk_type"]

        collection_summary = summary["collections"].setdefault(
            collection,
            {
                "chunk_count": 0,
                "documents": {},
            },
        )

        collection_summary["chunk_count"] += 1
        collection_summary["documents"][source_document] = (
            collection_summary["documents"].get(source_document, 0) + 1
        )

        summary["chunk_types"][chunk_type] = (
            summary["chunk_types"].get(chunk_type, 0) + 1
        )

    return summary


def save_document_chunks(
    chunks: list[DocumentChunk],
    failures: list[DocumentParsingFailure],
    output_path: Path,
) -> None:
    """Save parsed chunks and parsing summary to JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "chunks": [asdict(chunk) for chunk in chunks],
        "failures": [asdict(failure) for failure in failures],
        "summary": summarize_chunks(chunks),
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )