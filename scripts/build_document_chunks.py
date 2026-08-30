from __future__ import annotations

import logging
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.rag.document_chunker import (
    load_document_inventory,
    parse_inventory_documents,
    save_document_chunks,
    summarize_chunks,
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def main() -> None:
    configure_logging()

    settings = get_settings()

    inventory_path = Path(settings.processed_data_dir) / "document_inventory.json"
    output_path = Path(settings.processed_data_dir) / "document_chunks.json"

    logging.info("Starting Docling parsing and chunk creation.")
    logging.info("Inventory path: %s", inventory_path)

    inventory_items = load_document_inventory(inventory_path)

    logging.info("Documents found for parsing: %s", len(inventory_items))
    logging.info("First Docling run may take extra time while models initialize.")

    chunks, failures = parse_inventory_documents(inventory_items)
    save_document_chunks(chunks, failures, output_path)

    summary = summarize_chunks(chunks)

    print("\nDOCUMENT CHUNKING SUMMARY")
    print("-" * 50)
    print(f"Documents attempted: {len(inventory_items)}")
    print(f"Total chunks created: {summary['total_chunks']}")
    print(f"Failed documents: {len(failures)}")

    print("\nCHUNKS BY COLLECTION")
    print("-" * 50)
    for collection_name, collection_summary in summary["collections"].items():
        print(f"{collection_name}: {collection_summary['chunk_count']} chunks")

        for document_name, chunk_count in collection_summary["documents"].items():
            print(f"  - {document_name}: {chunk_count}")

    print("\nCHUNKS BY TYPE")
    print("-" * 50)
    for chunk_type, chunk_count in summary["chunk_types"].items():
        print(f"{chunk_type}: {chunk_count}")

    if failures:
        print("\nPARSING FAILURES")
        print("-" * 50)
        for failure in failures:
            print(f"{failure.file_name}: {failure.error_message}")

        raise RuntimeError(
            "One or more documents failed during parsing. "
            "Please review the failures above."
        )

    print("\nChunks saved to:")
    print(output_path)

    logging.info("Document parsing and chunk creation completed successfully.")


if __name__ == "__main__":
    main()