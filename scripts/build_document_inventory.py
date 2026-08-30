from __future__ import annotations

import logging
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.rag.document_inventory import (
    build_document_inventory,
    save_document_inventory,
    summarize_inventory,
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def main() -> None:
    configure_logging()

    settings = get_settings()
    output_path = Path(settings.processed_data_dir) / "document_inventory.json"

    logging.info("Building document inventory for Medical-AI Bot.")
    logging.info("Raw data directory: %s", settings.raw_data_dir)

    inventory_items = build_document_inventory(settings)
    save_document_inventory(inventory_items, output_path)

    summary = summarize_inventory(inventory_items)

    print("\nDOCUMENT INVENTORY SUMMARY")
    print("-" * 50)
    print(f"Total documents: {summary['total_documents']}")

    for collection_name, collection_details in summary["collections"].items():
        print(
            f"{collection_name}: "
            f"documents={collection_details['document_count']}, "
            f"roles={collection_details['access_roles']}"
        )

        for document_name in collection_details["documents"]:
            print(f"  - {document_name}")

    print("\nInventory saved to:")
    print(output_path)

    logging.info("Document inventory completed successfully.")


if __name__ == "__main__":
    main()