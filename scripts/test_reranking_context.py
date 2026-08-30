from __future__ import annotations

import argparse
import logging

from backend.app.core.config import get_settings
from backend.app.core.roles import UserRole, parse_user_role
from backend.app.rag.context_builder import (
    AnswerContextBuilder,
    citations_to_response_payload,
)
from backend.app.rag.embeddings import EmbeddingService
from backend.app.rag.hybrid_retriever import HybridRetriever
from backend.app.rag.reranker import CrossEncoderReranker
from backend.app.rag.vector_store import QdrantCollectionManager


DEFAULT_ROLE = UserRole.DOCTOR
DEFAULT_QUESTION = "What drug formulary guidance is available for antibiotic use?"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test cross-encoder reranking and LLM-ready context building."
    )

    parser.add_argument(
        "--role",
        type=str,
        default=DEFAULT_ROLE.value,
        help="User role for the test query.",
    )

    parser.add_argument(
        "--question",
        type=str,
        default=DEFAULT_QUESTION,
        help="Question to retrieve and rerank against.",
    )

    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=10,
        help="Number of hybrid retrieval candidates before reranking.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of reranked chunks to keep for context.",
    )

    return parser.parse_args()


def print_retrieved_candidates(candidates: list[object]) -> None:
    print("\nINITIAL HYBRID RETRIEVAL CANDIDATES")
    print("-" * 70)

    for index, chunk in enumerate(candidates, start=1):
        print(
            f"{index}. {chunk.collection} | {chunk.source_document} | "
            f"hybrid_score={chunk.score:.6f}"
        )


def print_reranked_chunks(reranked_chunks: list[object]) -> None:
    print("\nRERANKED TOP CHUNKS")
    print("-" * 70)

    for index, chunk in enumerate(reranked_chunks, start=1):
        print(f"\nRank {index}")
        print(f"Original rank: {chunk.original_rank}")
        print(f"Original hybrid score: {chunk.original_score:.6f}")
        print(f"Rerank score: {chunk.rerank_score:.6f}")
        print(f"Collection: {chunk.collection}")
        print(f"Source: {chunk.source_document}")
        print(f"Section: {chunk.section_title}")
        print(f"Access roles: {chunk.access_roles}")

        preview = chunk.text.replace("\n", " ").strip()
        if len(preview) > 260:
            preview = f"{preview[:260]}..."

        print(f"Preview: {preview}")


def print_context_preview(context_text: str) -> None:
    print("\nLLM-READY CONTEXT PREVIEW")
    print("-" * 70)

    preview = context_text

    if len(preview) > 1500:
        preview = f"{preview[:1500]}..."

    print(preview)


def main() -> None:
    configure_logging()
    args = parse_args()

    if args.candidate_limit <= 0:
        raise ValueError("--candidate-limit must be greater than zero.")

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero.")

    role = parse_user_role(args.role)
    question = args.question.strip()

    settings = get_settings()

    logging.info("Starting reranking + context builder test.")
    logging.info("Role: %s", role.value)
    logging.info("Question: %s", question)

    collection_manager = QdrantCollectionManager(settings=settings)
    embedding_service = EmbeddingService(settings=settings)
    retriever = HybridRetriever(
        settings=settings,
        collection_manager=collection_manager,
        embedding_service=embedding_service,
    )
    reranker = CrossEncoderReranker(settings=settings)
    context_builder = AnswerContextBuilder(settings=settings)

    try:
        candidates = retriever.retrieve(
            question=question,
            role=role,
            limit=args.candidate_limit,
        )

        reranked_chunks = reranker.rerank(
            question=question,
            retrieved_chunks=candidates,
            role=role,
            top_k=args.top_k,
        )

        answer_context = context_builder.build_context(
            question=question,
            reranked_chunks=reranked_chunks,
            role=role,
        )

        print("\nRERANKING + CONTEXT SUMMARY")
        print("=" * 70)
        print(f"Role: {role.value}")
        print(f"Question: {question}")
        print(f"Candidates retrieved: {len(candidates)}")
        print(f"Reranked chunks kept: {len(reranked_chunks)}")
        print(f"Context chunks used: {answer_context.chunks_used}")
        print(f"Context characters: {answer_context.total_chars}")

        print_retrieved_candidates(candidates)
        print_reranked_chunks(reranked_chunks)
        print_context_preview(answer_context.context_text)

        print("\nSOURCE CITATIONS PAYLOAD")
        print("-" * 70)
        for source in citations_to_response_payload(answer_context.sources):
            print(source)

        print("\nRERANKING CONTEXT CHECK: PASSED")
        logging.info("Reranking + context builder test completed successfully.")

    finally:
        collection_manager.close()


if __name__ == "__main__":
    main()