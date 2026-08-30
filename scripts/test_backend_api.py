from __future__ import annotations

import logging
from typing import Any

from fastapi.testclient import TestClient

from backend.app.main import app


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def assert_status(response: Any, expected_status_code: int) -> None:
    if response.status_code != expected_status_code:
        raise AssertionError(
            f"Expected status {expected_status_code}, got {response.status_code}. "
            f"Response: {response.text}"
        )


def login(client: TestClient, username: str, password: str) -> dict[str, Any]:
    response = client.post(
        "/login",
        json={
            "username": username,
            "password": password,
        },
    )
    assert_status(response, 200)
    return response.json()


def test_health(client: TestClient) -> None:
    print_section("HEALTH CHECK")

    response = client.get("/health")
    assert_status(response, 200)

    payload = response.json()
    print(payload)

    if payload["status"] != "ok":
        raise AssertionError("Health check failed.")


def test_login(client: TestClient) -> dict[str, Any]:
    print_section("LOGIN CHECK")

    payload = login(
        client=client,
        username="billing.ravi",
        password="billing_executive",
    )

    print(
        {
            "username": payload["username"],
            "role": payload["role"],
            "token_type": payload["token_type"],
            "has_access_token": bool(payload["access_token"]),
        }
    )

    if payload["role"] != "billing_executive":
        raise AssertionError("Unexpected role returned from login.")

    return payload


def test_collections(client: TestClient) -> None:
    print_section("COLLECTIONS CHECK")

    response = client.get("/collections/nurse")
    assert_status(response, 200)

    payload = response.json()
    print(payload)

    if payload["collections"] != ["general", "nursing"]:
        raise AssertionError("Unexpected nurse collection access.")


def test_sql_chat(client: TestClient, access_token: str) -> None:
    print_section("SQL CHAT CHECK")

    response = client.post(
        "/chat",
        json={
            "question": "How many claims are rejected?",
            "access_token": access_token,
        },
    )
    assert_status(response, 200)

    payload = response.json()
    print(
        {
            "answer": payload["answer"],
            "retrieval_type": payload["retrieval_type"],
            "role": payload["role"],
            "sources": payload["sources"],
        }
    )

    if payload["retrieval_type"] != "sql_rag":
        raise AssertionError("Expected SQL RAG response.")


def test_blocked_sql_chat(client: TestClient) -> None:
    print_section("BLOCKED SQL CHAT CHECK")

    nurse_login = login(
        client=client,
        username="nurse.priya",
        password="nurse",
    )

    response = client.post(
        "/chat",
        json={
            "question": "How many claims are rejected?",
            "access_token": nurse_login["access_token"],
        },
    )
    assert_status(response, 200)

    payload = response.json()
    print(
        {
            "answer": payload["answer"],
            "retrieval_type": payload["retrieval_type"],
            "role": payload["role"],
            "sources": payload["sources"],
        }
    )

    if payload["retrieval_type"] != "blocked":
        raise AssertionError("Expected blocked SQL response for nurse.")


def test_document_chat(client: TestClient) -> None:
    print_section("DOCUMENT CHAT CHECK")

    doctor_login = login(
        client=client,
        username="dr.mehta",
        password="doctor",
    )

    response = client.post(
        "/chat",
        json={
            "question": "What drug formulary guidance is available for antibiotic use?",
            "access_token": doctor_login["access_token"],
        },
    )
    assert_status(response, 200)

    payload = response.json()
    print(
        {
            "answer_preview": payload["answer"][:500],
            "retrieval_type": payload["retrieval_type"],
            "role": payload["role"],
            "sources": payload["sources"],
        }
    )

    if payload["retrieval_type"] != "hybrid_rag":
        raise AssertionError("Expected hybrid RAG response.")

    if not payload["sources"]:
        raise AssertionError("Expected document sources.")


def main() -> None:
    configure_logging()
    logging.info("Starting backend API checks.")

    client = TestClient(app)

    test_health(client)
    login_payload = test_login(client)
    test_collections(client)
    test_sql_chat(client, access_token=login_payload["access_token"])
    test_blocked_sql_chat(client)
    test_document_chat(client)

    print("\n" + "=" * 80)
    print("BACKEND API CHECK: PASSED")
    print("=" * 80)

    logging.info("Backend API checks completed successfully.")


if __name__ == "__main__":
    main()