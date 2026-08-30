from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.app.core.config import get_settings
from backend.app.core.exceptions import MedicalAIBotError
from backend.app.core.roles import get_accessible_collections, parse_user_role
from backend.app.schemas.auth import LoginRequest, LoginResponse
from backend.app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    CollectionsResponse,
    HealthResponse,
)
from backend.app.services.auth_service import AuthService
from backend.app.services.chat_service import ChatService


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return application health status."""

    settings = get_settings()

    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.app_env,
    )


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest) -> LoginResponse:
    """Authenticate a demo user and return a role-tagged token."""

    settings = get_settings()
    auth_service = AuthService(settings=settings)

    try:
        access_token, authenticated_user = auth_service.login(
            username=request.username,
            password=request.password,
        )
    except MedicalAIBotError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc

    return LoginResponse(
        access_token=access_token,
        username=authenticated_user.username,
        role=authenticated_user.role.value,
    )


@router.get("/collections/{role}", response_model=CollectionsResponse)
def get_collections(role: str) -> CollectionsResponse:
    """Return document collections accessible to a role."""

    try:
        user_role = parse_user_role(role)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    collections = [
        collection.value for collection in get_accessible_collections(user_role)
    ]

    return CollectionsResponse(
        role=user_role.value,
        collections=collections,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Main role-aware chat endpoint."""

    settings = get_settings()
    auth_service = AuthService(settings=settings)
    chat_service = ChatService(settings=settings)

    try:
        authenticated_user = auth_service.authenticate_token(request.access_token)
        response = chat_service.answer_question(
            question=request.question,
            role=authenticated_user.role,
        )
    except MedicalAIBotError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return ChatResponse(
        answer=response.answer,
        sources=response.sources,
        retrieval_type=response.retrieval_type,
        role=response.role,
    )