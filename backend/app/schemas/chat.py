from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat request sent by the frontend."""

    question: str = Field(min_length=1)
    access_token: str = Field(min_length=1)


class SourceResponse(BaseModel):
    """Source citation returned with document-based answers."""

    source_document: str
    section_title: str
    collection: str


class ChatResponse(BaseModel):
    """Main chat response."""

    answer: str
    sources: list[SourceResponse]
    retrieval_type: str
    role: str


class CollectionsResponse(BaseModel):
    """Collections available for a role."""

    role: str
    collections: list[str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    app_name: str
    environment: str