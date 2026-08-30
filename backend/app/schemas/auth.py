from __future__ import annotations

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    """Login request for demo users."""

    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class LoginResponse(BaseModel):
    """Login response containing a role-tagged access token."""

    access_token: str
    token_type: str = "bearer"
    username: str
    role: str