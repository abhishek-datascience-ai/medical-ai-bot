from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any

from backend.app.core.config import AppSettings
from backend.app.core.exceptions import RoleAccessError
from backend.app.core.roles import UserRole


@dataclass(frozen=True)
class DemoUser:
    """Demo account used for assignment login."""

    username: str
    password: str
    role: UserRole


@dataclass(frozen=True)
class AuthenticatedUser:
    """Authenticated user extracted from a signed token."""

    username: str
    role: UserRole


DEMO_USERS: dict[str, DemoUser] = {
    "dr.mehta": DemoUser(
        username="dr.mehta",
        password="doctor",
        role=UserRole.DOCTOR,
    ),
    "nurse.priya": DemoUser(
        username="nurse.priya",
        password="nurse",
        role=UserRole.NURSE,
    ),
    "billing.ravi": DemoUser(
        username="billing.ravi",
        password="billing_executive",
        role=UserRole.BILLING_EXECUTIVE,
    ),
    "tech.anand": DemoUser(
        username="tech.anand",
        password="technician",
        role=UserRole.TECHNICIAN,
    ),
    "admin.sys": DemoUser(
        username="admin.sys",
        password="admin",
        role=UserRole.ADMIN,
    ),
}


class AuthService:
    """Simple signed-token authentication service for demo users."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._secret_key = settings.secret_key or "development-only-secret"

    def login(self, username: str, password: str) -> tuple[str, AuthenticatedUser]:
        """Validate demo credentials and return signed token."""

        normalized_username = username.strip()
        user = DEMO_USERS.get(normalized_username)

        if user is None or not hmac.compare_digest(user.password, password):
            raise RoleAccessError("Invalid username or password.")

        authenticated_user = AuthenticatedUser(
            username=user.username,
            role=user.role,
        )
        token = self._create_token(authenticated_user)

        return token, authenticated_user

    def authenticate_token(self, token: str) -> AuthenticatedUser:
        """Validate signed token and return authenticated user details."""

        try:
            payload_b64, signature = token.split(".", maxsplit=1)
        except ValueError as exc:
            raise RoleAccessError("Invalid access token format.") from exc

        expected_signature = self._sign_payload(payload_b64)

        if not hmac.compare_digest(signature, expected_signature):
            raise RoleAccessError("Invalid access token signature.")

        payload = self._decode_payload(payload_b64)

        expires_at = int(payload.get("expires_at", 0))
        if expires_at < int(time.time()):
            raise RoleAccessError("Access token has expired.")

        username = str(payload.get("username", "")).strip()
        role_value = str(payload.get("role", "")).strip()

        if not username or not role_value:
            raise RoleAccessError("Access token is missing required claims.")

        try:
            role = UserRole(role_value)
        except ValueError as exc:
            raise RoleAccessError(f"Invalid role in access token: {role_value}") from exc

        return AuthenticatedUser(username=username, role=role)

    def _create_token(self, user: AuthenticatedUser) -> str:
        issued_at = int(time.time())
        expires_at = issued_at + (self._settings.access_token_expire_minutes * 60)

        payload = {
            "username": user.username,
            "role": user.role.value,
            "issued_at": issued_at,
            "expires_at": expires_at,
        }

        payload_b64 = self._encode_payload(payload)
        signature = self._sign_payload(payload_b64)

        return f"{payload_b64}.{signature}"

    @staticmethod
    def _encode_payload(payload: dict[str, Any]) -> str:
        payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        payload_bytes = payload_json.encode("utf-8")
        return base64.urlsafe_b64encode(payload_bytes).decode("utf-8").rstrip("=")

    @staticmethod
    def _decode_payload(payload_b64: str) -> dict[str, Any]:
        padded_payload = payload_b64 + "=" * (-len(payload_b64) % 4)

        try:
            payload_bytes = base64.urlsafe_b64decode(padded_payload.encode("utf-8"))
            payload = json.loads(payload_bytes.decode("utf-8"))
        except Exception as exc:
            raise RoleAccessError("Invalid access token payload.") from exc

        if not isinstance(payload, dict):
            raise RoleAccessError("Invalid access token payload type.")

        return payload

    def _sign_payload(self, payload_b64: str) -> str:
        digest = hmac.new(
            key=self._secret_key.encode("utf-8"),
            msg=payload_b64.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()

        return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")