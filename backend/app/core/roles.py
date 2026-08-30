from __future__ import annotations

from enum import Enum


class UserRole(str, Enum):
    """Supported Medical-AI Bot user roles."""

    DOCTOR = "doctor"
    NURSE = "nurse"
    BILLING_EXECUTIVE = "billing_executive"
    TECHNICIAN = "technician"
    ADMIN = "admin"


class DocumentCollection(str, Enum):
    """Supported document collections."""

    GENERAL = "general"
    CLINICAL = "clinical"
    NURSING = "nursing"
    BILLING = "billing"
    EQUIPMENT = "equipment"


ROLE_COLLECTIONS: dict[UserRole, tuple[DocumentCollection, ...]] = {
    UserRole.DOCTOR: (
        DocumentCollection.GENERAL,
        DocumentCollection.CLINICAL,
        DocumentCollection.NURSING,
    ),
    UserRole.NURSE: (
        DocumentCollection.GENERAL,
        DocumentCollection.NURSING,
    ),
    UserRole.BILLING_EXECUTIVE: (
        DocumentCollection.GENERAL,
        DocumentCollection.BILLING,
    ),
    UserRole.TECHNICIAN: (
        DocumentCollection.GENERAL,
        DocumentCollection.EQUIPMENT,
    ),
    UserRole.ADMIN: (
        DocumentCollection.GENERAL,
        DocumentCollection.CLINICAL,
        DocumentCollection.NURSING,
        DocumentCollection.BILLING,
        DocumentCollection.EQUIPMENT,
    ),
}


COLLECTION_ACCESS_ROLES: dict[DocumentCollection, tuple[UserRole, ...]] = {
    DocumentCollection.GENERAL: (
        UserRole.DOCTOR,
        UserRole.NURSE,
        UserRole.BILLING_EXECUTIVE,
        UserRole.TECHNICIAN,
        UserRole.ADMIN,
    ),
    DocumentCollection.CLINICAL: (
        UserRole.DOCTOR,
        UserRole.ADMIN,
    ),
    DocumentCollection.NURSING: (
        UserRole.NURSE,
        UserRole.DOCTOR,
        UserRole.ADMIN,
    ),
    DocumentCollection.BILLING: (
        UserRole.BILLING_EXECUTIVE,
        UserRole.ADMIN,
    ),
    DocumentCollection.EQUIPMENT: (
        UserRole.TECHNICIAN,
        UserRole.ADMIN,
    ),
}


SQL_RAG_ALLOWED_ROLES: tuple[UserRole, ...] = (
    UserRole.BILLING_EXECUTIVE,
    UserRole.ADMIN,
)


def get_accessible_collections(role: UserRole) -> tuple[DocumentCollection, ...]:
    """Return document collections accessible by a specific role."""

    return ROLE_COLLECTIONS[role]


def get_access_roles_for_collection(
    collection: DocumentCollection,
) -> tuple[UserRole, ...]:
    """Return roles allowed to access a specific document collection."""

    return COLLECTION_ACCESS_ROLES[collection]


def can_access_collection(role: UserRole, collection: DocumentCollection) -> bool:
    """Check whether a role can access a document collection."""

    return collection in ROLE_COLLECTIONS[role]


def can_use_sql_rag(role: UserRole) -> bool:
    """Check whether a role is allowed to use SQL RAG."""

    return role in SQL_RAG_ALLOWED_ROLES


def parse_user_role(role_value: str) -> UserRole:
    """Convert a raw role string into a validated UserRole enum."""

    normalized_role = role_value.strip().lower()

    try:
        return UserRole(normalized_role)
    except ValueError as exc:
        valid_roles = ", ".join(role.value for role in UserRole)
        raise ValueError(
            f"Invalid role '{role_value}'. Valid roles are: {valid_roles}."
        ) from exc