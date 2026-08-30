from __future__ import annotations


class MedicalAIBotError(Exception):
    """Base exception for Medical-AI Bot application errors."""


class ConfigurationError(MedicalAIBotError):
    """Raised when application configuration is invalid."""


class RoleAccessError(MedicalAIBotError):
    """Raised when a role tries to access unauthorized resources."""


class DatasetNotFoundError(MedicalAIBotError):
    """Raised when required dataset files or folders are missing."""


class RetrievalError(MedicalAIBotError):
    """Raised when the retrieval pipeline fails."""


class SQLRAGError(MedicalAIBotError):
    """Raised when SQL RAG processing fails."""


class LLMServiceError(MedicalAIBotError):
    """Raised when the external LLM service fails."""

class DocumentParsingError(MedicalAIBotError):
    """Raised when document parsing or chunking fails."""