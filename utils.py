"""Shared configuration, validation, and citation helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, TypedDict

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"
MANIFEST_PATH = VECTOR_STORE_DIR / "manifest.json"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
NO_ANSWER = "The provided documents do not contain enough information to answer this question."


class AppConfig(TypedDict):
    embedding_model: str
    openai_model: str
    min_score: float
    top_k: int


class ConfigurationError(ValueError):
    """Raised when an environment setting is missing or invalid."""


def load_config() -> AppConfig:
    """Load runtime configuration from the local environment."""
    load_dotenv(BASE_DIR / ".env")
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    if not embedding_model or not openai_model:
        raise ConfigurationError("EMBEDDING_MODEL and OPENAI_MODEL must not be empty.")
    try:
        min_score = float(os.getenv("MIN_SCORE", "0.25"))
        top_k = int(os.getenv("TOP_K", "5"))
    except ValueError as exc:
        raise ConfigurationError("MIN_SCORE must be a number and TOP_K must be an integer.") from exc
    if not 0 <= min_score <= 1:
        raise ConfigurationError("MIN_SCORE must be between 0 and 1.")
    if top_k < 1:
        raise ConfigurationError("TOP_K must be at least 1.")
    return {"embedding_model": embedding_model, "openai_model": openai_model, "min_score": min_score, "top_k": top_k}


def ensure_directories() -> None:
    """Create runtime directories used by uploads and the vector store."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


def validate_extension(filename: str) -> None:
    """Raise a useful error when a file type is unsupported."""
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{extension or 'none'}'. Supported: {supported}")


def parse_citation_ids(text: str) -> list[str]:
    """Return citation IDs in first-seen order, without duplicates."""
    found = re.findall(r"\[(S\d+)\]", text or "")
    return list(dict.fromkeys(found))


def page_label(page: int | None) -> str:
    """Format optional page metadata for display."""
    return f"Page {page}" if page is not None else "Page unavailable"


def clean_text(text: str) -> str:
    """Normalize extracted text while preserving paragraph boundaries."""
    paragraphs = [" ".join(line.split()) for line in text.splitlines()]
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    """Return existing files once, in stable name order."""
    return sorted({path.resolve() for path in paths if path.is_file()}, key=lambda path: path.name.lower())
