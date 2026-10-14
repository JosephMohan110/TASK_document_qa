"""Semantic retrieval over the synchronized FAISS index."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Load and reuse an embedding model for the current process."""
    return SentenceTransformer(model_name)


def retrieve(
    question: str,
    index_path: Path,
    metadata_path: Path,
    model_name: str,
    top_k: int = 5,
    min_score: float = 0.40,
) -> list[dict[str, Any]]:
    """Return above-threshold chunks with stable citation IDs and scores."""
    manifest_path = index_path.with_name("manifest.json")
    if not index_path.exists() or not metadata_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("The document index does not exist. Index documents before asking a question.")
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if top_k < 1 or not 0 <= min_score <= 1:
        raise ValueError("Retrieval settings are invalid. TOP_K must be positive and MIN_SCORE must be between 0 and 1.")
    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("embedding_model") != model_name:
        raise ValueError("The index was built with a different embedding model. Rebuild the document index.")
    if manifest.get("vector_count") != index.ntotal or index.ntotal != len(metadata):
        raise ValueError("The vector index and metadata are out of sync. Rebuild the document index.")
    if index.ntotal == 0 or not metadata:
        return []
    model = get_embedding_model(model_name)
    query_vector = model.encode([question], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    if query_vector.shape[1] != index.d:
        raise ValueError("The query embedding dimension does not match the index. Rebuild the document index.")
    scores, positions = index.search(query_vector, min(top_k, index.ntotal))
    results: list[dict[str, Any]] = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0 or float(score) < min_score:
            continue
        item = dict(metadata[position])
        item["citation_id"] = f"S{len(results) + 1}"
        item["score"] = round(float(score), 4)
        results.append(item)
    return results
