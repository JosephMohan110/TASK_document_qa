"""Grounded LLM generation and citation validation."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from utils import NO_ANSWER, page_label, parse_citation_ids

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a document question-answering assistant.
Answer the user's question using ONLY the provided sources.
Rules:
1. Do not use outside knowledge.
2. Do not invent or assume facts.
3. Every factual statement must include a citation.
4. Cite sources using [S1], [S2], etc.
5. Use only citation IDs that appear in the supplied context.
6. If the documents do not contain enough information, say so explicitly.
7. Do not create fake citations.
8. Prefer concise answers that directly answer the user's question.
"""


def format_context(passages: list[dict[str, Any]]) -> str:
    """Format retrieved passages with all metadata needed for traceability."""
    sections = []
    for passage in passages:
        sections.append(
            f"[{passage['citation_id']}]\n"
            f"Source: {passage['source']}\n"
            f"{page_label(passage['page'])}\n\n"
            f"{passage['text']}"
        )
    return "\n\n---\n\n".join(sections)


def validate_citations(answer: str, valid_ids: set[str]) -> tuple[bool, list[str]]:
    """Reject citations not present in the supplied retrieval context."""
    citations = parse_citation_ids(answer)
    invalid = [citation for citation in citations if citation not in valid_ids]
    if not citations and answer.strip() != NO_ANSWER:
        invalid.append("missing citation")
    if invalid:
        logger.warning("Invalid citation IDs returned by model: %s", invalid)
    return not invalid, invalid


def _extractive_answer(question: str, passages: list[dict[str, Any]]) -> str:
    """Build a conservative local answer from the highest-ranked evidence."""
    question_words = {
        word.lower()
        for word in re.findall(r"[A-Za-z0-9]{3,}", question)
        if word.lower() not in {"what", "when", "where", "which", "does", "have", "about", "from"}
    }
    selected: list[str] = []
    for passage in passages:
        sentences = re.split(r"(?<=[.!?])\s+", passage["text"].strip())
        ranked = sorted(
            sentences,
            key=lambda sentence: len(question_words & {word.lower() for word in re.findall(r"[A-Za-z0-9]{3,}", sentence)}),
            reverse=True,
        )
        if ranked and ranked[0].strip():
            selected.append(f"{ranked[0].strip()} [{passage['citation_id']}]")
        if len(selected) == 2:
            break
    return " ".join(selected) or NO_ANSWER


def _local_result(question: str, passages: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    """Return a cited local answer when a remote LLM is unavailable."""
    answer = _extractive_answer(question, passages)
    return {
        "answer": answer,
        "citation_ids": parse_citation_ids(answer),
        "valid": True,
        "error": None,
        "mode": "local",
        "notice": reason,
    }


def generate_answer(question: str, passages: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate a grounded answer and return only validated citation references."""
    if not passages:
        return {"answer": NO_ANSWER, "citation_ids": [], "valid": True, "error": None, "mode": "none", "notice": None}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _local_result(question, passages, "OpenAI is not configured. Showing a local extractive answer from the retrieved passages.")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Sources:\n\n{format_context(passages)}\n\nQuestion: {question}"},
            ],
        )
        answer = (response.choices[0].message.content or "").strip()
        valid, invalid = validate_citations(answer, {p["citation_id"] for p in passages})
        if not answer or not valid:
            return _local_result(question, passages, "The remote model returned an invalid answer. Showing a local cited answer instead.")
        return {"answer": answer, "citation_ids": parse_citation_ids(answer), "valid": True, "error": None, "mode": "openai", "notice": None}
    except Exception as exc:
        logger.exception("LLM request failed")
        return _local_result(question, passages, f"Remote answer generation failed. Showing a local cited answer instead. Details: {exc}")
