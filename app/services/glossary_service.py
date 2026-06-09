"""Business Glossary Resolution Service.

Purpose
-------
Domain users frequently use abbreviations, acronyms, and business terms that
have no direct representation in the database schema.  This service:

1. Searches the ``business_glossary_collection`` in Qdrant using the user's
   raw question embedding.
2. Returns the top-k matching glossary entries (each carrying a human-readable
   ``meaning`` and an optional ``sql_hint``).
3. Produces an **expanded query** that appends the matched context to the
   original question — so that downstream embedding, DDL retrieval, and
   re-ranking all benefit from the richer text.

Expanded query format
---------------------
::

    Give me MDCS count in Gujarat

    Business Meaning:
    MDCS = Dairy Cooperative Society registered after 2023-02-21
    PACS = Primary Agriculture Credit Society

    SQL Hints:
    MDCS: society_type='DAIRY' AND registration_date>'2023-02-21'

Pipeline position
-----------------
::

    User Question
         │
         ▼
    ★ Glossary Resolution (this service) ★
         │  expands query with business meanings & SQL hints
         ▼
    Embedding Generation  (uses expanded query)
         │
         ▼
    Qdrant DDL / QA / Docs retrieval
         │
         ▼
    BM25 pre-filter + CrossEncoder re-ranking
         │
         ▼
    LLM prompt
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from qdrant_client.models import PointStruct, ScoredPoint

from app.config import get_settings
from app.services.embedding_service import get_embedding
from app.services.qdrant_service import get_client, search_glossary

logger = logging.getLogger(__name__)
settings = get_settings()

# Minimum cosine similarity for a glossary match to be included.
# Tune lower (e.g. 0.3) for broader matching, higher (e.g. 0.7) for strict.
_SCORE_THRESHOLD: float = 0.45


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class GlossaryMatch:
    """A single resolved glossary entry.

    Attributes:
        term:     The business term / acronym (e.g. ``"MDCS"``).
        meaning:  Human-readable expansion (e.g. ``"Dairy Cooperative Society..."``).
        sql_hint: Optional SQL fragment that captures the business rule.
        score:    Qdrant cosine similarity score.
    """

    term: str
    meaning: str
    sql_hint: str
    score: float


@dataclass
class GlossaryResolutionResult:
    """Output of :func:`resolve_glossary`.

    Attributes:
        original_query: The user's original question unchanged.
        matches:        Glossary entries that matched the question.
        expanded_query: The enriched query ready for embedding and retrieval.
                        Equal to ``original_query`` when no matches are found.
    """

    original_query: str
    matches: list[GlossaryMatch]
    expanded_query: str


# ---------------------------------------------------------------------------
# Qdrant write helper
# ---------------------------------------------------------------------------
def push_glossary_record(
    term: str,
    meaning: str,
    sql_hint: str,
    category: str,
    embedding: list[float],
) -> str:
    """Upsert a single glossary entry into Qdrant.

    The embedding is generated from ``"{term} {meaning}"`` so that both the
    acronym and its expansion are searchable.

    Returns:
        The UUID string of the inserted point.
    """
    client = get_client()
    point_id = str(uuid.uuid4())
    client.upsert(
        collection_name=settings.qdrant_glossary_collection_name,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "term": term,
                    "meaning": meaning,
                    "sql_hint": sql_hint,
                    "category": category,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        ],
    )
    return point_id


# ---------------------------------------------------------------------------
# Core resolution logic
# ---------------------------------------------------------------------------
def resolve_glossary(
    user_question: str,
    top_k: int = 5,
    score_threshold: float = _SCORE_THRESHOLD,
) -> GlossaryResolutionResult:
    """Resolve business terms in *user_question* and expand it.

    Steps:
    1. Embed the original question.
    2. Search the glossary collection (semantic similarity).
    3. Convert raw Qdrant hits to :class:`GlossaryMatch` objects.
    4. Build the expanded query string.
    5. Log all matched terms for debugging.

    Args:
        user_question:   The raw user input.
        top_k:           Maximum number of glossary entries to retrieve.
        score_threshold: Minimum cosine similarity to accept a match.

    Returns:
        A :class:`GlossaryResolutionResult` with the original query, matches,
        and the expanded query string.
    """
    query_vector = get_embedding(user_question)

    try:
        hits: list[ScoredPoint] = search_glossary(
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
        )
    except Exception as exc:
        # Glossary collection might not exist yet — degrade gracefully.
        logger.warning(
            "Glossary search failed (collection may not exist yet): %s", exc
        )
        return GlossaryResolutionResult(
            original_query=user_question,
            matches=[],
            expanded_query=user_question,
        )

    matches: list[GlossaryMatch] = [
        GlossaryMatch(
            term=hit.payload.get("term", ""),
            meaning=hit.payload.get("meaning", ""),
            sql_hint=hit.payload.get("sql_hint", ""),
            score=hit.score,
        )
        for hit in hits
        if hit.payload.get("term")  # skip malformed entries
    ]

    # ── Logging ──────────────────────────────────────────────────────────────
    if matches:
        logger.info(
            "Glossary resolution: %d term(s) matched for query: %r",
            len(matches),
            user_question,
        )
        for m in matches:
            logger.info(
                "  [%.4f] %s = %s  |  sql_hint: %s",
                m.score,
                m.term,
                m.meaning,
                m.sql_hint or "(none)",
            )
    else:
        logger.info(
            "Glossary resolution: no matches above threshold=%.2f for query: %r",
            score_threshold,
            user_question,
        )

    expanded_query = _build_expanded_query(user_question, matches)

    if expanded_query != user_question:
        logger.info("Expanded query:\n%s", expanded_query)

    return GlossaryResolutionResult(
        original_query=user_question,
        matches=matches,
        expanded_query=expanded_query,
    )


# ---------------------------------------------------------------------------
# Query expansion builder
# ---------------------------------------------------------------------------
def _build_expanded_query(question: str, matches: list[GlossaryMatch]) -> str:
    """Append matched glossary context to the original question.

    Returns the original question unchanged when there are no matches.
    """
    if not matches:
        return question

    lines: list[str] = [question, ""]

    # Business Meaning section
    lines.append("Business Meaning:")
    for m in matches:
        lines.append(f"{m.term} = {m.meaning}")

    # SQL Hints section — only for entries that carry a hint
    hints = [(m.term, m.sql_hint) for m in matches if m.sql_hint.strip()]
    if hints:
        lines.append("")
        lines.append("SQL Hints:")
        for term, hint in hints:
            lines.append(f"{term}: {hint}")

    return "\n".join(lines)
