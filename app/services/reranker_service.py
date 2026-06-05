"""Cross-Encoder Re-ranking service for the Text-to-SQL retrieval pipeline.

Architecture
------------
- The ``CrossEncoder`` is loaded **once** at application startup via
  :func:`get_reranker` and kept as a module-level singleton.
- :func:`rerank_sync` is used by the existing synchronous ``run_chat_pipeline``.
- :func:`rerank` is an async-safe wrapper that offloads CPU-bound inference
  to a dedicated ``ThreadPoolExecutor``, preventing event-loop blocking for
  any future async callers.

Re-ranking flow
---------------
1. Retrieve ``top_k_candidates`` results **per collection** from Qdrant
   (caller's responsibility — see ``chat_service.py``).
2. Pass all candidates together as ``(ScoredPoint, source)`` tuples.
3. The CrossEncoder scores every ``(query, document)`` pair in one batch.
4. Return the ``top_k`` highest-scoring :class:`RerankedHit` objects.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sentence_transformers import CrossEncoder
from qdrant_client.models import ScoredPoint

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PRIMARY_MODEL: str = "BAAI/bge-reranker-base"
_FALLBACK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Truncate input pairs at this token length to match model capacity.
_MAX_LENGTH: int = 512

# Number of (query, doc) pairs sent to the model in a single forward pass.
# Larger batches amortise overhead; reduce if GPU/CPU memory is constrained.
_BATCH_SIZE: int = 64

# Dedicated thread pool so CrossEncoder inference never blocks the event loop.
_executor: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="reranker"
)

# Module-level singleton
_reranker: Optional[CrossEncoder] = None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RerankedHit:
    """A Qdrant :class:`~qdrant_client.models.ScoredPoint` enriched with
    re-ranking metadata.

    Attributes:
        hit:          The original Qdrant result (preserves all payload/metadata).
        source:       Which collection the hit came from: ``"qa"``, ``"ddl"``,
                      or ``"docs"``.
        rerank_score: Raw logit from the CrossEncoder (higher = more relevant).
    """

    hit: ScoredPoint
    source: str  # "qa" | "ddl" | "docs"
    rerank_score: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _extract_document_text(hit: ScoredPoint, source: str) -> str:
    """Normalise a Qdrant payload into a single text string for cross-encoding.

    Each collection stores different payload keys; this function assembles a
    dense, human-readable paragraph so the CrossEncoder can compare it directly
    against the user query.

    Args:
        hit:    A Qdrant ``ScoredPoint`` whose ``payload`` field is inspected.
        source: Collection tag — ``"qa"``, ``"ddl"``, or ``"docs"``.

    Returns:
        A plain-text document string ready for ``CrossEncoder.predict()``.
    """
    payload: dict = hit.payload or {}

    if source == "qa":
        question: str = payload.get("question", "")
        sql: str = payload.get("sql", "")
        return f"Question: {question}\nSQL: {sql}"

    if source == "ddl":
        table_name: str = payload.get("table_name", "")
        description: str = payload.get("description", "")
        ddl: str = payload.get("ddl", "")
        return f"Table: {table_name}\nDescription: {description}\nDDL: {ddl}"

    if source == "docs":
        return payload.get("content", "")

    # Generic fallback — concatenate all non-empty string values
    return " ".join(str(v) for v in payload.values() if v)


# ---------------------------------------------------------------------------
# Singleton loader
# ---------------------------------------------------------------------------
def get_reranker() -> CrossEncoder:
    """Return the singleton ``CrossEncoder``, loading it on first call.

    Model resolution order:
    1. ``settings.reranker_model`` (from ``.env`` / environment variable).
    2. Hard-coded primary: ``BAAI/bge-reranker-base``.
    3. Hard-coded fallback: ``cross-encoder/ms-marco-MiniLM-L-6-v2``.

    Raises:
        RuntimeError: If every candidate model fails to load.
    """
    global _reranker
    if _reranker is not None:
        return _reranker

    configured_model: str = getattr(settings, "reranker_model", _PRIMARY_MODEL)

    # dict.fromkeys preserves order and deduplicates if configured_model
    # happens to be equal to one of the built-in candidates.
    candidates = list(
        dict.fromkeys((configured_model, _PRIMARY_MODEL, _FALLBACK_MODEL))
    )

    for model_name in candidates:
        try:
            logger.info("Loading re-ranker model: %s", model_name)
            _reranker = CrossEncoder(model_name, max_length=_MAX_LENGTH)
            logger.info("Re-ranker model loaded successfully: %s", model_name)
            return _reranker
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load re-ranker model '%s': %s — trying next candidate.",
                model_name,
                exc,
            )

    raise RuntimeError(
        "All re-ranker models failed to load. "
        f"Tried in order: {candidates}"
    )


# ---------------------------------------------------------------------------
# Synchronous re-ranking (used by the existing blocking pipeline)
# ---------------------------------------------------------------------------
def rerank_sync(
    query: str,
    candidates: list[tuple[ScoredPoint, str]],
    top_k: int = 10,
) -> list[RerankedHit]:
    """Re-rank *candidates* against *query* and return the ``top_k`` results.

    This function is **synchronous** and safe to call from any blocking code
    path.  It blocks the calling thread while the CrossEncoder runs inference.

    Args:
        query:      The user's natural-language question.
        candidates: List of ``(ScoredPoint, source)`` tuples retrieved from
                    Qdrant.  ``source`` must be ``"qa"``, ``"ddl"``, or
                    ``"docs"``.
        top_k:      Number of highest-scoring hits to return.

    Returns:
        A list of :class:`RerankedHit` objects sorted by ``rerank_score``
        descending, with at most ``top_k`` elements.
    """
    if not candidates:
        logger.warning("rerank_sync called with an empty candidate list.")
        return []

    reranker = get_reranker()

    # Build (query, document) pairs for batch inference
    pairs: list[tuple[str, str]] = [
        (query, _extract_document_text(hit, source))
        for hit, source in candidates
    ]

    # CrossEncoder.predict returns an ndarray of shape (n,) — convert to list
    raw_scores: np.ndarray = reranker.predict(
        pairs,
        batch_size=_BATCH_SIZE,
        show_progress_bar=False,
    )
    scores: list[float] = (
        raw_scores.tolist() if hasattr(raw_scores, "tolist") else [float(raw_scores)]
    )

    # Attach scores, sort descending, take top-k
    ranked: list[RerankedHit] = sorted(
        (
            RerankedHit(hit=hit, source=source, rerank_score=score)
            for (hit, source), score in zip(candidates, scores)
        ),
        key=lambda r: r.rerank_score,
        reverse=True,
    )

    top: list[RerankedHit] = ranked[:top_k]

    if top:
        logger.info(
            "Re-ranking: %d candidates → top %d  "
            "(score range: %.4f – %.4f)",
            len(candidates),
            len(top),
            top[0].rerank_score,
            top[-1].rerank_score,
        )
        for i, r in enumerate(top, 1):
            logger.debug(
                "  [%d] score=%.4f  source=%-4s  payload_keys=%s",
                i,
                r.rerank_score,
                r.source,
                list((r.hit.payload or {}).keys()),
            )

    return top


# ---------------------------------------------------------------------------
# Async-safe re-ranking (for async endpoints / future async migration)
# ---------------------------------------------------------------------------
async def rerank(
    query: str,
    candidates: list[tuple[ScoredPoint, str]],
    top_k: int = 10,
) -> list[RerankedHit]:
    """Async-safe wrapper around :func:`rerank_sync`.

    Offloads CPU-bound ``CrossEncoder`` inference to the module-level
    ``ThreadPoolExecutor`` so the event loop is never blocked.

    Args:
        query:      The user's natural-language question.
        candidates: List of ``(ScoredPoint, source)`` tuples retrieved from
                    Qdrant.
        top_k:      Number of results to return after re-ranking.

    Returns:
        A list of :class:`RerankedHit` sorted by ``rerank_score`` descending.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor,
        rerank_sync,
        query,
        candidates,
        top_k,
    )
