import re
import logging
from datetime import datetime
from pathlib import Path
from app.services.embedding_service import get_embedding
from app.services.qdrant_service import search_similar, search_ddl, search_docs
from app.services.reranker_service import rerank_sync, bm25_prefilter, RerankedHit
from app.services.glossary_service import resolve_glossary, GlossaryResolutionResult
from app.services.llm_service import call_llm, call_llm_summary, _load_system_prompt
from app.services.sql_validator_service import validate_sql, ValidationResult
from app.services.input_sanitizer import sanitize_user_message
from app.services.pg_service import execute_query, fetch_schema_metadata
from app.schemas.chat_schema import ChatResponse
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_LOGS_DIR = Path(__file__).parent.parent.parent / "logs"


def _save_prompt_log(user_message: str, prompt: str) -> None:
    _LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w\s-]", "", user_message).strip()
    safe_name = re.sub(r"\s+", "_", safe_name)[:60]
    file_path = _LOGS_DIR / f"{timestamp}_{safe_name}.txt"
    system_prompt = _load_system_prompt()
    content = (
        f"==== SYSTEM PROMPT ====\n{system_prompt}\n\n"
        f"==== USER PROMPT ====\n{prompt}"
    )
    file_path.write_text(content, encoding="utf-8")
    logger.info("Prompt saved to logs: %s", file_path.name)


def _save_retrieval_debug_log(
    user_message: str,
    glossary_result: GlossaryResolutionResult,
    qa_hits: list,
    ddl_hits: list,
    docs_hits: list,
    reranked_qa: list,
    reranked_ddl: list,
) -> None:
    """Write a human-readable debug file with raw candidates and re-ranked results.

    Docs are logged as-is (no re-ranking applied).
    """
    _LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w\s-]", "", user_message).strip()
    safe_name = re.sub(r"\s+", "_", safe_name)[:60]
    file_path = _LOGS_DIR / f"{timestamp}_{safe_name}_retrieval_debug.txt"

    lines = []
    lines.append(f"USER QUESTION: {user_message}")
    lines.append("=" * 80)

    # ── Section 0: Glossary Resolution ───────────────────────────────────
    lines.append(f"\n{'=' * 80}")
    lines.append(f"SECTION 0 — GLOSSARY RESOLUTION ({len(glossary_result.matches)} terms matched)")
    lines.append("=" * 80)
    if glossary_result.matches:
        for m in glossary_result.matches:
            lines.append(
                f"  score={m.score:.4f} | {m.term} = {m.meaning}"
                + (f" | sql_hint: {m.sql_hint}" if m.sql_hint else "")
            )
        lines.append(f"\nEXPANDED QUERY:\n{glossary_result.expanded_query}")
    else:
        lines.append("  (no glossary terms matched — using original query)")

    # ── Section 1: Raw QA candidates from Qdrant ────────────────────────────
    lines.append(f"\n{'=' * 80}")
    lines.append(f"SECTION 1 — RAW QA CANDIDATES FROM QDRANT ({len(qa_hits)} results)")
    lines.append("=" * 80)
    for i, hit in enumerate(qa_hits, 1):
        lines.append(
            f"[{i:>2}] qdrant={hit.score:.4f} | "
            f"question={hit.payload.get('question', '')}"
        )

    # ── Section 2: Raw DDL candidates from Qdrant ───────────────────────────
    lines.append(f"\n{'=' * 80}")
    lines.append(f"SECTION 2 — RAW DDL CANDIDATES FROM QDRANT ({len(ddl_hits)} results) [BM25 pre-filter applied before CrossEncoder]")
    lines.append("=" * 80)
    for i, hit in enumerate(ddl_hits, 1):
        lines.append(
            f"[{i:>2}] qdrant={hit.score:.4f} | "
            f"table={hit.payload.get('table_name', '')} | "
            f"description={hit.payload.get('description', '')}"
        )

    # ── Section 3: Docs from Qdrant (sent directly to LLM) ──────────────────
    lines.append(f"\n{'=' * 80}")
    lines.append(f"SECTION 3 — DOCS FROM QDRANT → SENT DIRECTLY TO LLM ({len(docs_hits)} results)")
    lines.append("=" * 80)
    for i, hit in enumerate(docs_hits, 1):
        lines.append(
            f"[{i:>2}] qdrant={hit.score:.4f} | "
            f"category={hit.payload.get('category', '')} | "
            f"content={hit.payload.get('content', '')}"
        )

    # ── Section 4: Re-ranked QA sent to LLM ─────────────────────────────────
    lines.append(f"\n{'=' * 80}")
    lines.append(f"SECTION 4 — RE-RANKED QA SENT TO LLM ({len(reranked_qa)} results)")
    lines.append("=" * 80)
    for i, r in enumerate(reranked_qa, 1):
        lines.append(
            f"[{i:>2}] qdrant={r.hit.score:.4f} | rerank={r.rerank_score:.4f} | "
            f"question={r.hit.payload.get('question', '')} | "
            f"sql={r.hit.payload.get('sql', '')}"
        )

    # ── Section 5: Re-ranked DDL sent to LLM ────────────────────────────────
    lines.append(f"\n{'=' * 80}")
    lines.append(f"SECTION 5 — RE-RANKED DDL SENT TO LLM ({len(reranked_ddl)} results)")
    lines.append("=" * 80)
    for i, r in enumerate(reranked_ddl, 1):
        lines.append(
            f"[{i:>2}] qdrant={r.hit.score:.4f} | rerank={r.rerank_score:.4f} | "
            f"table={r.hit.payload.get('table_name', '')} | "
            f"description={r.hit.payload.get('description', '')} | "
            f"ddl={r.hit.payload.get('ddl', '')}"
        )

    file_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Retrieval debug log saved: %s", file_path.name)


def _build_qa_block(hits: list) -> str:
    if not hits:
        return "No similar examples found."
    lines = []
    for i, hit in enumerate(hits, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  Question : {hit.payload.get('question', '')}")
        lines.append(f"  SQL      : {hit.payload.get('sql', '')}")
        lines.append("")
    return "\n".join(lines).strip()


def _build_ddl_block(hits: list) -> str:
    if not hits:
        return "No DDL context found."
    lines = []
    for i, hit in enumerate(hits, 1):
        lines.append(f"Table {i}: {hit.payload.get('table_name', '')}")
        lines.append(f"  Description : {hit.payload.get('description', '')}")
        lines.append(f"  DDL         : {hit.payload.get('ddl', '')}")
        lines.append("")
    return "\n".join(lines).strip()


def _build_docs_block(hits: list) -> str:
    if not hits:
        return "No documentation context found."
    lines = []
    for i, hit in enumerate(hits, 1):
        lines.append(f"Doc {i} [{hit.payload.get('category', '')}]:")
        lines.append(f"  {hit.payload.get('content', '')}")
        lines.append("")
    return "\n".join(lines).strip()


def run_chat_pipeline(user_message: str) -> ChatResponse:
    """
    Full pipeline:
      1. Embed the user message.
      2. Retrieve top-5 similar questions from Qdrant.
      3. Build a context-aware prompt with the similar examples.
      4. Call the LLM to generate SQL.
      5. Return structured response.
    """
    # Step 0: Sanitize & validate user input
    is_valid, error_msg = sanitize_user_message(user_message)
    if not is_valid:
        logger.warning("Input sanitization failed: %s", error_msg)
        return ChatResponse(userMessage=user_message, error=error_msg)

    # Step 0b: Business Glossary Resolution
    # Expands acronyms/abbreviations before embedding so all downstream
    # retrieval (QA, DDL, Docs) benefits from the richer query text.
    logger.info("=== Glossary Resolution ===")
    glossary_result: GlossaryResolutionResult = resolve_glossary(user_message)
    expanded_query: str = glossary_result.expanded_query

    # Step 1: Embed expanded query (falls back to original if no matches)
    query_vector = get_embedding(expanded_query)

    # Step 2: Retrieve candidates
    # QA:  50 candidates — CrossEncoder selects best 10
    # DDL: 100 candidates — BM25 pre-filter →20, then CrossEncoder selects best 5
    # Docs: top-5 — passed directly to LLM (no re-ranking)
    qa_hits = search_similar(query_vector=query_vector, top_k=50)
    ddl_hits = search_ddl(query_vector=query_vector, top_k=100)
    docs_hits = search_docs(query_vector=query_vector, top_k=5)

    logger.info(
        "Qdrant candidates: %d QA (re-rank→10), %d DDL (BM25→20→CE→5), %d Docs (direct)",
        len(qa_hits), len(ddl_hits), len(docs_hits),
    )

    # Step 2b: Cross-encoder re-ranking — QA
    qa_candidates = [(hit, "qa") for hit in qa_hits]
    reranked_qa_hits: list[RerankedHit] = rerank_sync(
        query=user_message, candidates=qa_candidates, top_k=10
    )
    reranked_qa = [r.hit for r in reranked_qa_hits]

    # Step 2c: DDL — BM25 pre-filter (100 → 20) then CrossEncoder (20 → 5)
    ddl_candidates = [(hit, "ddl") for hit in ddl_hits]
    ddl_bm25_candidates = bm25_prefilter(
        query=user_message, candidates=ddl_candidates, top_n=20
    )
    reranked_ddl_hits: list[RerankedHit] = rerank_sync(
        query=user_message, candidates=ddl_bm25_candidates, top_k=5
    )
    reranked_ddl = [r.hit for r in reranked_ddl_hits]

    logger.info(
        "Re-ranking complete: QA %d→%d | DDL %d→BM25→%d→CE→%d",
        len(qa_hits), len(reranked_qa),
        len(ddl_hits), len(ddl_bm25_candidates), len(reranked_ddl),
    )
    for i, r in enumerate(reranked_ddl_hits, 1):
        logger.info(
            "[DDL %2d] qdrant=%.4f rerank=%.4f | table=%s",
            i,
            r.hit.score,
            r.rerank_score,
            r.hit.payload.get("table_name", ""),
        )

    # Step 3: Build context blocks for LLM
    # QA: re-ranked top-10  |  DDL: re-ranked top-5  |  Docs: direct Qdrant top-5
    _save_retrieval_debug_log(
        user_message=user_message,
        glossary_result=glossary_result,
        qa_hits=qa_hits,
        ddl_hits=ddl_hits,
        docs_hits=docs_hits,
        reranked_qa=reranked_qa_hits,
        reranked_ddl=reranked_ddl_hits,
    )
    qa_block = _build_qa_block(reranked_qa)
    ddl_block = _build_ddl_block(reranked_ddl)
    docs_block = _build_docs_block(docs_hits)

    user_prompt = (
        f"Question: {user_message}\n\n"
        f"### Similar Question-SQL Examples:\n{qa_block}\n\n"
        f"### Documentation Context:\n{docs_block}\n\n"
        f"### Database Schema (DDL):\n{ddl_block}\n\n"
        f"### Task:\n"
        f"Generate a SQL query for the following question:\n"
        f"Return ONLY the SQL query, nothing else."
    )

    # Step 4: Call LLM
    _save_prompt_log(user_message, user_prompt)
    generated_sql = call_llm([{"role": "user", "content": user_prompt}])

    logger.info("=== Generated SQL ===")
    logger.info("%s", generated_sql.strip())

    # Step 5: SQLGlot validation — SELECT-only, complexity, schema, auto-LIMIT
    schema_metadata: dict[str, list[str]] | None = None
    if settings.sql_schema_validation_enabled:
        schema_metadata = fetch_schema_metadata()

    validation: ValidationResult = validate_sql(
        generated_sql,
        schema_metadata=schema_metadata,
        max_joins=settings.sql_max_joins,
        max_subqueries=settings.sql_max_subqueries,
        max_query_length=settings.sql_max_query_length,
        auto_limit=settings.sql_auto_limit,
    )

    if not validation.is_valid:
        error_detail = " | ".join(validation.errors)
        logger.warning("SQL validation failed: %s", error_detail)
        return ChatResponse(userMessage=user_message, error=error_detail)

    # Use the validated (and potentially LIMIT-injected) SQL for execution
    final_sql = validation.final_sql
    logger.info(
        "Validation passed. Tables=%s | Columns=%s | Final SQL: %s",
        validation.tables,
        validation.columns,
        final_sql[:200],
    )

    # Step 6: Execute validated SQL against PostgreSQL
    db_result = execute_query(final_sql)

    if db_result["error"]:
        logger.warning("SQL execution error: %s", db_result["error"])
        return ChatResponse(
            userMessage=f"Database execution failed: {db_result['error']}",
            generatedSQL=final_sql,
            error=db_result["error"],
        )

    logger.info(
        "Query executed: %d rows, %d columns.",
        len(db_result["rows"]),
        len(db_result["columns"]),
    )

    # Step 7: Generate conversational NL summary
    summary = _generate_data_summary(
        question=user_message,
        sql=final_sql,
        columns=db_result["columns"],
        rows=db_result["rows"],
    )

    # Step 8: Build response
    return ChatResponse(
        userMessage=summary,
        generatedSQL=final_sql,
        columns=db_result["columns"],
        rows=db_result["rows"],
        rowCount=db_result["row_count"],
    )


def _generate_data_summary(question: str, sql: str, columns: list, rows: list) -> str:
    """Use the LLM to generate a natural language summary of the PostgreSQL query results."""
    if not rows:
        return "The query executed successfully but returned 0 results."
        
    sample_rows = rows[:20]
    summary_prompt = (
        "You are an expert conversational BI and database assistant.\n"
        f"The user asked: {question}\n"
        f"The system generated this SQL query to retrieve the answer:\n{sql}\n\n"
        "The database returned these results:\n"
        f"Columns: {columns}\n"
        f"Rows (showing first {len(sample_rows)} rows out of {len(rows)} total rows):\n"
        f"{sample_rows}\n\n"
        "Task:\n"
        "Provide a concise, conversational, and direct natural language summary answering the user's question based strictly on the retrieved database results.\n"
        "Rules:\n"
        "- Do not mention SQL syntax, table names, or technical code details.\n"
        "- Keep it simple, clear, and professional.\n"
        "- Be brief (2 to 4 sentences max)."
    )
    try:
        summary = call_llm_summary([{"role": "user", "content": summary_prompt}])
        return summary.strip()
    except Exception as e:
        logger.error("Failed to generate data summary: %s", str(e))
        return "Query executed successfully. Below are the results from your database:"
