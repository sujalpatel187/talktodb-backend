import re
import logging
from datetime import datetime
from pathlib import Path
from app.services.embedding_service import get_embedding
from app.services.qdrant_service import search_similar, search_ddl, search_docs
from app.services.reranker_service import rerank_sync, RerankedHit
from app.services.llm_service import call_llm, call_llm_summary, _load_system_prompt
from app.services.sql_validator import validate_sql
from app.services.input_sanitizer import sanitize_user_message
from app.services.pg_service import execute_query
from app.schemas.chat_schema import ChatResponse

logger = logging.getLogger(__name__)

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
    qa_hits: list,
    ddl_hits: list,
    docs_hits: list,
    reranked: list,
) -> None:
    """Write a human-readable debug file with raw candidates and re-ranked results."""
    _LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w\s-]", "", user_message).strip()
    safe_name = re.sub(r"\s+", "_", safe_name)[:60]
    file_path = _LOGS_DIR / f"{timestamp}_{safe_name}_retrieval_debug.txt"

    lines = []
    lines.append(f"USER QUESTION: {user_message}")
    lines.append("=" * 80)

    # ── Section 1: Raw QA candidates from Qdrant ────────────────────────────
    lines.append(f"\n{'=' * 80}")
    lines.append(f"SECTION 1 — RAW QA CANDIDATES FROM QDRANT ({len(qa_hits)} results)")
    lines.append("=" * 80)
    for i, hit in enumerate(qa_hits, 1):
        lines.append(
            f"[{i:>2}] score={hit.score:.4f} | "
            f"question={hit.payload.get('question', '')} | "
            
        )

    # ── Section 2: Raw DDL candidates from Qdrant ───────────────────────────
    lines.append(f"\n{'=' * 80}")
    lines.append(f"SECTION 2 — RAW DDL CANDIDATES FROM QDRANT ({len(ddl_hits)} results)")
    lines.append("=" * 80)
    for i, hit in enumerate(ddl_hits, 1):
        lines.append(
            f"[{i:>2}] score={hit.score:.4f} | "
            f"table={hit.payload.get('table_name', '')} | "
            f"description={hit.payload.get('description', '')} | "
            f"ddl={hit.payload.get('ddl', '')}"
        )

    # ── Section 3: Raw Docs candidates from Qdrant ──────────────────────────
    lines.append(f"\n{'=' * 80}")
    lines.append(f"SECTION 3 — RAW DOCS CANDIDATES FROM QDRANT ({len(docs_hits)} results)")
    lines.append("=" * 80)
    for i, hit in enumerate(docs_hits, 1):
        lines.append(
            f"[{i:>2}] score={hit.score:.4f} | "
            f"category={hit.payload.get('category', '')} | "
            f"content={hit.payload.get('content', '')}"
        )

    # ── Section 4: Re-ranked QA sent to LLM ─────────────────────────────────
    reranked_qa_items = [r for r in reranked if r.source == "qa"]
    lines.append(f"\n{'=' * 80}")
    lines.append(
        f"SECTION 4 — RE-RANKED QA SENT TO LLM ({len(reranked_qa_items)} results)"
    )
    lines.append("=" * 80)
    for i, r in enumerate(reranked_qa_items, 1):
        lines.append(
            f"[{i:>2}] rerank={r.rerank_score:.4f} | "
            f"question={r.hit.payload.get('question', '')} | "
        )

    # ── Section 5: Re-ranked DDL sent to LLM ────────────────────────────────
    reranked_ddl_items = [r for r in reranked if r.source == "ddl"]
    lines.append(f"\n{'=' * 80}")
    lines.append(
        f"SECTION 5 — RE-RANKED DDL SENT TO LLM ({len(reranked_ddl_items)} results)"
    )
    lines.append("=" * 80)
    for i, r in enumerate(reranked_ddl_items, 1):
        lines.append(
            f"[{i:>2}] rerank={r.rerank_score:.4f} | "
            f"table={r.hit.payload.get('table_name', '')} | "
            f"description={r.hit.payload.get('description', '')} | "
            f"ddl={r.hit.payload.get('ddl', '')}"
        )

    # ── Section 6: Re-ranked Docs sent to LLM ───────────────────────────────
    reranked_docs_items = [r for r in reranked if r.source == "docs"]
    lines.append(f"\n{'=' * 80}")
    lines.append(
        f"SECTION 6 — RE-RANKED DOCS SENT TO LLM ({len(reranked_docs_items)} results)"
    )
    lines.append("=" * 80)
    for i, r in enumerate(reranked_docs_items, 1):
        lines.append(
            f"[{i:>2}] rerank={r.rerank_score:.4f} | "
            f"category={r.hit.payload.get('category', '')} | "
            f"content={r.hit.payload.get('content', '')}"
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

    # Step 1: Embed user message
    query_vector = get_embedding(user_message)

    # Step 2: Retrieve candidates for re-ranking
    # QA: 50 candidates — widest net, re-ranker will select best 10
    # DDL / Docs: 10 candidates each
    qa_hits = search_similar(query_vector=query_vector, top_k=50)
    ddl_hits = search_ddl(query_vector=query_vector, top_k=10)
    docs_hits = search_docs(query_vector=query_vector, top_k=10)

    logger.info(
        "Qdrant candidates: %d QA, %d DDL, %d Docs (total %d)",
        len(qa_hits), len(ddl_hits), len(docs_hits),
        len(qa_hits) + len(ddl_hits) + len(docs_hits),
    )

    # logger.info("=== Raw Qdrant Candidates (before re-ranking) ===")
    # for i, hit in enumerate(qa_hits, 1):
    #     logger.info(
    #         "[QA  %2d] score=%.4f | question=%s",
    #         i, hit.score,
    #         hit.payload.get("question", ""),
    #         # hit.payload.get("sql", ""),
    #     )
    # for i, hit in enumerate(ddl_hits, 1):
    #     logger.info(
    #         "[DDL %2d] score=%.4f | table=%s | ddl=%s",
    #         i, hit.score,
    #         hit.payload.get("table_name", ""),
    #         hit.payload.get("ddl", ""),
    #     )
    # for i, hit in enumerate(docs_hits, 1):
    #     logger.info(
    #         "[DOC %2d] score=%.4f | category=%s | content=%s",
    #         i, hit.score,
    #         hit.payload.get("category", ""),
    #         hit.payload.get("content", ""),
    #     )

    # Step 2b: Cross-encoder re-ranking — score all 30 candidates together
    # and return the top-10 most relevant across all collections.
    candidates = (
        [(hit, "qa") for hit in qa_hits]
        + [(hit, "ddl") for hit in ddl_hits]
        + [(hit, "docs") for hit in docs_hits]
    )
    reranked: list[RerankedHit] = rerank_sync(
        query=user_message, candidates=candidates, top_k=10
    )

    # Partition reranked results back by source for the existing block builders
    reranked_qa = [r.hit for r in reranked if r.source == "qa"]
    reranked_ddl = [r.hit for r in reranked if r.source == "ddl"]
    reranked_docs = [r.hit for r in reranked if r.source == "docs"]

    # logger.info("=== Re-ranked Q&A Results (top %d) ===", len(reranked_qa))
    # for i, r in enumerate(
    #     (r for r in reranked if r.source == "qa"), 1
    # ):
    #     logger.info(
    #         "[%d] Rerank: %.4f | Question: %s | SQL: %s",
    #         i, r.rerank_score,
    #         r.hit.payload.get("question", ""),
    #         r.hit.payload.get("sql", ""),
    #     )

    # logger.info("=== Re-ranked DDL Results (top %d) ===", len(reranked_ddl))
    # for i, r in enumerate(
    #     (r for r in reranked if r.source == "ddl"), 1
    # ):
    #     logger.info(
    #         "[%d] Rerank: %.4f | Table: %s",
    #         i, r.rerank_score,
    #         r.hit.payload.get("table_name", ""),
    #     )

    # logger.info("=== Re-ranked Docs Results (top %d) ===", len(reranked_docs))
    # for i, r in enumerate(
    #     (r for r in reranked if r.source == "docs"), 1
    # ):
    #     logger.info(
    #         "[%d] Rerank: %.4f | Category: %s",
    #         i, r.rerank_score,
    #         r.hit.payload.get("category", ""),
    #     )

    # Step 3: Build context blocks for LLM from re-ranked hits
    _save_retrieval_debug_log(
        user_message=user_message,
        qa_hits=qa_hits,
        ddl_hits=ddl_hits,
        docs_hits=docs_hits,
        reranked=reranked,
    )
    qa_block = _build_qa_block(reranked_qa)
    ddl_block = _build_ddl_block(reranked_ddl)
    docs_block = _build_docs_block(reranked_docs)

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

    # Step 5: Validate SQL is SELECT-only
    is_valid, error_msg = validate_sql(generated_sql)
    if not is_valid:
        logger.warning("SQL validation failed: %s", error_msg)
        return ChatResponse(userMessage=user_message, error=error_msg)

    # Step 6: Execute SQL against PostgreSQL
    db_result = execute_query(generated_sql.strip())

    if db_result["error"]:
        logger.warning("SQL execution error: %s", db_result["error"])
        return ChatResponse(
            userMessage=f"Database execution failed: {db_result['error']}",
            generatedSQL=generated_sql.strip(),
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
        sql=generated_sql.strip(),
        columns=db_result["columns"],
        rows=db_result["rows"],
    )

    # Step 8: Build response
    return ChatResponse(
        userMessage=summary,
        generatedSQL=generated_sql.strip(),
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
