import re
import logging
from datetime import datetime
from pathlib import Path
from app.services.embedding_service import get_embedding
from app.services.qdrant_service import search_similar, search_ddl, search_docs
from app.services.llm_service import call_llm, _load_system_prompt
from app.services.sql_validator import validate_sql
from app.services.input_sanitizer import sanitize_user_message
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

    # Step 2: Retrieve context from all 3 collections in parallel-ish
    qa_hits = search_similar(query_vector=query_vector, top_k=5)
    ddl_hits = search_ddl(query_vector=query_vector, top_k=5)
    docs_hits = search_docs(query_vector=query_vector, top_k=5)

    logger.info("=== Similar Q&A Results ===")
    for i, hit in enumerate(qa_hits, 1):
        logger.info(
            "[%d] Score: %.4f | Question: %s | SQL: %s",
            i, hit.score,
            hit.payload.get("question", ""),
            hit.payload.get("sql", ""),
        )

    logger.info("=== Similar DDL Results ===")
    for i, hit in enumerate(ddl_hits, 1):
        logger.info(
            "[%d] Score: %.4f | Table: %s",
            i, hit.score,
            hit.payload.get("table_name", ""),
        )

    logger.info("=== Similar Docs Results ===")
    for i, hit in enumerate(docs_hits, 1):
        logger.info(
            "[%d] Score: %.4f | Category: %s",
            i, hit.score,
            hit.payload.get("category", ""),
        )

    # Step 3: Build context blocks for LLM
    qa_block = _build_qa_block(qa_hits)
    ddl_block = _build_ddl_block(ddl_hits)
    docs_block = _build_docs_block(docs_hits)

    user_prompt = (
        f"### Database Schema (DDL):\n{ddl_block}\n\n"
        f"### Documentation Context:\n{docs_block}\n\n"
        f"### Similar Question-SQL Examples:\n{qa_block}\n\n"
        f"### Task:\n"
        f"Generate a SQL query for the following question:\n"
        f"Question: {user_message}\n\n"
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

    # Step 6: Build response
    return ChatResponse(
        userMessage=user_message,
        generatedSQL=generated_sql.strip(),
    )
