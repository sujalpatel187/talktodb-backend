import logging
import psycopg2
import psycopg2.extras
from app.config import get_settings
from app.services.sql_validator import validate_sql

logger = logging.getLogger(__name__)

settings = get_settings()

MAX_ROWS = 500
STATEMENT_TIMEOUT_MS = 30_000  # 30 seconds


def _get_connection():
    """Create a new psycopg2 connection using settings from .env."""
    return psycopg2.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        dbname=settings.pg_database,
        user=settings.pg_user,
        password=settings.pg_password,
        options=f"-c statement_timeout={STATEMENT_TIMEOUT_MS}",
    )


def execute_query(sql: str) -> dict:
    """
    Execute a read-only SELECT query against PostgreSQL.

    Args:
        sql: The SQL query to execute (must be SELECT-only).

    Returns:
        dict with keys: columns, rows, row_count, error
    """
    # Defense in depth: re-validate SQL before executing
    is_valid, error_msg = validate_sql(sql)
    if not is_valid:
        logger.warning("SQL validation failed before execution: %s", error_msg)
        return {"columns": [], "rows": [], "row_count": 0, "error": error_msg}

    conn = None
    cursor = None
    try:
        conn = _get_connection()
        conn.set_session(readonly=True, autocommit=False)
        cursor = conn.cursor()

        logger.info("Executing SQL query: %s", sql.strip()[:200])
        cursor.execute(sql)

        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(MAX_ROWS)

        # Convert to serializable types
        serializable_rows = []
        for row in rows:
            serializable_rows.append(
                [_make_serializable(cell) for cell in row]
            )

        row_count = cursor.rowcount if cursor.rowcount >= 0 else len(serializable_rows)

        logger.info("Query returned %d rows, %d columns.", len(serializable_rows), len(columns))

        return {
            "columns": columns,
            "rows": serializable_rows,
            "row_count": row_count,
            "error": None,
        }

    except psycopg2.errors.ReadOnlySqlTransaction:
        logger.error("Blocked: attempted write operation in read-only mode.")
        return {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error": "Write operations are not allowed. Only SELECT queries can be executed.",
        }
    except psycopg2.errors.QueryCanceled:
        logger.error("Query timed out after %d ms.", STATEMENT_TIMEOUT_MS)
        return {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error": f"Query timed out after {STATEMENT_TIMEOUT_MS // 1000} seconds.",
        }
    except psycopg2.Error as e:
        logger.error("Database error: %s", str(e).strip())
        return {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error": f"Database error: {str(e).strip()}",
        }
    except Exception as e:
        logger.error("Unexpected error executing query: %s", str(e))
        return {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error": f"Unexpected error: {str(e)}",
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
            conn.close()


def _make_serializable(value):
    """Convert a database value to a JSON-serializable type."""
    if value is None:
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    # datetime, date, Decimal, etc.
    return str(value)
