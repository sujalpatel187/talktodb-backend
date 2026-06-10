import logging
import time
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


# ---------------------------------------------------------------------------
# Schema metadata cache
# ---------------------------------------------------------------------------

_schema_cache: dict[str, list[str]] = {}
_schema_cache_ts: float = 0.0


def fetch_schema_metadata(ttl: float | None = None) -> dict[str, list[str]]:
    """Fetch table→column metadata from ``information_schema.columns``.

    Returns a dict mapping lowercase table names to a list of lowercase
    column names (in ordinal order).  Results are cached for *ttl* seconds
    (defaults to ``settings.sql_schema_cache_ttl``, which is 300 s / 5 min).

    On failure the last successful (possibly stale) cache is returned so that
    a transient DB error never breaks the validation pipeline.
    """
    global _schema_cache, _schema_cache_ts

    effective_ttl = ttl if ttl is not None else float(settings.sql_schema_cache_ttl)
    now = time.monotonic()

    if _schema_cache and (now - _schema_cache_ts) < effective_ttl:
        logger.debug(
            "Schema metadata served from cache (%d tables).", len(_schema_cache)
        )
        return _schema_cache

    logger.info("Fetching schema metadata from information_schema …")
    query = """
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'

        UNION ALL

        SELECT
            mv.relname          AS table_name,
            attr.attname        AS column_name
        FROM pg_class mv
        JOIN pg_attribute attr ON attr.attrelid = mv.oid
        JOIN pg_namespace ns   ON ns.oid = mv.relnamespace
        WHERE mv.relkind  = 'm'
          AND ns.nspname  = 'public'
          AND attr.attnum > 0
          AND NOT attr.attisdropped

        ORDER BY table_name, column_name
    """
    conn = None
    cursor = None
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()

        metadata: dict[str, list[str]] = {}
        for table_name, column_name in rows:
            key = table_name.lower()
            if key not in metadata:
                metadata[key] = []
            metadata[key].append(column_name.lower())

        _schema_cache = metadata
        _schema_cache_ts = now

        logger.info(
            "Schema metadata loaded: %d table(s), %d total column(s).",
            len(metadata),
            sum(len(v) for v in metadata.values()),
        )
        return metadata

    except Exception as exc:
        logger.error("Failed to fetch schema metadata: %s", str(exc))
        # Return stale cache if available so validation can still proceed.
        return _schema_cache
    finally:
        if cursor:
            cursor.close()
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
            conn.close()
