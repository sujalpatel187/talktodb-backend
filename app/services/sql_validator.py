import re

_FORBIDDEN_KEYWORDS = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|MERGE|CALL|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)

_SELECT_PATTERN = re.compile(r"^\s*SELECT\b", re.IGNORECASE)


def validate_sql(sql: str) -> tuple[bool, str]:
    """
    Validate that the generated SQL is a safe SELECT-only query.

    Returns:
        (is_valid, error_message)
        - (True, "")  if the query is a SELECT statement.
        - (False, reason) if the query is forbidden or unrecognised.
    """
    cleaned = sql.strip()

    if not cleaned:
        return False, "No SQL query was generated."

    if _FORBIDDEN_KEYWORDS.match(cleaned):
        keyword = cleaned.split()[0].upper()
        return False, (
            f"Generated query contains a forbidden operation: {keyword}. "
            "Only SELECT queries are allowed."
        )

    if not _SELECT_PATTERN.match(cleaned):
        return False, (
            "Generated query does not appear to be a valid SELECT statement. "
            "Only SELECT queries are allowed."
        )

    return True, ""
