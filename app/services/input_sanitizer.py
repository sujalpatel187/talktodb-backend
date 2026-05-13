import re

# Patterns that indicate non-natural-language input
_SQL_PATTERN = re.compile(
    r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|EXEC|EXECUTE|UNION|WHERE|FROM|JOIN)\b",
    re.IGNORECASE,
)

_JSON_PATTERN = re.compile(r"^\s*[\[{].*[\]}]\s*$", re.DOTALL)

_CODE_PATTERN = re.compile(
    r"(import\s+\w+|def\s+\w+\s*\(|function\s*\(|<\?php|<script|#!/)",
    re.IGNORECASE,
)

_MIN_LENGTH = 3
_MAX_LENGTH = 500


def sanitize_user_message(message: str) -> tuple[bool, str]:
    """
    Validate that the user message is a natural language question.

    Returns:
        (is_valid, error_message)
        - (True, "")  if the message is acceptable.
        - (False, reason) if the message is rejected.
    """
    stripped = message.strip()

    if len(stripped) < _MIN_LENGTH:
        return False, "Message is too short. Please enter a valid question."

    if len(stripped) > _MAX_LENGTH:
        return False, f"Message is too long. Maximum allowed length is {_MAX_LENGTH} characters."

    if _JSON_PATTERN.match(stripped):
        return False, "Invalid input: JSON data is not accepted. Please enter a natural language question."

    if _SQL_PATTERN.search(stripped):
        return False, "Invalid input: SQL queries are not accepted. Please enter a natural language question."

    if _CODE_PATTERN.search(stripped):
        return False, "Invalid input: Code is not accepted. Please enter a natural language question."

    return True, ""
