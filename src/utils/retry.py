"""
Project Argus - Retry Utilities

Provides retry-with-backoff wrappers for Gemini API calls
to handle RESOURCE_EXHAUSTED rate-limit errors gracefully.
"""

import re
import time
import logging

logger = logging.getLogger(__name__)

MAX_RETRIES = 4
BASE_DELAY = 5
BACKOFF_MULTIPLIER = 2


def _parse_retry_delay(error_msg: str) -> float | None:
    """
    Parse the retryDelay value (in seconds) from a RESOURCE_EXHAUSTED error.

    The Gemini API returns errors like:
        ...'retryDelay': '27s'...

    Returns the delay in seconds, or None if not found.
    """
    match = re.search(r"'retryDelay':\s*'(\d+)s'", str(error_msg))
    if match:
        return float(match.group(1))
    return None


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception is a Gemini rate-limit error."""
    error_str = str(error)
    return "RESOURCE_EXHAUSTED" in error_str or "429" in error_str


def retry_on_rate_limit(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """
    Call `func(*args, **kwargs)` with retry logic for RESOURCE_EXHAUSTED errors.

    Uses the server-suggested retryDelay when available, otherwise
    exponential backoff starting at BASE_DELAY seconds.

    Args:
        func:        Callable to invoke.
        *args:       Positional args forwarded to func.
        max_retries: Maximum number of retry attempts.
        **kwargs:    Keyword args forwarded to func.

    Returns:
        Whatever func returns on success.

    Raises:
        The original exception if all retries are exhausted or the error
        is not a rate-limit error.
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e

            if _is_rate_limit_error(e) and attempt < max_retries:
                api_delay = _parse_retry_delay(str(e))
                wait_time = (api_delay + 2) if api_delay else (BASE_DELAY * (BACKOFF_MULTIPLIER ** attempt))
                logger.warning(
                    f"Rate-limited (attempt {attempt}/{max_retries}). "
                    f"Waiting {wait_time:.0f}s before retry..."
                )
                time.sleep(wait_time)
            else:
                raise

    raise last_error  # Should never reach here, but just in case
