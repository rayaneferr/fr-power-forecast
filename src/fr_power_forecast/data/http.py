"""HTTP helper shared by the data clients."""

import time
from collections.abc import Callable

import httpx


def get_with_retries(
    http: httpx.Client,
    url: str,
    params: dict,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """GET that retries rate limits (429) and server errors with exponential backoff.

    The last response is returned as is, whatever its status: callers decide how to handle it.
    """
    for attempt in range(max_retries + 1):
        response = http.get(url, params=params)
        retryable = response.status_code == 429 or response.status_code >= 500
        if not retryable or attempt == max_retries:
            return response
        sleep(base_delay * 2**attempt)
    raise AssertionError("unreachable")
