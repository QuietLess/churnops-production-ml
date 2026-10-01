"""API-key authentication and per-client rate limiting for the scoring endpoints.

- Auth: send `X-API-Key: <key>`. Keys come from the API_KEYS env var (comma-separated, so
  keys can be rotated: add the new key, roll clients, remove the old one). When API_KEYS is
  empty, auth is disabled and a warning is logged at startup; that is meant for local dev only.
- Rate limit: sliding one-minute window per client (API key if present, else client IP).
  Exceeding it returns 429 with a Retry-After header.

The limiter is in-process: with several workers or replicas each one counts separately.
A shared store (e.g. Redis) or the API gateway / load balancer should enforce the global
limit in a multi-instance deployment.
"""

from __future__ import annotations

import hashlib
import math
import secrets
import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from app.dependencies import get_settings
from app.settings import Settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False, description="Required when API_KEYS is set")
WINDOW_SECONDS = 60.0


def require_api_key(
    api_key: str | None = Depends(API_KEY_HEADER),
    settings: Settings = Depends(get_settings),
) -> str | None:
    """Return the caller's key (or None when auth is disabled); 401 if missing or wrong."""
    if not settings.auth_enabled:
        return None
    # compare_digest on every key: timing does not reveal how much of a key matched.
    if api_key and any(secrets.compare_digest(api_key, k) for k in settings.api_keys):
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key",
        headers={"WWW-Authenticate": "ApiKey"},
    )


class RateLimiter:
    """Thread-safe sliding-window limiter: at most `limit` hits per client per window."""

    def __init__(self, limit: int, window_seconds: float = WINDOW_SECONDS, clock=time.monotonic):
        self.limit = limit
        self.window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, client: str) -> float | None:
        """Record a request. Returns None if allowed, else seconds until the next slot frees up."""
        now = self._clock()
        with self._lock:
            hits = self._hits[client]
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return hits[0] + self.window - now
            hits.append(now)
            return None


def _client_id(request: Request, api_key: str | None) -> str:
    if api_key:  # never keep raw keys in memory structures that might be logged
        return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:16]
    return "ip:" + (request.client.host if request.client else "unknown")


def rate_limit(request: Request, api_key: str | None = Depends(require_api_key)) -> None:
    limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None or limiter.limit <= 0:
        return
    retry_after = limiter.hit(_client_id(request, api_key))
    if retry_after is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit of {limiter.limit} requests/minute exceeded",
            headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
        )


# Attach to routers that score or write data: auth first, then the limiter keyed by that key.
PROTECTED = [Depends(require_api_key), Depends(rate_limit)]
