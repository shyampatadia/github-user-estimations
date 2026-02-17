"""PAT token rotation and rate limit management.

All tokens share a SINGLE account-level rate limit (5,000 req/hr).
Rotation provides resilience (if one token has an issue), not extra throughput.
"""

import asyncio
import logging
import time

from . import config, database

logger = logging.getLogger(__name__)


class TokenRotator:
    def __init__(self, db_path: str = None):
        tokens = database.get_tokens(db_path)
        if not tokens:
            raise RuntimeError("No tokens found in database")
        self.tokens = tokens
        self._index = 0
        self._lock = asyncio.Lock()
        # Shared account-level rate limit (not per-token)
        self._remaining = config.RATE_LIMIT_PER_HOUR
        self._reset_time: float | None = None
        self._total_used = 0
        logger.info(
            f"Loaded {len(tokens)} tokens (shared {config.RATE_LIMIT_PER_HOUR} req/hr account limit)"
        )

    @property
    def total_used(self) -> int:
        return self._total_used

    async def get_token(self) -> dict:
        async with self._lock:
            # Check if reset time has passed
            if self._reset_time and time.time() >= self._reset_time:
                self._remaining = config.RATE_LIMIT_PER_HOUR
                self._reset_time = None

            # If we have budget, return next token (round-robin)
            if self._remaining > config.RATE_LIMIT_BUFFER:
                self._remaining -= 1
                self._total_used += 1
                token = self.tokens[self._index]
                self._index = (self._index + 1) % len(self.tokens)
                return token

            # Rate-limited: wait for reset
            if self._reset_time:
                wait = max(0, self._reset_time - time.time()) + 1
            else:
                wait = 60  # Fallback: wait 1 minute
            logger.warning(f"Account rate limit hit ({self._remaining} remaining). Waiting {wait:.0f}s...")

        # Release lock while sleeping
        await asyncio.sleep(wait)
        return await self.get_token()

    def update_from_headers(self, token_id: int, headers: dict):
        """Update shared rate limit state from any token's response headers."""
        remaining = headers.get("X-RateLimit-Remaining")
        reset_time = headers.get("X-RateLimit-Reset")
        if remaining is not None:
            self._remaining = int(remaining)
        if reset_time is not None:
            self._reset_time = float(reset_time)
