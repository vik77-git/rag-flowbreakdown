from __future__ import annotations

import asyncio

import pytest

from app.llm.key_rotator import KeyRotator
from app.utils.errors import RagError


def test_rotates_after_quota_and_cools_down():
    rotator = KeyRotator(["k1", "k2"], requests_per_key=2, cooldown_seconds=0.2)
    used = [asyncio.run(rotator.acquire()) for _ in range(4)]
    assert used == ["k1", "k1", "k2", "k2"]

    # both keys are cooling down; the next acquire waits, then succeeds
    assert asyncio.run(rotator.acquire()) in {"k1", "k2"}


def test_status_reports_cooldown():
    rotator = KeyRotator(["k1"], requests_per_key=1, cooldown_seconds=5)
    asyncio.run(rotator.acquire())
    status = rotator.status()[0]
    assert status["cooling_down"] and status["ready_in_seconds"] > 0


def test_missing_key_is_a_clear_error():
    with pytest.raises(RagError) as exc:
        asyncio.run(KeyRotator([]).acquire())
    assert "GROQ_API_KEY" in exc.value.message
