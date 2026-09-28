from __future__ import annotations

import asyncio
import json

from app.retrieval.base import Candidate
from app.reranking.groq_reranker import GroqReranker
from app.reranking.heuristic import heuristic_scores
from app.utils.errors import RagError


def candidates() -> list[Candidate]:
    return [
        Candidate(chunk_id="a", text="Multi-factor authentication is mandatory.", fusion_score=0.02),
        Candidate(chunk_id="b", text="Incident reviews are blameless.", fusion_score=0.01),
        Candidate(chunk_id="c", text="Vendors complete annual assessments.", fusion_score=0.005),
    ]


class FakeClient:
    def __init__(self, content: str | None = None, fail: bool = False):
        self.content, self.fail, self.calls = content, fail, 0

    async def complete(self, messages, model, **kwargs):
        self.calls += 1
        if self.fail:
            raise RagError("groq down", 502)
        return self.content, {"prompt_tokens": 100, "completion_tokens": 20}


def test_single_batched_call_and_ordering():
    client = FakeClient(json.dumps({"results": [
        {"chunk_id": "C01", "score": 0.4}, {"chunk_id": "C02", "score": 0.95}, {"chunk_id": "C03", "score": 0.1}]}))
    reranker = GroqReranker(client, "openai/gpt-oss-20b")
    top, usage = asyncio.run(reranker.rerank("authentication", candidates(), top_n=2))

    assert client.calls == 1  # one request for the whole candidate set
    assert [c.chunk_id for c in top] == ["b", "a"]
    assert top[0].rerank_score == 0.95
    assert usage == {"input_tokens": 100, "output_tokens": 20, "mode": "llm", "estimated": False}


def test_scores_normalized_from_percentages():
    client = FakeClient(json.dumps({"results": [{"chunk_id": "C01", "score": 90}]}))
    top, _ = asyncio.run(GroqReranker(client, "m").rerank("authentication", candidates(), top_n=3))
    assert 0.0 <= top[0].rerank_score <= 1.0


def test_malformed_response_falls_back():
    top, usage = asyncio.run(GroqReranker(FakeClient("not json at all"), "m").rerank("authentication mandatory", candidates(), 2))
    assert usage["mode"] == "heuristic"
    assert len(top) == 2 and top[0].rerank_score is not None


def test_provider_failure_falls_back():
    top, usage = asyncio.run(GroqReranker(FakeClient(fail=True), "m").rerank("authentication", candidates(), 3))
    assert usage["mode"] == "heuristic" and len(top) == 3


def test_heuristic_prefers_keyword_overlap():
    scores = heuristic_scores("multi-factor authentication", candidates())
    assert scores["a"] > scores["c"]
