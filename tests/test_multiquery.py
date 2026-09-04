"""Offline tests for the optional multi-query (RAG-Fusion) retrieval.

Everything network-bound is injected: the rewriter is a stub and the
retriever is a fake keyed on query text. What is tested is the fusion logic
and the fallbacks, which is what can go wrong silently.
"""

from __future__ import annotations

from sec_rag.retrieve.hybrid import Hit
from sec_rag.retrieve.multiquery import fuse, rewrite_queries, search_multiquery


def hit(cid: str, score: float = 1.0) -> Hit:
    return Hit(chunk={"chunk_id": cid, "text": cid, "ticker": "AAPL",
                      "form": "10-K", "report_date": "2024-09-28"}, score=score)


class FakeRetriever:
    def __init__(self, table: dict[str, list[str]]):
        self.table = table
        self.calls: list[str] = []

    def search(self, query, *, top_k, candidates, **flags):
        self.calls.append(query)
        return [hit(c) for c in self.table.get(query, [])][:top_k]


class TestFuse:
    def test_a_chunk_ranked_by_every_list_wins(self):
        fused = fuse([[hit("a"), hit("b")], [hit("b"), hit("c")]])
        assert [h.chunk_id for h in fused][0] == "b"

    def test_weights_break_ties_toward_the_original(self):
        # "a" is top of the original; "c" is top of a rewrite. Equal weights
        # tie them; the original's weight decides.
        fused = fuse([[hit("a")], [hit("c")]], weights=[2.0, 1.0])
        assert [h.chunk_id for h in fused] == ["a", "c"]

    def test_the_best_ranked_hit_object_is_kept(self):
        first = hit("a", score=0.1)
        first.rerank_logit = 3.5
        fused = fuse([[hit("z"), hit("a", score=0.9)], [first]])
        kept = next(h for h in fused if h.chunk_id == "a")
        assert kept.rerank_logit == 3.5, "fusion dropped the better hit's fields"

    def test_weight_count_must_match(self):
        import pytest
        with pytest.raises(ValueError):
            fuse([[hit("a")]], weights=[1.0, 1.0])


class TestSearchMultiquery:
    def test_rewrites_widen_the_pool(self):
        r = FakeRetriever({"q": ["a"], "r1": ["b"], "r2": ["c"]})
        out = search_multiquery(r, "q", top_k=5, rewriter=lambda q: ["r1", "r2"])
        assert {h.chunk_id for h in out} == {"a", "b", "c"}
        assert out[0].chunk_id == "a", "the original question's hit leads"
        assert r.calls == ["q", "r1", "r2"]

    def test_no_rewrites_degrades_to_single_query(self):
        r = FakeRetriever({"q": ["a", "b"]})
        out = search_multiquery(r, "q", top_k=5, rewriter=lambda q: [])
        assert [h.chunk_id for h in out] == ["a", "b"]

    def test_top_k_is_honoured_after_fusion(self):
        r = FakeRetriever({"q": ["a", "b", "c"], "r": ["d", "e"]})
        assert len(search_multiquery(r, "q", top_k=2, rewriter=lambda q: ["r"])) == 2


class TestRewriteQueries:
    def test_rewrites_exclude_the_question_and_duplicates(self):
        reply = {"rewrites": ["Q?", "net sales fiscal 2024", "Net sales fiscal 2024",
                              "", "R&D expense 2024"]}
        out = rewrite_queries("Q?", n=3,
                              chat_json_chain=lambda *a, **k: (reply, "m"))
        assert out == ["net sales fiscal 2024", "R&D expense 2024"]

    def test_a_failed_call_yields_no_rewrites_not_an_exception(self):
        def boom(*a, **k):
            raise RuntimeError("down")
        assert rewrite_queries("Q?", chat_json_chain=boom) == []
