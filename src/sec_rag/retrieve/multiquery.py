"""Optional multi-query retrieval with rank fusion (RAG-Fusion).

RAG-Fusion (Rackauckas, arXiv 2402.03367) generates several rewrites of the
user's question, retrieves for each, and fuses the ranked lists with
reciprocal rank fusion. The argument for it on this corpus is specific: the
numeric eval questions are written from XBRL concept labels ("research and
development expense") while filings phrase the same line item several ways
("Research and development", "R&D", a table row with no label near the
value), and the ablation showed dense retrieval alone reaching only 0.558
R@5. Rewrites that use filing vocabulary are the cheapest way to widen the
candidate pool without touching the index.

The argument against it is equally specific, and it is why this is optional
and off by default: `eval/fusion_sweep.py` established that RRF *hurts* when
its inputs are unequal, and rewrites are not equal to the original question.
The fusion here therefore weights the original question's list higher than
the rewrites (`original_weight`), and the whole thing has to be measured
before anyone believes it. `eval/multiquery_eval.py` does that measurement;
no number from it is published in this repository because it has not been
run against the hosted API.

Cost: one small LLM call per question for the rewrites, plus one embedding
per rewrite. Everything is injectable so the fusion logic is testable without
a network.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Sequence

from .hybrid import Hit, RRF_K, Retriever

REWRITE_PROMPT = """Rewrite the question below {n} different ways, so that each
rewrite would match how a US company's 10-K filing actually phrases the same
information. Use the vocabulary of financial statements and MD&A ("net sales",
"research and development", "property, plant and equipment"), keep the company
and the fiscal year exactly as given, and do not answer the question.

Question: {question}

Return JSON only: {{"rewrites": ["...", "..."]}}"""

DEFAULT_REWRITE_MODELS = (
    "deepseek-ai/deepseek-v4-flash-0731",
    "meta/llama-3.3-70b-instruct",
)


def _clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def rewrite_queries(question: str, *, n: int = 3,
                    models: Sequence[str] = DEFAULT_REWRITE_MODELS,
                    chat_json_chain: Callable[..., tuple[Any, str]] | None = None,
                    ) -> list[str]:
    """`n` rewrites of `question`, never including the question itself.

    Falls back to no rewrites when the model call fails: multi-query search
    then degrades to single-query search rather than to no search.
    """
    if chat_json_chain is None:
        from ..nvidia import chat_json_chain as _chain
        chat_json_chain = _chain
    try:
        data, _ = chat_json_chain(
            list(models),
            [{"role": "user", "content": REWRITE_PROMPT.format(
                n=n, question=question)}],
            validate=lambda d: isinstance(d.get("rewrites"), list),
            max_tokens=400,
        )
    except Exception:  # noqa: BLE001
        return []
    out: list[str] = []
    seen = {_clean(question).lower()}
    for r in data["rewrites"]:
        s = _clean(r)
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
        if len(out) >= n:
            break
    return out


def fuse(ranked_lists: Sequence[Sequence[Hit]], *,
         weights: Sequence[float] | None = None, k: int = RRF_K) -> list[Hit]:
    """Weighted reciprocal rank fusion over lists of Hits, best first.

    Weighted because the lists are not peers: the original question is the
    ground truth of intent and a rewrite is a guess about vocabulary.
    Unweighted RRF is the equal-weighting failure `fusion_sweep.py` measured.
    Each chunk keeps the Hit object from the list where it ranked best, so
    citations and rerank logits survive fusion.
    """
    weights = list(weights) if weights is not None else [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("one weight per ranked list")
    scores: dict[str, float] = {}
    best: dict[str, tuple[int, Hit]] = {}
    for w, hits in zip(weights, ranked_lists):
        for pos, h in enumerate(hits):
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + w / (k + pos + 1)
            if h.chunk_id not in best or pos < best[h.chunk_id][0]:
                best[h.chunk_id] = (pos, h)
    fused: list[Hit] = []
    for cid, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        h = best[cid][1]
        fused.append(Hit(chunk=h.chunk, score=score, dense_rank=h.dense_rank,
                         sparse_rank=h.sparse_rank, rerank_logit=h.rerank_logit))
    return fused


def search_multiquery(retriever: Retriever, question: str, *, top_k: int = 8,
                      candidates: int = 50, n_rewrites: int = 3,
                      original_weight: float = 2.0,
                      rewriter: Callable[[str], Sequence[str]] | None = None,
                      **flags: bool) -> list[Hit]:
    """Retrieve for the question and its rewrites; fuse; return the top_k.

    `flags` are the `Retriever.search` component switches. Reranking, if
    enabled, runs inside each single-query search; the fused order is then a
    fusion of reranked lists. Running the reranker once over the fused pool
    instead is a defensible alternative and is left to the eval script, where
    the choice can be measured rather than assumed.
    """
    rewriter = rewriter or (lambda q: rewrite_queries(q, n=n_rewrites))
    queries = [question, *rewriter(question)]
    lists = [retriever.search(q, top_k=candidates, candidates=candidates, **flags)
             for q in queries]
    weights = [original_weight] + [1.0] * (len(lists) - 1)
    return fuse(lists, weights=weights)[:top_k]


def describe(queries: Sequence[str]) -> str:
    """Debug helper: the query set as one line of JSON."""
    return json.dumps(list(queries), ensure_ascii=False)
