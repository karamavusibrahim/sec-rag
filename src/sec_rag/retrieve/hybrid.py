"""Hybrid retrieval: dense + BM25, fused with RRF, then cross-encoder reranked.

Every stage is switchable, because the point of this project is the ablation
table rather than any single configuration. `Retriever.search()` takes flags for
each component so the eval harness can sweep configurations without rebuilding.

**Why fuse with RRF instead of score-weighting.** Dense cosine similarity lives in
[-1, 1]; BM25 scores are unbounded and corpus-dependent. Normalizing them onto a
common scale requires a fitted mapping that shifts whenever the corpus changes.
Reciprocal Rank Fusion sidesteps the problem by discarding magnitudes and using
only rank:  score(d) = sum_r 1 / (k + rank_r(d)).  It has one hyperparameter,
needs no tuning per corpus, and is what the TREC literature keeps finding hard to
beat.

**Does hybrid actually help? Measured, not assumed -- and the answer is "no,
not on this workload."** The prior for this design was that filing queries mix
semantics ("main supply-chain risks") with exact identifiers ("ASC 606",
"fiscal 2024"), so dense and BM25 would be complementary. On the full
120-question XBRL-numeric split, with BM25 indexing the same breadcrumb+text
representation dense embeds, that prior is wrong:

    Dense only        nDCG@10 0.607   R@10 0.765
    Hybrid (RRF)      nDCG@10 0.528   R@10 0.709   <- worse, p=0.0007 paired
    Dense + rerank    nDCG@10 0.681   R@10 0.878
    Hybrid + rerank   nDCG@10 0.681   R@10 0.878   <- reranker erases the gap

And the narrative split (15 decontaminated prose questions) replicates the
direction with BM25 at its best: dense 0.757 vs hybrid 0.613. The mechanism is
not "BM25 is near-random" -- it recovers to 0.468 on prose -- but that RRF's
equal rank weighting degrades the fusion whenever one retriever is materially
better than the other, which nothing in RRF measures. The cross-encoder then
repairs the loss completely, which is why Hybrid+rerank looks fine and hides
the underlying problem. Both retrievers stay switchable so claims like these
stay checkable; full history (including two retracted earlier versions of
these numbers) in REPORT.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import bm25s
import numpy as np

from ..index.build import Index
from ..nvidia import RERANK_NEMOTRON, embed, rerank

RRF_K = 60  # standard constant from the original RRF paper


@dataclass
class Hit:
    chunk: dict[str, Any]
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_logit: float | None = None

    @property
    def chunk_id(self) -> str:
        return self.chunk["chunk_id"]

    @property
    def text(self) -> str:
        return self.chunk["text"]

    def citation(self) -> str:
        c = self.chunk
        item = f", Item {c['item']}" if c.get("item") else ""
        return f"{c['ticker']} {c['form']} {c['report_date'][:4]}{item}"


def _rrf(rank_lists: Sequence[Sequence[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion over lists of document indices, best-first."""
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for position, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + position + 1)
    return scores


class Retriever:
    def __init__(self, index: Index, *, rerank_model: str = RERANK_NEMOTRON):
        self.index = index
        self.rerank_model = rerank_model

    # -- individual retrievers -------------------------------------------

    def dense(self, query: str, top_k: int = 50) -> list[int]:
        # input_type="query" is not cosmetic: these are asymmetric encoders and
        # using the passage prefix for queries silently degrades results.
        qv = np.asarray(
            embed([query], model=self.index.embed_model, input_type="query")[0],
            dtype=np.float32,
        )
        qv /= np.linalg.norm(qv) or 1.0
        sims = self.index.vectors @ qv
        top = np.argpartition(-sims, min(top_k, len(sims) - 1))[:top_k]
        return [int(i) for i in top[np.argsort(-sims[top])]]

    def sparse(self, query: str, top_k: int = 50) -> list[int]:
        tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        k = min(top_k, len(self.index.chunks))
        idx, _ = self.index.retriever.retrieve(tokens, k=k, show_progress=False)
        return [int(i) for i in idx[0]]

    # -- the pipeline -----------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        candidates: int = 50,
        use_dense: bool = True,
        use_sparse: bool = True,
        use_rerank: bool = True,
    ) -> list[Hit]:
        """Retrieve, fuse, and optionally rerank.

        Flags exist so the eval harness can isolate each component's
        contribution; see eval/run_retrieval.py.
        """
        if not (use_dense or use_sparse):
            raise ValueError("at least one of use_dense / use_sparse must be True")

        dense_ranks = self.dense(query, candidates) if use_dense else []
        sparse_ranks = self.sparse(query, candidates) if use_sparse else []

        lists = [r for r in (dense_ranks, sparse_ranks) if r]
        if len(lists) == 1:
            # Single retriever: preserve its own ordering rather than passing a
            # one-element list through RRF, which would only rescale scores.
            ordered = [(i, 1.0 / (RRF_K + p + 1)) for p, i in enumerate(lists[0])]
        else:
            fused = _rrf(lists)
            ordered = sorted(fused.items(), key=lambda kv: -kv[1])

        d_pos = {i: p for p, i in enumerate(dense_ranks)}
        s_pos = {i: p for p, i in enumerate(sparse_ranks)}

        hits = [
            Hit(
                chunk=self.index.chunks[i],
                score=score,
                dense_rank=d_pos.get(i),
                sparse_rank=s_pos.get(i),
            )
            for i, score in ordered[:candidates]
        ]

        if use_rerank and hits:
            ranked = rerank(
                query,
                [h.chunk["breadcrumb"] + "\n" + h.text for h in hits],
                model=self.rerank_model,
            )
            reordered: list[Hit] = []
            for original_idx, logit in ranked:
                h = hits[original_idx]
                h.rerank_logit = logit
                h.score = logit
                reordered.append(h)
            hits = reordered

        return hits[:top_k]
