#!/usr/bin/env python
"""Convex-combination fusion sweep: the experiment the RRF result asks for.

The ablation established that RRF fusion loses to dense alone whenever one
retriever is materially better (numeric p=0.0007), because equal rank
weighting has no way to distrust a component. The standard published fix
(Bruch, Gai & Ingber, ACM TOIS 2023) is a convex combination of normalized
scores:

    fused(d) = alpha * norm(dense_score(d)) + (1 - alpha) * norm(bm25_score(d))

with min-max normalization computed *per query* over each retriever's own
top-k candidates (a per-query normalization bounds BM25's unbounded,
outlier-heavy scores -- the usual argument for preferring RRF). At the optimal
alpha this cannot lose to the better single retriever, which is precisely the
guarantee RRF lacks; alpha=1 is dense-only, alpha=0 is BM25-only.

Costs one query embedding per question (batched); every other number is
computed locally from the saved dense vectors and the BM25 index. RRF k=60 is
re-computed on the same candidate pools as the baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bm25s  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from run_retrieval import ndcg_at_k  # noqa: E402
from sec_rag.index.build import load as load_index  # noqa: E402
from sec_rag.nvidia import embed  # noqa: E402

CANDIDATES = 50
ALPHAS = [round(a / 10, 1) for a in range(11)]
RRF_K = 60


def minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi <= lo:
        return {d: 1.0 for d in scores}
    return {d: (s - lo) / (hi - lo) for d, s in scores.items()}


def sweep(questions, dense_ranked, sparse_ranked):
    """Per-alpha mean nDCG@10, plus an RRF row on the same candidate pools."""
    out = {a: 0.0 for a in ALPHAS}
    rrf_total = 0.0
    for q, dr, sr in zip(questions, dense_ranked, sparse_ranked):
        gold = set(q["gold_chunk_ids"])
        nd = minmax(dr)
        ns = minmax(sr)
        for a in ALPHAS:
            fused = {d: a * nd.get(d, 0.0) + (1 - a) * ns.get(d, 0.0)
                     for d in set(nd) | set(ns)}
            top = sorted(fused, key=fused.get, reverse=True)[:10]
            out[a] += ndcg_at_k(top, gold, 10)
        rrf = {}
        for ranked in (dr, sr):
            for rank, d in enumerate(sorted(ranked, key=ranked.get, reverse=True), 1):
                rrf[d] = rrf.get(d, 0.0) + 1.0 / (RRF_K + rank)
        top = sorted(rrf, key=rrf.get, reverse=True)[:10]
        rrf_total += ndcg_at_k(top, gold, 10)
    n = len(questions)
    return {a: v / n for a, v in out.items()}, rrf_total / n


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("eval/results/fusion_sweep.json"))
    args = ap.parse_args()

    index = load_index(args.index)
    ids = [c["chunk_id"] for c in index.chunks]
    bm25 = bm25s.BM25.load(str(args.index / "bm25"))

    splits = {
        "numeric": [json.loads(l) for l in
                    Path("data/eval/eval_set_v2.jsonl").read_text().splitlines() if l],
        "narrative": [json.loads(l) for l in
                      Path("data/eval/eval_narrative.jsonl").read_text().splitlines() if l],
    }

    results = {}
    for split, questions in splits.items():
        texts = [q["question"] for q in questions]
        qvecs = np.asarray(embed(texts, model=index.embed_model,
                                 input_type="query"), dtype=np.float32)
        qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True)
        sims = qvecs @ index.vectors.T  # (nq, nchunks), exact

        dense_ranked, sparse_ranked = [], []
        for i, q in enumerate(questions):
            top = np.argsort(-sims[i])[:CANDIDATES]
            dense_ranked.append({ids[j]: float(sims[i][j]) for j in top})
            toks = bm25s.tokenize([q["question"]], stopwords="en", show_progress=False)
            docs, scores = bm25.retrieve(toks, k=CANDIDATES, show_progress=False)
            sparse_ranked.append({ids[d]: float(s)
                                  for d, s in zip(docs[0], scores[0]) if s > 0})

        curve, rrf = sweep(questions, dense_ranked, sparse_ranked)
        best_a = max(curve, key=curve.get)
        results[split] = {"curve": curve, "rrf_k60": round(rrf, 4),
                          "best_alpha": best_a,
                          "best_ndcg": round(curve[best_a], 4),
                          "dense_only": round(curve[1.0], 4),
                          "bm25_only": round(curve[0.0], 4), "n": len(questions)}

        print(f"\n[{split}] n={len(questions)}")
        for a in ALPHAS:
            marker = "  <- best" if a == best_a else ""
            print(f"  alpha={a:.1f}  nDCG@10={curve[a]:.4f}{marker}")
        print(f"  RRF k=60   nDCG@10={rrf:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
