#!/usr/bin/env python
"""Optional DAT (Dynamic Alpha Tuning) fusion: per-query alpha via one LLM call.

The fixed-alpha convex sweep (`fusion_sweep.py`) showed weighted fusion repairs
RRF's damage, with optima at alpha = 0.8-0.9. DAT (arXiv 2503.23013) goes one
step further: instead of one global alpha, an LLM scores the *top-1 result of
each retriever* for the query at hand (0-5), and alpha is set per query:

    alpha = 0.5                     if Sv == 0 and Sb == 0
    alpha = 1.0                     if Sv == 5 and Sb != 5
    alpha = 0.0                     if Sb == 5 and Sv != 5
    alpha = Sv / (Sv + Sb)          otherwise, rounded to one decimal

Published gains are ~+3pp P@1 over the best fixed alpha on full sets, and
+6-8pp on the "hybrid-sensitive" subset where the two retrievers disagree at
rank 1 -- so results are reported both overall and stratified by disagreement.

Optional by design: fixed-alpha and dense-only baselines are computed on the
same candidate pools in the same run, and nothing in the default pipeline
changes. Cost: one small LLM call per query.
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

from fusion_sweep import minmax  # noqa: E402
from run_retrieval import ndcg_at_k  # noqa: E402
from sec_rag.index.build import load as load_index  # noqa: E402
from sec_rag.nvidia import chat_json, embed  # noqa: E402

CANDIDATES = 50
SCORER_MODEL = "deepseek-ai/deepseek-v4-flash"

SCORE_PROMPT = """Rate how effectively each passage answers the question, on a
0-5 scale (0 = irrelevant, 5 = directly and completely answers it).

Question: {question}

Passage A:
{passage_a}

Passage B:
{passage_b}

Respond with JSON only: {{"score_a": <0-5>, "score_b": <0-5>}}"""


def dat_alpha(sv: int, sb: int) -> float:
    if sv == 0 and sb == 0:
        return 0.5
    if sv == 5 and sb != 5:
        return 1.0
    if sb == 5 and sv != 5:
        return 0.0
    return round(sv / (sv + sb), 1)


def fused_ndcg(nd: dict, ns: dict, alpha: float, gold: set) -> float:
    fused = {d: alpha * nd.get(d, 0.0) + (1 - alpha) * ns.get(d, 0.0)
             for d in set(nd) | set(ns)}
    top = sorted(fused, key=fused.get, reverse=True)[:10]
    return ndcg_at_k(top, gold, 10)


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/processed"))
    ap.add_argument("--fixed-alpha", type=float, default=0.8,
                    help="the fixed-alpha baseline to beat (numeric optimum)")
    ap.add_argument("--out", type=Path, default=Path("eval/results/dat_fusion.json"))
    args = ap.parse_args()

    index = load_index(args.index)
    ids = [c["chunk_id"] for c in index.chunks]
    text_by_id = {c["chunk_id"]: c["text"] for c in index.chunks}
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
        sims = qvecs @ index.vectors.T

        rows = []
        for i, q in enumerate(questions):
            gold = set(q["gold_chunk_ids"])
            top = np.argsort(-sims[i])[:CANDIDATES]
            nd = minmax({ids[j]: float(sims[i][j]) for j in top})
            toks = bm25s.tokenize([q["question"]], stopwords="en",
                                  show_progress=False)
            docs, scores = bm25.retrieve(toks, k=CANDIDATES, show_progress=False)
            ns = minmax({ids[d]: float(s)
                         for d, s in zip(docs[0], scores[0]) if s > 0})

            dense_top1 = max(nd, key=nd.get) if nd else None
            sparse_top1 = max(ns, key=ns.get) if ns else None
            disagree = dense_top1 != sparse_top1

            try:
                data = chat_json(
                    SCORER_MODEL,
                    [{"role": "user", "content": SCORE_PROMPT.format(
                        question=q["question"],
                        passage_a=text_by_id.get(dense_top1, "")[:1200],
                        passage_b=text_by_id.get(sparse_top1, "")[:1200])}],
                    validate=lambda d: "score_a" in d and "score_b" in d,
                    max_tokens=100,
                )
                sv = max(0, min(5, int(data["score_a"])))
                sb = max(0, min(5, int(data["score_b"])))
                alpha = dat_alpha(sv, sb)
            except Exception as exc:  # noqa: BLE001
                print(f"  scorer failed on {q.get('qid')}: {exc}",
                      file=sys.stderr)
                sv = sb = -1
                alpha = args.fixed_alpha  # fall back to the baseline weighting

            rows.append({
                "qid": q.get("qid"), "disagree": disagree,
                "sv": sv, "sb": sb, "alpha": alpha,
                "dat": fused_ndcg(nd, ns, alpha, gold),
                "fixed": fused_ndcg(nd, ns, args.fixed_alpha, gold),
                "dense": fused_ndcg(nd, ns, 1.0, gold),
            })
            if (i + 1) % 20 == 0:
                print(f"  [{split}] {i + 1}/{len(questions)}")

        def mean(key, subset):
            return (sum(r[key] for r in subset) / len(subset)) if subset else 0.0

        sensitive = [r for r in rows if r["disagree"]]
        wins = sum(1 for r in rows if r["dat"] > r["fixed"])
        losses = sum(1 for r in rows if r["dat"] < r["fixed"])
        results[split] = {
            "n": len(rows),
            "nDCG@10": {"dat": round(mean("dat", rows), 4),
                        "fixed": round(mean("fixed", rows), 4),
                        "dense": round(mean("dense", rows), 4)},
            "hybrid_sensitive": {
                "n": len(sensitive),
                "dat": round(mean("dat", sensitive), 4),
                "fixed": round(mean("fixed", sensitive), 4)},
            "dat_vs_fixed_wins_losses": [wins, losses],
            "mean_alpha": round(mean("alpha", rows), 3),
            "per_question": rows,
        }
        r = results[split]
        print(f"\n[{split}] n={r['n']}  DAT {r['nDCG@10']['dat']}  "
              f"fixed(a={args.fixed_alpha}) {r['nDCG@10']['fixed']}  "
              f"dense {r['nDCG@10']['dense']}  mean-alpha {r['mean_alpha']}")
        print(f"  hybrid-sensitive n={r['hybrid_sensitive']['n']}: "
              f"DAT {r['hybrid_sensitive']['dat']} vs fixed "
              f"{r['hybrid_sensitive']['fixed']}  |  W/L {wins}/{losses}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
