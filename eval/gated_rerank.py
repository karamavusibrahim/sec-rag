#!/usr/bin/env python
"""Optional confidence-gated reranking: apply the reranker only when it is sure.

The v3 result left reranking in an odd place: it adds +0.074 nDCG@10 on the
numeric split but only +0.006 on narrative (3 wins, 3 losses, 9 ties). "Can LLM Rerankers
Predict Their Own Performance?" (arXiv 2606.03535) shows reranker output
carries a usable self-estimate of its own reliability; queries where the
reranker is unconfident are the ones where applying it hurts or wastes time.

Adaptation for a hosted cross-encoder that returns raw logits (no extra calls,
no extra tokens): compute per-query confidence statistics from the logits the
standard rerank pass already produces --

    top1     highest candidate logit (absolute relevance of the best hit)
    margin   top1 - top2 logit       (how decisively one candidate wins)
    entropy  softmax entropy over all candidate logits (overall ambiguity)

-- then gate: use the reranked order when the statistic clears a threshold,
keep the dense order otherwise. Thresholds are swept over the observed
percentiles of each statistic, so the curve is in-sample (same caveat as the
fusion-sweep alpha); the oracle row is the ceiling any gate could reach.

This is a quality gate, not a cost-saving pre-router: the logits only exist
after the reranker call, so every query still pays the reranking inference cost.
Optional by design: nothing in the default pipeline changes; dense-only and
rerank-always baselines are computed on the same candidate pools in the same
run. Cost: one rerank call per question (the same call the standard rerank
config already makes).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

from run_retrieval import ndcg_at_k  # noqa: E402
from sec_rag.index.build import load as load_index  # noqa: E402
from sec_rag.nvidia import rerank, embed  # noqa: E402

CANDIDATES = 50
K = 10
PERCENTILES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
STATS = ("top1", "margin", "entropy")


def softmax_entropy(logits: list[float]) -> float:
    arr = np.asarray(logits, dtype=np.float64)
    arr -= arr.max()
    p = np.exp(arr)
    p /= p.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def confident(stat: str, value: float, threshold: float) -> bool:
    """Entropy is inverted: low entropy = confident."""
    return value <= threshold if stat == "entropy" else value >= threshold


def gate_curves(rows: list[dict]) -> dict:
    """Sweep percentile thresholds per statistic; also compute the oracle."""
    curves: dict = {}
    for stat in STATS:
        values = [r["stats"][stat] for r in rows]
        curve = []
        for pct in PERCENTILES:
            thr = float(np.percentile(values, pct))
            scores, n_reranked = [], 0
            for r in rows:
                use = confident(stat, r["stats"][stat], thr)
                n_reranked += use
                scores.append(r["ndcg_rerank"] if use else r["ndcg_dense"])
            curve.append({
                "pct": pct, "threshold": round(thr, 4),
                "nDCG@10": round(sum(scores) / len(scores), 4),
                "frac_reranked": round(n_reranked / len(rows), 3),
            })
        best = max(curve, key=lambda c: c["nDCG@10"])
        curves[stat] = {"curve": curve, "best": best}
    oracle = sum(max(r["ndcg_rerank"], r["ndcg_dense"]) for r in rows) / len(rows)
    curves["oracle"] = round(oracle, 4)
    return curves


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/processed"))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap questions per split (smoke runs)")
    ap.add_argument("--out", type=Path,
                    default=Path("eval/results/gated_rerank.json"))
    args = ap.parse_args()

    index = load_index(args.index)
    ids = [c["chunk_id"] for c in index.chunks]
    by_id = {c["chunk_id"]: c for c in index.chunks}

    splits = {
        "numeric": [json.loads(l) for l in
                    Path("data/eval/eval_set_v2.jsonl").read_text().splitlines() if l],
        "narrative": [json.loads(l) for l in
                      Path("data/eval/eval_narrative.jsonl").read_text().splitlines() if l],
    }
    if args.limit:
        splits = {s: qs[:args.limit] for s, qs in splits.items()}

    results = {}
    for split, questions in splits.items():
        texts = [q["question"] for q in questions]
        qvecs = np.asarray(embed(texts, model=index.embed_model,
                                 input_type="query"), dtype=np.float32)
        qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True)
        sims = qvecs @ index.vectors.T

        rows: list[dict] = []
        failures: list[dict] = []
        for i, q in enumerate(questions):
            gold = set(q["gold_chunk_ids"])
            top = np.argsort(-sims[i])[:CANDIDATES]
            cand_ids = [ids[j] for j in top]
            dense_order = cand_ids[:K]

            passages = [by_id[cid]["breadcrumb"] + "\n" + by_id[cid]["text"]
                        for cid in cand_ids]
            try:
                ranked = rerank(q["question"], passages)
            except Exception as exc:  # noqa: BLE001
                # `continue` silently removed the query from every denominator
                # *and* from threshold selection, so a run where half the
                # rerank calls failed reported clean metrics over the surviving
                # half with nothing to distinguish it from a complete run. An
                # evaluation that quietly changes its own sample is worse than
                # one that stops.
                print(f"  rerank failed on {q.get('qid')}: {exc}",
                      file=sys.stderr)
                failures.append({"qid": q.get("qid"), "error": str(exc)})
                continue
            # An empty-but-successful response is a failure of the call, not
            # a zero-confidence result. Letting it through raised IndexError on
            # `logits[0]` outside the try block, killing the whole evaluation
            # instead of being counted as the one failed query it is.
            if not ranked:
                print(f"  rerank returned nothing for {q.get('qid')}",
                      file=sys.stderr)
                failures.append({"qid": q.get("qid"),
                                 "error": "empty rerank response"})
                continue
            logits = [lg for _, lg in ranked]
            rerank_order = [cand_ids[idx] for idx, _ in ranked[:K]]

            stratum = "table" if any(
                by_id.get(g, {}).get("kind") == "table" for g in gold) else "text"
            rows.append({
                "qid": q.get("qid"), "stratum": stratum,
                "stats": {
                    "top1": round(logits[0], 4),
                    "margin": round(logits[0] - logits[1], 4)
                              if len(logits) > 1 else 0.0,
                    "entropy": round(softmax_entropy(logits), 4),
                },
                "ndcg_dense": ndcg_at_k(dense_order, gold, K),
                "ndcg_rerank": ndcg_at_k(rerank_order, gold, K),
            })
            if (i + 1) % 20 == 0:
                print(f"  [{split}] {i + 1}/{len(questions)}")

        def mean(key, subset):
            return (sum(r[key] for r in subset) / len(subset)) if subset else 0.0

        strata = {}
        for name in ("table", "text"):
            sub = [r for r in rows if r["stratum"] == name]
            if sub:
                strata[name] = {
                    "n": len(sub),
                    "dense": round(mean("ndcg_dense", sub), 4),
                    "rerank": round(mean("ndcg_rerank", sub), 4),
                }
        results[split] = {
            # Both numbers, always. `n` alone cannot distinguish a complete run
            # from one that lost half its queries to rerank errors.
            "n_attempted": len(questions),
            "n": len(rows),
            "n_failed": len(failures),
            "failures": failures,
            "nDCG@10": {"dense": round(mean("ndcg_dense", rows), 4),
                        "rerank_always": round(mean("ndcg_rerank", rows), 4)},
            "gates": gate_curves(rows) if rows else {},
            "strata": strata,
            "per_question": rows,
        }
        r = results[split]
        if failures:
            print(f"\n[{split}] WARNING: {len(failures)}/{len(questions)} "
                  f"queries failed reranking and are excluded from every metric "
                  f"below. Treat these as complete-case results.", file=sys.stderr)
        print(f"\n[{split}] n={r['n']}/{r['n_attempted']}  "
              f"dense {r['nDCG@10']['dense']}  "
              f"rerank-always {r['nDCG@10']['rerank_always']}  "
              f"oracle {r['gates'].get('oracle')}")
        if not r["gates"]:
            print("  no queries succeeded; no gate curves to report",
                  file=sys.stderr)
        for stat in STATS:
            # `gates` is {} when every rerank call failed, and indexing it threw
            # KeyError while printing -- a crash in the reporting path that hid
            # the actual failure from whoever ran it.
            gate = r["gates"].get(stat)
            if not gate:
                continue
            b = gate["best"]
            print(f"  gate[{stat}]  best {b['nDCG@10']} at p{b['pct']} "
                  f"(reranks {b['frac_reranked']:.0%} of queries)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
