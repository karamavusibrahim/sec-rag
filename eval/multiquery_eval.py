#!/usr/bin/env python
"""Optional: measure multi-query rank fusion (RAG-Fusion) against dense-only.

`sec_rag/retrieve/multiquery.py` implements the method; this script is how
it gets a number, and it has not been run against the hosted API in this
repository -- there is no committed artifact and nothing in the README
quotes one. It exists so the claim can be made *after* it is measured, on
the same eval sets and the same candidate pools as every other arm.

Arms, per split:

    dense            the measured default (dense, no rerank)
    multiquery       dense per query, weighted RRF over original + rewrites
    dense+rerank     the strongest single-query arm
    multiquery+rerank
                     fused pool, reranked once

Everything except the rewrites and embeddings is computed locally. Cost:
one small LLM call and `n_rewrites` extra query embeddings per question,
plus the usual rerank calls for the reranked arms.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

from fusion_sweep import EVAL_SETS  # noqa: E402
from provenance import provenance  # noqa: E402
from run_retrieval import ndcg_at_k, recall_at_k  # noqa: E402
from sec_rag.index.build import load as load_index  # noqa: E402
from sec_rag.nvidia import rerank  # noqa: E402
from sec_rag.retrieve.hybrid import Retriever  # noqa: E402
from sec_rag.retrieve.multiquery import (  # noqa: E402
    DEFAULT_REWRITE_MODELS,
    fuse,
    rewrite_queries,
)

CANDIDATES = 50
K = 10


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/processed"))
    ap.add_argument("--n-rewrites", type=int, default=3)
    ap.add_argument("--original-weight", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path("eval/results/multiquery.json"))
    args = ap.parse_args()

    index = load_index(args.index)
    retriever = Retriever(index)
    by_id = {c["chunk_id"]: c for c in index.chunks}

    splits = {
        name: [json.loads(l) for l in path.read_text().splitlines() if l]
        for name, path in EVAL_SETS.items()
    }
    if args.limit:
        splits = {s: qs[:args.limit] for s, qs in splits.items()}

    results: dict = {}
    for split, questions in splits.items():
        if not questions:
            raise ValueError(f"no questions in the {split} split")
        rows = []
        n_rewrite_failures = 0
        for i, q in enumerate(questions):
            gold = set(q["gold_chunk_ids"])
            rewrites = rewrite_queries(q["question"], n=args.n_rewrites)
            n_rewrite_failures += not rewrites
            queries = [q["question"], *rewrites]
            lists = [retriever.search(qq, top_k=CANDIDATES, candidates=CANDIDATES,
                                      use_dense=True, use_sparse=False,
                                      use_rerank=False) for qq in queries]
            dense = [h.chunk_id for h in lists[0]]
            fused = fuse(lists, weights=[args.original_weight]
                         + [1.0] * (len(lists) - 1))
            multi = [h.chunk_id for h in fused]

            def reranked(ids: list[str]) -> list[str]:
                passages = [by_id[c]["breadcrumb"] + "\n" + by_id[c]["text"]
                            for c in ids[:CANDIDATES]]
                order = rerank(q["question"], passages, model=retriever.rerank_model)
                return [ids[idx] for idx, _ in order]

            arms = {"dense": dense, "multiquery": multi,
                    "dense+rerank": reranked(dense),
                    "multiquery+rerank": reranked(multi)}
            rows.append({
                "qid": q.get("qid"), "n_rewrites": len(rewrites),
                "rewrites": rewrites,
                **{f"ndcg::{a}": ndcg_at_k(ids, gold, K) for a, ids in arms.items()},
                **{f"r10::{a}": recall_at_k(ids, gold, K) for a, ids in arms.items()},
            })
            if (i + 1) % 20 == 0:
                print(f"  [{split}] {i + 1}/{len(questions)}")

        def mean(key):
            return round(sum(r[key] for r in rows) / len(rows), 4)

        summary = {a: {"nDCG@10": mean(f"ndcg::{a}"), "R@10": mean(f"r10::{a}")}
                   for a in ("dense", "multiquery", "dense+rerank",
                             "multiquery+rerank")}
        wins = sum(1 for r in rows if r["ndcg::multiquery"] > r["ndcg::dense"])
        losses = sum(1 for r in rows if r["ndcg::multiquery"] < r["ndcg::dense"])
        results[split] = {
            "n": len(rows), "n_rewrite_failures": n_rewrite_failures,
            "arms": summary,
            "multiquery_vs_dense_wins_losses": [wins, losses],
            "per_question": rows,
        }
        print(f"\n[{split}] n={len(rows)}  rewrite failures {n_rewrite_failures}")
        for a, m in summary.items():
            print(f"  {a:<18} nDCG@10 {m['nDCG@10']}  R@10 {m['R@10']}")
        print(f"  multiquery vs dense W/L {wins}/{losses}")

    results["provenance"] = provenance(
        args.index, eval_sets=EVAL_SETS.values(),
        models={"embed": index.embed_model, "rerank": retriever.rerank_model,
                "rewrite_chain": list(DEFAULT_REWRITE_MODELS)},
        extra={"candidates": CANDIDATES, "k": K, "n_rewrites": args.n_rewrites,
               "original_weight": args.original_weight, "limit": args.limit})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
