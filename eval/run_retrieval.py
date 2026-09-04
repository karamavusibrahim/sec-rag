#!/usr/bin/env python
"""Retrieval ablation sweep.

Runs the eval set through each configuration and reports Recall@k, nDCG@10 and
MRR@10, so the contribution of each pipeline component is a number rather than
an assertion.

nDCG uses binary relevance with the ideal ranking computed from the actual
number of gold chunks for that query -- so a question with 3 gold chunks is not
penalised for being unable to put 10 relevant results in the top 10.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

from provenance import provenance  # noqa: E402
from sec_rag.index.build import load as load_index  # noqa: E402
from sec_rag.retrieve.hybrid import Retriever  # noqa: E402

CONFIGS: list[tuple[str, dict[str, bool]]] = [
    ("BM25 only",              {"use_dense": False, "use_sparse": True,  "use_rerank": False}),
    ("Dense only",             {"use_dense": True,  "use_sparse": False, "use_rerank": False}),
    ("Hybrid (RRF)",           {"use_dense": True,  "use_sparse": True,  "use_rerank": False}),
    ("BM25 + rerank",          {"use_dense": False, "use_sparse": True,  "use_rerank": True}),
    ("Dense + rerank",         {"use_dense": True,  "use_sparse": False, "use_rerank": True}),
    ("Hybrid + rerank",        {"use_dense": True,  "use_sparse": True,  "use_rerank": True}),
]


def lexical_overlap(question: str, gold_texts: Sequence[str]) -> float:
    """Share of the question's content words that appear in its gold passages.

    This is the control that makes the numeric-vs-narrative comparison
    interpretable. BM25 can only match on shared terms, so its score across two
    splits is uninterpretable without knowing how many terms they share: a
    sparse-retrieval win on narrative questions means something quite different
    if those questions were written using the passage's own vocabulary.

    Reported per split rather than argued about.
    """
    q = {w for w in re.findall(r"[a-z0-9']+", question.lower()) if len(w) > 3}
    if not q:
        return 0.0
    g: set[str] = set()
    for t in gold_texts:
        g |= {w for w in re.findall(r"[a-z0-9']+", t.lower()) if len(w) > 3}
    return len(q & g) / len(q)


def _require_gold(gold: set[str]) -> None:
    # A question with no gold chunks has no defined score. Returning 0.0
    # folded such rows into every mean as failures, so a malformed eval set
    # lowered every configuration's number by the same amount and nothing
    # reported it. `report_tables.r_at_1_ceiling` already refuses these rows;
    # the metrics themselves now refuse them too, so the two cannot disagree.
    if not gold:
        raise ValueError("a question with empty gold_chunk_ids has no defined "
                         "retrieval score; fix the eval set rather than "
                         "scoring it as zero")


def recall_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    _require_gold(gold)
    return len(set(retrieved[:k]) & gold) / len(gold)


def ndcg_at_k(retrieved: Sequence[str], gold: set[str], k: int = 10) -> float:
    _require_gold(gold)
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, cid in enumerate(retrieved[:k])
        if cid in gold
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def mrr_at_k(retrieved: Sequence[str], gold: set[str], k: int = 10) -> float:
    _require_gold(gold)
    for i, cid in enumerate(retrieved[:k]):
        if cid in gold:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(retriever: Retriever, questions: list[dict[str, Any]],
             flags: dict[str, bool], *, k: int = 10, candidates: int = 50,
             ) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Aggregate metrics plus a per-question record.

    The per-question nDCG is persisted because aggregates cannot support the
    claims made from them: "A beats B" needs a paired comparison on the same
    questions (win/loss counts, sign test), and the first release of these
    results stored only means -- which made every gap uncheckable after the
    fact.
    """
    agg = {"R@1": 0.0, "S@1": 0.0, "R@5": 0.0, "R@10": 0.0,
           "nDCG@10": 0.0, "MRR@10": 0.0}
    per_q: list[dict[str, Any]] = []
    for q in questions:
        gold = set(q["gold_chunk_ids"])
        hits = retriever.search(q["question"], top_k=k, candidates=candidates, **flags)
        ids = [h.chunk_id for h in hits]
        nd = ndcg_at_k(ids, gold, 10)
        agg["R@1"] += recall_at_k(ids, gold, 1)
        # Success@1 -- was the top hit relevant? R@1 divides by |gold|, so with
        # multi-gold labels its attainable maximum is well below 1.0 and it is
        # routinely misread as a hit rate. Report the hit rate explicitly.
        agg["S@1"] += 1.0 if (ids and ids[0] in gold) else 0.0
        agg["R@5"] += recall_at_k(ids, gold, 5)
        agg["R@10"] += recall_at_k(ids, gold, 10)
        agg["nDCG@10"] += nd
        agg["MRR@10"] += mrr_at_k(ids, gold, 10)
        per_q.append({"qid": q.get("qid"), "nDCG@10": round(nd, 4)})
    # `max(len(questions), 1)` turned an empty eval set into a full set of
    # all-zero metrics and a successful exit -- a mis-specified --split or a
    # renamed file produced a plausible-looking artifact recording a total
    # failure that never happened. An evaluation with nothing to evaluate is a
    # configuration error, not a score of zero.
    if not questions:
        raise ValueError(
            "no questions to evaluate -- check --eval-set and --split; "
            "an empty eval set is a configuration error, not a zero score")
    return {m: v / len(questions) for m, v in agg.items()}, per_q


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/processed"))
    ap.add_argument("--eval-set", type=Path, nargs="+",
                    default=[Path("data/eval/eval_set_v2.jsonl")],
                    help="one or more jsonl files; each question's 'split' field groups it")
    ap.add_argument("--by-split", action="store_true",
                    help="report every configuration separately per split")
    ap.add_argument("--limit", type=int, default=None, help="subsample for a fast run")
    ap.add_argument("--out", type=Path, default=Path("eval/results/retrieval.json"))
    args = ap.parse_args()

    questions = []
    for path in args.eval_set:
        questions += [json.loads(l) for l in path.read_text().splitlines() if l]
    # Interleave by ticker before any --limit. The eval files are grouped
    # ticker-by-ticker (AAPL first), so a head slice silently became an
    # all-AAPL eval: the v2 headline numbers were computed on 40/40 AAPL
    # questions while the corpus framing said AAPL+MSFT+NVDA.
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for q in questions:
        by_ticker.setdefault(q.get("ticker", "?"), []).append(q)
    interleaved: list[dict[str, Any]] = []
    while any(by_ticker.values()):
        for t in sorted(by_ticker):
            if by_ticker[t]:
                interleaved.append(by_ticker[t].pop(0))
    questions = interleaved
    if args.limit:
        questions = questions[: args.limit]

    index = load_index(args.index)
    retriever = Retriever(index)
    print(f"corpus {len(index)} chunks | eval {len(questions)} questions\n")

    # Group by split. A question with no 'split' predates the field and is
    # numeric by construction.
    splits: dict[str, list[dict[str, Any]]] = {}
    for q in questions:
        splits.setdefault(q.get("split", "numeric"), []).append(q)

    text_by_id = {c["chunk_id"]: c["text"] for c in (
        json.loads(l) for l in
        (args.index / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l
    )}
    overlap_summary: dict[str, dict[str, Any]] = {}
    for name, qs in sorted(splits.items()):
        ov = [lexical_overlap(q["question"],
                              [text_by_id.get(g, "") for g in q["gold_chunk_ids"]])
              for q in qs]
        overlap_summary[name] = {
            "n": len(ov),
            "mean": sum(ov) / max(len(ov), 1),
            "per_question": [
                {"qid": q.get("qid"), "overlap": value}
                for q, value in zip(qs, ov)
            ],
        }
        print(f"split {name:<10} n={len(qs):<4} "
              f"mean query/gold word overlap {sum(ov) / max(len(ov), 1):.3f}")
    print()

    groups: list[tuple[str, list[dict[str, Any]]]] = [("all", questions)]
    if args.by_split and len(splits) > 1:
        groups = sorted(splits.items()) + [("all", questions)]

    rows: list[dict[str, Any]] = []
    per_question: dict[str, list[dict[str, Any]]] = {}
    for split_name, qs in groups:
        for name, flags in CONFIGS:
            t0 = time.time()
            scores, per_q = evaluate(retriever, qs, flags)
            elapsed = time.time() - t0
            rows.append({"split": split_name, "config": name, "n": len(qs),
                         **scores, "seconds": round(elapsed, 1)})
            if split_name == "all":
                per_question[name] = per_q
            print(
                f"[{split_name:<9}] {name:<20} R@1={scores['R@1']:.3f}  "
                f"S@1={scores['S@1']:.3f}  R@5={scores['R@5']:.3f}  "
                f"R@10={scores['R@10']:.3f}  "
                f"nDCG@10={scores['nDCG@10']:.3f}  MRR@10={scores['MRR@10']:.3f}  "
                f"({elapsed:.0f}s)"
            )
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"n_questions": len(questions), "n_chunks": len(index),
         "embed_model": index.embed_model,
         # Provenance. The v1/v2 mixups documented in REPORT.md had to be
         # reverse-engineered from file mtimes because none of this was stamped.
         # `provenance` adds the code revision, corpus and eval-set checksums,
         # and the reranker identity -- the fields every review asked for.
         "provenance": provenance(
             args.index, eval_sets=args.eval_set,
             models={"embed": index.embed_model,
                     "rerank": retriever.rerank_model},
             extra={"configs": {n: f for n, f in CONFIGS},
                    "candidates": 50, "top_k": 10}),
         "eval_sets": [str(p) for p in args.eval_set],
         "limit": args.limit,
         "tickers": {t: sum(1 for q in questions if q.get("ticker") == t)
                     for t in sorted({q.get("ticker", "?") for q in questions})},
         "splits": {k: len(v) for k, v in splits.items()},
         "lexical_overlap": overlap_summary,
         "results": rows,
         "per_question_ndcg": per_question}, indent=2))

    # Markdown table, ready to paste into the README.
    for split_name, _ in groups:
        sub = [r for r in rows if r["split"] == split_name]
        print(f"\n**{split_name}** (n={sub[0]['n']})\n")
        print("| Configuration | R@1 | S@1 | R@5 | R@10 | nDCG@10 | MRR@10 |")
        print("|---|---|---|---|---|---|---|")
        for r in sub:
            print(f"| {r['config']} | {r['R@1']:.3f} | {r['S@1']:.3f} | "
                  f"{r['R@5']:.3f} | "
                  f"{r['R@10']:.3f} | {r['nDCG@10']:.3f} | {r['MRR@10']:.3f} |")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
