#!/usr/bin/env python
"""Optional topic-compass steering: route queries to the right 10-K section.

MCompassRAG (arXiv 2606.18508) shows topic metadata acting as a "semantic
compass" for paragraph-level retrieval: predict where in the document an
answer should live, then steer scores toward chunks whose metadata agrees.
This corpus already carries the needed metadata on every chunk -- `item`
(10-K section) and `kind` (table vs prose) -- it is just unused at query time.
The v3 residual failure mode (numeric questions landing on prose instead of
the financial-statement tables, ~2x gap) is exactly a routing error.

Per query, one small LLM call predicts up to two 10-K items plus the expected
answer kind. Steering is a convex boost on min-max-normalized dense scores:

    steered(d) = norm_dense(d) + beta * match(d)
    match(d)   = (item(d) in predicted_items) / 2 + (kind(d) == predicted_kind) / 2

Beta is swept (beta=0 is the untouched dense baseline), and compass accuracy
(did the prediction contain the gold chunk's item?) is reported separately so
prediction errors are distinguishable from boost mechanics. Beta is in-sample,
same caveat as the fusion-sweep alpha.

Optional by design: nothing in the default pipeline changes; scorer failure
degrades to no boost for that query. Cost: one small LLM call per query.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

from fusion_sweep import EVAL_SETS, minmax  # noqa: E402
from provenance import provenance  # noqa: E402
from run_retrieval import ndcg_at_k  # noqa: E402
from sec_rag.index.build import load as load_index  # noqa: E402
from sec_rag.nvidia import chat_json_chain, embed  # noqa: E402

CANDIDATES = 50
K = 10
BETAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
COMPASS_MODELS = (
    "deepseek-ai/deepseek-v4-flash-0731",
    "nvidia/nemotron-3-super-120b-a12b",
    "meta/llama-3.3-70b-instruct",
)

COMPASS_PROMPT = """A question will be answered from a company's 10-K filing.
Predict where the answer lives.

Available 10-K sections:
{items}

Question: {question}

Respond with JSON only:
{{"items": [<up to 2 section numbers from the list, most likely first>],
  "kind": "table" or "text"}}
("table" = the answer is a figure in a financial-statement or data table;
 "text" = the answer is in prose/narrative discussion.)"""


def norm_item(value: object) -> str:
    """LLMs answer 'Item 8' or '8.'; chunks store bare '8'. Compare bare."""
    return re.sub(r"(?i)^item\s*", "", str(value)).strip().rstrip(".")


def compass_match(chunk: dict, items: list[str], kind: str) -> float:
    hit_item = str(chunk.get("item")) in items
    hit_kind = chunk.get("kind") == kind
    return (hit_item + hit_kind) / 2


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/processed"))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap questions per split (smoke runs)")
    ap.add_argument("--out", type=Path,
                    default=Path("eval/results/topic_compass.json"))
    args = ap.parse_args()

    index = load_index(args.index)
    ids = [c["chunk_id"] for c in index.chunks]
    by_id = {c["chunk_id"]: c for c in index.chunks}

    item_titles: dict[str, str] = {}
    for c in index.chunks:
        if c.get("item") and c.get("item_title"):
            item_titles.setdefault(str(c["item"]), c["item_title"])
    items_list = "\n".join(f"  Item {i}: {t}" for i, t in
                           sorted(item_titles.items(), key=lambda kv: kv[0]))

    splits = {
        name: [json.loads(l) for l in path.read_text().splitlines() if l]
        for name, path in EVAL_SETS.items()
    }
    if args.limit:
        splits = {s: qs[:args.limit] for s, qs in splits.items()}

    results: dict = {}
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

            try:
                data, model = chat_json_chain(
                    COMPASS_MODELS,
                    [{"role": "user", "content": COMPASS_PROMPT.format(
                        items=items_list, question=q["question"])}],
                    validate=lambda d: isinstance(d.get("items"), list)
                    and d.get("kind") in ("table", "text"),
                    max_tokens=100,
                )
                pred_items = [norm_item(x) for x in data["items"][:2]]
                pred_kind = data["kind"]
                failed = False
            except Exception as exc:  # noqa: BLE001
                print(f"  compass failed on {q.get('qid')}: {exc}",
                      file=sys.stderr)
                pred_items, pred_kind, model, failed = [], "", None, True

            gold_items = {str(by_id[g].get("item")) for g in gold if g in by_id}
            gold_kinds = {by_id[g].get("kind") for g in gold if g in by_id}

            ndcgs = {}
            for beta in BETAS:
                steered = {d: s + beta * compass_match(by_id[d], pred_items,
                                                       pred_kind)
                           for d, s in nd.items()}
                order = sorted(steered, key=steered.get, reverse=True)[:K]
                ndcgs[str(beta)] = ndcg_at_k(order, gold, K)

            rows.append({
                "qid": q.get("qid"), "failed": failed, "model": model,
                "pred_items": pred_items, "pred_kind": pred_kind,
                "gold_items": sorted(gold_items - {"None"}),
                "gold_kinds": sorted(k for k in gold_kinds if k),
                "item_hit": bool(set(pred_items) & gold_items),
                "kind_hit": pred_kind in gold_kinds,
                "ndcg": ndcgs,
            })
            if (i + 1) % 20 == 0:
                print(f"  [{split}] {i + 1}/{len(questions)}")

        n = len(rows)
        curve = {b: round(sum(r["ndcg"][str(b)] for r in rows) / n, 4)
                 for b in BETAS} if rows else {}
        best_b = max(curve, key=curve.get) if curve else None
        scored = [r for r in rows if not r["failed"]]
        results[split] = {
            "n": n, "n_compass_failed": n - len(scored),
            "configured_model_chain": list(COMPASS_MODELS),
            "served_models": {
                model: sum(r.get("model") == model for r in scored)
                for model in sorted({r.get("model") for r in scored if r.get("model")})
            },
            "curve": {str(b): v for b, v in curve.items()},
            "best_beta": best_b,
            "best_ndcg": curve.get(best_b),
            "dense_baseline": curve.get(0.0),
            "compass_accuracy": {
                "item": round(sum(r["item_hit"] for r in scored)
                              / len(scored), 3) if scored else None,
                "kind": round(sum(r["kind_hit"] for r in scored)
                              / len(scored), 3) if scored else None,
            },
            "per_question": rows,
        }
        r = results[split]
        print(f"\n[{split}] n={n}  dense {r['dense_baseline']}  "
              f"best beta={best_b} -> {r['best_ndcg']}  "
              f"compass acc item={r['compass_accuracy']['item']} "
              f"kind={r['compass_accuracy']['kind']}")

    results["provenance"] = provenance(
        args.index, eval_sets=EVAL_SETS.values(),
        models={"embed": index.embed_model,
                "compass_chain": list(COMPASS_MODELS)},
        extra={"candidates": CANDIDATES, "k": K, "betas": BETAS,
               "limit": args.limit})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
