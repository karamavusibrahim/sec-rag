#!/usr/bin/env python
"""Regenerate every published retrieval table and statistic from the artifacts.

Written because the report and the README drifted apart from
`eval/results/retrieval_v3.json` and from each other: three S@1 cells in
REPORT.md were higher than the artifact they were supposedly read off, and the
R@1 ceiling was quoted as ~0.43 when the eval set's own gold sizes put it at
0.477. Numbers that are transcribed by hand drift; numbers that are printed by
a script do not. Anything quoted in the docs should come out of here.

Nothing in this file makes a network call. It reads committed JSON and JSONL
only, so it can be run in CI and diffed against the docs.

    uv run python eval/report_tables.py                # tables + statistics
    uv run python eval/report_tables.py --check REPORT.md README.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ("R@1", "S@1", "R@5", "R@10", "nDCG@10", "MRR@10")


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

def retrieval_table(artifact: Path) -> str:
    d = json.loads(artifact.read_text())
    rows = ["| Configuration | " + " | ".join(COLUMNS) + " |",
            "|---" * (len(COLUMNS) + 1) + "|"]
    for r in d["results"]:
        cells = " | ".join(f"{r[c]:.3f}" for c in COLUMNS)
        rows.append(f"| {r['config']} | {cells} |")
    return "\n".join(rows)


def r_at_1_ceiling(eval_set: Path) -> tuple[float, int]:
    """R@1 divides by |gold|, so a perfect top-1 still scores 1/|gold|.

    The attainable mean is therefore the mean of 1/|gold| over questions --
    worth printing, because a reader who does not know the labels are
    multi-gold will read R@1 as a hit rate and conclude the system is far worse
    than it is.
    """
    sizes = [len(json.loads(l)["gold_chunk_ids"])
             for l in eval_set.read_text().splitlines() if l.strip()]
    return sum(1.0 / n for n in sizes if n) / len(sizes), len(sizes)


# --------------------------------------------------------------------------
# significance
# --------------------------------------------------------------------------

def sign_test(diffs: Sequence[float], *, eps: float = 1e-9) -> dict[str, Any]:
    wins = sum(1 for d in diffs if d > eps)
    losses = sum(1 for d in diffs if d < -eps)
    ties = len(diffs) - wins - losses
    n = wins + losses
    if n == 0:
        return {"wins": wins, "losses": losses, "ties": ties, "p": 1.0}
    tail = sum(comb(n, k) for k in range(min(wins, losses) + 1))
    return {"wins": wins, "losses": losses, "ties": ties,
            "p": min(1.0, 2 * tail / 2 ** n)}


def _qid_parts(qid: str) -> tuple[str, str, int]:
    ticker, rest = qid.split("-", 1)
    concept, fy = rest.rsplit("-", 1)
    return ticker, concept, int(fy)


def paired(artifact: Path, a: str, b: str) -> dict[str, float]:
    d = json.loads(artifact.read_text())["per_question_ndcg"]
    left = {r["qid"]: r["nDCG@10"] for r in d[a]}
    right = {r["qid"]: r["nDCG@10"] for r in d[b]}
    return {q: left[q] - right[q] for q in left if q in right}


def clustered(diffs: dict[str, float]) -> dict[str, dict[str, Any]]:
    """The same comparison at every unit of aggregation that is defensible.

    120 questions over 3 companies and 15 concepts are not 120 independent
    observations: the eval set is a template crossed over (ticker, concept,
    year), so questions about the same company or the same line item share
    almost everything. Pairing the two systems per question removes
    between-question variance but does nothing about that dependence, and a
    sign test over the questions therefore counts correlated observations as if
    they were independent trials.

    None of the aggregations below is uniquely correct. Printing all of them is
    the point: if the conclusion only survives at the question level, it is a
    property of the question template, not of the systems.
    """
    out = {"question": sign_test(list(diffs.values()))}
    for name, key in (("concept", lambda q: _qid_parts(q)[1]),
                      ("ticker", lambda q: _qid_parts(q)[0]),
                      ("ticker x concept", lambda q: _qid_parts(q)[:2])):
        groups: dict[Any, list[float]] = defaultdict(list)
        for q, v in diffs.items():
            groups[key(q)].append(v)
        means = [sum(v) / len(v) for v in groups.values()]
        out[name] = {"k": len(means), **sign_test(means)}
    out["question"]["k"] = len(diffs)
    return out


# --------------------------------------------------------------------------
# DAT
# --------------------------------------------------------------------------

def dat_summary(artifact: Path) -> str:
    d = json.loads(artifact.read_text())
    lines = ["| split | attempted | scorer failed | actually scored | "
             "mean alpha (all rows) | mean alpha (scored only) |",
             "|---|---|---|---|---|---|"]
    for split in ("numeric", "narrative"):
        s = d[split]
        ok = s.get("successful_only", {})
        lines.append(
            f"| {split} | {s['n']} | {s['n_scorer_failed']} | "
            f"{ok.get('n', '?')} | {s['mean_alpha']:.3f} | "
            f"{ok.get('mean_alpha', float('nan')):.3f} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# doc check
# --------------------------------------------------------------------------

_CELL = re.compile(r"^\|\s*([A-Za-z0-9 +()]+?)\s*\|(.+)\|\s*$")
# Only tables the docs explicitly claim to have taken from this artifact are
# checked. Matching on the config name alone is not enough: the narrative split
# uses the same six configuration names over a different eval set (n=15), and a
# name-only check flags every one of its rows as a contradiction. An anchor
# makes the claim explicit -- a table that says where it came from can be
# verified, and one that does not is not silently assumed to be this run.
ANCHOR = "<!-- generated by eval/report_tables.py from eval/results/retrieval_v3.json -->"


def check_docs(artifact: Path, docs: Iterable[Path]) -> int:
    """Fail if an anchored table disagrees with the artifact it cites."""
    truth = {r["config"]: r for r in json.loads(artifact.read_text())["results"]}
    bad = 0
    for doc in docs:
        lines = doc.read_text().splitlines()
        anchored = False
        for lineno, line in enumerate(lines, 1):
            if line.strip() == ANCHOR:
                anchored = True
                continue
            if anchored and not line.startswith("|"):
                if line.strip():
                    anchored = False
                continue
            m = _CELL.match(line)
            if not anchored or not m or m.group(1) not in truth:
                continue
            row = truth[m.group(1)]
            cells = [c.strip().strip("*") for c in m.group(2).split("|")]
            if len(cells) < len(COLUMNS):
                continue
            for col, cell in zip(COLUMNS, cells):
                try:
                    got = float(cell)
                except ValueError:
                    continue
                if abs(got - row[col]) > 5e-4:
                    print(f"{doc.name}:{lineno}: {m.group(1)} {col} "
                          f"= {got:.3f}, artifact says {row[col]:.3f}")
                    bad += 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path,
                    default=ROOT / "eval/results/retrieval_v3.json")
    ap.add_argument("--eval-set", type=Path,
                    default=ROOT / "data/eval/eval_set_v2.jsonl")
    ap.add_argument("--dat", type=Path,
                    default=ROOT / "eval/results/dat_fusion.json")
    ap.add_argument("--check", nargs="*", type=Path)
    args = ap.parse_args()

    if args.check:
        bad = check_docs(args.artifact, args.check)
        print("docs match the artifact" if not bad
              else f"{bad} cell(s) disagree with the artifact")
        return 1 if bad else 0

    print("## Retrieval (numeric split, n=120)\n")
    print(retrieval_table(args.artifact))

    if args.eval_set.exists():
        ceiling, n = r_at_1_ceiling(args.eval_set)
        print(f"\nR@1 attainable ceiling (mean of 1/|gold| over {n} questions): "
              f"{ceiling:.3f}")

    print("\n## Dense only vs Hybrid (RRF), by unit of aggregation\n")
    diffs = paired(args.artifact, "Dense only", "Hybrid (RRF)")
    print("| unit | n | wins | losses | ties | sign-test p |")
    print("|---|---|---|---|---|---|")
    for unit, r in clustered(diffs).items():
        print(f"| {unit} | {r['k']} | {r['wins']} | {r['losses']} | "
              f"{r['ties']} | {r['p']:.4f} |")

    if args.dat.exists():
        print("\n## DAT per-query fusion: what was actually scored\n")
        print(dat_summary(args.dat))
    return 0


if __name__ == "__main__":
    sys.exit(main())
