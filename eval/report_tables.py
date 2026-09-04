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
    rows = [json.loads(l) for l in eval_set.read_text().splitlines() if l.strip()]
    _reject_duplicate_qids(r.get("qid") for r in rows)
    sizes = [len(r["gold_chunk_ids"]) for r in rows]
    # A question with no gold chunks is invalid input, not a question whose
    # ceiling is zero. Dividing by every row while summing only the non-empty
    # ones quietly depressed the ceiling instead of reporting the problem.
    empty = [i for i, n in enumerate(sizes, 1) if not n]
    if empty:
        raise ValueError(
            f"{len(empty)} question(s) have empty gold_chunk_ids "
            f"(lines {empty[:5]}{'...' if len(empty) > 5 else ''}); "
            "an unanswerable question cannot contribute a ceiling")
    return sum(1.0 / n for n in sizes) / len(sizes), len(sizes)


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


def _reject_duplicate_qids(qids: Iterable[Any]) -> None:
    """Every per-question structure here is keyed by qid, so a repeated qid
    silently overwrites an earlier row: two questions with scores 1.0 and 0.0
    collapsed into one with whichever came last, and the sign test counted a
    trial that never happened. Refuse the input instead."""
    seen: set[Any] = set()
    dups: list[Any] = []
    for q in qids:
        if q in seen:
            dups.append(q)
        seen.add(q)
    if dups:
        raise ValueError(f"duplicate qid(s): {sorted(set(map(str, dups)))[:5]}"
                         f"{'...' if len(set(dups)) > 5 else ''}; per-question "
                         "rows are keyed by qid and cannot be aggregated")


def paired(artifact: Path, a: str, b: str) -> dict[str, float]:
    d = json.loads(artifact.read_text())["per_question_ndcg"]
    _reject_duplicate_qids(r["qid"] for r in d[a])
    _reject_duplicate_qids(r["qid"] for r in d[b])
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
DAT_ANCHOR = "<!-- generated by eval/report_tables.py from eval/results/dat_fusion.json -->"


def _table_after(lines: list[str], anchor_idx: int) -> list[str]:
    """The contiguous markdown table following an anchor line."""
    out: list[str] = []
    for line in lines[anchor_idx + 1:]:
        if line.startswith("|"):
            out.append(re.sub(r"\s+", " ", line.strip()))
        elif line.strip():
            break
    return out


def check_dat(dat: Path, docs: Iterable[Path]) -> int:
    """The DAT failure counts are the caveat that makes that section honest,
    and the retrieval checker never read them: changing "64 scorer failures"
    to "999" in the report passed. Every doc that anchors the DAT table now
    has to reproduce `dat_summary` cell for cell, and at least one doc must
    anchor it -- a caveat that no document claims is a caveat nobody checks.
    """
    expected = [re.sub(r"\s+", " ", l.strip()) for l in dat_summary(dat).splitlines()]
    bad = 0
    anchored = 0
    for doc in docs:
        lines = doc.read_text().splitlines()
        for a in (i for i, l in enumerate(lines) if l.strip() == DAT_ANCHOR):
            anchored += 1
            got = _table_after(lines, a)
            # Bold markers are presentation; compare the cells.
            got = [g.replace("**", "") for g in got]
            if got != expected:
                bad += 1
                print(f"{doc.name}:{a + 1}: anchored DAT table disagrees with "
                      f"{dat.name}")
                for e, g in zip(expected, got + [""] * (len(expected) - len(got))):
                    if e != g:
                        print(f"    expected {e}\n    found    {g}")
    if not anchored:
        print(f"no document anchors the DAT table (expected a line reading "
              f"{DAT_ANCHOR!r})")
        bad += 1
    return bad


def check_docs(artifact: Path, docs: Iterable[Path]) -> int:
    """Fail if an anchored table disagrees with, or fails to cover, the artifact.

    The first version only compared rows it recognised, and reported success for
    everything else. That made it fail *open*: a table truncated to one column
    passed, and so did a table of entirely wrong numbers that simply lacked the
    anchor. A checker that cannot fail is decoration, so it now requires the
    anchor to be present, every configuration to appear, and every row to carry
    a full set of columns.
    """
    truth = {r["config"]: r for r in json.loads(artifact.read_text())["results"]}
    bad = 0
    for doc in docs:
        lines = doc.read_text().splitlines()
        anchors = [i for i, l in enumerate(lines) if l.strip() == ANCHOR]
        if not anchors:
            print(f"{doc.name}: no anchored table found "
                  f"(expected a line reading {ANCHOR!r})")
            bad += 1
            continue
        for a in anchors:
            seen: set[str] = set()
            for off, line in enumerate(lines[a + 1:], start=a + 2):
                if not line.startswith("|"):
                    if line.strip():
                        break
                    continue
                m = _CELL.match(line)
                if not m or m.group(1) not in truth:
                    continue
                config = m.group(1)
                seen.add(config)
                row = truth[config]
                cells = [c.strip().strip("*") for c in m.group(2).split("|")]
                cells = [c for c in cells if c != ""]
                if len(cells) != len(COLUMNS):
                    print(f"{doc.name}:{off}: {config} has {len(cells)} "
                          f"columns, expected {len(COLUMNS)}")
                    bad += 1
                    continue
                for col, cell in zip(COLUMNS, cells):
                    try:
                        got = float(cell)
                    except ValueError:
                        print(f"{doc.name}:{off}: {config} {col} = {cell!r} "
                              f"is not a number")
                        bad += 1
                        continue
                    if abs(got - row[col]) > 5e-4:
                        print(f"{doc.name}:{off}: {config} {col} "
                              f"= {got:.3f}, artifact says {row[col]:.3f}")
                        bad += 1
            missing = set(truth) - seen
            if missing:
                print(f"{doc.name}:{a + 1}: anchored table is missing "
                      f"{len(missing)} configuration(s): {sorted(missing)}")
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
        if args.dat.exists():
            bad += check_dat(args.dat, args.check)
        print("docs match the artifacts" if not bad
              else f"{bad} cell(s)/table(s) disagree with the artifacts")
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
