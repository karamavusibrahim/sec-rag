#!/usr/bin/env python
"""Build a retrieval eval set with mechanically-derived gold labels.

The usual way to build a RAG eval set is to have an LLM write a question per
chunk and call that chunk the gold answer. That is circular: the question is
generated *from* the text it is then supposed to retrieve, so it inherits the
chunk's vocabulary and BM25 scores near 1.0 on it. You end up measuring
paraphrase distance, not retrieval.

This uses a different source of truth. SEC filings ship with XBRL: every
material number in the tables is also published as a structured fact with its
concept, unit, period, and originating accession number. So we can:

  1. take a fact (e.g. us-gaap:RevenueFromContractWithCustomer, FY2024, $391.0B)
  2. find which chunks actually contain that value, formatted as it appears
     in the document
  3. write a natural question about the concept, never showing the model the
     chunk text

Gold labels come from string-matching a number we obtained independently of the
corpus text. The question is written from the *concept label*, not the chunk, so
there is no lexical leakage from passage to query.

Limitations, stated honestly:
  - Only covers numeric/table retrieval. Narrative questions (risk factors,
    MD&A commentary) still need LLM generation or hand-authoring; those go in a
    separate split.
  - A value can legitimately appear in several chunks (summary table, detail
    table, MD&A prose). We treat all matches as gold -- multi-positive qrels,
    which is more honest than picking one arbitrarily.
  - Values appearing in too many chunks (>8) are dropped as non-discriminative;
    a number like "2024" or "100" is not a retrieval target.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from sec_rag.ingest.edgar import company_facts, iter_us_gaap_facts  # noqa: E402

# Concepts that make for meaningful questions. Filings publish thousands of XBRL
# facts, most of them plumbing (share counts by class, immaterial line items).
INTERESTING = {
    "Revenues": "total revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "total revenue",
    "NetIncomeLoss": "net income",
    "OperatingIncomeLoss": "operating income",
    "GrossProfit": "gross profit",
    "ResearchAndDevelopmentExpense": "research and development expense",
    "CostOfRevenue": "cost of revenue",
    "Assets": "total assets",
    "Liabilities": "total liabilities",
    "StockholdersEquity": "total shareholders' equity",
    "CashAndCashEquivalentsAtCarryingValue": "cash and cash equivalents",
    "OperatingExpenses": "total operating expenses",
    "SellingGeneralAndAdministrativeExpense": "selling, general and administrative expense",
    "EarningsPerShareDiluted": "diluted earnings per share",
    "PaymentsToAcquirePropertyPlantAndEquipment": "capital expenditures",
}


# Anchor terms per concept: a chunk only counts as gold if the number is
# accompanied by wording that identifies *which* figure it is. Kept
# deliberately loose -- filings phrase these many ways, and the goal is to
# reject a chunk that shares a digit string by coincidence, not to demand an
# exact row label.
_ANCHORS: dict[str, tuple[str, ...]] = {
    "Revenues": ("revenue", "net sales", "total sales"),
    "RevenueFromContractWithCustomerExcludingAssessedTax":
        ("revenue", "net sales", "total sales"),
    "NetIncomeLoss": ("net income", "net loss", "net earnings"),
    "OperatingIncomeLoss": ("operating income", "income from operations",
                            "operating loss"),
    "GrossProfit": ("gross profit", "gross margin"),
    "ResearchAndDevelopmentExpense": ("research and development",),
    "CostOfRevenue": ("cost of revenue", "cost of sales", "cost of goods"),
    "Assets": ("total assets", "assets"),
    "Liabilities": ("total liabilities", "liabilities"),
    "StockholdersEquity": ("shareholders' equity", "stockholders' equity",
                           "shareholders equity", "stockholders equity"),
    "CashAndCashEquivalentsAtCarryingValue": ("cash and cash equivalents",
                                              "cash equivalents"),
    "OperatingExpenses": ("operating expenses",),
    "SellingGeneralAndAdministrativeExpense":
        ("selling, general and administrative", "selling, general",
         "general and administrative"),
    "EarningsPerShareDiluted": ("diluted", "per share"),
    "PaymentsToAcquirePropertyPlantAndEquipment":
        ("property, plant and equipment", "capital expenditure",
         "purchases of property"),
}

# How far from the matched number the anchor may appear. A financial statement
# row is a label followed by two or three period columns, so the label sits
# within a few hundred characters of its value; a whole-chunk search would
# re-admit the coincidences this is meant to exclude, because a 10-K page
# mentioning revenue somewhere also contains many unrelated numbers.
ANCHOR_WINDOW = 400


def concept_supported(text: str, concept: str, variants: Sequence[str],
                      *, window: int = ANCHOR_WINDOW) -> bool:
    """Does this chunk say what the matched number *is*?

    Without this the qrels are built on digit strings alone, and `_formats`
    emits scaled variants -- so a total revenue of $1,234,000 generates the
    string "1,234", and a sentence reading "the company employed 1,234 people"
    becomes the gold passage for a revenue question. The retrieval metrics
    computed against such a label are measuring the wrong thing and there is no
    way to tell from the aggregate that they are.

    A chunk qualifies when at least one anchor phrase for the concept occurs
    within `window` characters of an occurrence of the value.
    """
    anchors = _ANCHORS.get(concept)
    if not anchors:
        return True
    low = text.lower()
    for variant in variants:
        start = 0
        while (i := text.find(variant, start)) != -1:
            lo, hi = max(0, i - window), min(len(text), i + len(variant) + window)
            near = low[lo:hi]
            if any(a in near for a in anchors):
                return True
            start = i + 1
    return False


def _formats(value: float) -> list[str]:
    """Ways a value plausibly appears in filing text.

    Filings report in thousands or millions depending on the statement, so we
    generate the scaled variants too and let the match decide.
    """
    out: set[str] = set()
    for scale in (1, 1_000, 1_000_000):
        v = value / scale
        if abs(v) < 1:
            continue
        if abs(v - round(v)) < 1e-9:
            n = int(round(v))
            out.add(f"{n:,}")
            if abs(n) >= 1000:
                out.add(str(n))
    if abs(value) < 1000:
        out.add(f"{value:,.2f}")
    return [s for s in out if len(s.replace(",", "")) >= 3]


def build(chunks: list[dict[str, Any]], tickers: list[str], *, max_per_ticker: int = 40,
          seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        by_ticker[c["ticker"]].append(c)

    questions: list[dict[str, Any]] = []
    for ticker in tickers:
        pool = by_ticker.get(ticker, [])
        if not pool:
            continue
        facts = company_facts(ticker)
        seen: set[tuple[str, int]] = set()
        candidates: list[dict[str, Any]] = []

        for fact in iter_us_gaap_facts(facts):
            concept, val = fact["concept"], fact["value"]
            if concept not in INTERESTING or not isinstance(val, (int, float)):
                continue
            if fact["form"] != "10-K" or not fact["annual"]:
                continue
            # `fiscal_year` comes from the period end, NOT from `fy`. See
            # edgar.fact_fiscal_year: a 10-K prints three comparative years and
            # tags all of them with the filing's `fy`, so keying on `fy` labels a
            # two-year-old figure as current. Here that produced questions asking
            # "in fiscal 2024" about a value that was really FY2022's -- the gold
            # chunk still contained the number, so retrieval metrics stayed
            # valid, but every question was quietly asking the wrong thing.
            fy = int(fact["fiscal_year"] or 0)
            key = (concept, fy)
            if key in seen or not fy:
                continue

            # A 10-K shows the current year plus two comparatives, so a value is
            # only plausibly *present* in a filing whose report year is within
            # that window. Without this guard, an FY2010 figure that happens to
            # share digits with something in a 2025 filing becomes a false gold
            # label -- silently corrupting every metric computed against it.
            variants = _formats(float(val))
            # Three conditions, and the third is the one that was missing:
            # the report year has to be one the filing could contain, the
            # number has to appear, and the chunk has to say what the number
            # *is*. Digit-only matching made "employed 1,234 people" a valid
            # gold passage for a $1,234,000 revenue question.
            matches = [
                c["chunk_id"] for c in pool
                if fy and 0 <= int(c["report_date"][:4]) - fy <= 2
                and any(v in c["text"] for v in variants)
                and concept_supported(c["text"], concept, variants)
            ]
            # Too many matches = the number is not discriminative; zero = the
            # value never appears in the text we indexed (different period).
            if not (1 <= len(matches) <= 8):
                continue

            seen.add(key)
            label = INTERESTING[concept]
            candidates.append({
                "qid": f"{ticker}-{concept}-{fy}",
                "question": f"What was {ticker}'s {label} in fiscal {fy}?",
                "ticker": ticker,
                "concept": concept,
                "fiscal_year": fy,
                "expected_value": val,
                "unit": fact["unit"],
                "gold_chunk_ids": matches,
                "split": "numeric",
                "source": "xbrl",
            })

        rng.shuffle(candidates)
        questions.extend(candidates[:max_per_ticker])

    return questions


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("data/eval/eval_set_v2.jsonl"))
    ap.add_argument("--max-per-ticker", type=int, default=40)
    args = ap.parse_args()

    chunks = [
        json.loads(l)
        for l in (args.index / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if l
    ]
    tickers = sorted({c["ticker"] for c in chunks})
    print(f"corpus: {len(chunks)} chunks, tickers={tickers}")

    qs = build(chunks, tickers, max_per_ticker=args.max_per_ticker)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in qs), encoding="utf-8"
    )

    n_gold = sum(len(q["gold_chunk_ids"]) for q in qs)
    print(f"wrote {len(qs)} questions -> {args.out}")
    print(f"  gold labels: {n_gold} total, {n_gold / max(len(qs), 1):.1f} avg per question")
    for q in qs[:3]:
        print(f"  e.g. {q['question']}  (gold: {len(q['gold_chunk_ids'])} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
