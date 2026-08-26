"""Regression tests for narrative gold-label expansion.

Narrative questions are LLM-written from one chunk, then propagated to
lexically near-identical chunks from the same company. Consecutive 10-Ks share
boilerplate, risk-factor phrasing and segment descriptions, so an *earlier*
filing clears any Jaccard threshold a later one does -- and an earlier filing
cannot answer a question about something that happened after it was published.

The concrete case: "How much did Nvidia lose in early 2026 because of new
export rules on its H20 chips?" carried a gold label pointing at the
2025-01-26 filing. A retriever that correctly ranked that chunk low was
penalised for being right.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from build_narrative_eval import near_duplicates  # noqa: E402

BODY = "export controls on H20 inventory charge gross margin impact " * 20


def chunk(cid: str, date: str, text: str = BODY) -> dict:
    return {"chunk_id": cid, "ticker": "NVDA", "report_date": date, "text": text}


def test_an_earlier_filing_is_never_a_gold_neighbour():
    src = chunk("NVDA_10K_2026-01-25#0038", "2026-01-25")
    pool = [chunk("NVDA_10K_2025-01-26#0094", "2025-01-26")]
    assert near_duplicates(src, pool) == []


def test_the_same_filing_still_expands():
    src = chunk("NVDA_10K_2026-01-25#0038", "2026-01-25")
    pool = [chunk("NVDA_10K_2026-01-25#0092", "2026-01-25")]
    assert near_duplicates(src, pool) == ["NVDA_10K_2026-01-25#0092"]


def test_a_later_filing_still_expands():
    # A later filing repeats the discussion and can answer the question.
    src = chunk("NVDA_10K_2026-01-25#0038", "2026-01-25")
    pool = [chunk("NVDA_10K_2027-01-24#0101", "2027-01-24")]
    assert near_duplicates(src, pool) == ["NVDA_10K_2027-01-24#0101"]


def test_a_different_company_is_still_excluded():
    src = chunk("NVDA_10K_2026-01-25#0038", "2026-01-25")
    other = chunk("AAPL_10K_2026-09-26#0011", "2026-09-26")
    other["ticker"] = "AAPL"
    assert near_duplicates(src, other and [other]) == []


def test_the_committed_eval_set_still_carries_the_old_labels():
    """Documents the known contamination rather than asserting it is gone.

    The fix changes how the set is *built*; the committed set predates it and
    rebuilding needs the corpus, which is not committed. This test fails if
    that count ever changes, so the caveat in README/REPORT cannot silently go
    stale in either direction.
    """
    path = ROOT / "data/eval/eval_narrative.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    affected = [
        r["qid"] for r in rows
        if any(g.split("#")[0].split("_")[-1] < r["qid"].split("#")[0].split("_")[-1]
               for g in r["gold_chunk_ids"])
    ]
    assert len(rows) == 15
    assert len(affected) == 3, affected
