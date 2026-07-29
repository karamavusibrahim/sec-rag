"""Regression tests for the eval set's ground truth.

Every bug these cover was found in production of the reports, and each one made
the *measurement* wrong while leaving the pipeline untouched. That is the failure
mode worth guarding: a broken retriever shows up as a bad score, but a broken
label set shows up as a good score for the wrong reason -- or, as happened here,
as a correct answer marked wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from sec_rag.ingest.edgar import fact_fiscal_year, is_annual  # noqa: E402

import build_narrative_eval as B  # noqa: E402


class TestFactFiscalYear:
    """`fy` is the filing's fiscal year, not the fact's.

    AAPL's FY2024 10-K reports three comparative years and tags all of them
    fy=2024. Keying on `fy` selected FY2022's R&D and labelled it FY2024, which
    made the agentic-rag eval score a correct 8.02% answer as wrong.
    """

    # The exact rows from AAPL's companyfacts that caused it.
    AAPL_RD = [
        ({"fy": 2024, "start": "2021-09-26", "end": "2022-09-24"}, 2022),
        ({"fy": 2024, "start": "2022-09-25", "end": "2023-09-30"}, 2023),
        ({"fy": 2024, "start": "2023-10-01", "end": "2024-09-28"}, 2024),
    ]

    @pytest.mark.parametrize("entry,expected", AAPL_RD)
    def test_year_comes_from_period_end_not_fy(self, entry, expected):
        assert fact_fiscal_year(entry) == expected

    def test_all_three_share_one_fy(self):
        """The premise of the bug: `fy` cannot distinguish these rows."""
        assert {e["fy"] for e, _ in self.AAPL_RD} == {2024}
        assert len({fact_fiscal_year(e) for e, _ in self.AAPL_RD}) == 3

    def test_fiscal_year_ending_in_january_labels_forward(self):
        """NVDA's FY2026 ended 2026-01-25 and is labelled FY2026."""
        assert fact_fiscal_year({"end": "2026-01-25"}) == 2026

    def test_missing_or_malformed_end(self):
        assert fact_fiscal_year({}) is None
        assert fact_fiscal_year({"end": ""}) is None
        assert fact_fiscal_year({"end": "n/a"}) is None


class TestIsAnnual:
    """A quarterly figure standing in for an annual one is the error the whole
    eval exists to catch; it must never enter the ground truth."""

    def test_full_year_duration(self):
        assert is_annual({"start": "2023-10-01", "end": "2024-09-28"})

    def test_quarter_is_rejected(self):
        assert not is_annual({"start": "2024-06-30", "end": "2024-09-28"})

    def test_instant_fact_is_annual(self):
        """Balance-sheet items have no start; from a 10-K they are year-end."""
        assert is_annual({"end": "2024-09-28"})

    def test_malformed_dates_are_rejected(self):
        assert not is_annual({"start": "oops", "end": "2024-09-28"})


class TestDecontamination:
    IDF = {"macroeconomic": 4.0, "headwinds": 4.0, "gross": 2.0, "margin": 2.0,
           "weak": 3.0, "economy": 3.0, "apple": 1.0}

    def test_copied_phrase_is_rejected(self):
        gold = ("Adverse macroeconomic headwinds could compress gross margin "
                "across the Company's product lines.")
        q = "How could adverse macroeconomic headwinds could compress gross margin?"
        assert B.contaminated(q, gold, self.IDF) == "shared 4-gram"

    def test_paraphrase_passes(self):
        gold = ("Adverse macroeconomic headwinds could compress gross margin "
                "across the Company's product lines.")
        q = "What happens to Apple's profit per unit when the economy weakens?"
        assert B.contaminated(q, gold, self.IDF) is None

    def test_threshold_is_calibrated_to_the_numeric_split(self):
        """Not an arbitrary constant.

        The first value, 0.34, rejected narrative questions *cleaner* than the
        XBRL questions they would be compared against -- handicapping BM25 on
        the split built to test whether BM25 helps. The cap belongs at the
        numeric split's median overlap, which was measured, not guessed.
        """
        assert B.MAX_IDF_OVERLAP == pytest.approx(0.725)
        assert B.MAX_IDF_OVERLAP > B.NUMERIC_SPLIT_MEAN_OVERLAP

    def test_stopwords_do_not_trigger_the_ngram_filter(self):
        """Function words are stripped before n-gramming.

        Otherwise "of the ... in the" style runs would make almost every
        question look like a copied phrase, and the filter would reject
        everything for a reason unrelated to leakage.
        """
        q = ("What supplier concentration exposure might disrupt manufacturing "
             "output and delivery schedules?")
        gold = ("It is the case that in the event of a downturn the value of "
                "the assets held by the entity may decline substantially.")
        assert B.contaminated(q, gold, {}) is None

    def test_a_question_with_too_few_content_words_is_rejected(self):
        """Length is checked on content words, not raw tokens: a question that
        is mostly stopwords carries too little signal to score retrieval."""
        assert B.contaminated("What is the risk to the value of the assets?",
                              "unrelated passage text", {}) == "too short"

    def test_idf_overlap_bounds(self):
        assert B.idf_overlap("weak economy", "a weak economy hurts", self.IDF) == 1.0
        assert B.idf_overlap("weak economy", "unrelated text", self.IDF) == 0.0


class TestDeadline:
    def test_generate_gives_up_rather_than_hanging(self, monkeypatch):
        """A stalled SSE stream never trips httpx's per-chunk read timeout, so
        the deadline has to live above the client. The obvious implementation
        (ThreadPoolExecutor as a context manager) does not work: shutdown waits
        for the worker, so the timeout fires and then blocks anyway."""
        import time
        monkeypatch.setattr(B, "_generate_inner", lambda chunk: time.sleep(30))
        chunk = {"chunk_id": "X", "report_date": "2025-01-01", "ticker": "T",
                 "text": "x", "item": "1A"}
        t0 = time.time()
        assert B.generate(chunk, deadline=1.0) is None
        assert time.time() - t0 < 5.0, "deadline did not actually abandon the call"
