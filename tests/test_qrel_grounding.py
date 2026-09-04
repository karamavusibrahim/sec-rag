"""Regression tests for the numeric ground-truth support diagnostic.

`concept_supported` is a DIAGNOSTIC, not a filter. It is recorded per question
as `gold_concept_supported` / `n_gold_unsupported` and never removes a label.
Filtering on it would require every gold chunk to contain the vocabulary the
question was written from, handing BM25 a lexical path from passage to query in
an ablation whose whole subject is sparse versus dense retrieval. These tests
pin the signal's behaviour; they do not assert that labels are filtered.

Original context:

The numeric eval set is built by finding chunks that contain the XBRL value.
`_formats` deliberately emits scaled variants, because filings report in
thousands or millions -- which means a total revenue of $1,234,000 generates
the search string "1,234". Matching on that alone made any chunk containing
"1,234" a gold passage for a revenue question, including one that was counting
employees. Retrieval metrics computed against such a label are measuring the
wrong thing, and nothing in the aggregate reveals it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from build_eval_set import _formats, concept_supported  # noqa: E402

REVENUE = "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_scaled_variants_are_still_generated():
    # The behaviour that makes the collision possible is intentional and stays.
    assert "1,234" in _formats(1_234_000.0)
    assert "1,234,000" in _formats(1_234_000.0)


def test_employee_count_is_not_gold_for_a_revenue_question():
    text = "As of the end of fiscal 2024 the company employed 1,234 people."
    assert not concept_supported(text, REVENUE, _formats(1_234_000.0))


def test_a_real_revenue_row_is_gold():
    text = "Total net sales 1,234 1,100 987 (in thousands)"
    assert concept_supported(text, REVENUE, _formats(1_234_000.0))


def test_a_nearby_mention_is_not_proof_the_number_is_the_figure():
    """The diagnostic is a weak signal and is documented as one.

    Proximity does not establish that the number *is* the concept. This case
    is why the signal is recorded rather than acted on: it reports support,
    and support is not correctness.
    """
    assert concept_supported("Revenue declined. The company employed 1,234 people.",
                             REVENUE, _formats(1_234_000.0))


def test_the_anchor_must_be_near_the_number_not_merely_present():
    # A 10-K page that mentions revenue somewhere also contains many unrelated
    # numbers; a whole-chunk search would re-admit exactly the coincidences
    # this check exists to reject.
    far = ("Revenue for the period is discussed below. " + "filler word " * 120
           + "the company employed 1,234 people.")
    assert not concept_supported(far, REVENUE, _formats(1_234_000.0))


def test_unknown_concepts_are_not_silently_dropped():
    # No anchor list means no opinion -- fall through rather than reject, so
    # adding a concept to INTERESTING never silently empties its qrels.
    assert concept_supported("anything 1,234", "SomeNewConcept", ["1,234"])


def test_net_income_and_research_anchors():
    assert concept_supported("Net income 5,678", "NetIncomeLoss", ["5,678"])
    assert not concept_supported("Cash paid for rent 5,678", "NetIncomeLoss",
                                 ["5,678"])
    assert concept_supported("Research and development 3,451",
                             "ResearchAndDevelopmentExpense", ["3,451"])


def test_empty_eval_set_is_an_error_not_a_zero_score():
    """An empty question list used to produce all-zero metrics and exit 0.

    That is the worst possible failure mode for an eval harness: a mis-typed
    --split or a renamed file writes a plausible artifact recording a total
    collapse that never happened, and the number then gets published.
    """
    import pytest
    from run_retrieval import evaluate

    with pytest.raises(ValueError, match="no questions"):
        evaluate(None, [], {})


# ---------------------------------------------------------------------------
# Driven through `build()`, not its helpers. The earlier tests in this file all
# called `concept_supported` directly, so deleting the diagnostic or restoring
# the rejected vocabulary filter would have left them green.
# ---------------------------------------------------------------------------

from unittest.mock import patch  # noqa: E402

from build_eval_set import build, value_occurs  # noqa: E402


def chunk(cid, text, ticker="AAPL", date="2024-09-28"):
    return {"chunk_id": cid, "text": text, "ticker": ticker,
            "report_date": date}


def fake_facts(concept, value, fy=2024):
    return [{"concept": concept, "value": value, "unit": "USD",
             "form": "10-K", "annual": True, "fiscal_year": fy}]


def run_build(chunks, concept="Revenues", value=1_234_000.0):
    with patch("build_eval_set.company_facts", return_value={}), \
         patch("build_eval_set.iter_us_gaap_facts",
               return_value=fake_facts(concept, value)):
        return build(chunks, ["AAPL"], max_per_ticker=10)


class TestValueBoundaries:
    """A value must occur as a standalone number, not inside a longer one."""

    def test_a_value_embedded_in_a_longer_number_is_not_a_match(self):
        assert not value_occurs("Total 1234", ["123"])
        assert not value_occurs("was 112,914", ["12,914"])
        assert not value_occurs("rate 16.16", ["6.16"])

    def test_a_standalone_value_matches(self):
        assert value_occurs("Total 123 units", ["123"])
        assert value_occurs("was 12,914 total", ["12,914"])

    def test_sentence_punctuation_is_not_numeric_continuation(self):
        # "were 123." and "123, compared" are prose, not longer numbers. The
        # first boundary guard treated every adjacent period/comma as numeric
        # continuation and silently dropped every figure that closed a
        # sentence -- which in filing prose is most of them.
        assert value_occurs("Total assets were 123.", ["123"])
        assert value_occurs("Total assets were 123, compared with prior", ["123"])
        assert value_occurs("was 12,914.", ["12,914"])
        # ...while a digit beyond the separator still means embedded:
        assert not value_occurs("grew 123,456 units", ["123"])
        assert not value_occurs("x 0.123 y", ["123"])

    def test_a_leading_attached_dot_is_a_decimal_not_punctuation(self):
        # ".123" is a decimal with its zero omitted; nothing ends a sentence
        # flush against the next number. The trailing side stays punctuation-
        # tolerant ("were 123.") -- direction matters.
        assert not value_occurs(".123", ["123"])
        assert not value_occurs("x .123 y", ["123"])
        assert value_occurs("vs. 123 units", ["123"])

    def test_a_dot_leader_is_not_a_decimal(self):
        # "Total assets .....123" is a formatted table row; only a SINGLE
        # leading dot reads as a decimal point.
        assert value_occurs("Total assets .....123", ["123"])

    def test_a_negated_printing_is_not_gold_for_a_positive_fact(self):
        # Filings write losses as -123 or (123). Same digits, different number.
        from build_eval_set import _formats
        assert not value_occurs("Net loss was -123.", _formats(123.0))
        assert not value_occurs("Net loss was (123).", _formats(123.0))

    def test_parentheses_are_style_not_sign_for_cash_outflows(self):
        """Capex is a POSITIVE fact the cash flow statement prints as (9,447).

        The first version of the sign rule assumed parentheses always negate.
        For Payments* concepts they are presentation, and the universal rule
        deleted every capital-expenditure question from the eval set. The real
        AAPL FY2024 printing:
        """
        from build_eval_set import _formats
        v = _formats(9_447_000_000.0)
        assert not value_occurs("Payments for acquisition of PP&E (9,447)", v)
        assert value_occurs("Payments for acquisition of PP&E (9,447)", v,
                            paren_is_sign=False)
        # ...while the minus sign stays a sign for everyone:
        assert not value_occurs("adjustment of -9,447", v, paren_is_sign=False)
        # ...while parens with intervening text are not a sign:
        assert value_occurs("(see note 123)", ["123"])
        # ...and negative facts still match their own parenthesised printing:
        assert value_occurs("Net loss (1,234)", _formats(-1_234_000.0))

    def test_a_value_at_either_end_of_the_chunk_matches(self):
        # "" in ",." is True in Python, so an emptiness check is required or
        # every value touching a chunk boundary is silently dropped.
        assert value_occurs("12,914", ["12,914"])
        assert value_occurs("12,914 and more", ["12,914"])
        assert value_occurs("ending in 12,914", ["12,914"])

    def test_a_parenthesised_loss_is_found(self):
        from build_eval_set import _formats
        assert value_occurs("Net loss (1,234)", _formats(-1_234_000.0))

    def test_a_sign_separated_by_currency_or_space_still_counts(self):
        """Filings print "($123)", "$ (1,234)" and "- 123"; PDFs emit U+2212.

        The adjacent-character sign check read all of those as unsigned, so a
        positive fact matched a negated printing and a negative fact could
        never match its own "($1,234)" row.
        """
        from build_eval_set import _formats
        pos, neg = _formats(123.0), _formats(-1_234_000.0)
        assert not value_occurs("Net loss was ($123).", pos)
        assert not value_occurs("Net loss was −123.", pos)
        assert not value_occurs("Net loss was - 123.", pos)
        assert value_occurs("Net loss was ($1,234).", neg)
        assert value_occurs("Net loss was $ (1,234).", neg)
        assert value_occurs("Net loss was − 1,234.", neg)
        # A closing paren far from the digits is prose, not a sign.
        assert value_occurs("(revenue of 123 in the period)", pos)


class TestBuildEndToEnd:
    def test_an_embedded_digit_string_does_not_become_gold(self):
        # 1,234,000 searches for "1,234" and "1234" among others. Inside
        # "12345" neither occurs as a standalone number, so no label is made.
        qs = run_build([chunk("X", "Headcount rose to 12345 people.")])
        assert qs == [], "a value inside a longer number became gold"

    def test_an_exact_standalone_collision_still_becomes_gold(self):
        """Documents what boundary matching does NOT fix.

        A revenue of $1,234,000 searches for "1,234" because filings report in
        thousands, and "the company employed 1,234 people" contains exactly
        that number standing alone. No amount of string matching separates
        those two facts -- only a label carrying the concept can, which is what
        inline-XBRL element-to-DOM mapping would provide. The diagnostic below
        is the interim measure, not a fix.
        """
        qs = run_build([chunk("X", "The company employed 1,234 people.")])
        assert len(qs) == 1 and qs[0]["gold_chunk_ids"] == ["X"]
        assert qs[0]["n_gold_unsupported"] == 1

    def test_a_real_revenue_row_becomes_gold(self):
        qs = run_build([chunk("X", "Total net sales 1,234 (in thousands)")])
        assert len(qs) == 1 and qs[0]["gold_chunk_ids"] == ["X"]

    def test_the_support_diagnostic_is_recorded_and_not_enforced(self):
        """Both fields must exist, and an unsupported label must survive.

        Filtering on concept vocabulary was tried and withdrawn: questions are
        written from the concept label, so requiring that vocabulary in the
        gold chunk hands BM25 a lexical path from passage to query. The signal
        is recorded so the contamination stays measurable.
        """
        qs = run_build([chunk("X", "The company employed 1,234 people.")])
        assert len(qs) == 1, "the label was filtered out rather than flagged"
        q = qs[0]
        assert q["gold_chunk_ids"] == ["X"]
        assert q["gold_concept_supported"] == []
        assert q["n_gold_unsupported"] == 1

    def test_presentation_parentheses_are_gold_for_expense_concepts(self):
        """Driven through build(), so reverting the production wiring fails.

        The seventh-pass test called `value_occurs(..., paren_is_sign=False)`
        directly and stayed green with the `build()` hookup reverted. These
        rows are the real printings: AAPL's cash flow statement for capex and
        its segment reconciliation for R&D, both positive XBRL facts.
        """
        capex = run_build([chunk("X", "Payments for acquisition of PP&E (9,447)")],
                          concept="PaymentsToAcquirePropertyPlantAndEquipment",
                          value=9_447_000_000.0)
        assert [q["gold_chunk_ids"] for q in capex] == [["X"]]
        rnd = run_build([chunk("X", "Research and development | (34,550) |")],
                        concept="ResearchAndDevelopmentExpense",
                        value=34_550_000_000.0)
        assert [q["gold_chunk_ids"] for q in rnd] == [["X"]]

    def test_parentheses_stay_a_sign_for_income_concepts(self):
        # The same printing under a concept that can be negative is a loss,
        # and a positive net-income fact must not claim it.
        qs = run_build([chunk("X", "Net income (loss) | (34,550) |")],
                       concept="NetIncomeLoss", value=34_550_000_000.0)
        assert qs == []


def test_empty_gold_is_an_error_in_every_metric():
    """recall/nDCG/MRR returned 0.0 for a question with no gold chunks, so a
    malformed row lowered every configuration's mean by the same amount and
    nothing reported it. `r_at_1_ceiling` already refuses such rows; the
    metrics now agree with it."""
    import pytest
    from run_retrieval import mrr_at_k, ndcg_at_k, recall_at_k

    for fn in (recall_at_k, ndcg_at_k, mrr_at_k):
        with pytest.raises(ValueError, match="empty gold"):
            fn(["a", "b"], set(), 10)


def test_duplicate_qids_are_refused(tmp_path):
    """Per-question rows are keyed by qid; a repeat silently overwrote the
    earlier row and the sign test counted a trial that never happened."""
    import json

    import pytest
    from report_tables import paired, r_at_1_ceiling

    art = tmp_path / "a.json"
    art.write_text(json.dumps({"per_question_ndcg": {
        "A": [{"qid": "dup", "nDCG@10": 1.0}, {"qid": "dup", "nDCG@10": 0.0}],
        "B": [{"qid": "dup", "nDCG@10": 0.5}]}}))
    with pytest.raises(ValueError, match="duplicate qid"):
        paired(art, "A", "B")
    es = tmp_path / "e.jsonl"
    es.write_text("\n".join(json.dumps({"qid": "q", "gold_chunk_ids": ["c"]})
                            for _ in range(2)))
    with pytest.raises(ValueError, match="duplicate qid"):
        r_at_1_ceiling(es)


def test_gated_rerank_fails_closed_on_total_failure(tmp_path, monkeypatch):
    """Every rerank call failing used to write survivor-only zeros and exit 0."""
    import json
    import sys
    import types

    import numpy as np
    import pytest

    import gated_rerank as gr

    chunks = [{"chunk_id": "c1", "breadcrumb": "b", "text": "t", "kind": "table"}]
    index = types.SimpleNamespace(chunks=chunks, embed_model="m",
                                  vectors=np.ones((1, 2), dtype=np.float32))
    monkeypatch.setattr(gr, "load_index", lambda p: index)
    monkeypatch.setattr(gr, "embed", lambda texts, **kw: [[1.0, 0.0]] * len(texts))

    def boom(*a, **kw):
        raise RuntimeError("boom")
    monkeypatch.setattr(gr, "rerank", boom)
    q = {"qid": "q1", "question": "?", "gold_chunk_ids": ["c1"]}
    monkeypatch.setattr(gr, "EVAL_SETS", {"numeric": tmp_path / "n.jsonl",
                                          "narrative": tmp_path / "r.jsonl"})
    for p in gr.EVAL_SETS.values():
        p.write_text(json.dumps(q))
    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["gated_rerank", "--out", str(out)])
    with pytest.raises(SystemExit, match="every one of 1 rerank calls failed"):
        gr.main()
    assert not out.exists(), "an artifact was written for a total failure"
