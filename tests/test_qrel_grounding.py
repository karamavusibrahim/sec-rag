"""Regression tests for numeric ground-truth construction.

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
