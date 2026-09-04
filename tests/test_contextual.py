"""Offline tests for optional contextual chunk headers."""

from __future__ import annotations

from sec_rag.ingest.chunk import Chunk
from sec_rag.ingest.contextual import context_coverage, contextualize, describe_chunk


def chunk(cid="c1", text="Research and development | 31,370 | 29,915") -> Chunk:
    return Chunk(doc_id="d", chunk_id=cid, ticker="AAPL", company="Apple Inc.",
                 form="10-K", report_date="2024-09-28", accession="0001",
                 item="8", item_title="Financial Statements", kind="table",
                 text=text, breadcrumb="AAPL 10-K FY2024 > Item 8",
                 source_url="https://example")


def test_context_enters_the_embedded_text_and_nothing_else_changes():
    c = chunk()
    assert c.embed_text == "AAPL 10-K FY2024 > Item 8\n\n" + c.text
    [out] = contextualize([c], describe=lambda ch: "Segment R&D, in millions.",
                          verbose=False)
    assert out.context == "Segment R&D, in millions."
    assert out.embed_text == ("AAPL 10-K FY2024 > Item 8\nSegment R&D, in millions."
                              "\n\n" + c.text)
    assert out.text == c.text and out.chunk_id == c.chunk_id
    assert c.context == "", "the input chunk was mutated"


def test_a_failed_description_keeps_the_chunk_uncontextualised():
    outs = contextualize([chunk("a"), chunk("b")],
                         describe=lambda ch: "" if ch.chunk_id == "a" else "ctx",
                         verbose=False)
    assert [o.context for o in outs] == ["", "ctx"]
    assert context_coverage(outs) == 0.5
    assert context_coverage([o.to_dict() for o in outs]) == 0.5


def test_describe_chunk_falls_through_a_failing_model_and_normalises():
    calls = []

    def chat(model, messages, **kw):
        calls.append(model)
        if model == "bad":
            raise RuntimeError("410 gone")
        assert "31,370" in messages[0]["content"]
        return "  The consolidated statement\nof operations, in millions.  "

    out = describe_chunk(chunk(), models=("bad", "good"), chat=chat)
    assert out == "The consolidated statement of operations, in millions."
    assert calls == ["bad", "good"]


def test_describe_chunk_returns_empty_when_every_model_fails():
    def chat(*a, **k):
        raise RuntimeError("down")
    assert describe_chunk(chunk(), models=("m",), chat=chat) == ""


def test_context_round_trips_through_to_dict():
    c = contextualize([chunk()], describe=lambda ch: "ctx", verbose=False)[0]
    assert c.to_dict()["context"] == "ctx"
    assert Chunk(**c.to_dict()).embed_text == c.embed_text
