#!/usr/bin/env python
"""Build the narrative half of the eval set, with lexical leakage removed.

The numeric split (`build_eval_set.py`) gets its gold labels from XBRL, so the
question never touches the passage's vocabulary. Narrative retrieval has no
equivalent structured channel: risk factors and MD&A commentary are prose, and
the only practical way to get questions is to generate them from the text.

That is exactly the setup that produces a dishonest eval. An LLM asked to write a
question about a passage reuses the passage's distinctive phrasing, BM25 then
matches on those exact terms, and the eval reports a sparse-retrieval win that is
really a measurement of copy-paste. Since the whole point of this split is to
test *whether BM25 earns its keep on prose*, letting that happen would make the
result worthless in the precise direction it is meant to inform.

So generation is not trusted. It is filtered:

  1. **Generate** a question from the chunk, prompted for investor vocabulary.
  2. **Decontaminate** -- reject any question sharing a content n-gram of length
     >= `NGRAM_N` with its own gold chunk. This is mechanical; the prompt is
     merely a first pass, and the filter is what actually enforces the property.
  3. **Cap rare-term overlap** -- reject questions whose IDF-weighted unigram
     overlap with the gold chunk exceeds `MAX_IDF_OVERLAP`. n-grams catch copied
     phrases; this catches copied jargon scattered across a reworded sentence.
  4. **Expand gold over near-duplicates** -- the corpus holds two consecutive
     fiscal years per ticker, and risk factors are largely rewritten year over
     year rather than rewritten from scratch. A question about a risk in FY2025
     is legitimately answered by the FY2024 chunk covering the same risk, so
     marking it wrong would penalise correct retrieval. Near-duplicates are
     found by deterministic token Jaccard, not by embedding similarity, which
     would smuggle the dense retriever into its own ground truth.

What survives is still weaker evidence than the XBRL split, and the README says
so. The retained leakage is reported as a number (`idf_overlap` per question) so
the reader can discount rather than guess.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from sec_rag.nvidia import chat_json_chain  # noqa: E402

WRITER_MODELS = (
    "deepseek-ai/deepseek-v4-flash-0731",
    "nvidia/nemotron-3-super-120b-a12b",
    "openai/gpt-oss-120b",
)

# Items whose content is genuinely narrative. Item 8 is financial statements and
# Item 15 is exhibits -- both are the numeric split's territory.
NARRATIVE_ITEMS = {"1", "1A", "1C", "3", "7", "7A"}

NGRAM_N = 4  # a shared 4-gram of content words is a copied phrase

# Calibrated against the numeric split, not chosen by intuition.
#
# The first version of this file used 0.34, on the reasoning that a low overlap
# means a well-paraphrased question. Measuring the XBRL split -- which is clean
# *by construction*, since its questions are written from concept labels and
# never see the passage -- showed mean overlap 0.683 and median 0.725. Asking
# "what was NVDA's total revenue in fiscal 2025" unavoidably reuses NVDA,
# revenue, fiscal and 2025, all of which are in the chunk.
#
# So lexical overlap is not contamination; it is what asking a specific question
# looks like. A 0.34 cap was rejecting narrative questions *cleaner* than the
# numeric ones they would be compared against -- handicapping BM25 on precisely
# the split built to test whether BM25 helps.
#
# The threshold's real job is narrower: ensure narrative questions are no more
# contaminated than the numeric baseline. So it sits at the numeric median.
# Verbatim phrase copying, which the numeric split cannot have at all, is
# handled separately and absolutely by the n-gram filter.
MAX_IDF_OVERLAP = 0.725
NUMERIC_SPLIT_MEAN_OVERLAP = 0.683  # measured, for reference in the README

MIN_CHARS = 900
NEAR_DUP_JACCARD = 0.55

# Bounded: a burst large enough to trip a 429 costs more in retries than the
# parallelism saves. OVERSAMPLE covers the questions the filter will reject.
GEN_WORKERS = 4
OVERSAMPLE = 2

_WORD_RE = re.compile(r"[a-z0-9']+")

# Function words carry no retrieval signal; leaving them in would make every
# question look contaminated and every n-gram check trivially fire.
_STOP = frozenset("""
a an the and or but if then than that this these those of in on at to for from by with
without within into over under about across after before during between is are was were
be been being do does did doing have has had having will would can could may might must
shall should its it their his her our your they we you i he she as not no nor so such
what which who whom whose when where why how any all each both few more most other some
company companys business results operations financial condition
""".split())


def tokens(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 2]


def ngrams(seq: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)}


def build_idf(chunks: list[dict[str, Any]]) -> dict[str, float]:
    """Corpus IDF, used both to weight overlap and to define 'rare term'."""
    n_docs = len(chunks)
    df: Counter[str] = Counter()
    for c in chunks:
        df.update(set(tokens(c["text"])))
    return {w: math.log(n_docs / (1 + d)) for w, d in df.items()}


def idf_overlap(question: str, gold_text: str, idf: dict[str, float]) -> float:
    """Share of the question's IDF mass that also appears in the gold chunk.

    1.0 means every distinctive word in the question was copied from the
    passage; near 0 means the question was genuinely reworded. This is the
    leakage statistic, reported per question rather than assumed away.
    """
    q = set(tokens(question))
    if not q:
        return 0.0
    g = set(tokens(gold_text))
    total = sum(idf.get(w, 0.0) for w in q)
    if total <= 0:
        return 0.0
    return sum(idf.get(w, 0.0) for w in q & g) / total


def contaminated(question: str, gold_text: str, idf: dict[str, float]) -> str | None:
    """Return a rejection reason, or None if the question is clean."""
    q_toks, g_toks = tokens(question), tokens(gold_text)
    if len(q_toks) < 4:
        return "too short"
    if ngrams(q_toks, NGRAM_N) & ngrams(g_toks, NGRAM_N):
        return f"shared {NGRAM_N}-gram"
    ov = idf_overlap(question, gold_text, idf)
    if ov > MAX_IDF_OVERLAP:
        return f"idf overlap {ov:.2f}"
    return None


def near_duplicates(source: dict[str, Any], pool: Iterable[dict[str, Any]]) -> list[str]:
    """Chunks covering the same material, by token Jaccard.

    Deliberately lexical. Using embeddings here would put the dense retriever's
    own notion of similarity into the labels it is then scored against.

    **Lexical similarity is not enough on its own, and this is where the first
    version was wrong.** Two 10-Ks from consecutive years share boilerplate,
    risk-factor phrasing and segment descriptions, so an earlier filing clears
    any Jaccard threshold a later one does. But a question about an event is
    only answerable by a filing published *after* the event. The question
    "how much did Nvidia lose in early 2026 because of new H20 export rules"
    was labelled with a chunk from the 2025-01-26 filing -- which predates the
    rules and cannot answer it. A retriever that correctly ranks that chunk low
    was being penalised for being right.

    So a near-duplicate must also be temporally capable of answering: it may be
    the same filing or a *later* one, never an earlier one. This is a
    necessary condition, not a sufficient one -- a later filing may still not
    discuss the material -- so these remain candidate labels that deserve
    review, not automatic gold.
    """
    src = set(tokens(source["text"]))
    if not src:
        return []
    src_date = str(source.get("report_date") or "")
    out = []
    for c in pool:
        if c["chunk_id"] == source["chunk_id"] or c["ticker"] != source["ticker"]:
            continue
        if src_date and str(c.get("report_date") or "") < src_date:
            continue
        other = set(tokens(c["text"]))
        if not other:
            continue
        j = len(src & other) / len(src | other)
        if j >= NEAR_DUP_JACCARD:
            out.append(c["chunk_id"])
    return out


PROMPT = """You are writing evaluation questions for a search system over SEC 10-K filings.

Below is one passage from {ticker}'s {year} Form 10-K, {item_title}.

<passage>
{text}
</passage>

Write ONE question that this passage answers.

Requirements:
- Ask it the way an investor or analyst would ask a colleague, out loud, without
  the filing in front of them. Plain vocabulary.
- Do NOT reuse distinctive wording from the passage. If the passage says
  "adverse macroeconomic conditions", ask about "a weak economy". Paraphrase
  every specialised term.
- The question must be answerable ONLY by this passage's subject matter -- it
  should be specific enough that a different company's filing would not answer
  it. Name the company.
- One sentence. No preamble.

Return JSON only:
{{"question": "...", "topic": "three or four words naming the subject"}}"""


def generate(chunk: dict[str, Any], *, deadline: float = 90.0) -> dict[str, Any] | None:
    """Generate one question, or give up.

    The wall-clock deadline is not belt-and-braces over the socket timeout -- it
    covers a failure the socket timeout structurally cannot. httpx's timeout is
    per-chunk on a stream, so a response that trickles keepalives with no content
    never trips it: the call blocks forever, silently, with no error to log and
    no output to show. Observed here as a build that produced ten questions and
    then sat on the eleventh for over ten minutes.

    A generation this small either returns quickly or is not returning. Skipping
    the chunk costs one eval question; hanging costs the whole run.
    """
    # A daemon thread, not ThreadPoolExecutor. The obvious version --
    #
    #     with ThreadPoolExecutor(max_workers=1) as pool:
    #         return pool.submit(_generate_inner, chunk).result(timeout=deadline)
    #
    # does not work, and fails in a way that looks like the bug it was meant to
    # fix: leaving the `with` block calls shutdown(wait=True), which blocks until
    # the worker finishes. The timeout fires, then the context manager waits out
    # the very hang it just detected. Written and observed here before it was
    # caught -- the run sat silent for four minutes with a 90s deadline set.
    #
    # A blocked socket read cannot be interrupted from outside, so the only
    # options are to abandon the thread or to wait for it. Daemon threads are
    # abandonable and do not hold up process exit.
    box: dict[str, Any] = {}

    def _work() -> None:
        try:
            box["result"] = _generate_inner(chunk)
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    thread.join(timeout=deadline)
    if thread.is_alive():
        print(f"    deadline ({deadline:.0f}s) exceeded for {chunk['chunk_id']}, "
              f"skipping", file=sys.stderr)
        return None
    if "error" in box:
        print(f"    generation failed for {chunk['chunk_id']}: {box['error']}",
              file=sys.stderr)
        return None
    return box.get("result")


def _generate_inner(chunk: dict[str, Any]) -> dict[str, Any] | None:
    year = chunk["report_date"][:4]
    try:
        data, _ = chat_json_chain(
            WRITER_MODELS,
            [{"role": "user", "content": PROMPT.format(
                ticker=chunk["ticker"], year=year,
                item_title=chunk.get("item_title") or f"Item {chunk.get('item')}",
                text=chunk["text"][:3500])}],
            validate=lambda d: isinstance(d, dict) and isinstance(d.get("question"), str),
            max_tokens=400,
            # Short by design. The client's default 120s is a per-chunk *read*
            # timeout on a stream, so a stalled response costs 120s x retries x
            # models before the chain gives up -- observed as an apparent hang
            # with no output. A generation this small either comes back quickly
            # or is not coming back; failing fast into the next model is
            # strictly better than waiting.
            timeout=45.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    generation failed for {chunk['chunk_id']}: {exc}", file=sys.stderr)
        return None
    return data


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("data/eval/eval_narrative.jsonl"))
    ap.add_argument("--per-ticker", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    chunks = [
        json.loads(l)
        for l in (args.index / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if l
    ]
    idf = build_idf(chunks)

    eligible = [
        c for c in chunks
        if c.get("item") in NARRATIVE_ITEMS
        and c.get("kind") == "text"
        and c["n_chars"] >= MIN_CHARS
    ]
    by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in eligible:
        by_ticker[c["ticker"]].append(c)

    print(f"corpus {len(chunks)} chunks | {len(eligible)} narrative-eligible")

    rng = random.Random(args.seed)
    questions: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()

    # Candidates are generated in parallel rather than one at a time.
    #
    # Serially, a single slow call blocks the whole build, and on this API "slow"
    # is not rare: identical call shapes were observed at 4s and at 50s, and a
    # stream that stalls blocks until its deadline. Generating a batch
    # concurrently means the batch costs the *slowest* call, not the sum -- and a
    # chunk that has to be abandoned costs nothing but itself.
    #
    # Over-generate by `OVERSAMPLE` so the decontamination filter has candidates
    # to discard without forcing another serial round trip.
    for ticker, pool in sorted(by_ticker.items()):
        rng.shuffle(pool)
        batch = pool[: args.per_ticker * OVERSAMPLE]
        with ThreadPoolExecutor(max_workers=GEN_WORKERS) as ex:
            results = list(ex.map(generate, batch))

        kept = 0
        for chunk, data in zip(batch, results):
            if kept >= args.per_ticker:
                break
            if not data:
                continue
            question = data["question"].strip()
            reason = contaminated(question, chunk["text"], idf)
            if reason:
                rejected[reason.split()[0]] += 1
                print(f"    reject [{reason}] {question[:70]}")
                continue

            gold = [chunk["chunk_id"], *near_duplicates(chunk, eligible)]
            questions.append({
                "qid": f"{chunk['chunk_id']}-narr",
                "question": question,
                "topic": str(data.get("topic") or "").strip(),
                "ticker": ticker,
                "item": chunk.get("item"),
                "fiscal_year": chunk["report_date"][:4],
                "gold_chunk_ids": gold,
                "idf_overlap": round(idf_overlap(question, chunk["text"], idf), 3),
                "split": "narrative",
                "source": "llm-generated, decontaminated",
            })
            kept += 1
            print(f"  [{ticker} {kept}/{args.per_ticker}] {question[:80]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in questions), encoding="utf-8"
    )

    n_gold = sum(len(q["gold_chunk_ids"]) for q in questions)
    mean_ov = sum(q["idf_overlap"] for q in questions) / max(len(questions), 1)
    print(f"\nkept {len(questions)} questions -> {args.out}")
    print(f"  rejected: {dict(rejected)} ({sum(rejected.values())} total)")
    print(f"  gold labels: {n_gold} ({n_gold / max(len(questions), 1):.1f} avg per question)")
    print(f"  mean residual IDF overlap with gold: {mean_ov:.3f} "
          f"(cap {MAX_IDF_OVERLAP})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
