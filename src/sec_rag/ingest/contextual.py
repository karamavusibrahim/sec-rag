"""Optional contextual chunk headers ("contextual retrieval").

Anthropic's contextual retrieval (engineering post, September 2024) prepends
to every chunk a short, model-written description of what the chunk is within
its document before embedding and BM25-indexing it. The failure it targets is
exactly the one this corpus has: a financial-statement row reads "Research and
development | 31,370 | 29,915" with nothing on the chunk saying which company,
which statement or which fiscal years those columns are. The breadcrumb
already supplies company, form and item; the context sentence is meant to
supply what the breadcrumb cannot -- the statement name, the column periods,
the unit -- in vocabulary a question would use.

Two reasons it is optional and unmeasured here, both stated up front:

- The eval set's whole design is that gold chunks must not inherit the
  question's vocabulary. A context sentence written by a model that has read
  the chunk can smuggle in concept labels ("total revenue") and so open the
  lexical path the eval set was built to close. That is not a reason not to
  try it; it is a reason to read a BM25 gain under contextualisation with
  suspicion and to look at the lexical-overlap control alongside it.
- It costs one LLM call per chunk at index time, so it has not been run
  against the hosted API in this repository and no number is claimed for it.

The description call is injectable so the wiring is testable offline.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import Any, Callable, Sequence

from .chunk import Chunk

CONTEXT_PROMPT = """Below is a passage from {company}'s {form} for the fiscal year
ending {report_date} (section: {breadcrumb}). Write one or two sentences that
situate the passage within the filing: which statement, schedule or discussion
it belongs to, what periods and units its figures are in, and what it is about.
Do not restate the figures. Reply with the sentences only.

<passage>
{text}
</passage>"""

DEFAULT_CONTEXT_MODELS = (
    "deepseek-ai/deepseek-v4-flash-0731",
    "meta/llama-3.3-70b-instruct",
)

# A chunk's own text is truncated for the prompt; tables run long and the
# model needs the shape, not every row.
MAX_PASSAGE_CHARS = 2500


def describe_chunk(chunk: Chunk, *, models: Sequence[str] = DEFAULT_CONTEXT_MODELS,
                   chat: Callable[..., str] | None = None) -> str:
    """One or two situating sentences for `chunk`, or "" if the call fails."""
    prompt = CONTEXT_PROMPT.format(
        company=chunk.company, form=chunk.form, report_date=chunk.report_date,
        breadcrumb=chunk.breadcrumb, text=chunk.text[:MAX_PASSAGE_CHARS])
    if chat is None:
        from ..nvidia import chat as _chat
        chat = _chat
    for model in models:
        try:
            out = chat(model, [{"role": "user", "content": prompt}],
                       max_tokens=160, temperature=0.0)
        except Exception as exc:  # noqa: BLE001
            print(f"  context failed ({model}) for {chunk.chunk_id}: {exc}",
                  file=sys.stderr)
            continue
        text = " ".join(str(out).split())
        if text:
            return text
    return ""


def contextualize(chunks: Sequence[Chunk], *,
                  describe: Callable[[Chunk], str] | None = None,
                  verbose: bool = True) -> list[Chunk]:
    """Return copies of `chunks` with `context` filled in.

    A failed description leaves the chunk uncontextualised rather than
    dropping it: a partially contextualised index is still a complete corpus,
    and `chunks.jsonl` records per chunk which ones got context so the
    partiality is visible afterwards.
    """
    describe = describe or describe_chunk
    out: list[Chunk] = []
    n_ok = 0
    for i, c in enumerate(chunks, 1):
        ctx = describe(c) or ""
        n_ok += bool(ctx)
        out.append(replace(c, context=ctx))
        if verbose and i % 100 == 0:
            print(f"  contextualised {i}/{len(chunks)} ({n_ok} with context)")
    if verbose:
        print(f"  contextualised {len(out)} chunks; {n_ok} with context, "
              f"{len(out) - n_ok} without")
    return out


def context_coverage(chunks: Sequence[dict[str, Any] | Chunk]) -> float:
    """Share of chunks carrying a context sentence -- for provenance."""
    if not chunks:
        return 0.0
    n = sum(1 for c in chunks
            if (c.get("context") if isinstance(c, dict) else c.context))
    return n / len(chunks)
