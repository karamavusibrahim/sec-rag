"""Structure-aware chunking for SEC filings.

Two decisions carry most of the retrieval quality here.

**1. Tables are serialized, not flattened.** A 10-K is largely financial tables,
and that is where the numbers live. Running an HTML-to-text pass over
`<table>` markup produces a wall of orphaned digits with no row or column
context -- the model then cheerfully attributes "12,345" to the wrong line item.
We serialize each table to pipe-delimited rows with its header retained, and keep
it as an atomic chunk so a row is never split from its column names.

**2. Chunks are scoped to Items, with the Item as a breadcrumb.** 10-K/10-Q text
is heavily anaphoric ("as discussed above", "the aforementioned agreement"), so a
bare paragraph is often unusable standalone. Prefixing
`AAPL 10-K FY2024 > Item 7. MD&A >` gives the embedder topical context and gives
the reader a citation.

Chunks are sized by **characters**, not tokens: token-budget sizing makes chunk
content vary with the tokenizer, and a character budget is stable and debuggable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Iterator

from selectolax.parser import HTMLParser

# 10-K and 10-Q item headings. Matched case-insensitively at line start after
# whitespace normalization. The `\.?` and `[\s:.\-]` slop is necessary: real
# filings write "ITEM 1A." / "Item 1A -" / "Item&nbsp;1A:" interchangeably.
_ITEM_RE = re.compile(
    r"^item\s+(\d{1,2}[A-C]?)\s*[.:\-–—]?\s*(.{0,120})$",
    re.I,
)

# Boilerplate lines that add nothing and dilute embeddings.
_NOISE_RE = re.compile(
    r"^(table of contents|index|page\s+\d+|\d+|"
    r"see accompanying notes.*|the accompanying notes.*)$",
    re.I,
)

# Private-use codepoint: cannot occur in filing text, survives HTML text
# extraction, and (unlike NUL) is a legal character in Python source.
_SENTINEL = "\ue000"

MAX_CHARS = 2400
MIN_CHARS = 200
OVERLAP_CHARS = 200


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    ticker: str
    company: str
    form: str
    report_date: str
    accession: str
    item: str | None          # "7", "1A", ... or None if before the first item
    item_title: str | None
    kind: str                 # "text" | "table"
    text: str                 # the raw content
    breadcrumb: str           # human/embedder-facing path
    source_url: str
    n_chars: int = 0
    # Optional situating sentence(s) written by `ingest.contextual`: what this
    # chunk is within its filing, in words the chunk itself may not contain.
    # Empty unless the index was built with --contextual. Persisted to
    # chunks.jsonl so an index records whether it was built with context.
    context: str = ""

    def __post_init__(self) -> None:
        self.n_chars = len(self.text)

    @property
    def embed_text(self) -> str:
        """What actually gets embedded: breadcrumb (+ context) + content.

        The breadcrumb is included deliberately -- it supplies the topical anchor
        that an anaphoric paragraph lacks on its own. The optional context goes
        in the same place for the same reason, and into BM25 too: both
        retrievers index `embed_text`, so contextualising helps or hurts them
        together rather than reintroducing the asymmetry fixed in index/build.
        """
        head = f"{self.breadcrumb}\n{self.context}" if self.context else self.breadcrumb
        return f"{head}\n\n{self.text}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# HTML -> structured blocks
# --------------------------------------------------------------------------

def _serialize_table(node: Any) -> str:
    """Render an HTML table as pipe-delimited rows, dropping empty columns.

    Filing tables are riddled with spacer cells (`<td>&nbsp;</td>`) used for
    visual alignment; naive extraction turns one logical row into a dozen empty
    fields. We drop all-empty columns so the result reads like a table.
    """
    rows: list[list[str]] = []
    for tr in node.css("tr"):
        cells = [
            re.sub(r"\s+", " ", (td.text() or "")).strip()
            for td in tr.css("td, th")
        ]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [i for i in range(width) if any(r[i] for r in rows)]
    if not keep:
        return ""

    lines = [" | ".join(r[i] for i in keep) for r in rows]
    return "\n".join(line for line in lines if line.replace("|", "").strip())


def _iter_blocks(html: str) -> Iterator[tuple[str, str]]:
    """Yield ("text"|"table", content) in document order.

    Tables are pulled out first and replaced with a placeholder so they don't get
    smeared into surrounding prose, then re-emitted in position.
    """
    tree = HTMLParser(html)
    for tag in tree.css("script, style, ix\\:header"):
        tag.decompose()

    body = tree.body or tree.root
    if body is None:
        return

    # selectolax has no ordered mixed-node walk, so we tag tables with a marker,
    # extract their text, and split on the marker to recover order.
    tables: list[str] = []
    for i, node in enumerate(body.css("table")):
        serialized = _serialize_table(node)
        marker = f"\n{_SENTINEL}TABLE{i}{_SENTINEL}\n"
        tables.append(serialized)
        node.replace_with(marker)

    full = body.text(separator="\n")
    for part in re.split(f"{_SENTINEL}TABLE" + r"(\d+)" + _SENTINEL, full):
        if part is None:
            continue
        if part.isdigit() and int(part) < len(tables):
            content = tables[int(part)]
            if content.strip():
                yield "table", content
        else:
            cleaned = _clean_text(part)
            if cleaned:
                yield "text", cleaned


def _clean_text(raw: str) -> str:
    lines: list[str] = []
    for line in raw.splitlines():
        line = re.sub(r"[\xa0\u2007\u202f\u2009\u200a]", " ", line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line or _NOISE_RE.match(line):
            continue
        lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Blocks -> chunks
# --------------------------------------------------------------------------

def _split_paragraphs(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Pack lines into <= max_chars windows, never splitting mid-sentence.

    Adds a small overlap between windows so a fact spanning a boundary survives
    in at least one chunk.
    """
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for line in text.splitlines():
        if size + len(line) + 1 > max_chars and buf:
            out.append("\n".join(buf))
            tail = "\n".join(buf)[-OVERLAP_CHARS:]
            buf = [tail, line] if tail else [line]
            size = sum(len(b) + 1 for b in buf)
        else:
            buf.append(line)
            size += len(line) + 1
    if buf:
        out.append("\n".join(buf))
    return [c for c in out if len(c.strip()) >= 1]


def chunk_filing(html: str, filing: Any) -> list[Chunk]:
    """Turn one filing's HTML into structure-aware chunks."""
    doc_id = filing.slug
    item: str | None = None
    item_title: str | None = None
    chunks: list[Chunk] = []
    seq = 0

    def make(kind: str, text: str) -> Chunk:
        nonlocal seq
        fy = filing.report_date[:4]
        crumbs = [f"{filing.ticker} {filing.form} {fy}"]
        if item:
            crumbs.append(f"Item {item}. {item_title}" if item_title else f"Item {item}")
        if kind == "table":
            crumbs.append("[financial table]")
        seq += 1
        return Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}#{seq:04d}",
            ticker=filing.ticker,
            company=filing.company,
            form=filing.form,
            report_date=filing.report_date,
            accession=filing.accession,
            item=item,
            item_title=item_title,
            kind=kind,
            text=text.strip(),
            breadcrumb=" > ".join(crumbs),
            source_url=filing.url,
        )

    for kind, content in _iter_blocks(html):
        if kind == "table":
            # Tables stay atomic: a row split from its header is worse than a
            # long chunk. Oversized tables are windowed but keep row 1 as header.
            if len(content) <= MAX_CHARS:
                chunks.append(make("table", content))
            else:
                rows = content.splitlines()
                header = rows[0] if rows else ""
                for window in _split_paragraphs("\n".join(rows[1:]), MAX_CHARS - len(header) - 1):
                    chunks.append(make("table", f"{header}\n{window}"))
            continue

        # Text: watch for Item headings, which reset the breadcrumb.
        buf: list[str] = []
        for line in content.splitlines():
            m = _ITEM_RE.match(line)
            if m and len(line) < 140:
                if buf:
                    for window in _split_paragraphs("\n".join(buf)):
                        if len(window.strip()) >= MIN_CHARS:
                            chunks.append(make("text", window))
                    buf = []
                item = m.group(1).upper()
                item_title = (m.group(2) or "").strip(" .:-–—") or None
                continue
            buf.append(line)
        if buf:
            for window in _split_paragraphs("\n".join(buf)):
                if len(window.strip()) >= MIN_CHARS:
                    chunks.append(make("text", window))

    return chunks
