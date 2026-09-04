#!/usr/bin/env python
"""Ingest filings and build the hybrid index.

Usage:
    export SEC_USER_AGENT="you@example.com"
    export NVIDIA_API_KEY="nvapi-..."
    uv run python scripts/build_index.py --tickers AAPL MSFT NVDA --per-ticker 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from sec_rag.index.build import build  # noqa: E402
from sec_rag.ingest.chunk import chunk_filing  # noqa: E402
from sec_rag.ingest.edgar import fetch_filing_html, list_filings  # noqa: E402


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA"])
    ap.add_argument("--forms", nargs="+", default=["10-K"])
    ap.add_argument("--per-ticker", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--contextual", action="store_true",
                    help="optional: prepend a model-written situating sentence "
                         "to every chunk before indexing (one LLM call per "
                         "chunk; see sec_rag/ingest/contextual.py)")
    args = ap.parse_args()

    all_chunks = []
    for ticker in args.tickers:
        filings = list_filings(ticker, forms=tuple(args.forms), limit=args.per_ticker)
        if not filings:
            print(f"  {ticker}: no filings matched {args.forms}")
            continue
        for f in filings:
            html = fetch_filing_html(f)
            chunks = chunk_filing(html, f)
            all_chunks.extend(chunks)
            print(f"  {f.slug}: {len(chunks):4d} chunks  ({len(html):>9,} chars html)")

    if not all_chunks:
        print("no chunks produced; aborting", file=sys.stderr)
        return 1

    print(f"\ntotal: {len(all_chunks)} chunks from {len(args.tickers)} tickers")
    if args.contextual:
        from sec_rag.ingest.contextual import contextualize
        print("writing contextual headers (optional; one LLM call per chunk) ...")
        all_chunks = contextualize(all_chunks)
    build(all_chunks, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
