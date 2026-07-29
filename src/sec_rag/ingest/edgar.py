"""SEC EDGAR client.

EDGAR is free, public-domain, and has no API key — but it does have a fair-access
policy that is enforced:

- **A descriptive User-Agent with contact info is required.** Requests with a
  default library UA get 403'd. Set `SEC_USER_AGENT` (e.g. "you@example.com").
- **10 requests/second maximum.** Exceeding it earns a temporary IP block, so
  this client rate-limits itself rather than relying on good luck.

Two host families, and they are not interchangeable:
    data.sec.gov  -> JSON APIs (submissions, XBRL company facts)
    www.sec.gov   -> the Archives (actual filing documents)

The XBRL endpoint is the interesting one for evaluation: it returns the same
numbers that appear in the filing tables, as structured facts. That gives us
verifiable ground truth for numeric questions without hand-labelling.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from datetime import date
from typing import Any, Iterator

import httpx

DATA_BASE = "https://data.sec.gov"
WWW_BASE = "https://www.sec.gov"
TICKER_MAP_URL = f"{WWW_BASE}/files/company_tickers.json"

# SEC fair-access limit is 10 req/s; we stay comfortably under it.
_MIN_INTERVAL = 0.15
_last_request = 0.0


def _user_agent() -> str:
    ua = os.getenv("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError(
            "SEC_USER_AGENT is not set. EDGAR requires a contact string or it "
            "returns 403. Example: export SEC_USER_AGENT='you@example.com'"
        )
    return f"sec-rag/0.1 ({ua})"


def _throttle() -> None:
    global _last_request
    delta = time.monotonic() - _last_request
    if delta < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - delta)
    _last_request = time.monotonic()


def _get(url: str, *, timeout: float = 60.0) -> httpx.Response:
    _throttle()
    headers = {
        "User-Agent": _user_agent(),
        "Accept-Encoding": "gzip, deflate",
    }
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url, headers=headers)
    if resp.status_code == 403:
        raise RuntimeError(
            f"EDGAR returned 403 for {url}. This is almost always a User-Agent "
            "problem — SEC_USER_AGENT must contain real contact info."
        )
    resp.raise_for_status()
    return resp


# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Filing:
    cik: str
    ticker: str
    company: str
    form: str            # "10-K", "10-Q", ...
    accession: str       # dashless
    filing_date: str     # ISO
    report_date: str     # period of report, ISO
    primary_doc: str

    @property
    def url(self) -> str:
        cik_int = int(self.cik)
        return f"{WWW_BASE}/Archives/edgar/data/{cik_int}/{self.accession}/{self.primary_doc}"

    @property
    def slug(self) -> str:
        return f"{self.ticker}_{self.form.replace('-', '')}_{self.report_date}"


def ticker_to_cik(ticker: str, *, cache_dir: Path | None = None) -> tuple[str, str]:
    """Resolve a ticker to (zero-padded CIK, company name)."""
    ticker = ticker.upper()
    cache = (cache_dir or Path("data/raw")) / "company_tickers.json"
    if cache.exists():
        payload = json.loads(cache.read_text())
    else:
        payload = _get(TICKER_MAP_URL).json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload))

    for entry in payload.values():
        if entry["ticker"].upper() == ticker:
            return str(entry["cik_str"]).zfill(10), entry["title"]
    raise KeyError(f"ticker {ticker!r} not found in EDGAR company_tickers.json")


def list_filings(
    ticker: str,
    *,
    forms: tuple[str, ...] = ("10-K", "10-Q"),
    limit: int = 8,
    cache_dir: Path | None = None,
) -> list[Filing]:
    """Recent filings for a ticker, newest first.

    Only reads the `recent` block of the submissions API (~1000 filings), which
    is plenty for our window. Older filings live in separate paginated files.
    """
    cik, company = ticker_to_cik(ticker, cache_dir=cache_dir)
    data = _get(f"{DATA_BASE}/submissions/CIK{cik}.json").json()
    recent = data["filings"]["recent"]

    out: list[Filing] = []
    for i, form in enumerate(recent["form"]):
        if form not in forms:
            continue
        out.append(
            Filing(
                cik=cik,
                ticker=ticker.upper(),
                company=company,
                form=form,
                accession=recent["accessionNumber"][i].replace("-", ""),
                filing_date=recent["filingDate"][i],
                report_date=recent["reportDate"][i] or recent["filingDate"][i],
                primary_doc=recent["primaryDocument"][i],
            )
        )
        if len(out) >= limit:
            break
    return out


def fetch_filing_html(filing: Filing, *, cache_dir: Path | None = None) -> str:
    """Download a filing's primary document, caching to disk."""
    cache_dir = cache_dir or Path("data/raw/filings")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{filing.slug}.html"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    html = _get(filing.url).text
    path.write_text(html, encoding="utf-8")
    return html


def company_facts(ticker: str, *, cache_dir: Path | None = None) -> dict[str, Any]:
    """XBRL company facts — structured financials straight from the filings.

    This is the eval-set generator: every fact here is a number that appears in
    a filing table, with its unit, period, and source accession number. Numeric
    questions built from these have machine-checkable answers.
    """
    cik, _ = ticker_to_cik(ticker, cache_dir=cache_dir)
    cache_dir = cache_dir or Path("data/raw")
    path = cache_dir / f"facts_{ticker.upper()}.json"
    if path.exists():
        return json.loads(path.read_text())
    payload = _get(f"{DATA_BASE}/api/xbrl/companyfacts/CIK{cik}.json").json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload


def fact_fiscal_year(entry: dict[str, Any]) -> int | None:
    """The fiscal year a fact is *about*, which is not `fy`.

    This is the single most dangerous field in the companyfacts API. `fy` is the
    fiscal year of the **filing the fact appeared in**, not of the value. A 10-K
    prints three years of the income statement, so AAPL's FY2024 filing yields:

        fy=2024  start=2021-09-26  end=2022-09-24  val=26,251,000,000
        fy=2024  start=2022-09-25  end=2023-09-30  val=29,915,000,000
        fy=2024  start=2023-10-01  end=2024-09-28  val=31,370,000,000

    All three are `fy=2024`. Keying on `fy` therefore silently picks a
    two-year-old figure and labels it current -- which is exactly what happened
    here: an eval reported the agent "wrong" for answering 8.02% (31,370/391,035,
    correct) against a gold value built from the FY2022 number.

    The period a fact describes is carried by `end`, and for these filers the
    fiscal-year label matches the calendar year of the period end (AAPL Sep 2024
    -> FY2024, MSFT Jun 2024 -> FY2024, NVDA Jan 2026 -> FY2026).
    """
    end = entry.get("end")
    if not end or len(end) < 4:
        return None
    try:
        return int(end[:4])
    except ValueError:
        return None


def is_annual(entry: dict[str, Any]) -> bool:
    """True for a full-year duration fact.

    Balance-sheet items are instants (no `start`) and are treated as annual when
    they come from a 10-K. Flow items must span roughly a year, so that a
    quarterly figure is never mistaken for an annual one -- the substitution this
    whole eval exists to catch.
    """
    start, end = entry.get("start"), entry.get("end")
    if not start:
        return True
    try:
        d0 = date(int(start[:4]), int(start[5:7]), int(start[8:10]))
        d1 = date(int(end[:4]), int(end[5:7]), int(end[8:10]))
    except (ValueError, TypeError):
        return False
    return (d1 - d0).days >= 300


def iter_us_gaap_facts(facts: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Flatten us-gaap XBRL facts into rows.

    Yields dicts with concept, label, value, unit, fiscal period, form, and the
    accession number the value came from — so a generated question can be traced
    back to the exact document.

    `fiscal_year` is the year the fact *describes*, derived from the period end.
    Prefer it over `fy`, which is the filing's year — see `fact_fiscal_year`.
    """
    for concept, body in (facts.get("facts", {}).get("us-gaap") or {}).items():
        label = body.get("label") or concept
        for unit, entries in (body.get("units") or {}).items():
            for e in entries:
                if e.get("form") not in ("10-K", "10-Q"):
                    continue
                yield {
                    "concept": concept,
                    "label": label,
                    "value": e.get("val"),
                    "unit": unit,
                    "fy": e.get("fy"),
                    "fiscal_year": fact_fiscal_year(e),
                    "annual": is_annual(e),
                    "fp": e.get("fp"),
                    "start": e.get("start"),
                    "end": e.get("end"),
                    "form": e.get("form"),
                    "accession": (e.get("accn") or "").replace("-", ""),
                    "filed": e.get("filed"),
                }
