# sec-rag

Hybrid retrieval over SEC filings, with an evaluation harness that measures each
pipeline component instead of assuming it helps.

Built on NVIDIA NIM: `nemotron-3-embed-1b` for embeddings,
`llama-nemotron-rerank-1b-v2` for reranking.

## The finding

This project has an eval harness, and the harness disproved the design
assumption behind the project. Measured over all 120 XBRL-grounded questions
(40 per ticker, interleaved) against a 1,511-chunk corpus, with BM25 indexing
the same breadcrumb+text representation dense embeds:

| Configuration | R@1 | S@1 | R@5 | R@10 | nDCG@10 | MRR@10 |
|---|---|---|---|---|---|---|
| BM25 only | 0.043 | 0.108 | 0.227 | 0.351 | 0.226 | 0.225 |
| Dense only | 0.195 | 0.575 | 0.558 | 0.765 | **0.607** | 0.676 |
| Hybrid (RRF) | 0.175 | 0.442 | 0.478 | 0.709 | **0.528** | 0.555 |
| BM25 + rerank | 0.212 | 0.550 | 0.622 | 0.755 | 0.620 | 0.662 |
| Dense + rerank | 0.197 | 0.508 | 0.676 | 0.878 | 0.681 | 0.682 |
| Hybrid + rerank | 0.195 | 0.483 | **0.692** | **0.878** | 0.681 | 0.671 |

(S@1 = was the top hit relevant. R@1 divides by gold-set size, so its ceiling
here is ~0.43 — don't read it as a hit rate.)

**Hybrid fusion made retrieval worse — significantly.** Dense alone scores
0.607 nDCG@10; adding BM25 through Reciprocal Rank Fusion drops it to 0.528.
Paired per-question: dense wins 69, loses 34, ties 17 — sign-test p=0.0007.
The received wisdom that "hybrid beats either component alone" is not true on
this workload.

The cause is visible in row 1. Even given the metadata, BM25 scores 0.226 in
isolation, because the gold passages are financial **tables** — grids of
numbers sharing little natural-language vocabulary with a question like *"What
was AAPL's total assets in fiscal 2024?"*. RRF deliberately discards score
magnitudes and fuses on rank alone, which is what makes it robust across
corpora, and also what lets a much weaker ranker drag a much stronger one down.

**Reranking is where the value is**: +0.074 nDCG@10 and +0.113 R@10 over dense
alone (p=0.0003). It also *completely masks* the fusion problem — Hybrid+rerank
and Dense+rerank land on the same 0.681 (paired 28/32/60, p=0.70), because the
cross-encoder re-reads the passages and repairs everything RRF broke. That is
exactly how a bad component survives in a pipeline nobody measures: ship
"hybrid + rerank", pay for BM25 on every query, and never learn it contributes
nothing.

> **These numbers are v3.** Two prior versions were retracted after audits.
> v1 keyed XBRL facts on the API's `fy` field — the fiscal year of the
> **filing**, not the fact, so questions quietly asked about the wrong year.
> v2 fixed that but was silently computed on a 40/40-AAPL head slice of a
> ticker-grouped file, with BM25 indexing bare text while dense embedded
> breadcrumb+text — an asymmetry worth 3× of BM25's apparent collapse
> (0.112 → 0.353 on identical questions once fixed). The headline conclusion
> survived both corrections and is now the strongest it has been (it carries a
> p-value). Details in REPORT.md §2.3; the runner now interleaves tickers and
> stamps provenance + per-question scores into every results file.

## The narrative split: the conclusion survived a test designed to break it

The obvious objection to the table above is that the split is **numeric-only**,
which structurally disadvantages lexical retrieval. So a second split was built
over Item 1/1A/7 prose — 15 questions, LLM-generated but mechanically
decontaminated (see below).

| Configuration | R@1 | S@1 | R@5 | R@10 | nDCG@10 | MRR@10 |
|---|---|---|---|---|---|---|
| BM25 only | 0.300 | 0.333 | 0.511 | 0.567 | 0.468 | 0.432 |
| Dense only | 0.556 | 0.667 | 0.778 | 0.844 | **0.757** | 0.757 |
| Hybrid (RRF) | 0.456 | 0.533 | 0.578 | 0.733 | 0.613 | 0.586 |
| BM25 + rerank | 0.522 | 0.667 | 0.711 | 0.711 | 0.683 | 0.713 |
| Dense + rerank | 0.556 | 0.667 | 0.844 | 0.844 | **0.763** | 0.769 |
| Hybrid + rerank | 0.556 | 0.667 | 0.844 | 0.844 | 0.763 | 0.769 |

**BM25 is ~2× better on prose than on tables under fair conditions** — 0.226
numeric vs 0.468 narrative. (An earlier "5.2× recovery" claim conflated this
with the indexing asymmetry: giving BM25 the breadcrumb tripled its numeric
score and barely moved its narrative score, because prose already carries its
own topical vocabulary. The residual 2× is the part genuinely about tables.)

**And fusion still hurts anyway**: dense 0.757 vs hybrid 0.613. That falsifies
half of my own explanation. I attributed RRF's damage to it giving a *near-random*
ranker equal standing — but BM25 at 0.468 is not near-random, and fusion still
costs 0.14 (5W/1L/9T for dense — directional at n=15, consistent with the
numeric split's p=0.0007). The real condition is broader:

> RRF degrades results whenever one retriever is **materially better** than the
> other, not only when one is useless. Equal rank weighting bets that the
> components are comparable; 0.757 vs 0.468 is not comparable.

So the bar for hybrid retrieval to pay is not "both components work" but "both
components work *about equally well*" — and nothing in RRF measures whether you
have cleared it. The literature agrees: Bruch, Gai & Ingber (TOIS 2023) show a
tuned convex score combination beats RRF, which is the next experiment
(REPORT §5).

**Reranking's value turns out to be table-specific**: +0.074 nDCG@10 on numeric
(p=0.0003), **+0.006** on narrative (3W/3L/9T — nothing). On prose the first
stage already puts a right passage at rank 1 for two thirds of queries, so the
cross-encoder has little to fix. That argues for routing reranking by query
type rather than paying its ~3× latency on every query — a conclusion not
available from the numeric split alone.

### Why the narrative questions can be trusted

They are LLM-generated, which is exactly the setup that normally inflates BM25.
So generation is filtered rather than trusted: any question sharing a content
4-gram with its gold chunk is rejected outright, and IDF-weighted term overlap is
capped. The cap is calibrated against the numeric split rather than by intuition
— an earlier value of 0.34 was rejecting narrative questions *cleaner* than the
XBRL questions they'd be compared against, which would have handicapped BM25 on
the very split built to test it.

The control that matters, measured:

```
mean query/gold content-word overlap:   numeric (n=120) 0.475   narrative 0.393
```

The narrative questions share **fewer** words with their gold passages. BM25's
prose advantage happened despite less lexical signal, not because of more.

```bash
uv run python eval/build_narrative_eval.py --per-ticker 5
uv run python eval/run_retrieval.py --eval-set data/eval/eval_narrative.jsonl
```

## How the eval set avoids being circular

The standard way to build a RAG eval set is to have an LLM write a question from
each chunk and call that chunk the gold answer. That measures paraphrase
distance, not retrieval — the question inherits the chunk's vocabulary, so
lexical retrieval scores near 1.0 for free.

This uses XBRL instead. Every material number in a filing is *also* published as
a structured fact carrying its concept, unit, period and source accession:

1. Take a fact — `us-gaap:Assets`, FY2024, `364,980,000,000`
2. Find which chunks contain that value formatted as it appears in the document
3. Write the question from the **concept label**, never showing the model the chunk

Gold labels come from string-matching a number obtained independently of the
corpus text, so there is no lexical path from passage to question.

One bug worth recording: the first version matched values across all fiscal
years, so an FY2010 figure that happened to share digits with something in a 2025
filing became a false gold label. A 10-K shows the current year plus two
comparatives, so the fix is a window guard — `0 <= report_year - fiscal_year <= 2`.
Without it, every number in the table above would have been quietly wrong.

## Pipeline

```
EDGAR ─▶ structure-aware chunking ─▶ dense (NVIDIA) ─┐
                                  └─▶ BM25 ──────────┼─▶ RRF ─▶ rerank ─▶ top-k
                                                     ┘
```

Chunking follows the filing's structure rather than a token budget:

- **Tables are serialized, not flattened.** HTML-to-text over `<table>` markup
  yields a wall of orphaned digits whose column association is lost, and the
  model then attributes numbers to the wrong line item. Each table becomes
  pipe-delimited rows with its header retained, kept atomic so a row is never
  split from its column names.
- **Chunks carry an Item breadcrumb.** Filing prose is heavily anaphoric ("as
  discussed above"), so a bare paragraph is often unusable alone.
  `AAPL 10-K 2025 > Item 1A. Risk Factors >` is prepended to the embedded text.
- **Sized in characters, not tokens** — a character budget doesn't shift when the
  tokenizer does.

**Why a flat numpy index and not a vector DB.** At 1,511 chunks × 2048 dims, a
normalized dot product is an *exact* nearest-neighbour search in single-digit
milliseconds. FAISS or pgvector would add a dependency, a build step, and
approximation error to solve a problem that doesn't exist yet. That switch
belongs around ~1M vectors, and the retrieval interface is narrow enough to keep
it a one-file change.

## Setup

```bash
uv sync
cp .env.example .env    # then fill in both values
```

```
NVIDIA_API_KEY=nvapi-...          # https://build.nvidia.com
SEC_USER_AGENT=you@example.com    # EDGAR returns 403 without a contact string
```

EDGAR requires a descriptive User-Agent and enforces 10 req/s; the client
rate-limits itself rather than relying on luck.

## Usage

```bash
# Build the corpus (6 10-Ks -> ~1,500 chunks)
uv run python scripts/build_index.py --tickers AAPL MSFT NVDA --per-ticker 2

# Generate the eval set from XBRL facts
uv run python eval/build_eval_set.py

# Run the ablation
uv run python eval/run_retrieval.py --limit 40
```

## Layout

```
src/sec_rag/
  nvidia.py           NIM client: chat (streaming), embeddings, reranking
  ingest/edgar.py     EDGAR API, rate limiting, XBRL facts
  ingest/chunk.py     structure-aware chunking, table serialization
  index/build.py      dense vectors + BM25
  retrieve/hybrid.py  RRF fusion + cross-encoder rerank, all stages switchable
eval/
  build_eval_set.py   XBRL-grounded question generation
  run_retrieval.py    the ablation sweep
```

## Notes on the NVIDIA endpoints

Three endpoint families that are **not** uniform:

| | host | shape |
|---|---|---|
| chat | `integrate.api.nvidia.com/v1` | OpenAI-compatible |
| embeddings | `integrate.api.nvidia.com/v1` | OpenAI-ish, **plus a required `input_type`** |
| reranking | `ai.api.nvidia.com/v1/retrieval` | bespoke `{query:{text}, passages:[{text}]}` |

`input_type` must be `"query"` or `"passage"` — these are asymmetric encoders,
and using the wrong one degrades retrieval **silently**, with no error. The
OpenAI SDK cannot pass the field, which is why this client uses httpx directly.
Rerank models never appear in `/v1/models` because they live on a different host.

`build.nvidia.com` is a rate-limited free tier under an evaluation-only Terms of
Service; production use requires NVIDIA AI Enterprise. Model availability is also
not stable — see the sibling `agentic-rag` project, where two models hit HTTP 410
end-of-life mid-development.

## Limitations

- 6 filings, 3 tickers. Scaling is a config change, not a code change.
- 120 numeric + 15 narrative questions, heavily correlated (~15 concepts ×
  years × tickers). Paired sign tests are reported for the central claims;
  only the numeric-split comparisons are individually significant.
- No answer-generation stage — this project is scoped to retrieval, the part
  that's cleanly measurable. Generation lives in `agentic-rag`.
- The fix indicated by the data has been measured (`eval/fusion_sweep.py`):
  convex fusion `α·dense + (1−α)·BM25` beats RRF at every α ≥ 0.6 on both
  splits (best: numeric α=0.8 → 0.616, narrative α=0.9 → 0.766, vs RRF
  0.528/0.613). RRF's failure was equal weighting, not fusion as such.
  See REPORT §3.4.

## License

MIT. SEC filings are US government works in the public domain.
