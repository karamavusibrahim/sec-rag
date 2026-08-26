# Audit — retrieval claims against committed artifacts

A pass over what this repository claims against what its own result files and
code support. Everything below was recomputed offline from committed JSON and
JSONL — no API calls, no re-runs. Where a claim had to come down, the retracted
value and the reason stay in the text.

`eval/report_tables.py` is the durable half of this audit. It prints every
published table and statistic from the artifacts, and `--check` fails when a
doc drifts from them. The transcription errors below are the kind that only
happen to hand-copied numbers.

## Corrected

### Three S@1 cells were higher than the artifact they came from

`REPORT.md` published dense / hybrid / BM25+rerank S@1 as **0.575 / 0.442 /
0.550**. `eval/results/retrieval_v3.json` says **0.542 / 0.425 / 0.500**. Every
other cell in that six-row table matched exactly, and README had already been
corrected — so this was transcription drift in one file, and all three errors
ran in the flattering direction. The R@1 ceiling was quoted as "~0.43"; the
mean of `1/|gold|` over the eval set's own gold sizes is **0.477**.

Both tables now carry a generation anchor and are verified by
`eval/report_tables.py --check`.

### "Fusion made retrieval worse — significantly" overstated what was measured

The sign test reproduces exactly: 69/34/17, p=0.0007. But the 120 questions are
a template crossed over (ticker, concept, year) across **three companies**, so
they are not 120 independent trials. Aggregating so each cluster contributes
one observation:

| unit | n | W/L/T | p |
|---|---|---|---|
| question | 120 | 69/34/17 | 0.0007 |
| ticker × concept | 38 | 25/11/2 | 0.0288 |
| concept | 15 | 11/4/0 | 0.1185 |
| ticker | 3 | 3/0/0 | 0.2500 |

What survives is arguably better than the p-value: **the direction is
unanimous** — dense wins on all 3 companies and 11 of 15 concepts. What does
not survive is the word "significantly", and the fix is more companies, not a
different test. (Sign and Wilcoxon tests are also known to run high Type-I
error rates relative to bootstrap and randomization tests at large n — another
reason not to lean on the question-level figure.)

### The DAT replication was mostly measuring its own fallback

`eval/results/dat_fusion.json` records `n_scorer_failed` — and it is large.
Every failure falls back to the fixed α, which by construction ties the
baseline:

| split | attempted | scorer failed | actually scored |
|---|---|---|---|
| numeric | 120 | 64 | **56** |
| narrative | 15 | 12 | **3** |

The report presented these as n=120 and n=15. The narrative row is a 3-query
result wearing an n=15 label.

Worse, the *interpretation* was circular on that split. The report's headline
diagnostic was that DAT's mean α landed "almost exactly on the swept optimum"
(0.797 / 0.76) — evidence that the scorer finds the right global weight and
loses value to per-query variance. But the fallback **is** α=0.8, so with 12 of
15 rows falling back the narrative mean was being dragged to 0.8 by the
failures themselves. Over the 3 rows actually scored, mean α is **0.600**.

The numeric split survives the check (0.797 attempted vs 0.795 over the 56
scored) and its W/L direction holds on the complete cases, so that reading
stands there and only there. Both docs now state the failure counts next to the
numbers.

### A retracted effect size was still in the report

REPORT still cited fair indexing moving BM25 `0.112 → 0.353` on the old slice.
README had already retracted that exact figure as never committed. Removed from
REPORT, with the retraction stated rather than the number silently deleted. The
full-set figures (`0.075 → 0.226`) do reproduce and stay.

## Fixed in code

### Numeric gold labels never checked what the number was

`eval/build_eval_set.py`

`_formats` deliberately emits scaled variants, because filings report in
thousands and millions — so a total revenue of $1,234,000 also searches for
`1,234`. Matching on the digits alone, a chunk reading *"the company employed
1,234 people"* satisfied every condition the builder checked and became the
gold passage for a revenue question. Reproduced directly against the builder.

The builder now requires a concept anchor — "net sales", "research and
development", "total assets" — within ~400 characters of the matched value.
Near, not merely present: a 10-K page mentioning revenue somewhere also
contains dozens of unrelated numbers, and a whole-chunk search would re-admit
exactly the coincidences this rejects. Concepts with no anchor list fall
through rather than being rejected, so adding one to `INTERESTING` can never
silently empty its qrels.

**Prevalence is unknown, not estimated.** The corpus is fetched at run time and
not committed, so the affected fraction of the 120 questions cannot be
measured here. It is bounded by the existing discriminativeness filter (a value
matching >8 chunks is dropped) and the ±2-year window — bounded is not
measured. Stated in README and REPORT §1.1 rather than glossed.

### Narrative labels propagated backwards in time

`eval/build_narrative_eval.py`

Gold was expanded to lexically near-identical chunks from the same company.
The reasoning is sound for *standing* risk factors, which are edited rather
than rewritten year over year — and the report argues it well. It fails for
*event* questions, which the generator produces from the same prose: "how much
did Nvidia lose in early 2026 because of new export rules on its H20 chips"
was labelled with a chunk from the **2025-01-26** filing, which predates the
rules. A retriever ranking that chunk low was penalised for being right.

Expansion is now restricted to the same filing or later. Unlike the numeric
case this one **is** measurable from committed data: **3 of 15** narrative
questions carry an earlier-filing label. A test pins the count so the caveat
cannot go stale in either direction.

### Two evaluations that changed their own denominators

- `eval/gated_rerank.py` — a rerank exception hit `continue`, dropping the
  query from every metric *and* from threshold selection. A run that lost half
  its queries reported clean numbers over the survivors, indistinguishable from
  a complete run. Now records `n_attempted`, `n_failed`, the failing qids, and
  warns loudly.
- `eval/run_retrieval.py` — `max(len(questions), 1)` turned an empty eval set
  into a full set of all-zero metrics and a **successful exit**. A mis-typed
  `--split` wrote a plausible artifact recording a collapse that never
  happened. Now raises.

## Examined and rejected

**"The confidence gate cannot save reranker calls."** Raised as a finding
elsewhere in this audit cycle; it does not hold. REPORT §3.6 claims "zero extra
API calls", which is exactly true — the gate reuses logits the rerank pass has
already returned — and it never claims the gate *avoids* rerank calls.
`eval/gated_rerank.py`'s own docstring already says so: *"a quality gate, not a
cost-saving pre-router."* No change made.

## Not fixed — needs the corpus

Both qrel fixes change how the eval set is *built*; the committed set predates
them and rebuilding needs filings that `.gitignore` excludes. The eval set is
also regenerated from a run-time fetch of "the newest two filings", with no
accession manifest pinning which six were used, and `retrieval_v3.json` records
no corpus hash, BM25 configuration, reranker model, or git revision.

**Committing a corpus manifest — accessions, URLs, report dates, checksums,
chunker version — is the highest-value change left in this repository.** Until
it exists, aggregate metrics can be recomputed from the saved JSON but the
retrieval itself cannot be rerun, and corrections like the two above can be
applied but never verified.
