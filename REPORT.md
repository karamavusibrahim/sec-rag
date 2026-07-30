# sec-rag — technical report

**Question.** Does hybrid retrieval — dense embeddings fused with BM25 — actually
beat its components on SEC filings, and how much does a cross-encoder reranker
add on top?

**Answer.** On numeric/table questions, hybrid fusion is actively harmful: dense
alone scores 0.607 nDCG@10, hybrid 0.528, and the paired comparison is
significant (dense wins 69 questions, loses 34, sign-test p=0.0007). Reranking
is where the value is, and it *completely masks* the fusion damage — so a
pipeline nobody measured would ship the broken component and look fine. A
second, narrative split confirmed the mechanism while falsifying half of my
explanation of it (§3.2) — and a subsequent audit (§2.3) found the original
measurement itself had two flaws, both of which were fixed and both of which
the headline conclusion survived.

| | |
|---|---|
| Corpus | 6 × 10-K filings (AAPL, MSFT, NVDA; two fiscal years each), 1,511 chunks |
| Embeddings | `nvidia/nemotron-3-embed-1b`, 2048-dim, asymmetric (`input_type`) |
| Sparse | `bm25s`, native save/load |
| Reranker | `nvidia/llama-nemotron-rerank-1b-v2` |
| Fusion | Reciprocal Rank Fusion, k=60 |
| Eval | XBRL-grounded numeric split + decontaminated narrative split |

---

## 1. Why the eval set is the interesting part

The standard way to build a RAG eval set is to have an LLM write a question from
each chunk and treat that chunk as the gold answer. This is circular in a way
that specifically corrupts the question being asked here: the generated question
inherits the chunk's vocabulary, so BM25 matches on copied terms and scores near
1.0 for free. An ablation built that way would have concluded that sparse
retrieval is excellent.

### 1.1 The numeric split: XBRL as an independent channel

Every material number in a filing is *also* published as a structured XBRL fact
carrying its concept, unit, period and source accession. So:

1. Take a fact — `us-gaap:Assets`, FY2024, `364,980,000,000`
2. Find which chunks contain that value formatted as it appears in the document
3. Write the question from the **concept label**, never showing the model the chunk

Gold labels come from string-matching a number obtained independently of the
corpus text. There is no lexical path from passage to query.

**A bug this caught.** The first version matched a value against any chunk
containing it, and produced false gold labels: an FY2010 figure sharing digits
with something in a 2025 filing became "correct". Since a 10-K prints the current
year plus two comparatives, a value is only plausibly present in a filing whose
report year is within that window:

```python
matches = [
    c["chunk_id"] for c in pool
    if fy and 0 <= int(c["report_date"][:4]) - fy <= 2
    and any(v in c["text"] for v in variants)
]
```

Without that guard every metric downstream is computed against partly fictional
labels, and nothing about the output looks wrong.

**A worse bug, found later, in the same three lines.** `fact["fy"]` is not the
fiscal year of the fact. It is the fiscal year of the **filing the fact appeared
in**. A 10-K prints three comparative years, so AAPL's FY2024 filing yields:

```
fy=2024  start=2021-09-26  end=2022-09-24  val=26,251,000,000
fy=2024  start=2022-09-25  end=2023-09-30  val=29,915,000,000
fy=2024  start=2023-10-01  end=2024-09-28  val=31,370,000,000
```

All three are `fy=2024`. Keying on it and taking the first match labels FY2022's
R&D as FY2024's — so questions read *"What was AAPL's research and development
expense in fiscal 2024?"* while the expected value was two years stale.

The retrieval metrics survived this, because gold labels are chunk ids found by
string-matching the value, and the FY2022 figure genuinely appears in the FY2024
filing as a comparative. The ablation's conclusions are unaffected. But every
question was quietly asking about the wrong year, and in the sibling
`agentic-rag` project the same bug did real damage: it scored the agent **wrong**
for answering 8.02% — which was exactly right — because gold had been computed
from the stale figure.

The fix derives the year from the period `end` date and requires the duration to
span roughly a year, so a quarterly value can never stand in for an annual one:

```python
def fact_fiscal_year(entry):
    end = entry.get("end")
    return int(end[:4]) if end else None
```

Two lessons. First, a field named `fy` in a financial API is not necessarily the
fiscal year of the number next to it — check what the period fields say. Second,
and more general: **this bug was invisible for as long as nothing cross-checked
the gold value against an independently-derived answer.** It surfaced only when
an agent computed 8.02% from the filing and disagreed with the label.

### 1.2 The narrative split: decontamination instead of trust

Narrative retrieval (risk factors, MD&A) has no XBRL equivalent — it is prose,
and questions have to be generated. Rather than trusting a prompt to avoid
copying, generation is filtered mechanically:

1. **Generate** a question from the chunk, prompted for investor vocabulary.
2. **Reject any shared content 4-gram** with the gold chunk. This is absolute:
   a copied phrase is a copied phrase.
3. **Cap IDF-weighted unigram overlap** — catches copied jargon scattered across
   a reworded sentence, which n-grams miss.
4. **Expand gold over near-duplicates** by token Jaccard. The corpus holds two
   consecutive fiscal years per ticker, and risk factors are largely *edited*
   year over year rather than rewritten, so a question about an FY2025 risk is
   legitimately answered by the FY2024 chunk. Marking that wrong would penalise
   correct retrieval. Deliberately lexical: using embeddings here would put the
   dense retriever's own notion of similarity into the labels it is scored
   against.

### 1.3 A calibration error worth recording

The IDF overlap cap was initially set to **0.34**, on the intuition that a low
overlap means a well-paraphrased question. Then the numeric split — which is
clean *by construction* — was measured:

```
numeric split (n=120):  mean 0.683   median 0.725   p10 0.407   p90 0.875
```

*(Measured on the v1 numeric set — the calibration predates the v2 rebuild.
The v3 full-set lexical-overlap control, 0.475 vs narrative 0.393 in §3.1,
confirms the direction is unchanged.)*

Asking *"What was NVDA's total revenue in fiscal 2025?"* unavoidably reuses
`NVDA`, `revenue`, `fiscal` and `2025`, all of which appear in the target chunk.

**Lexical overlap is not contamination. It is what asking a specific question
looks like.** The 0.34 cap was rejecting narrative questions *cleaner* than the
numeric questions they would be compared against — handicapping BM25 on precisely
the split built to test whether BM25 helps, and biasing the sample toward
questions that avoid naming their own subject, which are worse questions.

The threshold's real job is narrower: ensure the narrative split is no more
contaminated than the numeric baseline. It now sits at the numeric median
(0.725). Verbatim phrase copying, which the numeric split cannot have at all,
remains blocked absolutely by the n-gram filter. Acceptance yield went from
roughly 15% to 86%.

The general lesson: a filter threshold picked by intuition is an unmeasured
parameter in the middle of a measurement.

---

## 2. Results — numeric split

All 120 XBRL-grounded questions (40 per ticker, interleaved), 1,511 chunks,
BM25 indexing the same breadcrumb+text representation dense embeds (§2.3
explains why both of those clauses have to be said):

| Configuration | R@1 | S@1 | R@5 | R@10 | nDCG@10 | MRR@10 |
|---|---|---|---|---|---|---|
| BM25 only | 0.043 | 0.108 | 0.227 | 0.351 | 0.226 | 0.225 |
| Dense only | 0.195 | 0.575 | 0.558 | 0.765 | **0.607** | 0.676 |
| Hybrid (RRF) | 0.175 | 0.442 | 0.478 | 0.709 | **0.528** | 0.555 |
| BM25 + rerank | 0.212 | 0.550 | 0.622 | 0.755 | 0.620 | 0.662 |
| Dense + rerank | 0.197 | 0.508 | 0.676 | 0.878 | 0.681 | 0.682 |
| Hybrid + rerank | 0.195 | 0.483 | **0.692** | **0.878** | 0.681 | 0.671 |

(S@1 is Success@1 — was the top hit relevant. R@1 divides by the number of gold
chunks, so with multi-gold labels its attainable ceiling here is ~0.43; it is
reported for continuity but S@1 is the number to read as a hit rate.)

### 2.1 Fusion made retrieval worse — significantly

Dense alone: 0.607 nDCG@10. Add BM25 through RRF: 0.528. Per-question, dense
beats hybrid on 69 questions and loses on 34 (17 ties): sign-test **p=0.0007**.
This is the corrected, fair-conditions version of the finding and it is not
noise.

The mechanism is visible in row 1. Even with full access to the metadata
(§2.3), BM25 scores **0.226** in isolation, because the gold passages are
financial *tables* — grids of numbers sharing little natural-language
vocabulary with a question. RRF deliberately discards score magnitudes and
fuses on rank alone. That rank-only property is what makes RRF robust across
corpora with incomparable score scales, and it is also what lets a far weaker
ranker pull a far stronger one down. RRF has no way to express "this retriever
is much worse here."

This is worth separating from the folk claim it contradicts. "Hybrid beats
either component" is true when both components are individually competent. It
is a statement about the components, not about fusion, and it gets repeated as
though it were a property of the method.

### 2.2 Reranking is the real gain — and it hides the bug

Dense + rerank adds **+0.074 nDCG@10** and **+0.113 R@10** over dense alone
(paired: 72 wins / 34 losses, p=0.0003).

More importantly, Hybrid + rerank and Dense + rerank land on **the same
nDCG@10 (0.681)**, with a paired record of 28/32/60 — p=0.70,
indistinguishable. The cross-encoder re-reads the actual passages and repairs
everything RRF broke: a 0.079 nDCG gap before reranking becomes nothing after
it. A team shipping "hybrid + rerank" without running the ablation would carry
a component that costs latency and money, contributes nothing, and is
invisible because a later stage cleans up after it.

That is the general shape of the finding: **a pipeline stage can be actively
harmful and undetectable if a downstream stage is strong enough.** Only
component-wise ablation finds it.

### 2.3 The audit: the first version of this measurement had two flaws

A 2026-07-29 audit of this eval found that the previously published numeric
table (BM25 0.112, dense 0.695, hybrid 0.562) was computed under two conditions
the report did not state, because I did not know them:

**The "40-question" slice was 40/40 AAPL.** The eval file groups questions
ticker-by-ticker, AAPL first, and `--limit 40` took the head. MSFT and NVDA
served only as distractors, 80 of 120 built questions went unused, and the
effective sample was ~15 concepts for one company. The runner now interleaves
by ticker before any limit and stamps eval-set path, limit, and per-ticker
counts into the results file.

**BM25 was structurally handicapped.** Dense embedded breadcrumb+text
(`NVDA 10-K 2025 > Item 7 …`) while BM25 indexed bare text — and **zero** of
the gold chunks contain the company name in their raw text, while every
question opens with it. The two retrievers were not searching the same
documents. Isolating the effect offline (BM25 is local, so this costs
nothing): fair indexing alone moves BM25 from 0.112 to 0.353 on the exact old
slice, and from 0.075 to 0.226 on the full set — a **3× effect that had been
attributed to "BM25 is bad at this corpus."**

Both fixes applied, the conclusion gets *stronger*, not weaker: fusion's cost
is now measured under fair conditions with p=0.0007, and the reranker's total
repair reproduces. But the numbers themselves changed (0.695 → 0.607 for dense
is mostly the harder 3-ticker question mix), which is exactly why results
files now record their provenance and per-question scores — the v1/v2 history
of this table had to be reconstructed from file modification times.

---

## 3. Results — narrative split

15 decontaminated questions over Item 1/1A/7 prose. The hypothesis under test:
BM25's collapse on the numeric split is caused by the gold passages being
**tables**, not by BM25 being unsuited to the corpus. If so, sparse retrieval
should recover on prose — and the "fusion hurts" conclusion might reverse.

| Configuration | R@1 | S@1 | R@5 | R@10 | nDCG@10 | MRR@10 |
|---|---|---|---|---|---|---|
| BM25 only | 0.300 | 0.333 | 0.511 | 0.567 | 0.468 | 0.432 |
| Dense only | 0.556 | 0.667 | 0.778 | 0.844 | **0.757** | 0.757 |
| Hybrid (RRF) | 0.456 | 0.533 | 0.578 | 0.733 | 0.613 | 0.586 |
| BM25 + rerank | 0.522 | 0.667 | 0.711 | 0.711 | 0.683 | 0.713 |
| Dense + rerank | 0.556 | 0.667 | 0.844 | 0.844 | **0.763** | 0.769 |
| Hybrid + rerank | 0.556 | 0.667 | 0.844 | 0.844 | 0.763 | 0.769 |

(Run against the fair BM25 index; the narrative numbers barely moved — BM25
0.449 → 0.468 — which is itself informative, see below.)

### 3.1 The explanation was half right — and the audit split it in two

| | numeric | narrative |
|---|---|---|
| BM25 only, nDCG@10 (fair index) | 0.226 | **0.468** |
| Dense only, nDCG@10 | 0.607 | 0.757 |

Under fair conditions, BM25 is still **~2× better on prose than on tables** —
the tables mechanism from §2.1 is real. But the audit showed the previously
reported "BM25 recovers 4–5× on prose" conflated two different effects:

- **Metadata blindness** (the indexing asymmetry, §2.3) — worth ~3× on the
  numeric split, where gold chunks are tables that never name their company,
  and almost nothing on the narrative split (0.449 → 0.468), where prose
  passages already carry topical vocabulary.
- **Vocabulary mismatch on tables** — the residual ~2× numeric-vs-narrative
  gap that survives fair indexing. This part is genuinely about tables being
  grids of digits.

**The leakage control rules out the remaining alternative explanation.** A
sceptic should suspect the narrative questions were easier for BM25 because
they were generated from the passages and inherited their vocabulary. Measured
query/gold content-word overlap says otherwise:

```
numeric split (n=120)   0.475
narrative split (n=15)  0.393
```

The narrative questions share *fewer* words with their gold passages than the
XBRL-grounded ones do. BM25's prose advantage happened **despite** less lexical
signal, not because of more. If anything the comparison is conservative.

### 3.2 The conclusion did not reverse — and that is the real finding

Dense only 0.757, Hybrid 0.613. **Fusion still costs 0.14 nDCG@10, on a split
where BM25 is at its best.**

This falsifies half of my own §2.1 explanation. I attributed RRF's damage to it
giving a *near-random* ranker equal standing with a good one. But BM25 at 0.468
is not near-random, and fusion still hurts. The mechanism is broader than stated:

> RRF degrades results whenever one retriever is **materially better** than the
> other, not only when one is near-useless. Equal rank weighting is a bet that
> the components are comparable, and 0.757 vs 0.468 is not comparable.

(Honesty about sample size: per-question this is 5 wins / 1 loss / 9 ties for
dense over hybrid — directionally consistent with the numeric split's p=0.0007
but not significant on its own at n=15, p=0.22. The narrative split's role is
replication of direction, not independent proof.)

The condition for hybrid retrieval to pay is narrower than the folk claim
suggests. It is not "both components work" — it is "both components work *about
equally well*." That is a much harder bar, and nothing about RRF measures whether
you have cleared it. This matches the published analysis of fusion functions:
Bruch, Gai & Ingber (ACM TOIS 2023, arXiv:2210.11934) show RRF is sensitive to
its parameters and that a tuned convex combination of normalized scores
outperforms it — the fix this data points at (§5).

### 3.3 Reranking's value is concentrated on tables

| | numeric | narrative |
|---|---|---|
| Dense only | 0.607 | 0.757 |
| Dense + rerank | 0.681 | 0.763 |
| gain | **+0.074** (p=0.0003) | **+0.006** (3W/3L/9T, p=1.0) |

On prose the cross-encoder adds nothing detectable — dense retrieval already
puts a right passage first for two thirds of questions (S@1 0.667), and the
paired record is an even 3/3 with 9 ties. Reranking earns its 3× latency cost
specifically where first-stage ranking is poor, which here means numeric and
table retrieval.

The practical consequence is that reranking should arguably be *conditional* on
query type rather than always-on, which is not a conclusion available from the
numeric split alone.

Note also that Hybrid+rerank and Dense+rerank are **identical on every metric
and every question** (15 ties of 15) on narrative. The reranker completely
erases the difference between the two first stages, which is §2.2's masking
effect appearing again in a second, independent split, and about as clean a
demonstration of it as one could ask for.

### 3.4 The fix, measured: weighted fusion repairs what RRF breaks

The convex-combination sweep (`eval/fusion_sweep.py`) — `α·norm(dense) +
(1−α)·norm(BM25)`, min-max normalized per query over each retriever's top-50,
α swept 0→1, RRF k=60 recomputed on the same candidate pools:

| | numeric (n=120) | narrative (n=15) |
|---|---|---|
| BM25 only (α=0) | 0.226 | 0.468 |
| RRF k=60 | 0.528 | 0.613 |
| Dense only (α=1) | 0.607 | 0.757 |
| **best α** | **0.8 → 0.616** | **0.9 → 0.766** |

Three readings, in decreasing order of confidence:

1. **Tuned convex fusion beats RRF decisively on both splits** (+0.088
   numeric, +0.153 narrative) — at *every* α ≥ 0.6, not just the optimum.
   This is the actionable result: if you must fuse, weight.
2. **The optima land at α = 0.8–0.9** — heavily dense-weighted, exactly where
   "one retriever is materially better" predicts, and nowhere near the
   equal-ish weighting RRF implements (RRF's score sits between the α=0.5 and
   α=0.6 points on both curves).
3. **At its optimum, fusion edges past dense alone** (+0.009 on both splits).
   Read this one cautiously: α was tuned and evaluated on the same questions,
   and a gain that size is within noise. The honest claim is "weighted fusion
   stops losing", not "weighted fusion wins".

So the earlier framing survives intact and gains its constructive half: RRF's
failure was never fusion-as-such, it was **equal weighting with no mechanism
to learn it is wrong** — hand the fusion one tunable weight and the damage
disappears.

### 3.5 Optional: DAT per-query fusion — a measured negative

Deep-research (2026-07-30) surfaced DAT — Dynamic Alpha Tuning
(arXiv 2503.23013), verified 3-0 in adversarial review: instead of one global
α, an LLM scores each retriever's top-1 passage per query (0–5) and sets
α = Sv/(Sv+Sb). Published gains: +2.8–3.3pp P@1 over the *best* fixed α with
GPT-4o as the scorer, concentrated on queries where the retrievers disagree.

Implemented as `eval/dat_fusion.py` (optional; fixed-α remains the default)
with `deepseek-v4-flash` as the hosted scorer, and measured against the same
candidate pools:

| | DAT | fixed α=0.8 | dense only |
|---|---|---|---|
| numeric (n=120) | 0.600 | **0.616** | 0.607 |
| narrative (n=15) | 0.729 | 0.752 | **0.757** |

**DAT lost to the fixed α on both splits** (numeric W/L 17/22), including on
the hybrid-sensitive subset where the paper's gains concentrate. The
diagnostic detail: DAT's *mean* α came out at 0.797 numeric / 0.76 narrative —
almost exactly the swept optimum — so the per-query scorer finds the right
global weight on average and then subtracts value through per-query variance.
A 0–5 judgment of two top-1 passages by a small hosted model is noisier than
the signal it is trying to add; the paper used GPT-4o.

The mode ships as optional and off. A published, verified technique that
fails to transfer under an honest replication is a result, not a bug — and it
is precisely the kind of result the fixed-α sweep's in-sample caveat (§3.4)
predicted this corpus might produce.

---

## 4. Engineering notes

**Structure-aware chunking.** Chunks carry an Item breadcrumb (`NVDA 10-K 2025 >
Item 1A. Risk Factors`) prepended to their text, and tables are serialised to
pipe-delimited rows rather than flattened. `MAX_CHARS=2400`, `MIN_CHARS=200`,
`OVERLAP_CHARS=200`.

**Asymmetric embeddings.** `nemotron-3-embed-1b` requires `input_type` to be
`"query"` or `"passage"`. Passing the wrong one degrades retrieval *silently* —
no error, just worse results. The OpenAI SDK cannot pass this field, which is why
the client is raw `httpx`.

**Endpoint non-uniformity.** Three NVIDIA endpoint families with three different
shapes: chat at `integrate.api.nvidia.com/v1` (OpenAI-compatible), embeddings on
the same host but with the extra required field, reranking on a *different host*
(`ai.api.nvidia.com/v1/retrieval`) with a bespoke schema and no presence in
`/v1/models`.

**A flat numpy index, deliberately.** At 1,511 chunks a vector database is
infrastructure for its own sake; exact search over a numpy array is faster than
an approximate index and has no recall error to account for. The honest reason to
adopt one is scale that does not exist here.

**EDGAR requires a descriptive User-Agent** (403 otherwise) and rate-limits to 10
req/s; the client throttles to 0.15s between calls.

---

## 5. Limitations, and the next experiment

- **120 numeric + 15 narrative questions, heavily correlated.** The numeric
  questions are ~15 concepts × years × three companies, not 120 independent
  draws; paired sign tests (which respect the pairing) are reported for the
  central claims, and only the numeric-split comparisons are individually
  significant. The narrative split's *direction* is the result, not its
  magnitudes.
- **Three companies, two fiscal years, one filing type.** All large-cap US tech.
  Nothing here speaks to smaller filers, other sectors, or 10-Qs. In particular,
  `fact_fiscal_year` (year of the period `end`) is correct for AAPL/MSFT/NVDA
  but would mislabel a January-ending filer that names its fiscal year for the
  prior calendar year.
- **Retrieval only.** Whether the right chunk in the top 10 becomes a right
  answer is measured in the sibling `agentic-rag` project, not here.
- **The narrative split is LLM-generated and filtered, not independently
  grounded.** It is strictly weaker evidence than the XBRL split, and the
  residual per-question overlap is recorded in the eval set so it can be
  discounted rather than assumed away. (Provenance note: the 0.725 IDF cap was
  calibrated on the v1 numeric split; the full-set value is 0.475 mean overlap,
  so the calibration direction is unchanged.)
- **A few chunks exceed `MAX_CHARS`** — a single unbroken line is never split
  and the overlap tail re-adds length (observed up to ~3,950 chars). With
  `truncate: "END"` on embed/rerank calls, oversized tails are silently dropped
  by the model.
- **RRF k=60 was not tuned, and RRF ties break by dict insertion order**
  (dense-first — deterministic, and slightly flattering to hybrid).

**The experiment the data asked for has been run** — §3.4: the convex-α sweep
confirms weighted fusion beats RRF at every α ≥ 0.6 on both splits, with
optima at 0.8–0.9. What remains open on fusion: an out-of-sample α (tune on
one ticker, evaluate on the others) to check whether the small
optimum-vs-dense edge is real, and the query-type router (numeric vs prose
optima differ by only 0.1 here, so the router may not pay on this corpus —
the reranker gate from §3.3 is the stronger use of the same signal).

## 6. Conclusion

The reusable result is not "hybrid is bad." It is that **RRF's central design
property — fusing on rank while discarding scores — has no mechanism for
distrusting a component**, and that a strong reranker will conceal the
consequences. Both facts are only visible with per-component ablation against
non-circular labels.

The narrative split sharpened this rather than reversing it. BM25 is at its
best on prose and fusion *still* lost 0.14 nDCG@10. So the requirement is not
that both retrievers be competent, but that they be **comparably** competent,
and RRF provides no way to check. The split also showed that reranking's large
gain is specific to table retrieval (+0.074 numeric vs +0.006 narrative), which
suggests routing it by query type instead of paying for it on every query.

Building the second split was worth it precisely because it could have falsified
the first conclusion. It falsified half of the stated *mechanism* while
confirming the *finding*. The audit (§2.3) then split the mechanism again:
roughly 3× of BM25's numeric collapse was an indexing asymmetry this project
introduced, and only the residual ~2× is about tables. The finding survived
both corrections with a p-value attached — which is the strongest state it has
been in — but the path there is the real lesson: **every large effect in this
report eventually decomposed into a smaller true effect plus a measurement
artifact**, and the artifact was only ever found by someone assuming it was
there.

The eval harness, not the retriever, is the deliverable. Building the gold labels
from a channel independent of the corpus text (XBRL here, LaTeX source in
`multimodal-rag`) is what made every other number in this report worth reading.
