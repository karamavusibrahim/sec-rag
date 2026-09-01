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
0.550**. `eval/results/retrieval_v3.json` says **0.542 / 0.425 / 0.500**. Every other
cell in that six-row table matched exactly, so this was transcription drift in
three cells, and all three ran in the flattering direction.

An earlier draft of this file said README "had already been corrected". That was
wrong. `git show main:README.md` carries the same 0.575 / 0.442 / 0.550. The
error came from reading the working tree and describing it as the committed
state; both files are corrected by this branch, not one. The R@1 ceiling was quoted as "~0.43"; the
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

What survives is arguably better than the p-value: the direction is
**consistent**, and unanimous at company level — dense wins on all 3 companies,
and on 11 of 15 concepts. (An earlier draft called it "unanimous" flatly. Four
concepts favour hybrid; only the company-level aggregation is unanimous.) What
does not survive is the word "significantly", and the fix is more companies,
not a different test. (Sign and Wilcoxon tests are also known to run high Type-I
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
Removed from REPORT, with the retraction stated rather than the number silently
deleted.

Two corrections to an earlier draft of this entry. README on `main` cited
`0.353` too and had *not* retracted it; the retraction is made by this branch,
not inherited from it. And the full-set pair `0.075 → 0.226` does **not**
reproduce either: `0.226` is in `retrieval_v3.json`, but `0.075` appears in no
committed artifact, so the before-and-after cannot be checked offline. Only the
corrected endpoint is supported.

## Fixed in code

### Numeric gold labels never checked what the number was

`eval/build_eval_set.py`

`_formats` deliberately emits scaled variants, because filings report in
thousands and millions — so a total revenue of $1,234,000 also searches for
`1,234`. Matching on the digits alone, a chunk reading *"the company employed
1,234 people"* satisfied every condition the builder checked and became the
gold passage for a revenue question. Reproduced directly against the builder.

**The obvious fix was the wrong one, and this branch does not ship it.** The
first attempt required a concept anchor — "net sales", "research and
development" — within ~400 characters of the matched value, and dropped labels
that lacked one. Two problems killed it:

- **It biases the eval it is grading.** Questions are written *from the concept
  label*, so requiring gold chunks to contain that same vocabulary guarantees a
  lexical path from passage to question. That is precisely the circularity the
  XBRL design exists to avoid, and it would have handed free keyword matches to
  BM25 in an ablation whose entire subject is sparse versus dense retrieval.
- **It did not work.** `"Revenue declined. The company employed 1,234 people."`
  satisfies a proximity check for a revenue question. The original false
  positive survives it.

So the support signal is now **recorded, not enforced**: each question carries
`gold_concept_supported` and `n_gold_unsupported`, which makes the
contamination measurable and lets metrics be recomputed on the supported subset
without letting labels inherit the question's words. Fixing it properly needs
qrels built from inline-XBRL element-to-DOM mappings, which carry the concept
with the value instead of inferring it from nearby prose.

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
case the *count* is measurable from committed data: **3 of 15** narrative
questions carry an earlier-filing label, pinned by a test.

That count is not the same as 3 wrong labels, and an earlier draft blurred the
two by calling it "20% known-contaminated". At least one of the three is a
standing Apple developer-ecosystem question of exactly the kind REPORT §1.2
argues an earlier filing *can* answer. Without committed chunk text the honest
statement is "3 labels point backwards in time"; how many are wrong is unknown.
The date rule now applied is correspondingly blunt — it rejects backward
propagation for standing risks too, which is stricter than the reasoning
warrants.

### Two evaluations that changed their own denominators

- `eval/gated_rerank.py` — a rerank exception hit `continue`, dropping the
  query from every metric *and* from threshold selection. A run that lost half
  its queries reported clean numbers over the survivors, indistinguishable from
  a complete run. Now records `n_attempted`, `n_failed`, the failing qids, and
  warns loudly. **This makes the loss visible; it does not stop it.** Metrics
  are still computed over survivors and the run still exits 0, so these remain
  complete-case results. Failing closed is the better behaviour and is not done
  here.
- `eval/run_retrieval.py` — `max(len(questions), 1)` turned an empty eval set
  into a full set of all-zero metrics and a **successful exit**, writing a
  plausible artifact recording a collapse that never happened. Now raises.
  (An earlier draft illustrated this with a mis-typed `--split`; there is no
  such option. The reachable route is an eval file that exists and is empty.)

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


---

## Third pass

The second pass converted the concept anchor from a filter to a diagnostic.
That was right about the leakage and wrong to stop there: the original false
positive came straight back, and nothing else had been put in its place.

### Numbers are now matched at boundaries, which does not leak vocabulary

Plain substring matching accepted a value *inside* a longer number: `"123"`
matched `"1234"`, `"6.16"` matched `"16.16"`, `"12,914"` matched `"112,914"`.
Every one of those produced a gold label for a figure the passage never states,
and unlike the concept anchor, fixing it costs no independence — a digit
boundary says nothing about the question's wording.

A bug inside that fix is worth recording, because it is the failure direction
that matters. The first version tested `before in ",."`, and `"" in ",."` is
`True` in Python: the empty string is a substring of everything. So a value at
the very start or end of a chunk read as *embedded in a longer number* and was
silently dropped. Real gold disappearing is worse than false gold appearing,
and only an explicit emptiness check prevents it.

### Losses in parentheses were finding no gold at all

Filings print a loss as `(1,234)`, never `-1,234`. `_formats` emitted only the
signed form, so every negative figure matched zero chunks and dropped out of
the eval set — skewing the question mix toward profitable years with nothing
recording that it had. Both forms are generated now.

### What boundary matching still does not fix

A revenue of $1,234,000 searches for `"1,234"` because filings report in
thousands, and *"the company employed 1,234 people"* contains exactly that
number standing alone. No string-matching rule separates those two facts. Only
a label that carries the concept can, which is what inline-XBRL element-to-DOM
mapping provides and why it remains the right fix. A test now pins this case
explicitly as known-unfixed rather than leaving it implied.

### Tests now drive `build()`

Every earlier test in `test_qrel_grounding.py` called `concept_supported`
directly, so deleting the diagnostic or restoring the withdrawn filter would
have left them green. `TestBuildEndToEnd` runs the builder with stubbed XBRL
facts and asserts on the labels it produces.

### Two crashes in the reporting path

`gated_rerank` raised `IndexError` on `logits[0]` when a rerank call returned
an empty-but-successful response — killing the whole evaluation rather than
counting the one failed query it was. And when *every* call failed, `gates` was
`{}` and printing the summary raised `KeyError`, hiding the actual failure
behind a crash. Both handled.

`r_at_1_ceiling` divided by every question while summing only those with gold
chunks, so an empty `gold_chunk_ids` quietly depressed the ceiling instead of
being reported. It raises now.

### Also corrected

- REPORT still published `0.075 → 0.226` after AUDIT.md had already retracted
  the `0.075`. Only the endpoint is quoted now.
- "Direction is unanimous" survived in README and REPORT next to the `11/4`
  concept split that contradicts it. Both now say consistent, unanimous across
  companies.
- REPORT's closing section called having a p-value the finding's "strongest
  state" while concept and company p-values are 0.12 and 0.25.

## Still open after three passes

- **No corpus manifest.** `retrieval_v3.json` still records no accessions,
  checksum, BM25 configuration, reranker model or git revision, and the corpus
  is gitignored. Nothing here is rerunnable; this remains the highest-value
  change in the repo.
- **Committed artifacts predate their producers.** `gated_rerank.json` has none
  of the failure fields the code now writes, and `dat_fusion.json` has no
  `configured_scorer_model` while the code names `deepseek-v4-flash-0731` and
  REPORT says `deepseek-v4-flash`.
- **`report_tables --check` only guards the numeric table.** Narrative, DAT and
  gate tables are unanchored and unchecked; wrong numbers there still pass.
- **Failure policy is inconsistent across harnesses**: `run_retrieval` raises,
  `fusion_sweep` raises `ZeroDivisionError`, DAT writes zero-like summaries,
  gated rerank reports complete-case metrics and exits 0.
- **Gated rerank is still complete-case.** Failures are recorded and warned
  about; they are not fatal, and the denominator still shrinks.
- **The narrative date rule ignores question type**, so standing-risk prose
  from an earlier filing is rejected along with genuinely impossible labels.
- `paired()` silently collapses duplicate qids; empty qrels still score as zero
  in `recall_at_k` / `ndcg_at_k`.
- The `concept_supported` diagnostic is itself imprecise: it calls
  `"Turnover | 1,234"` unsupported and the employee sentence supported when a
  concept word happens to be nearby. It is a rough signal, labelled as one.


---

## Fourth pass

### The boundary guard was rejecting sentence-final figures

The third-pass matcher treated *every* adjacent period or comma as numeric
continuation, so `"Total assets were 123."` and `"123, compared with last
year"` found no gold — and in filing prose, most figures close a sentence or a
clause. The dangerous direction again: real gold silently disappearing. A comma
or period now only counts as continuation when a digit sits on its far side
(`123,456` embeds `123`; `123.` does not), with tests for both directions.

The reviewer also re-flagged that the exact-collision case ("employed 1,234
people" as a standalone number) is back to being gold. That is the documented
position, not an oversight: the vocabulary filter that caught it was withdrawn
for injecting a lexical path into the one ablation this repo is about, the case
is pinned by a test as known-unfixed, and the fix that does not bias the eval —
inline-XBRL element-to-DOM qrels — needs the corpus.

### Wording drift

REPORT §1.2 still called the three backward-pointing narrative labels a "20%
known-contaminated fraction" after this file had already corrected that to a
count. Aligned: three labels an earlier filing cannot necessarily support, at
least one of which (a standing risk) an earlier filing *may* answer — "may",
not "can": no chunk text is committed, so answerability is established by
nothing in this repository.

## Still open after four passes

Unchanged from the third-pass list: no corpus manifest (still the
highest-value change), committed artifacts predating their producers
(`gated_rerank.json`, `dat_fusion.json` scorer-model field), `--check` covering
only the anchored numeric table, inconsistent failure policy across harnesses,
complete-case gated rerank, the blunt narrative date rule, duplicate-qid
collapsing in `paired()`, and empty qrels scoring zero.


---

## Fifth pass

One wording overclaim of the fourth pass, corrected: the standing-risk
narrative question was described as one an earlier filing "can" answer. No
chunk text is committed, so nothing here can establish that — "may" is the
supportable word, and the point it serves is unchanged (the blanket date rule
cannot tell events from standing risks). The punctuation boundary matcher
survived independent review with no defect found; a leading attached dot
(".123") was additionally excluded as decimal continuation, with tests. Known
remaining imprecision, noted so it is not rediscovered: `concept_supported`'s
backward window is ~400 chars minus the anchor's own length — tolerable for a
diagnostic that is recorded rather than enforced.
