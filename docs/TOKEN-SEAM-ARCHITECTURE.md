> **Home note (2026-08-24).** This document and its probe were written in
> `AdaWorldAPI/paperless-rs`, which cannot be pushed to. They now live here, in
> `crates/tesseract-paperless` (feature `token`). Where the text below says
> "this crate" it means that crate; where it names `paperless-*` crates, read
> them as the `kv` / `intake` / `token` modules. The measurements are unchanged
> and were re-run in this repo: **ALL 41 GATES GREEN**, identical numbers.

# The token seam — one tokenization receipt, many borrowed consumers

**Status:** bounded architecture + probe. `PROBE-TOKEN-SEAM-1`, 37 gates, 9
disable-runs verified red-then-green. Nothing here is a production carrier.

**The question, exactly as posed.** Can Tantivy + one versioned BPE
`TokenStreamView` become the shared lexical intake seam for document retrieval,
DeepNSM-v2 reasoning and forward autocomplete, while structured evidence enters
lance-graph directly and Polars disappears from the online path unless
measurement proves it is genuinely needed?

**The answer.** Yes for the seam, measured end to end. The three consumers ran
off ONE tokenization per span and added ZERO further tokenizations — not by
discipline but by construction: Tantivy is handed a receipt handle instead of
text, and the DeepNSM projection's function signature has no source parameter.
Neither Tantivy nor DeepNSM-v2 needed a single line changed. Polars is not in
the online path and never was — there is nothing to remove, which is a weaker
and more honest headline than the one the framing invited.

What is NOT settled is listed as eight named gaps in §7. Two of them are
load-bearing: the OCR boundary hands over no byte offsets to attach a span to
(G1), and the 8-bit vocabulary lane saturates at 75 KB of ordinary English
(G6). Neither is a defect in the seam; both are the next thing to measure.

---

## 1. The shape

```
     retina: tesseract-rs (pixels) │ spider-rs (DOM)
                             │
                             ▼
              ogar_doc_ir::DocIr   (AUTHORITATIVE)
        content_sha256 · pages · regions · reading_order
                             │
                             │  a span IS a Region; its text is Region::text
                             │  exactly one pass per span
                             ▼
                   versioned BPE contract
              (codebook + normalisation rule + contract id)
                             │
                             ▼
        resident lane:  [u8;12] particles  +  receipts
                             │
        ┌────────────────────┼────────────────────┐
        │ borrowed           │ borrowed           │ borrowed
        ▼                    ▼                    ▼
    Tantivy              DeepNSM-v2          forward predictor
  terms + positions   WordId → PoS → SPO    (context, next) windows
   DERIVED, deletable   DERIVED, deletable    EPHEMERAL, owns nothing
```

and, in parallel and never through this seam:

```
   structured rows → typed intake (arm-discovery: FeatureSpec + Vec<Vec<u32>>)
                   → SPO / evidence / provenance
```

The two meet at shared `(source_id, span_id, byte_from)` identity — not at a
shared table.

| layer | owns | never |
|---|---|---|
| `docir` | the span population, read from `ogar-doc-ir` | mints an identity |
| `contract` | the codebook and its identity | reads the lane |
| `lane` | resident particles + framing | owns text |
| `lexical` | the DeepNSM projection | reads the source |
| `seam_tantivy` | the index seam | owns offsets |
| `forward` | the prediction input surface | owns the sequence |

Each "never" is proven by a gate, not asserted — §8 lists which disable turns
each one red.

## 1b. The identity is the document layer's, not this crate's

An earlier cut of this crate minted `source_id: u32` and `span_id: u32`. That
was a second population wearing the document layer's job, and `ogar-doc-ir`
already answers all three questions a receipt has to ask:

| the receipt needs | `ogar-doc-ir` supplies |
|---|---|
| WHICH document | `DocIr::content_sha256` — sha256 of the ORIGINAL bytes |
| WHICH span | a `Region`, addressed by `(DocPage::number, Region::reading_order)` |
| the span's text | `Region::text` — each region owns its own canonical text |

Three consequences, and the second is the one that removes a gap this report
previously carried:

- **`content_sha256` is a PER-ACQUISITION dedup key, not a cross-retina
  identity.** The crate's own docs correct its plan's first sketch on exactly
  this point: a scan and an HTML page of the same invoice have different bytes
  and therefore different hashes. For a *tokenization* receipt that is the
  right reading — you tokenize bytes, so different bytes are a different
  tokenization. Cross-retina convergence is a facts question
  (`converges_on_facts`) and is not this seam's business.
- **Byte offsets are REGION-LOCAL, so the "no offsets at the OCR boundary" gap
  largely dissolves.** The boundary supplies no PAGE-wide offset and does not
  need to: the region owns its text, and `ogar-from-docv1::region_text` is
  where tesseract's `leading_space`-aware join already happens.
- **The seam is source-agnostic for free.** `docir.rs` contains no line that
  knows which retina produced the IR, so a crawled page presents the same span
  population as a scan.

The receipt interns the 32-byte hash **once per document** and carries a `u16`
index. At these span sizes a receipt is already about a third of the resident
bytes; stamping the hash on every one would have more than doubled that for no
addressing gain.

Two things `docir::spans` deliberately does NOT do, both gated: a `Figure`
contributes nothing (it has no text), and a `Table`'s cells are **not**
flattened into the token stream — a cell is typed `(row, col)` data the
structured path consumes, and pouring it into text destroys exactly the
structure that path exists to read.

## 2. What was measured

Two committed, real, hermetic corpora. `kjv-genesis-scene` is the same in-tree
text `PROBE-TOKEN-BPE-GEOMETRY-1` (#1012) used, carried verbatim so the two
probes are comparable; `alice-paragraphs` is Project Gutenberg's *Alice's
Adventures in Wonderland*, carried from `tantivy/benches/alice.txt`.

### lane

| corpus | bytes | spans | tokens | ratio | uniq ids | particles | resident B | particles/span p50/p95/max | continuation |
|---|---|---|---|---|---|---|---|---|---|
| kjv-genesis-scene | 1 125 | 8 | 352 | 3.20× | 135 | 32 | 864 | 4 / 8 / 8 | 87.5 % |
| alice-paragraphs | 75 513 | 300 | 36 994 | 2.04× | 245 | 3 214 | 55 400 | 8 / 30 / 43 | 100 % |

Encode cost, in #1012's honest unit (merge-table probes, never wall time):
85 748 and 7 879 322.

The `uniq` column is the number of distinct ids that APPEAR in the lane, and
that is not the vocabulary size — a distinction worth stating because an
earlier draft of this report conflated them. The trained table is **full at
255 of 255** on Alice (and still full on the whole 170 KB file); 245 of those
ids occur. On the KJV fixture the table is **180 of 255** — there the CORPUS,
not the cap, set the size, which is what #1016's own record of the fixture
says. The two corpora sit on opposite sides of that line, and that is the
interesting fact rather than either number alone.

**The resident lane is ~73 % of the source text, not a fraction of it.** For
Alice the particles alone are 38 568 B (51 % of source), the 300 receipts add
16 800 B, and the interned document hash adds 32. At these span sizes the FRAMING, not the payload, is where the bytes
go: a 56-byte receipt against 12-byte particles is 30 % of the resident total
at paragraph granularity and 54 % at verse granularity. #1012 could not see
this — it had no receipt.

**Continuation is the norm, confirmed at a second scale.** #1012 measured every
verse overflowing one particle; here every one of 300 paragraphs does, with a
p95 of 30 particles. Any production design budgets continuation from the start.

### the three consumers

| consumer | added source tokenizations | added query tokenizations | evidence |
|---|---|---|---|
| Tantivy (both term modes, both corpora) | 0 | 0 | indexed value is `rcpt:<n>`, never text |
| DeepNSM-v2 projection | 0 | 0 | `project()` takes no source parameter |
| forward windows | 0 | 0 | slices pointer-identical to the lane |

Whole-run totals: **313 source tokenizations** — 8 + 300, one per span, plus 5
by deliberate fixtures inside the contract and framing gates — and **1 query
tokenization**, counted on a separate counter because a query is different
bytes and folding it into one number would make the claim a lie.

### Tantivy, driven by the receipt

| index | segments | terms | docs | index tokens | bytes | phrase hits |
|---|---|---|---|---|---|---|
| kjv / surface | 1 | 137 | 8 | 354 | 2 913 | 1 |
| kjv / token-id | 1 | 137 | 8 | 354 | 2 334 | 1 |
| alice / surface | 1 | 244 | 300 | 37 149 | 82 904 | 2 |
| alice / token-id | 1 | 247 | 300 | 37 149 | 81 897 | 2 |

Three things fall out of that table.

- **The term dictionary is bounded by the vocabulary, not by the corpus.** 247
  terms for 75 KB, because a term is a BPE token. That is a different retrieval
  regime from word indexing and it is a trade, not a win: recall rises (a query
  fragment matches inside a word) and per-term selectivity falls.
- **Surface terms silently merge distinct ids.** Alice indexes 247 ids as 244
  surface terms: three ids decode to a string another id also decodes to (BPE
  can reach the same string by different merge paths). Id-mode keeps them
  apart, and is also smaller on disk — two-hex-digit terms beat surfaces.
- **Positions are the receipt's positions.** Asserted directly: the analyzer
  registered on the index yields `0..token_count` in order for the receipt
  under test. There is one segmentation and the index consumed it.

**Tantivy structurally cannot become the owner of offsets.** In this fork the
indexer's `index_text` reads `Token::text` and `Token::position`, uses
`position_length` transiently, and reads `offset_from`/`offset_to` NOWHERE
outside its own tests (`src/postings/postings_writer.rs`). Byte offsets are
consumed only by snippet generation, which re-tokenizes the STORED text at
query time (`src/snippet/mod.rs:211`). That is the demarcation this
architecture wants, handed to us by the index's own design — and it has a
price, recorded as G7: under the handle design Tantivy's built-in highlighter
has nothing to highlight, so snippets must be served from the canonical text
through the receipt.

The route NOT taken is worth recording. `PreTokenizedString { text: String,
tokens: Vec<Token> }` allocates a `String` per token, and `segment_writer.rs`
deep-clones the whole boxed value before indexing it — about `4 + 2N`
allocations for N tokens. That is precisely the materialised token-object
population the root memory law forbids. The custom tokenizer reuses ONE `Token`
buffer, the way Tantivy's own `SimpleTokenizer` does.

### DeepNSM-v2, projected from ids alone

| corpus | lexical units | resolved to a WordId | OOV | tokens/unit p50/p95/max | straddling tokens | SPO triples |
|---|---|---|---|---|---|---|
| kjv-genesis-scene | 225 | 186 | 17.3 % | 1 / 4 / 7 | 30 | 4 |
| alice-paragraphs | 13 108 | 10 586 | 19.2 % | 2 / 6 / 15 | 58 | 671 |

**The cardinality is not 1:1 in either direction, measured.** A lexical unit
spans a median of 2 tokens and up to 15; and 58 tokens carry the start of more
than one unit — a single BPE token straddling a word boundary. So BPE sequence
identity and the DeepNSM word coordinate are different id spaces, and the
projection is a real function rather than a relabelling. Nothing in the seam
assigns a `WordId` to a BPE token; the ids that reach the FSM are DeepNSM's
own, resolved from the reconstructed surface against its own frequency-ranked
vocabulary.

**DeepNSM-v2 needed no change.** Its library is already tokenizer-free:
`parse_to_spo(&[Tagged])` consumes `(WordId, Pos)` pairs and touches no string.
The `split_whitespace`/`normalise` logic lives only in its examples. That is
the single most load-bearing fact in this report — the seam is adoptable
because the consumer was already shaped for it.

The OOV figure is against an 18 559-word academic vocabulary (20 845 COCA rows,
2 286 of them duplicate surface forms). ~19 % is what an academic word list
does on Victorian prose; it is a property of the vocabulary, not of the seam.

### the forward surface

| corpus | best order | top-1 | k=1 / k=2 / k=3 | positions scored | unseen context | positions with a DeepNSM coordinate |
|---|---|---|---|---|---|---|
| kjv-genesis-scene | 1 | 0.0 % | 0.0 / 0.0 / 0.0 | 81 | 19 | 95.8 % |
| alice-paragraphs | 2 | 26.4 % | 13.3 / 26.4 / 25.4 | 5 093 | 399 | 72.0 % |

This is a **counting baseline, not a language model**, and it is here to prove
the input surface rather than to predict anything. Two readings are legitimate:
the windows are slices INTO the resident particle array (asserted
pointer-identical, and a disable that copies them turns the gate red), and the
sequence has enough structure for order-2 context to double order-1 accuracy.
The KJV row is 0.0 % on 81 scored positions — at 6 training verses that is
indistinguishable from chance and is reported rather than dropped.

Of the three candidate input representations named in the brief: **(A)** the
token id is present by construction; **(B)** the DeepNSM `(basin, identity)`
palette coordinate is available for 72–96 % of token positions, so the hybrid
**(C)** is CONSTRUCTIBLE. Which one a trained model should prefer is **not
measured here and no claim is made** — that needs training, a real corpus, and
its own probe.

## 3. The ten answers

**1. Can one versioned BPE tokenization receipt drive Tantivy directly?**
Yes, and with no patch to Tantivy. A custom `Tokenizer` reads the resident lane
and yields borrowed ids; the indexed field value is a receipt handle, so the
index is never handed the source at all. Indexing 308 documents across four
index builds added 0 tokenizations of either kind. Positions in the index ARE
the receipt's positions. Costs: sub-word term semantics, and no built-in
snippets (G7).

**2. Can the same receipt project into DeepNSM-v2 without raw-source
re-tokenization?** Yes. `project()` takes a borrowed view and no source bytes —
re-reading the source is unavailable, not merely avoided. It reads the
contract's per-id surface table, which is at most 255 short byte strings.
DeepNSM-v2's library is unchanged. The one thing the seam is forced to
duplicate is the COCA part-of-speech mapping — and the reason is a deliberate
earlier deletion, not an oversight (G3).

**3. Can the same ordered token ids serve as forward-prediction input?** Yes,
as borrowed slices of the resident particles. Order-2 context reaches 26.4 %
top-1 against 13.3 % at order 1 on held-out spans with a counting baseline. The
predictor owns nothing and is dropped.

**4. What production framing/codebook metadata is missing from #1012's
fixture-scale `[u8;12]` result?** Five things, all now specified and gated:
- a **contract id** — a digest over the canonical serialisation of the table
  AND the normalisation rule id. Without it a stored `u8` is not a weak
  reference, it is a wrong one: decoding corpus A's ids under corpus B's
  codebook returned 551 bytes of garbage where 1 126 were expected.
- a **framing triple** `first_particle + particle_count + token_count`. There
  is no shipped token continuation mechanism anywhere in lance-graph; the
  nearest precedent in shape, `rail_geometry::RailCarving::AxisSlab { reg, cont
  }`, chains one register to one continuation and caps at 24 levels — under
  the measured p50 of 4 particles, so it does not fit. Honestly stated, TWO
  framings are lawful: `particle_count` alone bounds the run and a PAD scan
  inside that bound is exact because PAD is RESERVED (cost: one vocabulary
  slot); or `token_count` carries the length in 4 bytes and frees the slot for
  a full 256-id alphabet. What is unlawful is inferring the end from padding
  with no bound — measured, a lane-wide PAD scan overshoots receipt 0 by 10
  tokens straight into receipt 1, and every span whose length is a multiple of
  12 is that case rather than a corner one.
- a **per-id decoded-length table**, which is what makes byte offsets a DERIVED
  quantity. The receipt stores no offset column at all; an offset is a prefix
  sum taken during the walk, checked here against a decode-the-prefix ground
  truth.
- a **per-id surface table**, which is what lets every downstream projection
  run without the source.
- **span identity — taken from `ogar-doc-ir`, not minted**: `content_sha256`
  interned once per document, plus `(page, reading_order)` per span, plus a
  region-local `byte_from`. The measured consequence is that at these span
  sizes the receipt is ~30 % of resident bytes, so the receipt's own layout
  matters more than the particle's — which is also why the hash is interned
  rather than stamped per span.

**5. Does any online step still require Polars?** No — and the honest form of
that answer is that it never did. A sweep of nine checkouts found **zero**
occurrences of `polars` in any manifest or source file; every `DataFrame`
mention is prose (a "Kuzu/Polars pattern" note, an unimplemented R-bridge
sketch, and Tantivy's own columnar test naming). `tesseract-paperless` and
`tesseract-rs` — the two repos that constitute this online path — declare none
of `arrow`, `datafusion`, `lance`, `lancedb`. The online steps are: parse →
normalise → tokenize → index → project. Not one is a groupby, a join, a window
function, or a columnar expression. Where a DataFrame WOULD genuinely be
reached for — structured evidence — the typed path already exists and is not
tabular algebra either: `lance-graph-arm-discovery` takes `Dataset { spec:
FeatureSpec, rows: Vec<Vec<u32>> }`, category-index rows against a schema, with
its own module doc stating "no one-hot float vector and no embedding". The
remaining honest gap is that "parse a table" has no in-tree answer at all (no
`calamine`, no `csv` anywhere), and the answer to that is a parser, not a
DataFrame (G8).

**6. What remains authoritative?** In order, and the order is enforced rather
than described:
`ogar_doc_ir::DocIr` (authoritative — it owns the document's identity, its span
population and each span's canonical text) → `contract` (authoritative for what
an id MEANS) → `receipt + lane` (exact and reconstructible; byte-identical
decode on both corpora) → `Tantivy index` (derived, deletable) → `DeepNSM
projection` (derived, deletable) → `model state` (ephemeral, owns nothing).

**7. Can Tantivy be deleted and rebuilt without semantic loss?** Yes. It holds
terms and positions derived from the lane, stores a handle rather than text,
and persists no offsets. Rebuilding is a re-walk of the lane at 0 source
tokenizations. What is lost on deletion is query latency — and, under this
design, the built-in highlighter (G7).

**8. Can the LSTM be deleted and rebuilt without token or document loss?**
Yes, trivially: it owns nothing. Windows are borrowed, the predictor is built
from them and dropped. Weights, hidden state and logits never enter the lane.

**9. Can the token stream be consumed entirely as borrowed/resident SoA views
after intake?** As BORROWED views, yes — every consumer here takes
`&TokenStreamView` and the forward windows are pointer-identical to the lane's
storage. As a lawful **resident SoA lane**, not yet: this lane is a probe-local
`Vec<[u8;12]>`. A production lane must either implement `SoaEnvelope`
(`ColumnDescriptor`, `ENVELOPE_LAYOUT_VERSION = 2`, `verify_layout`,
`mailbox_owner`) or land as a new `ValueTenant` — whose enum currently has 16
variants and none for tokens. That is the boundary, stated rather than crossed
(G2).

**10. Does the combined path reduce representation changes versus the
Pandas/Polars + Elasticsearch style pipeline?** Yes, and it is countable rather
than rhetorical.

| | representation changes | tokenizations of the source |
|---|---|---|
| classic: text → DataFrame → analyzer → embeddings → ANN → graph adapter | 5 | ≥ 2 (ingest + the engine's own analyzer) |
| this seam: text → ids → {terms, lexical units, windows} | 1 owned + 3 derived views | **1**, asserted per span |

The measured figure is 313 source tokenizations for 308 spans plus 5 by
deliberate fixtures, and 0 added by any consumer.

## 4. Falsifier verdicts

| # | falsifier | verdict |
|---|---|---|
| F1 | Tantivy requires independent tokenization | **REFUTED** — 0 tokenizations across 4 index builds; the value is a handle |
| F2 | codebook identity not frozen/versioned | **REFUTED** — contract id digests the table AND the rule; `T-CONTRACT-RULE` |
| F3 | continuation cannot reconstruct boundaries | **REFUTED** — adjacent receipts decode independently and concatenate exactly |
| F4 | DeepNSM cannot project without re-tokenizing the source | **REFUTED** — the projection has no source parameter |
| F5 | the DeepNSM vocabulary is weakened to accommodate BPE | **REFUTED** — crate unchanged; the ids that reach the FSM are its own |
| F6 | the LSTM needs a second canonical population | **REFUTED** — windows pointer-identical to the lane |
| F7 | embeddings become canonical identity | **NOT REACHED** — no embedding, ANN or vector store exists here; the cam96 codebook is ABSENT |
| F8 | Tantivy becomes the only owner of positions/offsets | **STRUCTURALLY IMPOSSIBLE** — the indexer never reads offsets at all |
| F9 | deleting Tantivy loses canonical state | **REFUTED** — the index is derived from the lane |
| F10 | structured evidence flattened into text | **NOT EXERCISED** — no structured corpus is present; MecCog is absent |
| F11 | a DataFrame survives in the online path | **REFUTED** — zero occurrences, workspace-wide |
| F12 | BPE ids or tokenizer content leak into a class address | **REFUTED** — `T-FENCE` over 730 lines of non-comment code |
| F13 | source reconstruction is not exact | **REFUTED** — byte-exact on both corpora, both framings |
| F14 | offsets drift between Tantivy, DeepNSM and the receipt | **REFUTED** — positions asserted equal; spans re-normalise to the source; Tantivy holds no offsets to drift |
| F15 | one source is tokenized more than once | **REFUTED** — 1 per span, asserted per corpus |
| F16 | the project expands into a generic LLM/embedding architecture | **HELD** — no model, no embedding, no vector database, no token service |

F7 and F10 are honest non-results, not passes. F16 is a boundary this document
is responsible for keeping.

## 5. What the seam costs, stated plainly

- The resident lane is **~74 % of the source text** at a 255-id vocabulary —
  not a compression story. Compression fell from 3.18× on 1 KB to 2.03× on
  75 KB as the vocabulary saturated.
- The Tantivy index for Alice is **82 904 B against 75 514 B of source** —
  110 %. An index is an index.
- Framing overhead is **30–54 % of resident bytes** at paragraph and verse
  granularity respectively.
- Encode cost is quadratic-ish in the naive form used here: 7.9 M merge-table
  probes for 75 KB, because encoding applies all 255 merges as full passes. A
  production encoder uses a priority queue; this one was carried unchanged from
  #1012 deliberately, so the two probes' numbers stay comparable.

None of these are arguments against the seam. They are the numbers a production
decision needs and did not have.

## 6. What this does NOT change

- **HHTL is address geometry; BPE is tokenization.** #1012 measured that a BPE
  merge tree is pair-ENCODABLE but is not a lawful radix prefix partition
  (same-depth tokens are prefixes of one another). Nothing here revisits that.
- **Content never travels in a class address.** The contract id is a FIELD on
  the receipt. `T-FENCE` greps the library's own non-comment source for
  `classid`/`class_id` and finds none.
- **Tokenization is not span-local.** Encoding the KJV text whole yields 346
  tokens; encoding it as 8 spans yields 354. Merges cross span boundaries, so
  the span partition is part of what the receipt identifies. Re-spanning a
  document changes its ids and therefore its index and its projection — a
  property to pin, not a bug (G5).

## 7. The named gaps

**G1 — RESOLVED by reading `ogar-doc-ir`; what remains is much smaller.** This
report previously carried "the OCR boundary supplies no byte offsets" as the
gap that blocked real documents. It is not one. `doc.v1` carries no page-wide
offset, but the perceptual IR does not need one: a span is a `Region` and
`Region::text` is its own canonical text, so an offset is region-local and
`ogar-from-docv1::region_text` is where the `leading_space`-aware join already
lives. What actually remains: a SUB-region span (half a paragraph) needs a
non-zero `byte_from`, which the receipt already carries and no producer yet
emits.

**G2 — no resident SoA carrier.** The lane is a probe-local `Vec`. Lawful
options: implement `SoaEnvelope`, or mint a new `ValueTenant` (16 variants
today, none for tokens). The measured framing overhead says the receipt's
column layout is the thing to design first.

**G3 — there is no shipped, callable part-of-speech surface anywhere, and the
reason is a decision already taken.** `coca_pos`, `archaic_pos` and `normalise`
are byte-identical in BOTH of DeepNSM-v2's examples, and the doc comment above
them explains why: *"`deepnsm_v2::lexicon` was deleted after an audit found the
planner's `insight_coca_read` already grounds this in the master COCA
`lexicon.tsv` (with lemmatisation)."* Re-adding the module would re-litigate
that audit, so this seam did not — it restates the minimal tagger and says so.

But the grounding the deletion relied on does not hold for a lean consumer, and
that is the finding rather than the complaint. `insight_coca_read` is itself an
**example binary** in `lance-graph-planner` — not a library API — and that
crate pulls `serde`/`serde_yml`/`tokio`/`ndarray`, so it is outside the BBB
dependency set a document-intake binary can carry. Its master `lexicon.tsv` is
absent from this checkout. So a consumer has exactly three options, none of
which is "call the shipped one": depend on the planner (violates the barrier),
restate the minimal tagger (what this seam does, ~20 lines, example-grade), or
re-open the deletion with this consumer as the new evidence. The third is the
right one if a second consumer ever needs it; one consumer restating twenty
lines is not yet a case.

**G4 — the semantic half is unexercised.** `cam96_codebook.bin` /
`cam96_codes.bin` are release assets, absent here, so DeepNSM's palette256²
DISTANCE was not run. Only the lexical/grammar half was.

**G5 — re-spanning changes the tokens** (see §6). Needs a pin.

**G6 — the 8-bit lane saturates, and this is the biggest unknown.** Alice used
247 of 255 ids on 75 KB; the vocabulary was full and compression was already
falling. The canon's answer is the hi byte of each `(8:8)` pair as a PAGE lane —
two separate bytes, never a widened `u16` — which is untested. That is the next
probe, and until it runs no scale claim should be made.

**G7 — snippets.** Under the handle design Tantivy's highlighter has no text to
highlight; highlighting must be served from the canonical text through the
receipt.

**G9 — the projection's whitespace rule is ASCII where DeepNSM's is Unicode.**
`project()` walks bytes and splits on `u8::is_ascii_whitespace`; DeepNSM splits
on `char::is_whitespace`. They agree on every corpus here — the probe counts
non-ASCII whitespace and measures **0** across both — but they would disagree
on a non-breaking space or an en-quad. The clean fix needs char-boundary
tracking across token boundaries, since a BPE token can split a multi-byte
character; until a corpus that exercises it exists, the divergence is bounded
by that count rather than closed.

**G8 — no table parser in-tree.** Neither `calamine` nor `csv` appears in any of
the nine checkouts. "Parse a table" is genuinely unanswered — and the answer is
a parser, not a DataFrame, because parsing a table does not require a DataFrame
to own cognition.

## 8. The disable table

Every gate below was verified RED under the named change and GREEN without it.
A gate nobody has tried to break is not evidence.

| disable | gate that went red |
|---|---|
| index the raw text instead of the receipt handle | `T-TANTIVY-NOSRC` |
| receipt loses `particle_count` (PAD-scan to lane end) | `T-RECON` |
| `byte_len` off by one on one id | `T-OFFSET` |
| contract id ignores the normalisation rule | `T-CONTRACT-RULE` |
| lexical unit span end off by one | `T-DEEPNSM-SPAN` |
| a class address enters the token path | `T-FENCE` |
| `view()` stops checking the contract id | `T-CONTRACT-GATE` |
| the projection re-encodes the source | `T-DEEPNSM` |
| forward windows copy instead of borrowing | `T-FORWARD` |
| drop the CRLF normalisation (the original bug) | `T-CORPUS` |
| every lexical unit claims a single-token span | `T-DEEPNSM-CARD` |
| the PoS-blind control keeps the real tags | `T-DEEPNSM-FSM` |
| build the phrase from receipt 1, assert receipt 0 | `T-TANTIVY-PHRASE` |
| `spans()` renumbers by position instead of reading `reading_order` | `T-DOCIR-KEY` |
| `intern_document` never dedups | `T-DOCIR-KEY` |
| the IR hashes something other than the source bytes | `T-DOCIR` |
| `spans()` flattens table cells into the token stream | `T-DOCIR-SPANS` |
| a text-less container emits an empty span | `T-DOCIR-SPANS` |

The last four exist because an independent vacuity audit of the finished probe
found five holes, and all five were real:

- `T-CORPUS` asserted byte counts and never the SPAN count — so the CRLF bug
  that collapsed 300 paragraphs into one span would have re-passed it.
- `T-DEEPNSM-CARD` asserted `q50 >= 1`, which is true of any lexical unit that
  exists at all, while its message claimed the cardinality was *measured*. It
  now asserts the two facts it actually claims: a maximum above 1, and a
  non-zero count of tokens straddling a word boundary.
- `T-DEEPNSM-FSM` asserted only that some triple came out — a statement about a
  type signature, not about behaviour. It now runs a **PoS-blind control**: the
  same word ids with every tag flattened to `Noun` must produce a DIFFERENT
  triple count, which is what makes "the FSM consumes `(WordId, Pos)`" a
  measured claim.
- `T-TANTIVY-PHRASE` checked `stored.starts_with("rcpt:")`, which is
  unconditionally true of every document in that index. It now asserts equality
  with the receipt the phrase was taken from.
- `project()` splits on `u8::is_ascii_whitespace` where DeepNSM splits on
  `char::is_whitespace`. Real divergence, unexercised here — now BOUNDED by
  measurement (G9) rather than by hope.

**Three of the disables were themselves wrong first, and that is the part worth
keeping.** The third is from the `ogar-doc-ir` re-cut: the first version of
`T-DOCIR-KEY` compared each receipt's key against the SAME `spans()` call it
was validating — an implementation checked against itself, which stayed green
when `spans()` was changed to renumber by position. It now walks the `DocIr`
independently, and the fixture's `reading_order` is deliberately `2i+1` rather
than the positional index, so "reads the field" and "renumbers by position" are
distinguishable at all. A fourth disable (flattening table cells) initially
found nothing for a different reason — neither text corpus contains a table or
a figure, so no fixture could express it. That is a coverage gap, not a sound
gate, and it was closed by building a three-region page that has both. (a) A first attempt at the framing disable replaced the
`token_count` trim with a PAD scan *inside the receipt's own particle range* and
stayed green — correctly, because within a bounded run a scan for a RESERVED id
is exact. The gate's prose had over-claimed; it now states both lawful framings
and the disable targets the actual unlawful one. (b) A first attempt at the
codebook disable removed the rule id from the digest and stayed green, because
on mixed-case text the two rules train different tables anyway. The gate now
trains both rules on an already-lowercase corpus, where the tables are
identical, and additionally asserts they behave differently on mixed-case
input. **A knob that does not bind on the fixture is not a disable, and a
fixture's SHAPE is part of a test's coverage.**

A third instance was pure apparatus: an early disable batch reported "no
failure" for six changes in a row because the probe binary path was wrong and
nothing ran. A null result is a claim about the measurement apparatus until
proven otherwise.

## 9. Next rungs, in order

1. **G6 — the paged vocabulary.** Does `hi:lo` as `(page, id)` extend the
   alphabet without widening a lane, and what does compression do at 1 MB?
   Until measured, no scale claim.
2. **A real retina.** This probe builds its `DocIr` from text; the next one
   should take `ogar-from-docv1`'s output on an actual scan and a
   `spider_doc_ir` crawl of the same content, and check that both present the
   same span population shape to the tokenizer.
3. **G2 — the lawful lane.** `SoaEnvelope` or a tenant, with the receipt column
   layout designed against the measured 30–54 % framing overhead.
4. **A real forward arm.** The seam supplies the input; whether representation
   A, B or C wins is untested and needs training, not assertion.
5. **F10 — a structured-evidence corpus** to exercise the parallel typed path
   and the shared span identity where the two meet.
