# Paperless substrate — one receipt lane, N borrowers, addressed by DocIr, rendered by a2ui

> **This supersedes the first draft of this file**, which was a paperless-ngx
> parity plan (flat lancedb row + a separate Tantivy text copy + a proposed
> blob store + a reconciler to heal the resulting duality). The operator
> rejected that framing directly: *"why blob, you didnt even care about
> making the design for paperless tokenized BPE deepnsm-v2 + tantivy +
> a2ui.rs ogar-doc-ir … dont dare to tell me what you cant do."*
>
> The rejection was correct and the miss was structural: this repo's own
> survey had already handed over the real architecture, and the first draft
> filed it under "Named boundaries — deliberately NOT built" instead of
> designing to it. OGAR's ingestion spine states the rule the draft violated
> — **NT-4**: *"text/key/value strings live out-of-line in value-slab
> stores keyed by classid+identity. An index over **that**, joined by
> `document_guid`, is a lens — never a parallel source of truth."* The
> `tesseract-paperless/src/token/` seam already **proves** a lens can be
> built this way (Tantivy indexes a *receipt handle*, never the text); the
> shipped `store.rs`+`search.rs` path does the opposite — four copies of the
> same bytes, no shared transaction — and that shipped path is the thing
> that should not exist, not the blob store I proposed to patch around it.
>
> Grounding for this revision: four Opus-tier readers, each assigned one
> surface, reading in full and citing `file:line`
> (run id `wf_00543ac0-20d`, 2026-08-24): the token seam
> (`crates/tesseract-paperless/src/token/`), the OGAR DocIr subtree canon
> (`ogar-doc-ir`, the W4-2 build spec, NT-4/S-5 verbatim), a2ui-rs's wire +
> render + RBAC + paint tiers, and deepnsm/deepnsm-v2 + the `bible_wave`
> whole-corpus precedent. Every claim below traces to one of those four
> reports or was checked directly. Where a claim is spec-only rather than
> built, this document says so — that is not deferral, it is the map.

---

## 0. The one-sentence architecture

**A document is a `DocIr` subtree (addresses come from OGAR's own tree —
page number, region reading order — never a second minted population). Its
text lives in exactly one place: a persisted BPE token receipt lane keyed by
that same address. Tantivy and deepnsm-v2 are borrowers of that lane through
a receipt-handle indirection, not copies of it. a2ui addresses the
structure — regions, fields, actions — by the same keys; document TEXT rides
the existing HTML render path until the wire is deliberately widened.**

Every wave below either builds a missing piece of this sentence or names,
honestly, an OGAR/a2ui prerequisite this repo cannot build for itself.

## 1. What already proves the sentence — read before designing anything else

This is not aspiration. `crates/tesseract-paperless/src/token/` already
runs the whole chain, end to end, on committed corpora, with 41 gates:

```
ogar_doc_ir::Region.text          (the ONE copy of the bytes — DocIr's own)
        │  docir::spans(ir, doc) -> SpanRef{ key: SpanKey{doc,page,reading_order}, text }
        ▼
TokenizerContract::try_encode(text)         -- ONE tokenization, counted
        │
        ▼
TokenLane::append(key, byte_from, contract, ids) -> TokenStreamReceipt
        │  (particles: Vec<[u8;12]>, receipts: Vec<TokenStreamReceipt>)
        ├──────────────────────────────┬──────────────────────────────┐
        ▼                              ▼                              ▼
ReceiptTokenizer (tantivy)      lexical::project (deepnsm-v2)   forward::windows
  term = "rcpt:<i>", NEVER        -> LexicalUnit{surface,..}      borrowed (ctx,next)
  the text; token_stream()          NO source bytes taken          pairs, zero alloc
  resolves the handle back          -> PaletteVocab::id(surface)
  into lane.view(receipt,           -> Tagged{id, pos}
  contract) and decodes             -> fsm::parse_to_spo(&[Tagged])
  surface bytes ON READ                -> Vec<Spo>   (deepnsm-v2, ONE dep)
```

Measured, not asserted: the DeepNSM-v2 gate asserts `passes == 0 && units >
0` — the lexical projection runs with **zero additional source
tokenizations** (`probe_token_seam.rs:513-525`). Tantivy's term dictionary
holds handles, never text (`seam_tantivy.rs:78-80,133-165`). A blind control
that flattens every PoS tag to `Noun` yields **zero** triples on the same
ids — proving the FSM gate is behavioral, not a type-checked no-op
(`probe_token_seam.rs:491-508`).

**And the shipped path in the very same crate does the opposite.**
`search.rs:119-121` indexes full `text` into Tantivy as `TEXT | STORED`;
`store.rs:114-116,122` writes `text` + `preview` + the whole `doc_ir_json`
into lancedb — four copies, no shared transaction, and this crate's own
CLAUDE.md already names the resulting archived-but-unsearchable crash
window as a consequence. The seam that proves NT-4 sits unused three
directories over from the code that violates it. **Wave B closes that gap
directly — it is the single highest-leverage change in this document.**

## 2. What is real, what is spec, per surface — read before assuming anything is buildable

### 2a. The token seam — a PROBE, feature-gated, in-RAM

Real and tested: `TokenizerContract::train/try_encode/try_encode_query/
decode` (`contract.rs`), `TokenLane` + `TokenStreamReceipt` (`lane.rs`),
`docir::spans` (the DocIr bridge, `docir.rs`), the Tantivy `ReceiptTokenizer`
(`seam_tantivy.rs`), the deepnsm-v2 `lexical::project` bridge (`lexical.rs`),
and a count-table `forward::windows` baseline (`forward.rs`). 41 gates, 18
disable-verified.

**Not real yet, and load-bearing:**
- **Nothing persists.** `TokenLane`/`TokenStreamReceipt`/`TokenizerContract`
  derive no `Serialize`. `canonical_bytes()` (the codebook hasher) is
  private with no inverse. The only constructor is `train(corpus, norm)` —
  reopening after restart today means re-tokenizing from scratch.
- **The Tantivy handle is positional, not content-addressed.**
  `handle(i)` formats the *index into `lane.receipts()`*
  (`seam_tantivy.rs:78-80`). A receipt carries a real key
  (`SpanKey{doc,page,reading_order}`) but nothing maps a stored handle back
  to it — a persisted index built today would break the moment append order
  changes.
- **The codebook is alphabet-closed per corpus.** `try_encode` refuses any
  byte outside the trained alphabet; a query containing an unseen byte
  silently returns an **empty** token stream via `unwrap_or_default()` — a
  zero-hit search with no error (`seam_tantivy.rs:145-152`,
  `contract.rs:276-285`). One contract per corpus does not scale to a
  growing archive: a new document's unseen byte cannot be encoded at all,
  and retraining mints a new `contract_id`, invalidating every stored id.
- **Re-spanning changes the tokens.** Measured: the KJV encoded whole yields
  346 tokens; the same bytes as 8 spans yield 354 — merges cross span
  boundaries. The span partition is part of what a receipt identifies.
- **The 12 bytes are 12 u8 ids, and the cap is real.** `VOCAB_CAP = 255`;
  Alice's 75 KB corpus already saturates it (255/255).
- **The projection's whitespace rule (ASCII) and deepnsm's (Unicode)
  agree on every committed corpus and would not on a non-breaking space.**

### 2b. `ogar-doc-ir` — the observation IR is shipped; the persisted subtree is a build spec

Shipped, serde-only, canon-free: `DocIr{version, source, geometry,
content_sha256, mime, pages, fields}` → `DocPage{number,width,height,
regions}` → recursive `Region{kind, bbox:BBoxRail, reading_order, text,
cells, children}`. **No node carries an id of its own** — `DocPage::number`
and `Region::reading_order` are the only ordering keys anything can join on,
and the token seam's `SpanKey` already uses exactly those, correctly (a past
draft "invented parallel source_id/span_id integers instead of reading them
— a second population wearing the document layer's job", per the token
crate's own recorded correction).

Also shipped: the *composition* IR — `compose::{DocNode, ObjectSlot,
ResolutionMode}` (arena-indexed, stable ids), `resolve::{ResolvedDoc,
DocObjectSource}`, and `project::{field_mask, masked_values}` behind feature
`classview` — which projects `ir.fields` through a `ClassView` +
`WideFieldMask` into exactly the `(position, value)` shape a2ui's own
`project_node` produces from a facet. **This is the real, working seam
between DocIr and an addressed render surface** — but it covers only
`ir.fields` (harvested key/value pairs). Regions, lines, and table cells
have **no mask projection at all** today.

**Spec-only, verified by direct grep — build before depending on it:**
- **`typed_field` (`0x080A`) is not minted.** `ogar-vocab`'s own table
  literally says "reserved for a future OCR-plane kind" at two locations.
- **No `ogar-doc` crate, no `DocRenderer`/`ProjectionRenderer` trait exists
  anywhere in OGAR.** The "fourth adapter" a2ui-rs's own CLAUDE.md names is
  doctrine, not an interface — there is nothing to implement or consume.
- **The three persistence `ActionDef`s do not exist.** `OCR_ACTION_NAMES`
  is const-asserted to exactly 14 names; `persist_document`/
  `read_document`/`reconstruct_document` are not among them.
  `recognize_document`'s own `produces` stops at `doc_json, fields` — the
  shipped pipeline halts exactly where the operator boundary always said it
  would.
- **S-5's actual text is not "bytes before address."** Verbatim: *"the
  addressable record and its bytes must not be able to diverge."* Its
  mechanism in the reference case is a **transaction** (row + file placement
  commit together); "bytes-first" is the recommendation only for the
  no-transaction fallback. This matters for Wave D below.
- **`doc.v1`'s `checks` array is dropped by `ogar-from-docv1`.** Verified
  both sides: `structured.rs` emits it; `V1Field{key,value,bbox,conf}` never
  parses it, and `TypedField` has no field to hold it. Also dropped: page
  `quality`, `plain_text`, `fields_map`, all per-line typography metrics.

### 2c. a2ui-rs — real addressed rendering; no text on the wire yet

Real and tested: the LE frame wire (`NodeDelta`/`ActionInvoke`, `ogar-a2ui-
frame`), server-side RBAC-by-projection (`surface ∩ role`, fail-closed on an
empty grant), the wasm client's L1/L2 nested resolution, and a 4-skin paint
tier (`Form/Flow/Grid/Tile`) that takes only `&[FieldView] + &[ActionRef] +
Viewport + Skin` — **producer-agnostic**, proven against a wholly synthetic
non-MedCare class. `Skin::Tile` places a surface at an (x,y) read from a
u8:u8 facet rail — which is **exactly** `ogar-doc-ir::BBoxRail`'s encoding
(two rails, const-asserted 2/4 bytes). Every click is already a Klickweg
telemetry edge lowering to an `ActionInvocation` with no new vocabulary.

**The one structural blocker, stated plainly: a2ui's value lane is 12
bytes — one u8 per field position, 0..11. There is no path down the wire
for a document's TEXT.** `FACET_LEN = 12` (`render_stream.rs:58`); the
client renders `state.facet[i].to_string()` — a string produced from ONE
byte. A `FieldView.value` today is a stringified number, not a body of
text. This is not a missing feature at the edges — it is the wire's whole
shape. Rendering `doc.v1` through a2ui *as it stands* would show twelve
numbers per node.

**Also unbuilt:** the widened `enum FieldView{Text,Badge,Table,ObjectSlot,
…}` (still the 4-field struct); nesting via `child_links` is populated only
client-side (`link_child`), no frame or server API ever sends a slot link,
and it is capped at 12 children per node by the same facet ceiling; there is
**no field-write/edit frame** anywhere (`FrameKind` is a closed 2-variant
vocabulary; correcting an OCR'd value has no representation); the mask
ceiling is inconsistent (`WideFieldMask`/`FieldView.position` are u8-bounded
at 256, the wire's own `mask_words` are u32-native); `ClassRbac::field_mask`
in `lance-graph-contract` is still narrow (`FieldMask`, 64-bit) **and
fail-open** (`FieldMask::FULL` default) — a2ui-rs sidesteps this by never
consuming `ClassRbac` at all, which means the charter's own retype is a real
open item, just not one this design depends on.

### 2d. deepnsm-v2, not deepnsm v1 — the currently-wired reasoning uses the WRONG one

Two different crates, different id spaces, different `Pos` enums:
- **`deepnsm` (v1)** — the one `tesseract-ogar/src/reasoning.rs` currently
  wires per-document. It **owns a tokenizer**
  (`Vocabulary::tokenize(&str)`), splits words itself, and cannot consume a
  token lane without re-reading text. Has the documented homograph bug
  (context-free PoS from a frequency table).
- **`deepnsm-v2`** — one dependency (`lance-graph-contract`), deliberately
  tokenizer-free and string-free. `PaletteVocab::id(&str) -> WordId` is an
  exact-match lookup; `fsm::parse_to_spo(&[Tagged])` consumes `(WordId,
  Pos)` pairs only. **This is the crate the token seam already drives, with
  zero library changes needed** — proven above.

**PoS is the missing ingredient, and it is load-bearing, not optional.**
Neither the lane nor deepnsm-v2 supplies a tagger; the probe (mirroring
`bible_wave`) hand-builds a side table from COCA CSVs. Its own blind control
(all-`Noun`) proves the dependency is real, not a type signature.

**The lane's unit is a span (Region), not a sentence.** The probe injects
exactly one `Pos::Stop` per span — meaning SPO triples chain **serially
across every real sentence boundary inside a region** unless a sentence
splitter injects `Stop` at the actual boundaries. `tesseract-ogar::
sentences::assemble_sentences` is the stack's only sentence splitter, and it
takes a `DocPage` (the typed recognizer output), not a `DocIr` and not a
`TokenStreamView` — there is currently **no path from a persisted lane back
to a sentence boundary.**

**Cross-document accumulation does NOT require `lance-graph-planner`.**
`deepnsm_v2::belief::BeliefArena` — `observe`/`revise_at`/
`close_transitive`, Copula-gated so verbs never falsely compose — lives in
the same one-dependency crate the lane already drives. This corrects
CLAUDE.md's own AS-IS BOUNDARY entry, which drew the planner boundary one
layer too early (the boundary is the five NARS **tactics** + `TruthValue` +
stance/insight — not the belief arena itself). Two real caveats: `Stamp` is
a 64-source bitset that folds by modulo past 64 documents (silently
downgrades revision to choice — conservative, never wrong, but an archive
will exceed 64 sources); there are **two `BeliefArena`s** with identical
export names over different truth types (`NarsTruth` here, `TruthValue` in
the planner) — a design must name which one and never let a call site drift
between them.

**`bible_wave.rs` is the whole-archive precedent, and it is bigger than
"whole-corpus SPO."** Its real stage list: split into a semantic unit (it
had verses for free; an archive must mint one — ingest order or a Lance
version, since neither the lane nor DocIr supplies one) → PoS side-table →
FSM → `TemporalStream` keyed by that unit → a **bounded** derivation arena
(`DERIV_HORIZON`, because transitive closure over hub predicates is O(N²)
and does not terminate unbounded) → a shape router → basin self-codes → an
evidence composite. A design should mirror this stage list, not re-derive
it, and must ship the horizon from day one.

## 3. The waves

Sequencing rule: **build the lens before building anything that needs a
lens.** Wave A (persistence) blocks everything; Wave B (kill the
duplication) is the highest-leverage single change and should land next;
C/D/E are then largely independent.

### Wave A — the lane persists, keyed correctly, closes its two structural holes

**OBJECTIVE.** `TokenLane`/`TokenStreamReceipt`/`TokenizerContract` survive
a restart, and the two probe-honest gaps (positional handles, alphabet
closure) get real answers rather than silently shipping as-is.

**DESIGN.**
- Serialize the three types. The contract's codebook (`expand`, `base_of`,
  `merges`, `strings`, `byte_len`) is small (≤255 entries) — a plain
  bincode/serde blob keyed by `contract_id` is sufficient; no new storage
  system needed. The lane's particles/receipts/docs are flat `Vec`s already
  — the same shape.
- **Fix the handle.** Change the Tantivy-indexed term from `rcpt:<i>`
  (positional) to a **content address**: `rcpt:<hex(content_sha256)>:
  <page>:<reading_order>` — i.e. the `SpanKey` the receipt already carries,
  turned into a stable string. Resolution becomes a lookup by that key into
  the lane's own receipt-by-key index (a `HashMap<SpanKey, usize>` is
  sufficient; the lane already knows its receipts). This is the fix the
  design survey named as missing and it is a small, local change — not a
  new architecture.
- **The alphabet-closure decision, made explicitly rather than inherited.**
  Two lawful options, and the wave must pick one and say why: (i) a
  **per-archive contract**, retrained periodically as new byte alphabets
  appear, with every prior receipt's ids remapped through an explicit
  migration pass (expensive, correctness-preserving); or (ii) a **superset
  alphabet trained once over a representative multi-document corpus**
  up front (the Alice/KJV precedent already shows this saturates around
  255 ids on ordinary English prose — so a real archive's first-ingest
  training corpus should be sized deliberately, not accidentally, and the
  cap's saturation should be logged, not silently hit). Recommend (ii) for
  the first archive-scale build; log every `try_encode` refusal so
  saturation is visible before it becomes a search-silently-empty bug.
- **The empty-token-stream-on-unknown-byte trap gets a signal, not a
  swallow.** `ReceiptTokenizer`'s query path must distinguish "genuinely no
  match" from "byte outside the trained alphabet" — surface the latter as a
  reported condition (a warning field on the search response), never a
  silent empty result.

**Falsifiers.** Round-trip: train, encode a corpus, persist, restart, reopen
→ every receipt resolves to the same surface bytes as before persistence
(disable: skip persistence, assert the test fails on restart). Handle
stability: append two spans, delete-and-reinsert the lane's internal order,
assert the SAME content-addressed handle still resolves (disable: revert to
positional handles, assert this goes red). Saturation signal: train past
255 distinct bytes worth of alphabet, assert a logged/reported saturation
event exists (disable: remove the log call, assert silence).

### Wave B — kill the shipped duplication; the receipt lane becomes THE authority

**OBJECTIVE.** `search.rs` and `store.rs` stop each holding their own copy
of the text. One authority; Tantivy is a lens over it, per NT-4, using the
mechanism Wave A just made durable.

**DESIGN.**
- `LanceStore`'s schema drops the `text` and `preview` columns as
  independently-written data. What it keeps: `content_sha256_hex`,
  `document_guid`, structural metadata (`page_count`, `mime`, timestamps),
  and — this is the point — a reference to the token lane's document
  interning key (`lane.intern_document(content_sha256)`'s `u16`, or the
  hash itself, which is already the row's own key). `doc_ir_json` stays
  (it is the structural authority, not a text copy); `preview` is
  **derived on read** from the lane via `lexical::project` /
  `TokenStreamView::tokens()`, not stored.
- `search.rs`'s Tantivy schema drops its `TEXT | STORED` text field and adds
  the receipt-handle field from Wave A. Indexing a document means: for each
  `SpanRef` from `docir::spans`, tokenize once (already the seam's job),
  append to the lane, index the handle. **No second tokenization, no second
  text copy** — this is the exact chain in §1, now with a persisted lane
  and stable handles underneath it.
- **Snippets**, which the token-seam survey names as a real consequence
  (Tantivy's built-in highlighter re-tokenizes *stored text*, which no
  longer exists in the index): serve the snippet by resolving the matched
  receipt's handle back to the lane, decoding the surrounding tokens'
  surface bytes via `TokenStreamView`, and highlighting the matched term
  in that decoded string. This is a few lines against Wave A's `view()` —
  not new architecture, but it must be built, and it is the honest price of
  removing the duplication.
- The two-store consistency problem the first draft tried to heal with a
  reconciler **shrinks to nothing here, structurally**: there is only one
  place text lives (the lane), and lancedb + Tantivy both hold references
  into it rather than copies of it. A crash mid-ingest can still leave the
  lane ahead of one reference or the other, but there is no *content* to
  reconcile — only a missing/stale reference, which is a cheap, bounded
  repair (re-derive the reference from the lane, which is authoritative),
  not a text-recovery problem.

**Falsifiers.** Ingest a document, kill the process before the Tantivy
handle write, restart, reconcile the reference → search finds it with the
correct decoded snippet (disable: skip the reconcile step, assert search
misses it). Delete a document → its lane entries become unreachable from
either store (disable: skip clearing the reference, assert a stale receipt
is still findable). A malformed/unknown-byte query on a real archive →
reports the saturation signal from Wave A, not a bare empty result.

### Wave C — deepnsm-v2 off the lane, replacing the v1 per-document wiring

**OBJECTIVE.** `tesseract-ogar/src/reasoning.rs`'s current per-document
path (deepnsm v1, re-reads text, has the homograph bug, invisible to the
lane) is superseded by driving deepnsm-v2 directly off the persisted lane —
the chain in §1, already proven by the probe, now run in production.

**DESIGN.**
- Replace the `SentenceReasoner` body: instead of `Vocabulary::tokenize(&
  sentence.text)` (v1, re-reads), consume the lane's `TokenStreamView` for
  the span, run `lexical::project` → `PaletteVocab::id` → `Tagged` →
  `fsm::parse_to_spo`. This requires the PoS side table (§2d) — build it
  once from the same COCA CSVs `SentenceReasoner::from_vocab_dir` already
  loads; the data is already on disk, only the tagging step changes.
- **Fix the span/sentence unit mismatch, not paper over it.** Before
  handing a span's tokens to the FSM, run `assemble_sentences`-shaped
  sentence-boundary detection over the span's *text* (available from
  `Region::text` — the lane's own upstream source) to find real sentence
  ends, and inject `Pos::Stop` at those boundaries rather than only once
  per span. This is the fix the design survey names as necessary to avoid
  fabricated cross-sentence triples — build it as a small pre-pass, not a
  redesign of the sentence splitter.
- `sentence_nars_truth`'s formula stays (it is this module's own honest
  construction, not a transcode); the input signal (mean OCR confidence +
  parse coverage) is unaffected by the tagger swap.

**Falsifiers.** The FSM's blind control (all-`Noun`) must still yield zero
triples through the new wiring (proves the tagger, not the plumbing, is
doing the work — the exact gate the probe already runs). A two-sentence
span with real sentence-ending punctuation must yield triples that do NOT
chain the second sentence's subject from the first's object (disable: skip
the Stop-injection pre-pass, assert cross-sentence fabrication reappears).

### Wave D — cross-document belief accumulation, without the planner

**OBJECTIVE.** The AS-IS BOUNDARY's stated gap — "reasoning over what was
recognized is absent by design" — narrows honestly, using what §2d shows is
actually reachable at the one-dependency tier.

**DESIGN.**
- Wire `deepnsm_v2::belief::BeliefArena` (not the planner's namesake) as an
  archive-scoped accumulator: each document's SPO triples from Wave C
  become `observe` calls, stamped by a source id derived from the
  document's interned lane id (`u16`, already ≤ 64 for a while, then
  degrading conservatively per §2d's `Stamp` finding — log when the archive
  crosses 64 distinct sources so the degradation is visible, not silent).
- **Mint the version tick `bible_wave` got for free.** No natural monotone
  tick exists for a document archive; use ingest order (a simple
  monotonic counter) as the `TemporalStream` version, documented as a
  policy choice, not a measurement — re-pin if a better tick (a Lance
  version, once the lane persists through lancedb) turns out to matter.
- **Ship the derivation horizon from day one** (`bible_wave`'s own
  lesson): `DerivationArena::derive_transitive_capped` with an explicit
  cap, never the uncapped variant, and assert only soundness (every
  premise resolvable, no cycles) at any archive size — never completeness.
- This wave explicitly does NOT touch `lance-graph-planner`. The five NARS
  tactics, `TruthValue`, and the stance/insight surface stay named,
  deliberately out of scope, exactly where the AS-IS BOUNDARY already put
  them — this wave corrects *how far the free tier reaches*, not the
  boundary's location.

**Falsifiers.** Two documents making the same claim → `observe` revises
(NARS revision), not choice (disable: force disjoint stamps unconditionally,
assert revision never fires). A derivation run at archive scale terminates
within the horizon and reports only soundness, never completeness (disable:
remove the cap, assert an unbounded genealogy-shaped input does not
terminate in a reasonable bound — this IS the falsifier, run it once
deliberately to see the failure, then confirm the cap prevents it).

### Wave E — a2ui addresses the structure; text stays on the render path until the wire is widened

**OBJECTIVE.** State exactly what a2ui *can* do for this archive today
(real, immediate value) without pretending the wire carries document text
that it structurally cannot yet carry.

**DESIGN — what ships now, using only what §2c shows is real:**
- **Structural addressing.** A document's `DocIr` regions project through
  `ogar_doc_ir::project::masked_values` (already built, already RBAC-aware
  via `WideFieldMask`) into the same `(position, value)` shape a2ui's
  `project_node` expects. `TypedField`s (harvested key/value pairs) are the
  first-class content this path already carries end to end — an invoice's
  IBAN, date, amount as addressed, RBAC-projected fields, live today
  without new wire work.
- **`Skin::Tile` for page/region navigation.** A region's `BBoxRail` (two
  const-asserted u8:u8 rails) is *already* the encoding `Skin::Tile` reads
  for placement — so a page-overview surface (click a region, jump to it)
  is a rendering wave, not a wire wave. Every click is free Klickweg
  provenance ("who opened which region, when") with no new audit type.
- **Actions by ordinal.** Open / reprocess / delete / export each become an
  `ActionDef` ordinal on the document's class (once §2b's minted `document
  0x080B` class carries them — see Wave F) resolved server-side, never a
  URL — the T2 rule this repo already enforces elsewhere.
- **Quality and loss ARE renderable — because a status is one byte.** The
  same §2c ceiling that rules text out rules these in. A document's
  confidence must reach a person as a *reading* (clean / check / suspect),
  never as a raw `mean_conf` — the usability argument
  (`AdaWorldAPI/paperless-rs` `docs/USABILITY.md`, from MedCare-rs's
  `LabValue::classify` → `LabFlag`: a clinician reads "normal", not "13.2",
  and this repo's own measurements put `mean_conf 99.47` on a page at
  `CER 0.6154`). A status enum discriminant is **1 byte**; a raw `f32` conf
  is not; text is not. So the shape usability wants is exactly the shape the
  facet lane already carries, with no value-slab extension and no Wave F
  dependency. Two fields, appended to `ogar-vocab::document()` at positions
  7-8 (currently 7 attributes at 0..6, so nothing moves — the class-view
  append-only rule): `quality: DocQuality` (1-byte enum) and `dropped: u8`
  (knowingly-discarded content, **saturating at 255** — the signal is "look
  at this", not an exact inventory). Both are source-agnostic: a DOM retina
  reports `dropped = 0`. Both are `Badge`-shaped for the day the upstream
  `enum FieldView{Text,Badge,Table,ObjectSlot,Geometry,…}` widening lands —
  consumed then, never pre-empted now (**T1**). The class change itself is
  an OGAR prerequisite, listed in Wave F.
- **Document TEXT is explicitly NOT rendered through the a2ui wire in this
  wave.** It is served the way it already is — HTML, through the existing
  `Projection{delta,html}` path a2ui-server already ships, which needs zero
  new types. This is not a workaround; it is the correct scoping given
  §2c's finding that the wire's value lane is 12 bytes: pretending
  otherwise would either truncate text to twelve stringified numbers or
  require inventing the value-slab wire extension a2ui's own CLAUDE.md
  already names as future, unbuilt work (Wave F).

**Falsifiers.** A region's rail-derived Tile placement matches its `DocIr`
bbox within the rail's own quantization (disable: swap the rail axes,
assert placement is visibly wrong). An unauthorized field is genuinely
absent from the projected surface, not merely hidden client-side (disable:
remove the `WideFieldMask` intersection, assert the field reappears). An
action invocation resolves by ordinal and rejects an out-of-range one
(disable: skip the range check, assert an invalid ordinal panics instead of
refusing cleanly). `quality` and `dropped` round-trip through a single facet
byte unchanged, and positions 0..6 still resolve to the same seven field
names after the append (disable: widen `dropped` past `u8` or add a
`DocQuality` variant past 255, assert the value is truncated rather than
silently wrong — this is the regression the one-byte finding exists to
prevent).

### Wave F — named OGAR/a2ui prerequisites this repo does not build

Recorded so a future session does not re-derive them, and so no wave above
silently assumes they exist:

- **Mint `typed_field 0x080A`** in `ogar-vocab` — a small, scoped OGAR PR.
  Blocks: any field-node addressing beyond `ir.fields`' current flat
  projection.
- **The `persist_document`/`read_document`/`reconstruct_document`
  `ActionDef`s** and, underneath them, a real `ogar-doc` crate implementing
  the W4-2 GUID-keyed subtree (one node per page/region/figure/field,
  reusing `PAGE_LAYOUT 0x0807`/`PAGE_IMAGE 0x0808`, cells as facets not a
  new class — the spec already answers "one node per what"). Waves A–E do
  **not** depend on this landing; they use the lane + DocIr's own nesting
  directly. This crate would let the archive's *structural* address space
  (not just its text) survive as an OGAR-canonical subtree rather than an
  opaque `doc_ir_json` blob — a real upgrade, not required for anything
  above.
- **The `DocRenderer`/`ProjectionRenderer` trait** — currently doctrine, no
  code. a2ui, askama, and Typst today share the `FieldView` struct and
  `ogar_doc_ir::project`'s masking with no common trait between them; a
  design that says "implement `DocRenderer` for a2ui" would be inventing
  the trait, not consuming one.
- **The a2ui value-slab wire extension** — the mechanism that would let
  document TEXT travel down the `NodeDelta` wire instead of the HTML
  fallback Wave E uses. `render_stream.rs` already reserves the design
  space ("a future value-slab render path WILL emit them") but nothing
  implements it. This is the one piece whose absence is a genuine capability
  gap rather than a scoping choice — worth a dedicated a2ui-rs arc once
  Waves A–E are live and the HTML fallback's limits are actually felt.
- **`ClassRbac::field_mask`'s retype** to `WideFieldMask` in
  `lance-graph-contract` (currently narrow + fail-open). a2ui-rs's own
  session-role RBAC sidesteps this today; it becomes load-bearing only if
  a future wave routes document RBAC through `ClassRbac` instead of a2ui's
  own session mask.
- **Append `quality` + `dropped` to `ogar-vocab::document()`**
  (`lib.rs:4288-4307`) at positions 7-8 — the two status fields Wave E
  renders. A small, scoped OGAR PR in the same shape as the
  `typed_field 0x080A` mint above, and **append-only**: positions 0..6 are
  the shipped basis and must not move (`ogar-class-view/src/lib.rs:34-48`).
  It mints no classid — `DOCUMENT = 0x080B` already exists
  (`ogar-vocab/src/lib.rs:1960`); this adds attributes to an existing
  class. Blocks: Wave E's quality/loss bullet only — every other Wave E
  deliverable is independent of it.

## 4. Corrections this design makes to the standing record

- **CLAUDE.md's AS-IS BOUNDARY drew the reasoning boundary one layer too
  early.** `deepnsm_v2::belief::BeliefArena` is reachable at the one-
  dependency tier, not only the planner. Append a dated correction there
  when Wave D lands, per this repo's append-only convention — do not edit
  the original entry.
- **The shipped `search.rs`/`store.rs` text duplication is not a tolerable
  interim shape; it is the thing NT-4 already forbids, sitting three
  directories from the seam that proves the right answer.** Wave B is not
  an optimization — it is closing a violation the crate's own probe already
  disproves.
- **`lance-graph-arm-discovery` remains correctly out of scope** (unrelated
  tabular rule-mining, already corrected in CLAUDE.md) — nothing in this
  design revisits that.
- **A competing plan was started against the crates instead of against this
  one, and is now subordinate to it (2026-08-25).** `a2ui-rs`
  `.claude/plans/document-fields-to-a2ui-render-v1.md` began as a
  standalone "wire the document fields into a2ui" plan. It duplicated Wave
  E and got three things wrong that §2c already answers: it assumed the
  wire could carry a status *string* (the lane is 12 bytes, one `u8` per
  position); it reasoned carefully about `FieldMask`-64 vs `WideFieldMask`
  while missing that the **12-byte value lane** is the ceiling that
  actually binds; and it proposed hand-building a field basis that
  `ogar_doc_ir::project::masked_values` + `Skin::Tile`/`BBoxRail` already
  provide. It also filed Wave F's `typed_field 0x080A` mint as a surprise
  precondition rather than the known prerequisite it is. That file is now
  an **addendum subordinate to this Wave E**, carrying only the one-byte
  finding folded in above. The generalizable miss: *design against the
  plans, not only against the crates* — the answer was already written
  down.

## 5. Execution model (unchanged from the operator's ruling this session)

- **Opus — filigree planning.** Authoring each wave's verbatim worker spec
  when it fans out, sequencing, adjudicating disable-run results, all
  central gating (`cargo fmt`/`clippy -D warnings`/scoped tests run ONCE,
  `--release` for anything touching real recognition), every
  `git commit`/`push`.
- **Sonnet — grind.** Bounded per-file implementation against a written
  spec, disjoint files only; never claims compilation or test results it
  did not run; never runs cargo or git write commands.
- **Haiku — churn, contract-gated.** Pre-written commands only (a scoped
  `-p` build/test run, a specified disable-arm re-run, tag-file collection);
  never authors, decides, or edits any file but its own tag-file.
- See `CLAUDE.md`'s dated 2026-08-24 amendment for the full three-tier
  contract this supersedes.
