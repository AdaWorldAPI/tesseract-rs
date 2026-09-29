# Archive metadata + AUTO matching wiring — SPEC v1 (5+3 council input)

**Status:** SPEC v1, Phase 0 of a 5+3 council
(`lance-graph/.claude/agents/5plus3-council.md`). Nothing is built.
**Parent:** `arm-discovery-and-coca-prior-v1.md`, piece 1. The miner shipped in
PR #101 (`tesseract-paperless::auto_match`). This spec wires it into the
archive.
**Why council-grade:** it spans `tesseract-paperless` (store, search, new
modules) and `tesseract-paperless-web`. It also decides what a *training label*
is, and a wrong answer there produces suggestions that look fine and are wrong,
silently.

## The problem in one paragraph

The archive stores no correspondent, document type or tags. The schema
(`tesseract-paperless/src/store.rs:107-123`) has hash, guid, filename, mime,
source, page count, confidence, `doc_ir_json`, timestamp and `spo_json`, and
nothing else. So `AutoMatcher::mine` has no labels to learn from, and the web
app has no way to assign one. Wiring AUTO in means building a small metadata
layer first: definitions, assignments with provenance, a term vocabulary, and a
suggestion/accept surface.

## 1. Frozen decisions

| # | decision | source |
|---|---|---|
| F1 | AUTO suggestions are never applied automatically. The miner's API is `&self` and returns values. | `auto_match.rs` module doc; PR #101 |
| F2 | Rule antecedents must be observable at ingest. Terms and field keys are cues, never targets. | `auto_match.rs` (`Cue`, `Layout::target`); PR #101 review P1 |
| F3 | Rules must clear lift ≥ `min_lift_ppm`. At most `MAX_BINARY_FEATURES = 256` tags + field keys + terms per mining run. | `auto_match.rs` (`AutoMatchParams`, `MAX_BINARY_FEATURES`); PR #101 review |
| F4 | `doc_ir_json` is the only stored copy of a document's text. The Tantivy index is disposable and reconciled on start. | `tesseract-rs/CLAUDE.md` § Paperless Wave B; `reconcile.rs` |
| F5 | lance / lancedb come from crates.io upstream, pinned per the lance-graph line. | lance-graph `CLAUDE.md` § carve-out; `tesseract-paperless/Cargo.toml` `store` feature |
| F6 | Storage and KV live in `tesseract-paperless` (+ `-web`). Recognition crates (`tesseract-core`, `-recognizer`, `-ocr`) gain no dependency. | `tesseract-rs/CLAUDE.md` § `tesseract-paperless` |
| F7 | S-8 matching semantics are paperless-ngx's `matching_algorithm`. | `matching.rs` |
| F8 | Axis encoding: correspondent / type ordinal `0` = none, `c + 1` = id `c`; tags are a set. | `axes.rs:17-21` |
| F9 | No model identifier in any artifact; the record is `tesseract-rs/CLAUDE.md` + the commit. | `tesseract-rs/CLAUDE.md` Iron rule 3; session rules |

## 2. Input inventory

| file:line | what it is | relevance |
|---|---|---|
| `tesseract-paperless/src/store.rs:50` | `const TABLE = "documents"` | the only table today |
| `store.rs:107-123` | `schema()` — 11 columns, no metadata | the gap |
| `store.rs:268-311` | `LanceStore::put` — `merge_insert(hash).when_matched_update_all()` | **re-ingest overwrites every column of the row** |
| `store.rs:224` | `LanceStore::connect` — opens or creates, migrates legacy columns (`:797`) | schema-evolution precedent |
| `store.rs:408` | `LanceStore::delete` | must cascade to assignments |
| `tesseract-paperless/src/search.rs:153-155` | Tantivy schema: `hash` STRING, `filename` TEXT, `text` TEXT (default tokenizer) | vocabulary source |
| `tesseract-paperless/src/reconcile.rs:25,37` | `ReconcileReport`, `ReconcileError` | extend for orphan assignments |
| `tesseract-paperless/src/matching.rs:86-161` | `MatchRule`, `CompiledRule`, `matching()` | S-8 source of `rule` assignments |
| `tesseract-paperless/src/auto_match.rs:83-151` | `ArchiveDomains`, `ArchiveRow`, `Target`, `Cue`, `Suggestion` | the miner's input/output |
| `auto_match.rs:488,575` | `AutoMatcher::mine`, `suggest` | the calls this spec wires |
| `tesseract-paperless/src/axes.rs:48-60` | axis `FieldId`s (correspondent, type, storage path, month, owner) | same axes, report side |
| `tesseract-paperless-web/src/routes.rs:24-30` | routes: `/`, `/upload`, `/documents`, `/documents/:hash`, `/documents/:hash/delete` | no metadata UI |
| `tesseract-paperless-web/src/state.rs:14,52,92` | `AppState`, `load`, startup `reconcile` | where rules are built |
| `tesseract-paperless-web/src/ingest.rs:130-260` | `ingest` — dedup → OCR → `store.put` → `search.index_document` | where S-8 runs |

## 3. Proposed resolution (committed)

**R1 — Definitions table `taxonomy`** (new Lance table in the same database as
`documents`). One row per `(kind, id)`: `kind ∈ {correspondent, document_type,
tag}`, `id: u32` dense per kind, `name: Utf8`, `match_algorithm: u8`
(`MatchAlgorithm::from_paperless`), `match_pattern: Utf8`,
`case_insensitive: bool`, `retired: bool`. Ids are allocated as
`max(id) + 1` per kind and **never reused**; deleting a definition sets
`retired` (so `ArchiveDomains` sizes stay monotonic and old rules never point
at a recycled id). Names are unique per kind among non-retired rows.

**R2 — Assignments table `assignments`**, keyed `(content_sha256_hex, kind,
id)`, with `source ∈ {manual, rule, auto_accepted}` and `assigned_at_unix_ms`.
A separate table, not columns on `documents`, **because `put` overwrites the
whole row on re-ingest** (`store.rs:305-309`): metadata stored there would be
wiped by an identical re-upload. Single-valued kinds (correspondent, type): at
most one non-retired assignment per document; assigning replaces. `delete`
cascades. `reconcile` gains a pass that removes assignments whose document is
gone.

**R3 — What trains AUTO.** Training labels are assignments with
`source ∈ {manual, auto_accepted}`. **`rule` (S-8) assignments are excluded
from training entirely**, neither cue nor target. Reasons: (a) AUTO would
relearn the S-8 pattern and report it back with inflated lift, adding nothing;
(b) when a user edits an S-8 rule, the old learned echo would linger;
(c) paperless-ngx keeps the two tiers separate too. `auto_accepted` counts,
because a human confirmed it.

**R4 — What a suggestion is keyed on.** At suggestion time a document's row
is built the same way as a training row: its `manual` + `auto_accepted`
assignments (none for a fresh upload), its terms, its field keys. The
"already assigned" check in `suggest` uses **all** sources, so AUTO never
suggests a target S-8 or a human already set.

**R5 — Term vocabulary from the Tantivy index, not a second tokenizer.**
Terms are read from the `text` field's term dictionary with document frequency
(DF), so they are exactly the tokens search uses. Vocabulary =
the top `MAX_BINARY_FEATURES − tags − field_keys` terms by DF, after dropping
terms with DF `< min_evidence` (cannot form a rule) or DF `> 50%` of documents
(stop-word-like, spends the cap). Per-document presence comes from each
vocabulary term's postings list, so no text is re-tokenized. Term ids are
ranks in that run's vocabulary; **the vocabulary and the mined rules are one
artifact, rebuilt together**, never mixed across runs. If `tags + field_keys ≥
MAX_BINARY_FEATURES`, mining is refused with `TooManyFeatures` and the UI says
so.

**R6 — Field keys.** The field-key domain is the distinct harvested field keys
across the archive (doc.v1 `fields[].key`), read from `doc_ir_json`. *Open
point for the council (question C2): whether `DocIr` retains field keys at
all.*

**R7 — Rules are derived and disposable.** Mined at startup (after
`reconcile`) and on demand (`POST /auto/mine`), held in `AppState` behind an
`RwLock`, never persisted. Same footing as the Tantivy index.

**R8 — Surface.** The document detail page shows suggestions: target name,
truth expectation, evidence count, and the cues that fired. "Accept" writes one
assignment with `source = auto_accepted`; "dismiss" writes nothing (not
persisted in v1). Manual assign/unassign and a minimal definitions page
(create / rename / retire, edit S-8 pattern) are added. Ingest runs S-8 and
writes `rule` assignments; it never writes AUTO results.

**R9 — Features.** New store tables live behind the existing `store` feature.
`tesseract-paperless-web` enables `matching` and `auto-match` on
`tesseract-paperless`.

**Order.** (1) R1 + R2 store tables with their tests; (2) R3/R4 row builder
(store → `ArchiveRow`s) as a pure function; (3) R5 vocabulary from Tantivy;
(4) S-8 at ingest; (5) R7 mining in `AppState`; (6) R8 routes + templates.

## 4. Non-goals

- COCA prior (plan piece 2): gated on its own measurement.
- Per-class table mining (piece 3): deferred to Wave D.
- Storage path, owner, permissions, month cue: not needed for AUTO v1.
- Persisting dismissals or rules: rules are derived (R7); dismissals wait for a
  measured need.
- Automatic application of suggestions: forbidden by F1.
- Tuning `min_evidence` / `min_confidence` / `min_lift` / `k`: stays policy pins
  until a real archive is measured.

## 5. Pre-registered gates

| # | gate | falsifier (red-then-green) |
|---|---|---|
| G1 | Re-ingesting an archived document keeps its assignments. | store assignments as `documents` columns → red |
| G2 | `rule` (S-8) assignments never appear in a training row, as cue or target. | include them in the row builder → red |
| G3 | The Tantivy-derived vocabulary's DF equals a brute-force DF over the archived text with the same analyzer, on a fixture. | off-by-one the DF filter → red |
| G4 | Viewing a document (GET) writes nothing: assignment count unchanged. | have the detail route accept the top suggestion → red |
| G5 | Accept writes exactly one assignment with `source = auto_accepted`. | write `manual` → red |
| G6 | Deleting a document removes its assignments; `reconcile` removes orphans. | skip the cascade → red |
| G7 | `tags + field_keys ≥ 256` → no mining, error surfaced; vocabulary never exceeds the remaining cap. | drop the cap check → red |
| G8 | A retired id is never reallocated. | allocate `count` instead of `max + 1` → red |
| G9 | `clippy -D warnings`, fmt, every `tesseract-paperless` feature line and `tesseract-paperless-web` tests green. | — |

## 6. Per-savant question sets

**S1 — prior art** (does it exist already?)
1. Does any crate in `tesseract-rs`, `lance-graph` or `OGAR` already define a taxonomy / tag / correspondent store or type? (`PRIOR-ART-AT` with path)
2. Does `ogar-doc-ir` or `ogar-vocab` already have a concept for tags / correspondents / document types that R1 should reuse instead of new kinds?
3. Does `axes.rs` (or any lance-graph report code) already assume an assignment representation that R2 must match?
4. Does Tantivy (the pinned `AdaWorldAPI/tantivy` rev) expose term-dictionary iteration with `doc_freq` and postings per term, as R5 assumes? Name the API.
5. Is there an existing "derived, rebuilt on start" pattern besides `reconcile` that R7 should reuse?

**S2 — iron rules / frozen decisions**
1. Does any R-item apply a suggestion without a human action (F1)?
2. Can a term or field key become a target anywhere in R3-R5 (F2)?
3. Does R5 keep the vocabulary within `MAX_BINARY_FEATURES` in every case (F3)?
4. Does any R-item store a second copy of document text (F4)?
5. Does any R-item add a dependency to `tesseract-core`, `-recognizer` or `-ocr` (F6)?
6. Does R8/R1 change S-8 semantics away from paperless-ngx (F7)?

**S3 — code truth** (is every file:line claim real?)
1. Is `store.rs:305-309` really `merge_insert(...).when_matched_update_all()`, so a re-`put` overwrites all columns of the row?
2. Is `store.rs:107-123` the full schema, with no metadata column?
3. Does the Tantivy `text` field use the default tokenizer, and is that analyzer reachable for G3's brute-force count?
4. Does `ogar_doc_ir::DocIr` (as stored in `doc_ir_json`) retain harvested field keys? If not, where do they survive? (R6 / C2)
5. Is `AutoMatcher::suggest`'s "already assigned" check driven only by `ArchiveRow`, so R4's all-sources rule needs the caller to pass all sources?
6. Does lancedb 0.39 support a second table in the same connection (`create_empty_table` / `open_table`) as `connect` does for `documents`?

**S4 — cascade impact** (what must change?)
1. List every file that must change for R1-R9, mandatory vs follow-up.
2. Which existing tests break or need updating (store, reconcile, web routes)?
3. Which CI lines in `.github/workflows/rust.yml` change?
4. Does the Dockerfile (`tesseract-paperless-web/Dockerfile`) or the Railway volume layout change?
5. Does `tesseract-rs/CLAUDE.md` need a new section, and does any existing claim become false?

**S5 — different views** (strongest alternative reading; no redesign)
1. Is excluding `rule` assignments from training (R3) the right call, or does it starve AUTO on archives where S-8 does most of the labelling? Name the consequence either way.
2. Is Tantivy DF the right vocabulary source, or does its tokenizer (no stemming, no stop list) make the vocabulary poor? Name the consequence.
3. Does "rules never persisted" (R7) create a user-visible inconsistency (suggestions change after a restart)?
4. What is the second-order effect of `auto_accepted` counting as a training label (a feedback loop that entrenches early mistakes)?
5. Is a separate `assignments` table the right shape, or would a `documents`-side column with a non-clobbering `put` be simpler? State the trade-off, don't redesign.

## Output contract (savants)

≤ 10 findings, each `(question #, verdict, file:line evidence, ≤ 2 sentences)`.
Verdicts: `CONFIRMS / VIOLATES / GAP / PRIOR-ART-AT / RISK`. No redesigns: a
savant that wants one files a single `RISK` and stops. Read-only.
