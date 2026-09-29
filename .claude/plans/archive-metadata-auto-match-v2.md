# Archive metadata + AUTO matching wiring — DRAFT v2 (5+3 council, Phase 2)

**Status:** DRAFT v2. It consolidates the five savants' findings on SPEC v1
(`archive-metadata-auto-match-v1.md`, kept unchanged). The three reviewers see
this document only. Nothing is built.
**Parent:** `arm-discovery-and-coca-prior-v1.md`, piece 1. The miner is
`tesseract-paperless::auto_match` (PR #101).

## The problem in one paragraph

The archive stores no correspondent, document type or tags. The documents
table (`tesseract-paperless/src/store.rs:107-123`, 11 columns) has no metadata
column. So `AutoMatcher::mine` has no labels to learn from, and the web app
cannot assign one. Wiring AUTO in needs a small metadata layer first:
definitions, assignments with provenance, a review flag, a term vocabulary, and
a suggest/accept surface.

## 1. Frozen decisions

| # | decision | source |
|---|---|---|
| F1 | AUTO suggestions are never applied without a human action. | `auto_match.rs` module doc; PR #101. **A deliberate difference from paperless-ngx**, which merges its classifier's prediction into the matching list and applies it at consumption (`paperless-ngx/src/documents/signals/handlers.py:126-128`). |
| F2 | Rule antecedents must be observable at ingest; terms and field keys are cues, never targets. | `auto_match.rs` (`Layout::target`, `:389-397`); PR #101 review |
| F3 | Rules clear lift ≥ `min_lift_ppm`. `mine` refuses more than `MAX_BINARY_FEATURES = 256` tags + field keys + terms. | `auto_match.rs:497-503` |
| F4 | `doc_ir_json` is the only stored copy of text; the Tantivy index is disposable. | `CLAUDE.md` § Paperless Wave B |
| F5 | lance / lancedb from crates.io upstream. | lance-graph `CLAUDE.md` carve-out |
| F6 | Storage lives in `tesseract-paperless` (+ `-web`); recognition crates gain no dependency. | `CLAUDE.md` § `tesseract-paperless` |
| F7 | S-8 matching semantics are paperless-ngx's. | `matching.rs` |
| F8 | Axis encoding: ordinal `0` = none, `c + 1` = id `c`; tags are a set. | `axes.rs:49-52` |
| F9 | No model identifier in any artifact; the record is `CLAUDE.md` + the commit. | `CLAUDE.md` Iron rule 3 |
| F10 | *(new, S1)* Correspondent, document type and tag are **consumer-local filing metadata**, never minted OGAR classids. | `OGAR/docs/OGAR-DOC-INGESTION-SPINE.md:274-277` (NT-2) |

## 2. Input inventory (verified by S3; one row corrected)

| file:line | what it is |
|---|---|
| `tesseract-paperless/src/store.rs:50` | `const TABLE = "documents"` |
| `store.rs:107-123` | `schema()`: 11 columns, no metadata |
| `store.rs:268-311` | `put`: `merge_insert(hash).when_matched_update_all()` rewrites all 11 columns |
| `store.rs:224-251` | `connect`, holds `db`; **legacy-column migration at `:240-250`** (`drop_columns` `:249`). *(v1 cited `:797`, which is a test.)* |
| `store.rs:408` | `delete` |
| `search.rs:153-155` | Tantivy schema: `hash` STRING\|STORED, `filename` TEXT, `text` TEXT. `text` is **not stored**. Default analyzer = `SimpleTokenizer` + `RemoveLongFilter(40)` + `LowerCaser`, no stemmer (`tantivy/src/tokenizer/tokenizer_manager.rs:60-67`), reachable via `Index::tokenizer_for_field` (`tantivy/src/index/index.rs:552`). |
| `reconcile.rs:75` | `reconcile(store, index) -> ReconcileReport { indexed, removed, unreadable }`, compiled only under `store` + `search` (`lib.rs:67-69`) |
| `matching.rs:84-161` | `MatchAlgorithm`, `MatchRule`, `CompiledRule`, `matching()`. `MatchAlgorithm::Auto` compiles to never-match. |
| `auto_match.rs:83-151, 488, 575-596` | `ArchiveDomains`, `ArchiveRow`, `Target`, `Cue`, `Suggestion`, `mine`, `suggest` (cues and "already assigned" both come from the one `ArchiveRow`) |
| `ogar-doc-ir/src/lib.rs:175,275` | `DocIr.fields: Vec<TypedField>` keeps `key: String`; `ogar-from-docv1/src/lib.rs:253-255` copies it |
| `lancedb-0.39.0/src/connection.rs:533,552`, `table.rs:1569` | several tables per connection; `merge_insert` takes composite keys |
| `tesseract-paperless-web/src/routes.rs:24-30, 547` | routes; `document_delete` at `:547` |
| `web/src/state.rs:14, 41, 52, 86-99` | `AppState` (no `RwLock` today; `Arc<Semaphore>` at `:41`), `load`, startup `reconcile` |
| `web/src/ingest.rs:130-260` | `ingest`: `preflight` returns `Duplicate` before OCR/`put` (`:136-148`), then OCR → `put` → `index_document` |
| `paperless-ngx/src/documents/classifier.py:336-390` | trains on non-inbox documents only; a label counts only if its definition's `matching_algorithm == MATCH_AUTO` |
| `paperless-ngx/src/documents/signals/handlers.py:123-133` | an assigned correspondent is kept unless `replace`; several matches → the first is taken, the count logged |

## 3. Resolution (committed)

**R1 — Definitions table `taxonomy`.** One row per `(kind, id)`: `kind ∈
{correspondent, document_type, tag}`, `id: u32` allocated as `max(id) + 1` per
kind and never reused, `name`, the three `MatchRule` fields (`match_algorithm`,
`match_pattern`, `case_insensitive`; S1: reuse `MatchRule`, no parallel type),
`retired: bool`. Defaults follow paperless-ngx: `case_insensitive = true`,
algorithm ANY (`models.py:70-76`). **Excluded on purpose:** the tag tree,
colour and `is_inbox_tag` (the review flag, R3, plays the inbox role). Kinds are
consumer-local (F10).

**R2 — Assignments table `assignments`**, keyed `(content_sha256_hex, kind,
id)` (composite `merge_insert` key), with `source ∈ {manual, rule,
auto_accepted}` and `assigned_at_unix_ms`. A separate table rather than
columns on `documents`. **The exposure is store-level, not the web path**
(S3): the web path returns `Duplicate` before `put`, but `put` rewrites the
whole row, so any direct `put`, a concurrent double upload or a future writer
would wipe metadata held there. Single-valued kinds hold at most one
assignment per document. **The cascade lives inside `LanceStore::delete`**
(S4), not in a route. Orphans are removed by a new
`reconcile_assignments(store)` under the `store` feature alone (S3: the
existing `reconcile` is `store`+`search` only), called from `AppState::load`.

**R3 — What is a target, and what trains it.** *(Rewritten; resolves S5 Q1
and Q4, S2 Q1 and Q6, S3 Q5 and S1 Q3.)*
- **Targets are only definitions whose `match_algorithm` is AUTO**, as in
  paperless-ngx (`classifier.py:366-389`). Every other definition belongs to
  S-8 and is never an AUTO target.
- **Training rows are reviewed documents only** (R3a below), again as in
  paperless-ngx (non-inbox, `classifier.py:336-338`). An unreviewed upload with
  no tags is not evidence that it has none.
- **Labels for AUTO targets** come from `manual` and `auto_accepted`
  assignments. S-8 cannot produce them: an AUTO definition compiles to
  never-match (`matching.rs`). So v1's "exclude `rule` from training" is no
  longer a special case. The S-8 echo that v1 R3(a) worried about cannot occur,
  because an S-8 definition is never a target.
- **Cues:** terms, field keys, and every assignment of a *non-AUTO* definition,
  whatever its source (S-8 results are observable at ingest, F2). An AUTO
  definition's own assignment is also a legal cue for other targets.
- **"Already assigned"** now comes out of the one `ArchiveRow` correctly. AUTO
  targets can only carry manual or accepted assignments, and those are in the
  row. So **no API change to `suggest` is needed**. The caller does keep only
  suggestions whose target is an AUTO definition.
- **The `auto_accepted` feedback loop** (S5 Q4, VIOLATES v1). An accepted
  suggestion is a human label, so it trains. That is the same loop paperless-ngx
  has. It is bounded two ways. (1) Negative evidence exists now: a reviewed
  document that lacks an AUTO tag is a real negative for it, so "dismiss, then
  mark reviewed" lowers the rule's confidence. (2) G10 pins that a rubber-stamp
  accept cannot move a rule by more than its own evidence count. The residual
  risk (a user who accepts without reading) is **accepted and stated**, not
  solved.

**R3a — Review flag.** New table `reviews(content_sha256_hex,
reviewed_at_unix_ms)`. It replaces paperless's inbox tag. A document is
reviewed once a human saves its metadata (assign, unassign, accept, or an
explicit "mark reviewed"). Deletion cascades inside `LanceStore::delete`.

**R4 — Rows.** A pure function builds one `ArchiveRow` per reviewed document,
plus a suggestion-time row for any document, from `taxonomy` + `assignments` +
the document's terms and field keys. Ids are remapped to dense per-run ids over
**non-retired** definitions only. That fixes S2/S3's "retired ids spend the cap
forever", and old rules never survive a run anyway (R7).

**R5 — Term vocabulary.** *(Rewritten; resolves S1/S2/S5 on deleted-doc DF,
postings→hash mapping and analyzer identity.)* Terms come from each reviewed
document's **archived text** (`DocIr`, F4), tokenized with **the index's own
analyzer** (`Index::tokenizer_for_field(text)`), so vocabulary tokens are
exactly the search tokens. DF is counted over live archived documents.
Deleted or re-indexed documents cannot inflate it, and no doc-store lookup per
posting is needed. Filters, all policy pins: keep tokens of length ≥ 3 that
contain no digit (drops OCR fragments and amounts); drop DF < `min_evidence`;
drop DF > 50% of reviewed documents. Keep the top `256.saturating_sub(tags +
field_keys)` by DF. The refusal matches the miner: `tags + field_keys + terms >
256` is refused (`mine` already does this); `tags + field_keys = 256` mines
with zero terms. **Accepted risk (S5 Q2):** no stemming, so the German model's
inflections take separate slots. The fix is registering a stemmed analyzer on
the index, which changes search too, and is deferred as its own decision.

**R6 — Field keys.** The domain is the distinct `DocIr.fields[].key` values
across reviewed documents (S3 Q4: they survive into `doc_ir_json`), remapped to
dense per-run ids.

**R7 — The model is derived, disposable, and swapped whole.** One artifact
`AutoModel { vocabulary, field_keys, id maps, matcher }`, built at startup
right after `reconcile` in `AppState::load` (S1 Q5), and on `POST /auto/mine`.
It is held in `AppState` behind one `RwLock` and replaced in a single swap, so
a reader never mixes one run's term ids with another's rules. It is not
persisted. **Accepted consequence (S5 Q3):** suggestions change after a
re-mine, never between mines. paperless persists its model, but here rebuilding
at startup is cheap at the cap (256 features).

**R8 — Surface.** The detail page shows AUTO suggestions (target name, truth
expectation, evidence, the cues that fired). "Accept" writes one `auto_accepted`
assignment and marks the document reviewed. "Dismiss" writes nothing by itself.
A minimal definitions page covers create, rename, retire and editing the S-8
rule. **S-8 at ingest:** for each single-valued kind, if the document has no
assignment, take the first matching definition in id order and log the match
count when there are several. Never overwrite an existing assignment. This is
paperless-ngx's `handlers.py:123-133` behaviour. Tags: every S-8 match is
assigned with `source = rule`.

**R9 — Features and CI.** New tables live behind `store`.
`tesseract-paperless-web` adds `matching` and `auto-match` to its
`tesseract-paperless` features. The row builder and vocabulary need a new
combined CI test + clippy line `--features store,search,matching,auto-match`
(S4 Q3: nothing compiles them otherwise).

**Order.** (1) the `taxonomy`, `assignments` and `reviews` tables with the
delete cascade and `reconcile_assignments`; (2) the row builder (pure);
(3) the vocabulary (DocIr + index analyzer); (4) S-8 at ingest; (5) `AutoModel`
in `AppState`; (6) **a router-level test harness first** (`tower::oneshot` over
`router()` with a real `AppState` on a tempdir; S4: none exists, and G4/G5 need
it); (7) routes and templates; (8) `CLAUDE.md`, including correcting the claims
S4 lists: the "not yet wired" lines, `lib.rs:22` "holds no store", and the
state.rs log formatting.

## 4. Non-goals

- COCA prior (plan piece 2): gated on its own measurement.
- Per-class table mining (piece 3): Wave D.
- **Report/axes wiring** (S1 Q3): `axes.rs` needs a dense row index and a FAST
  u64 field that does not exist. v1's "same axes, report side" line is dropped.
- Tag tree, colour, storage path, owner and permissions.
- A stemmed analyzer (R5): its own decision, because it changes search.
- Persisting dismissals as their own records (R3's review negatives cover the
  feedback need).
- Automatic application (F1).
- Tuning the policy pins (`min_evidence`, `min_confidence`, `min_lift`, `k`, the
  R5 token filters).

## 5. Pre-registered gates

| # | gate | falsifier (red-then-green) |
|---|---|---|
| G1 | A direct re-`put` of an archived document keeps its assignments and review flag. | store assignments as `documents` columns → red |
| G2 | A definition whose algorithm is not AUTO is never a suggestion target. | drop the AUTO filter → red |
| G3 | Vocabulary tokens equal the index analyzer's tokens for the same text, and a deleted or re-ingested document is counted once, or not at all. | tokenize with a second analyzer, or count from the term dictionary → red |
| G4 | Viewing a document (GET) writes nothing: assignment and review counts unchanged. | let the detail route accept the top suggestion → red |
| G5 | Accept writes exactly one `auto_accepted` assignment and marks the document reviewed. | write `manual`, or skip the review mark → red |
| G6 | Delete removes assignments and the review row; `reconcile_assignments` removes orphans. | skip the cascade → red |
| G7 | `tags + field_keys + terms` never exceeds 256; `tags + field_keys = 256` mines with no terms. | drop `saturating_sub` → red |
| G8 | A retired definition's id is never reallocated and never reaches `ArchiveDomains`. | allocate `count`, or keep retired ids in the domain → red |
| G9 | Unreviewed documents are not training rows. | include them → red on a fixture where unreviewed uploads would dilute a rule below the floor |
| G10 | Accepting a suggestion on k documents raises the rule's `cooccur` by exactly k, never more. | count the accept twice (assignment + review) → red |
| G11 | S-8 never overwrites an existing assignment, and on several matches for a single-valued kind takes the first by id. | overwrite → red |
| G12 | `clippy -D warnings`, fmt, every `tesseract-paperless` feature line (incl. the new combined line) and `tesseract-paperless-web` tests green. | — |

## 6. Change ledger (v1 → v2)

| finding | verdict | disposition |
|---|---|---|
| S5 Q1, S2 Q6, S1 prior art: paperless-ngx separates by the definition's algorithm, not the assignment's source | GAP / PRIOR-ART-AT | **adopted**: R3 rewritten |
| S5 Q1: excluding `rule` encodes S-8-tagged docs as false negatives | GAP | resolved by R3: S-8 definitions are cues, never targets |
| S5 Q1: paperless trains on reviewed documents only | GAP | adopted: R3a review flag, G9 |
| S5 Q4: v1 R3(a)'s echo argument applies to `auto_accepted` | VIOLATES | v1's reasoning retired, because the echo is now impossible (targets are AUTO definitions only). The accept loop is bounded by review negatives and G10; the residual risk is accepted and stated |
| S5 Q4: no negative feedback | GAP | review negatives (R3); separate dismissal records stay a non-goal |
| S2 Q1, S3 Q5, S1 Q3: `suggest` cannot tell cues from "assigned" | GAP / RISK | resolved without an API change (R3). **Losing option recorded:** add an assigned-set parameter to `suggest`. Unnecessary once targets are AUTO-only |
| S2 Q6: several S-8 matches on a single-valued kind | GAP | R8 + G11, verified against `handlers.py:123-133` |
| S2 Q6, S5: v1 R3(c) misattributed paperless | RISK | removed |
| S1/S2/S5: Tantivy DF counts deleted docs; postings→hash needs doc-store reads | RISK | resolved by R5: DF from archived text with the index analyzer. **Losing option:** term dictionary + `doc_freq_given_deletes` + stored-hash reads |
| S1/S3: G3 needs the index analyzer, not a second tokenizer | RISK | G3 rewritten |
| S5 Q2: no stemming → inflections fill the cap | RISK | partly mitigated (length and digit filters); stemming is an explicit non-goal and an accepted risk |
| S5 Q3: model changes on re-mine; vocab and rules must swap together | RISK | R7: one artifact, one swap; the change after re-mine is an accepted consequence |
| S2 Q3, S3 INV: retired ids spend the cap forever; `>=` vs `>` mismatch | RISK | R4 dense per-run ids over non-retired definitions; R5 refusal aligned with `mine`; G7, G8 |
| S2 Q3: correspondent and type cardinality is uncapped | RISK | **accepted**: it adds probe cost linearly per category, not quadratically; revisit if an archive exceeds a few hundred |
| S3 INV: R2's rationale overstated web exposure | RISK | R2 rationale corrected to store-level |
| S3 INV: `:797` is a test | VIOLATES | inventory corrected to `:240-250` |
| S3 INV, S4 Q1: `reconcile` is store+search only | GAP | new `reconcile_assignments` under `store` |
| S4 Q1: put the cascade in `LanceStore::delete` | RISK | adopted (R2) |
| S4 Q2: no router-level test harness | GAP | order step (6), a prerequisite of G4/G5 |
| S4 Q3: no CI line compiles the row builder | GAP | R9 combined line |
| S4 Q5: CLAUDE.md claims become false; `lib.rs:22` already false | VIOLATES | order step (8) |
| S4 Q4: Dockerfile header omits `lance-graph-arm-discovery` | RISK | follow-up comment fix, same PR |
| S1 Q1: reuse `MatchRule`; state paperless defaults and exclusions | PRIOR-ART-AT / GAP | R1 |
| S1 Q2: OGAR NT-2 | RISK | F10 |
| S1 Q3: `axes.rs` needs a row index and FAST field | GAP | non-goal; v1's claim dropped |
| S1 Q5: hook after `reconcile` in `AppState::load` | PRIOR-ART-AT | R7 |
| S4 Q4: no Dockerfile or volume change | CONFIRMS | noted |
| S2 Q2/Q4/Q5, S3 Q1/Q2/Q4/Q6, S5 Q5 | CONFIRMS | noted |
