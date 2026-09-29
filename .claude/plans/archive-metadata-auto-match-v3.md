# Archive metadata + AUTO matching wiring — v3 (RATIFIED, 5+3 council)

**Status:** RATIFIED 2026-09-29. This is the build spec. v1 (spec) and v2
(consolidated draft) stay unedited as the record.
**Council:** 5 savants on v1 (S1 prior art/OGAR, S2 miner semantics,
S3 storage/inventory, S4 build/CI/web, S5 paperless parity), then 3 reviewers on
v2: overclaim-auditor (1 BLOCK, 9 FIX), dilution-collapse sentinel (1 BLOCK,
9 FIX), firewall-warden (1 BLOCK, 6 FIX). CodeRabbit on PR #101 raised two
findings that duplicate reviewer findings (§7). Every BLOCK is resolved below.
Where reviewers disagreed, the stricter verdict won.

## The problem

The archive stores no correspondent, document type or tag
(`tesseract-paperless/src/store.rs:107-123`). `AutoMatcher::mine` therefore has
nothing to learn from, and the web app cannot assign anything. The work adds:
- definitions, assignments and a review flag;
- a term vocabulary;
- a suggest/accept surface.

## 1. Frozen decisions

| # | decision | source |
|---|---|---|
| F1 | AUTO suggestions are never applied without a human action. **S-8 at ingest never writes an assignment of an AUTO definition.** This is a deliberate difference from paperless-ngx, which merges the classifier's prediction into its match list (`paperless-ngx/src/documents/matching.py:47-70`) and applies it at consumption (`signals/handlers.py:126-143`). | `auto_match.rs` module doc; PR #101 |
| F2 | Rule antecedents are observable at ingest. Terms and field keys are cues and never targets. **A suggestion is never chained back as a cue:** no assignment of an AUTO definition is ever a cue. | `arm-discovery-and-coca-prior-v1.md` correction 1 |
| F3 | Every rule must clear lift ≥ `min_lift_ppm`. `mine` refuses more than 256 binary features. | `auto_match.rs:497-503` |
| F4 | `doc_ir_json` is the canonical text. `spo_json` also carries derived sentence text (`tesseract-paperless-web/src/ingest.rs:293-294`). The vocabulary reads `DocIr` only. The Tantivy index is disposable. | CLAUDE.md § Paperless Wave B (the Wave B "stored once" claim does not cover `spo_json`) |
| F5 | lance and lancedb come from crates.io upstream. | lance-graph CLAUDE.md carve-out |
| F6 | Everything lives in `tesseract-paperless` and `-web`. No recognition crate gains a dependency. | CLAUDE.md § `tesseract-paperless` |
| F7 | S-8 semantics are paperless-ngx's, including match order. Candidates are ordered by name, as the model default `ordering = ("name",)` does (`models.py:78-80`). With `use_first`, the first match is taken and the count is logged (`handlers.py:126-143`). | `matching.rs` |
| F8 | Axis encoding: ordinal 0 means none, `c + 1` means id `c`. Tags are a set. | `axes.rs:49-52` |
| F9 | No model identifier in any artifact. The record is CLAUDE.md plus the commit. | session rule; `.claude/prompts/2026-06-16-cpp-spo-tesseract-session-prompt.md:65` |
| F10 | Correspondent, document type and tag are consumer-local filing metadata. A `definition_id` is never an OGAR classid and is never passed to `NodeGuid`, `FacetCascade` or `mint_for`. | `OGAR/docs/OGAR-DOC-INGESTION-SPINE.md:274-278` (NT-2) |

## 2. Inventory (verified; corrected cites marked ✎)

| file:line | fact |
|---|---|
| `store.rs:50`, `:107-123`, `:224-251` (migration `:240-250` ✎), `:268-311`, `:408-414` | table name, schema, `connect`, `put` (a whole-row `merge_insert`), `delete` |
| `store.rs:209-213` | `LanceStore` fields are private, so its public signatures stay unchanged |
| `search.rs:151-170` | `SearchIndex.index` and `Fields.text` are private. `text` is TEXT and not stored. |
| tantivy fork rev `5fc9fcc` (`Cargo.toml:109`): `index.rs:457` ✎ `tokenizer_for_field`; `tokenizer_manager.rs:58-67` | the default analyzer is SimpleTokenizer + RemoveLong(40) + LowerCaser, with no stemmer |
| `lib.rs:53-54`, `Cargo.toml:181,186` | `MatchRule` exists only under `matching`, and `store` does not imply `matching` |
| `lib.rs:67-69` | `reconcile` needs both `store` and `search` |
| `matching.rs:63,78,108` | `Auto = 6`; `from_paperless`; `compile()` maps None and Auto to never-match |
| `auto_match.rs:98-111, 266, 343-358, 488-560, 575-628, 630` | row with one correspondent slot and one type slot; oracle build; distance; `mine`; `suggest`; `keep_best_of_kind` |
| `lance-graph-arm-discovery/src/aerial/extract.rs:87-100,132-147` | the probe proposes the nearest category per feature (first minimum wins), looping over the categories of every feature |
| `web/src/routes.rs:24-30, 539` ✎ | routes; `document_delete` |
| `web/src/state.rs:38, 52-100` | `reasoner: None` degrade pattern; `load` + startup `reconcile` |
| `web/Cargo.toml:37`, `Dockerfile:32-37,106` | `tower` 0.5 `util` dev-dep; the Dockerfile's dependency comment; the full lance-graph clone |
| paperless-ngx `classifier.py:336-392`, `models.py:70-80`, `matching.py:47-70`, `handlers.py:95-143`, UI `data/matching-model.ts:10` | non-inbox training; labels only for MATCH_AUTO; name ordering; `use_first`; **UI default `DEFAULT_MATCHING_ALGORITHM = MATCH_AUTO`** |

## 3. Resolution

**R1 — Definitions table `taxonomy`.** One row per `(kind, definition_id)`.
- `kind ∈ {correspondent, document_type, tag}`.
- `definition_id: u32` is `max + 1` per kind and never reused.
- `name` is unique per kind (refused on create or rename).
- The rule is stored as **plain columns**:
  - `match_algorithm: u8` (paperless numbering)
  - `match_pattern: String`
  - `case_insensitive: bool`
- `retired: bool`, `created_at_unix_ms`.
- The conversion to `MatchRule` (`MatchAlgorithm::from_paperless`) and the AUTO test live in glue under `all(store, matching)`. Adding `matching` to `store` would break the `store`-only CI line (firewall F1 BLOCK).
- **Create defaults to AUTO with `case_insensitive = true`**, as the paperless-ngx UI does. The model's own default is ANY with an empty pattern, which would leave a new definition dead on both tiers (sentinel F4).
- Left out on purpose: the tag tree, colour, `is_inbox_tag`, owner and permissions.

**R2 — Assignments table `assignments`.** One row per assignment.
- Columns: `content_sha256_hex`, `kind`, `definition_id`, `source ∈ {manual, rule, auto_accepted}`, `assigned_at_unix_ms`.
- **Single-valued write path:** assigning a correspondent or document type deletes that `(hash, kind)`'s existing rows, then inserts. A composite key alone cannot enforce single-valuedness (overclaim F5, sentinel F7).
- `source` is provenance and is shown in the UI. It has no effect on mining (R3).
- It is a separate table because `put` rewrites the whole document row. The exposure is at the store level (a direct `put`, a double upload, or a future writer), not on the web path, which dedups before `put`.

**R2a — Delete and reconcile.**
- `LanceStore::delete` removes the document row **first**, then cascades to its assignments and review rows. The worst case is then orphans, never a document with its metadata gone (firewall F3).
- `reconcile_metadata(store)` sweeps orphan assignment and review rows. It lives in a **new module `archive_meta.rs`** under `store` only; `reconcile.rs` needs `store` + `search`.
- It runs from `AppState::load`, so a re-upload of a deleted hash never inherits a stale review flag.
- Each of the three new tables has its own open-or-create in `connect`.

**R3 — Targets, labels, cues.** *(Rewritten; resolves both BLOCKs on R3/R3a and CodeRabbit's two findings.)*
- **Targets** are exactly the non-retired definitions whose `match_algorithm` is AUTO, as in paperless-ngx (`classifier.py:366-392`).
- **Eligibility enters the miner, not the caller.** `AutoMatcher::mine_eligible(domains, rows, params, eligible)` makes the oracle distance to an ineligible target item `u32::MAX`, so the probe never proposes it as a consequent. It can still be an antecedent.
  - Before this, a stronger non-AUTO correspondent was the nearest category, so the AUTO one was never proposed. `keep_best_of_kind` in `suggest` could also keep a non-AUTO winner that the caller then dropped (overclaim F1, sentinel F2, CodeRabbit).
  - `mine` is `mine_eligible` with everything eligible (unchanged behaviour, existing tests).
- **Labels:** every assignment of an AUTO definition on a reviewed document counts, whatever its source. This is paperless parity: its labels are all current assignments of MATCH_AUTO objects on non-inbox documents.
  - Switching a definition from S-8 to AUTO is therefore consistent. A `rule` assignment on a reviewed document was kept by the human who reviewed it, so it is a positive, not a false negative (sentinel F5).
- **Cues:**
  - terms;
  - field keys;
  - assignments of non-AUTO definitions, from any source.
  - An AUTO definition's assignment is **never** a cue (F2). This closes the accept → cue → accept loop that would let lift be measured on labels the rules produced (sentinel F3).
- **Retired definitions:** each single-valued kind gets one shared, never-eligible category, "retired". A document assigned to a retired correspondent is therefore still "assigned", gets no correspondent suggestion, and is not mined as unassigned. A retired tag drops out of the row entirely (overclaim R4 note).

**R3a — Review flag.** *(Rewritten; resolves the sentinel BLOCK.)*
- Table `reviews(content_sha256_hex, reviewed_at_unix_ms)`, written **only** by an explicit "mark reviewed" action. It is removed by "mark unreviewed".
- Assign, unassign and accept do **not** touch it. Accepting tag X therefore never turns tag Y into a negative.
- Training rows are reviewed documents only (`classifier.py:336-338`).
- An AUTO tag missing from a reviewed document is a negative. This is the only negative channel, and the person who marks a document reviewed is confirming its whole metadata. That is what the inbox tag means in paperless.
- **Accepted, not solved:** a definition that becomes AUTO is scored against documents reviewed before it existed, where its absence was never assessed. The effect is lower confidence and lift, so it biases toward **silence**, never toward a wrong suggestion. Given F1 that is the safe direction. The sentinel proposed counting negatives only after `created_at`; that needs per-target missingness, which the miner's dense binary encoding cannot express. Recorded as the losing option in §7.

**R4 — Row builder**, a pure function over plain structs (`auto-match` only, never typed with `DocumentRow`).
- Inputs are definitions, assignments, reviewed hashes and per-document terms and field keys.
- Output is `ArchiveDomains` + `Vec<ArchiveRow>` + the eligible set + dense id maps.
- The maps cover non-retired definitions plus the per-kind retired category, and are rebuilt per run.
- A suggestion-time row is built the same way for any document.

**R5 — Term vocabulary.** *(Rewritten; resolves overclaim F7 and sentinel F6.)*
- **One population:** the reviewed documents. Every DF and ratio below is over that set.
- Text comes from `DocIr` (F4) and is tokenized by the index's own analyzer. A new public `SearchIndex::text_analyzer() -> TextAnalyzer` under `search` exposes it.
- The vocabulary module takes a tokenizer closure. It then needs no tantivy, and the analyzer is wired in only in glue under `all(search, auto-match)` (firewall F2).
- Filters, all policy pins, each with an inertness test (switching it off admits something):
  - token length ≥ 3;
  - the token contains no digit;
  - `DF ≥ vocab_min_df` (its own knob, default 5 — no longer coupled to `min_evidence`);
  - `DF ≤ 50%` of reviewed documents;
  - keep the top `term_budget` by DF (tie: lexicographic).
- **Budget order:** tags first, then field keys, then terms, all within 256.
  - Field keys pass the same DF filter and top-N as terms (overclaim F8).
  - If the tags alone exceed 256, the mine is refused with `TooManyFeatures`.
  - If no content term fits, the mine is refused with a new `NoContentCues` error, which the UI shows. A model with no content cues cannot fire on a new upload (sentinel F6; v1's surfaced refusal restored).
- **Accepted risk:** no stemming. Registering a stemmed analyzer changes search and is its own decision (§4).

**R6 — Field keys.** The domain is the distinct `DocIr.fields[].key` values on reviewed documents, filtered and capped as in R5.

**R7 — Model artifact.** `AutoModel { vocabulary, field_keys, id maps, eligible, matcher }` lives in `tesseract-paperless` under `all(store, search, matching, auto-match)`. `-web` holds no mining logic.
- `AppState` holds `std::sync::RwLock<Option<Arc<AutoModel>>>`. Readers clone the `Arc` and never hold the guard across `.await`. A re-mine replaces the whole `Arc` in one swap.
- `AutoModel::suggest_for(&DocIr, assignments)` maps text to term ids **with its own vocabulary**, so a caller can never mix one run's ids with another's rules.
- Mining runs on `spawn_blocking` and **off the boot path**. The model is `None` until the first mine finishes, and the UI says "no suggestions yet", following the `reasoner: None` pattern (overclaim F6).
- Re-mining is triggered by `POST /auto/mine`.
- **Cost is unmeasured and labelled so.** A pre-registered measurement (§5 M1) times a mine on a synthetic 10,000-document archive at 256 features before any claim about cost is made.

**R8 — Surface.**
- **Detail page:**
  - shows AUTO suggestions: target name, expectation, evidence, and the cues that fired;
  - **Accept** writes one `auto_accepted` assignment (single-valued: replace) and **nothing else**;
  - **Dismiss** writes nothing;
  - **Mark reviewed** and **Mark unreviewed** toggle the flag.
- **Definitions page:** create (defaults to AUTO), rename (unique per kind), retire, and edit the rule.
- **S-8 at ingest** (with `matching`):
  - For each single-valued kind with no assignment, match the non-retired, non-AUTO definitions in **name order** (F7). The first match is assigned with `source = rule`, and the match count is logged when there are several.
  - An existing assignment is never overwritten.
  - For tags, every S-8 match is assigned.
  - AUTO definitions never match, and there is a gate for it (F1).

**R9 — Features and CI.**
- `-web` enables `matching` and `auto-match` on its `tesseract-paperless` dependency.
- New CI lines:
  - `test` + `clippy` on `--features store,search,matching,auto-match`;
  - `check` + `clippy` on `store,auto-match` and on `search,auto-match`, to catch a mis-gated module.
- `-web`'s existing build/test/clippy lines compile the union. They do not run the paperless unit tests, which is why the combined line exists.

**Module placement:**

| module | gate |
|---|---|
| `archive_meta.rs` (tables, loaders, `reconcile_metadata`, single-valued write) | `store` |
| `auto_rows.rs` (R4 row builder, R5/R6 vocabulary via a tokenizer closure) | `auto-match` |
| `auto_model.rs` (`AutoModel`, the definitions-to-`MatchRule` glue, the analyzer glue) | `all(store, search, matching, auto-match)` |
| `auto_match.rs` + `mine_eligible` | `auto-match` |

**Order.**
1. `auto_match::mine_eligible` and its gates.
2. `auto_rows.rs` (pure).
3. `archive_meta.rs`, including the delete cascade in `store.rs`.
4. `SearchIndex::text_analyzer`.
5. `auto_model.rs`.
6. S-8 at ingest.
7. The `AppState` model slot and background mine.
8. A router harness: `tower::oneshot` over `router()` with an `AppState` on a tempdir, **seeding store rows directly**. It never runs OCR, because `-web` tests run in debug.
9. Routes and templates.
10. CLAUDE.md, including correcting the stale claims: `lib.rs:22`, the Wave B "stored once" versus `spo_json`, and the "not yet wired" lines.
11. The Dockerfile dependency comment (`Dockerfile:32-37`: add `lance-graph-arm-discovery` and `deepnsm-v2`).

## 4. Non-goals

- COCA prior (piece 2) and per-class tables (piece 3).
- Report/axes wiring: it needs a dense row index and a FAST field.
- The tag tree, colour, storage path and permissions.
- A stemmed analyzer.
- Dismissal records.
- Automatic application (F1).
- Tuning the policy pins.
- Per-target missingness in the miner (R3a accepted risk).

## 5. Gates (each a red-then-green disable run)

| # | gate | disable |
|---|---|---|
| G1 | A direct re-`put` of a document keeps its assignments and review flag. | store them as `documents` columns |
| G2a | A non-AUTO definition is never a target, even when it is the only rule. | drop eligibility |
| G2b | A strong non-AUTO correspondent rule and a weaker AUTO rule on the same cue: the AUTO suggestion comes back (recall twin). | filter after `suggest` instead of in the oracle |
| G3 | Vocabulary DF equals a brute-force count over the reviewed documents' `DocIr` text with the same analyzer, and a re-ingested or deleted document is counted once or not at all. | use a second tokenizer, or count from the term dictionary |
| G3b | Each R5 filter pin is inert-tested: switching it off admits a token. | per pin |
| G4 | GET of the detail page writes nothing. | detail route auto-accepts |
| G5 | Accept writes exactly one `auto_accepted` row and **no** review row. | also mark reviewed |
| G6 | Delete removes document then metadata; `reconcile_metadata` removes orphans, and a re-uploaded hash is unreviewed. | skip the cascade, or skip the review sweep |
| G7a | `tags + field_keys = 256` gives `NoContentCues`. | build the matcher with zero terms |
| G7b | 257 gives `TooManyFeatures`, and the budget arithmetic never underflows. | drop `saturating_sub` (panics in debug) |
| G8 | A retired definition's id is never reallocated and is never a target. A document on a retired correspondent gets no correspondent suggestion. | allocate by count; map retired to none |
| G9 | Unreviewed documents are not training rows: on a fixture where they would pull a rule's confidence below the confidence floor, the rule survives. | include them |
| G10 | Accepting a suggestion on k unreviewed documents changes **no** rule statistic until they are marked reviewed. Marking them reviewed raises that pair's `cooccur` by exactly k. | accept also marks reviewed |
| G10b | Marking reviewed a document the rule fires on but which lacks the target lowers that rule's confidence (the negative channel can fire). | exclude reviewed-negative rows |
| G11a | S-8 never overwrites an existing assignment. | overwrite |
| G11b | Several S-8 matches for a single-valued kind take the first by name. | order by id |
| G11c | S-8 never writes an AUTO definition. | compile Auto to always-match |
| G12 | Assigning a correspondent twice leaves exactly one correspondent row. | skip the delete |
| G13 | An AUTO assignment is never a cue. | admit AUTO assignments as cues |
| G14 | The default-create path yields an AUTO definition that, given evidence, produces a suggestion. | default to ANY |
| G15 | Two models with disjoint vocabularies each answer from their own vocabulary through `suggest_for`. | pass term ids in from outside |
| G16 | clippy `-D warnings`, fmt, every feature line including the new ones, `-web` tests. | — |

**M1 (measurement, pre-registered):** the wall time of `mine_eligible` + the
vocabulary pass on 10,000 synthetic reviewed documents at 256 features, in
release. It is reported, not gated. The cost claim in R7 stays "unverified"
until M1 has run.

## 6. What v3 still does not know

- Whether the policy pins suit a real archive: none of them has been measured on one.
- Whether the silence bias from R3a matters in practice.
- The M1 cost.

## 7. Change ledger (v2 → v3)

| finding | verdict | disposition |
|---|---|---|
| firewall F1: `MatchRule` in a `store`-gated table | BLOCK | R1 plain columns, with glue under `all(store, matching)` |
| sentinel F1: "reviewed" set by accept makes every other tag a negative | BLOCK | R3a: set only by an explicit mark; G5, G10, G10b |
| overclaim F1: mining pre-emption and best-of-kind make "no API change" false | BLOCK | R3 `mine_eligible` with oracle-level eligibility; G2a/G2b. v2's "no API change" is withdrawn. |
| CodeRabbit #4137902355 (filter before per-kind selection) | major | same fix as the row above |
| CodeRabbit #4137832421 (v1: `suggest` blind to rule assignments) | major | already resolved in v2 by the AUTO-only targets; still holds in v3, since rule rows can never collide with a target of the same definition |
| sentinel F3: AUTO-as-cue widens the frozen correction; G10 vacuous | FIX | R3/F2: AUTO never a cue; G13; G10 rewritten |
| sentinel F4: default ANY leaves definitions inert | FIX | R1 defaults to AUTO (UI parity); G14 |
| sentinel F5: flipping a definition to AUTO creates false negatives | FIX | R3 labels are all AUTO assignments on reviewed documents (paperless parity) |
| sentinel F5 alternative: negatives only after `created_at` | losing | needs per-target missingness; recorded as the R3a accepted silence bias |
| overclaim F5, sentinel F7: single-valuedness unenforced | FIX | R2 delete-then-insert; G12 |
| sentinel F7: orphan review rows | FIX | R2a sweep; G6 |
| overclaim F4, sentinel F8: "first in id order" is wrong; paperless orders by name | FIX | F7/R8 name order; G11a/b split |
| overclaim F2: G10 overclaims | FIX | G10 rewritten, G10b added; the "bounded by its own evidence" sentence is dropped |
| overclaim F3: G7 disable changes nothing at 256 | FIX | G7a/G7b split |
| overclaim F6: "cheap at the cap" unmeasured; the probe is quadratic in total width | FIX | R7 labelled unverified, off the boot path, M1. The v2 ledger's "linearly per category" wording is withdrawn: the cost is quadratic in total width, not measured. |
| overclaim F7, sentinel F6: wrong analyzer cite, private index, two DF populations, pins untested, `min_evidence` coupling | FIX | R5 rewritten; G3, G3b |
| overclaim F8: field keys uncapped | FIX | R5 budget order |
| overclaim F9: F1/F4/F9 and routes cites | FIX | corrected in §1/§2 |
| overclaim F10: ledger rows resolved in name only | FIX | this ledger restates each one with its R-item and gate |
| sentinel F9: v1 items lost without rows | FIX | restored: S-8 never writes AUTO (F1, G11c); name uniqueness (R1); assigning replaces (R2, G12); surfaced refusal (R5 `NoContentCues`); brute-force DF (G3) |
| sentinel F10: nothing gates the one swap | FIX | R7 `suggest_for`; G15 |
| firewall F2: per-module gating | FIX | module placement table |
| firewall F3: delete order | FIX | R2a |
| firewall F4: CI lines | FIX | R9 |
| firewall F5: web state | FIX | R7 |
| firewall F6: `id` looks like a classid | FIX | renamed `definition_id`; F10 wording |
| firewall F7: Dockerfile comment | FIX | order step 11 |
| overclaim R9 wording: "nothing compiles them" should be "nothing tests them" | FIX | R9 |
| firewall Q1/Q3/Q4/Q5; overclaim cites verified; sentinel R4/R6/R9/G1/G4 | CONFIRMS | noted |
