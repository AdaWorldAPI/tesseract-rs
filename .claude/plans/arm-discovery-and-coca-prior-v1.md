# Association-rule discovery and a COCA prior for the paperless archive — v1

**Status:** PROPOSAL. Nothing is built. Written 2026-09-29, after PR #100
(homograph context rule) merged. Convergence with a parallel session is
invited: see `.claude/prompts/2026-09-29-arm-discovery-convergence.md`.

**Scope:** two independent pieces, plus a deferred third. Each can ship on its
own.

## Facts this proposal rests on (read from source, not assumed)

- `lance-graph-arm-discovery` (sibling repo, `crates/lance-graph-arm-discovery`)
  has **no dependencies** and is **excluded** from the lance-graph workspace. A
  path dependency from here costs almost nothing.
- Input: `FeatureSpec::new(cardinalities: Vec<u32>)` + `Dataset::new(spec, rows:
  Vec<Vec<u32>>)`: one categorical value per feature per row.
- Mining: `AerialProposer::new(data, oracle: impl CodebookDistance, params).mine()
  -> Vec<CandidateRule>`. The probe needs a distance oracle between items
  (`MatrixDistance`, or sparse `TopKDistance::new(spec, miss, edges)`); candidates
  are then confirmed on the data with integer counts.
- `CandidateRule { antecedent, consequent, cooccur, antecedent_count, window }`:
  support = cooccur / window, confidence = cooccur / antecedent_count.
- Truth: `arm_to_nars(rule, k)` / `arm_to_truth_u8(rule, k)`; NARS confidence
  `c = m / (m + k)` with `m = cooccur`. Output edge: `CandidateTriple` +
  `to_ndjson` → `{s,p,o,f,c}`.
- COCA (`deepnsm/word_frequency/word_forms.csv`):
  `lemRank,lemma,PoS,lemFreq,wordFreq,word`. **Unigram only.** There are no word
  pairs and no context, so no contextual rule can be mined from COCA itself.
- paperless-ngx `matching_algorithm = AUTO` is the classifier tier. S-8
  (`tesseract-paperless::matching`) transcribed every other algorithm and left
  AUTO as never-matching.

## Piece 1 — archive-level rule mining as the paperless AUTO tier (recommended first)

**What.** One row per archived document. Features:

| feature | cardinality | source |
|---|---|---|
| correspondent | number of correspondents + 1 (none) | archive metadata |
| document type | number of types + 1 | archive metadata |
| one feature per tag | 2 (absent / present) | archive metadata |
| one feature per harvested field key (IBAN, amount, …) | 2 | `doc.v1` `fields` |
| month bucket | 12 + 1 | ingest time |

Mine rules whose consequent is a tag, correspondent or document type, for
example `correspondent=Stadtwerke → tag:Energie (f 0.94, c 0.91)`. At ingest,
a document with no manual assignment gets the rule's consequent **suggested**,
carrying its `NarsTruth`, never silently applied.

**Why.** It fills a gap that already exists (AUTO), and it replaces paperless's
trained classifier with integer, reproducible rules that come with evidence
counts.

**Where.** New module `tesseract-paperless/src/auto_match.rs`, behind a new
feature `auto-match = ["dep:lance-graph-arm-discovery"]`, with its own CI test
and clippy line (every tier gets a CI line — see CLAUDE.md, paperless sections).

**Open design points.**
1. **The distance oracle.** Aerial+'s probe needs an item-to-item distance.
   There is no codebook for archive metadata. Options: (a) a uniform oracle, so
   every item is a candidate and the integer confirm step does all the work
   (fine at archive scale, feature count in the tens to low hundreds); (b) a
   distance table from co-occurrence counts. Start with (a) and measure the
   candidate count.
2. **Minimum evidence.** A rule from 2 documents is noise. Default proposal:
   `cooccur >= 5` and `k` chosen so that 5 documents give `c ≈ 0.5`, i.e. `k = 5`.
   This is a policy pin until measured on a real archive.
3. **Retraining.** Re-mine on every ingest, on a schedule, or on demand. Mining
   is cheap at archive scale; start with on-demand plus after N new documents.
4. **Where rules live.** Keep them derived and disposable (like the Tantivy
   index, rebuilt from the archive), never a second source of truth.

**Falsifiers.**
- On a synthetic archive where correspondent X always carries tag Y, the rule
  `X → Y` appears with confidence 1.0 (can fire).
- On a synthetic archive where tags are assigned independently of correspondent,
  no rule clears the evidence floor (can stay silent). Must use a non-trivial
  archive, not an empty one.
- Raising the evidence floor removes rules; lowering it admits rules
  (the knob is not decoration).
- A suggestion is never written as an assignment (the ingest path is read-only
  with respect to tags unless the caller accepts).

## Piece 2 — COCA frequency as a prior, revised with the context rule

**What.** PR #100's `SentenceReasoner::disambiguate` makes a hard choice from
the previous token. Make it a weighted one:

- Prior: from COCA, per ambiguous form, `f = wordFreq(noun) / (wordFreq(noun) +
  wordFreq(verb))` for "is a noun", with confidence from the total count
  (e.g. `bites`: 5275 noun vs 1559 verb → f ≈ 0.77).
- Context evidence: the rule's cue (determiner/adjective/preposition → noun;
  noun/pronoun/modal, with no following verb → verb) as a second truth value
  with a fixed, stated confidence.
- Combine with NARS revision (`lance-graph-contract`'s `NarsTruth`, already a
  dependency). Pick the reading whose expectation is higher.

**Why.** A strong context cue should beat a weak prior, and an overwhelming
prior should survive a weak cue. Today the cue always wins.

**Where.** `tesseract-ogar/src/reasoning.rs` only. No new dependency.

**Falsifiers.**
- The five existing disable-verified tests stay green (behaviour on the pinned
  sentences does not change).
- A new case where the prior should win over a weak cue flips under revision and
  does not flip under the hard rule. It has to be found by measurement over the
  side table; if no such case exists, Piece 2 changes nothing and is not worth
  shipping. **State the result either way.**

**Not this:** mining context rules with arm-discovery from our own parsed
sentences. The parser that would tag those rows is the one being corrected, so
the mined rules would largely echo its own mistakes.

## Piece 3 — tables per document class (deferred to Wave D)

Each `doc.v1` table row becomes a row, columns become features after numeric
binning. One lab report has too few rows for support; this only works across
many documents of the same class, which is the cross-document belief work
(plan `paperless-archive-integration-v1.md`, Wave D). Deferred; listed so it is
not re-proposed as new.

## Order

1. Piece 1 (fills the AUTO gap, own feature flag).
2. Piece 2 (small, one file, reuses what exists), gated on its own measurement.
3. Piece 3 with Wave D.

## Still open, independent of this proposal

- `tesseract-paperless/src/consistency.rs` does not call the PR #100
  disambiguator yet.
- S-8 matching is not wired into ingest.
