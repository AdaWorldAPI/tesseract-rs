# Prompt — converge your ideas into the arm-discovery / COCA-prior proposal

**For:** the other session working on this workspace (tesseract-rs,
lance-graph, deepnsm, paperless). Paste this as your opening message.

---

A proposal was written in `AdaWorldAPI/tesseract-rs` on branch
`claude/brave-mayer-65y3cy`:

    .claude/plans/arm-discovery-and-coca-prior-v1.md

It proposes three pieces:
1. `lance-graph-arm-discovery` mining rules over the paperless archive
   (one row per document: correspondent, type, tags, field keys, month) to fill
   paperless-ngx's `AUTO` matching tier with integer rules that carry a
   `NarsTruth`.
2. COCA word-form frequency as a prior, combined by NARS revision with the
   context rule from tesseract-rs PR #100 (`SentenceReasoner::disambiguate`).
3. Per-class table mining, deferred to Wave D.

**What I'd like from you:**

1. Read the proposal in full, then read the sources it cites before judging it
   (`lance-graph/crates/lance-graph-arm-discovery/src/{encode,rule,translator}.rs`,
   `src/aerial/mod.rs`, `tesseract-ogar/src/reasoning.rs`,
   `tesseract-paperless/src/matching.rs`). A claim about absence needs the file
   opened, not only a grep.
2. Add the ideas you already have that bear on this, especially anything about
   SPO frequency/confidence over tabular data, COCA frequency and PoS, the
   distance oracle for non-codebook items, or where mined rules should live.
3. For each of your ideas say which piece it changes, and whether it
   **replaces**, **extends** or **conflicts with** what is written. Name
   conflicts plainly; do not merge two designs into one that does neither.
4. Correct any fact in the "Facts this proposal rests on" section that is
   wrong, with the file and line that shows it.
5. Do not build anything yet. The output is a revised proposal.

**How to hand it back:**

- Write `.claude/plans/arm-discovery-and-coca-prior-v2.md` next to v1 (do not
  edit v1). Put a short "What changed from v1 and why" section at the top.
- Keep the falsifier discipline: every new mechanism names a can-fire test and a
  can-stay-silent test on non-trivial input, and every threshold is marked as a
  policy pin until measured.
- Commit on your own branch and open a draft PR against `master`; link it from
  the PR description to v1 so the two can be read side by side.
- Leave open points open. "Unknown" is an acceptable answer.
