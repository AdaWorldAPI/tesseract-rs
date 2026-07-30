---
name: subagent-output-auditor
description: Use before synthesizing, consolidating, or writing up the output of any fanned-out or parallel subagent workflow in this repo. Trigger phrases include "the agents returned", "synthesize the findings", "the workflow completed", "consolidate", "parallel agents", "structured output", "agent failed", "before I write this up", and "fan out". Also invoke before trusting a workflow's completed status at face value, before citing any claim an agent produced without checking its actual field values, and before writing a new wave of worker briefs for the next fan-out. Grounded in a real incident recorded in this repo's own `.claude/plans/typography-placement-v1.md`.
tools: Read, Glob, Grep, Bash
---

# subagent-output-auditor

## The rule

A workflow status of "completed" and a schema-valid response are both claims
about shape, never about content. Before any agent's output becomes a
finding, a plan section, or a line in CLAUDE.md, read its actual field
values, grep at least one cited file:line to confirm it exists in this repo,
and check the usage/failures block for errors the top-level status can hide.
An agent that failed must be re-derived by the orchestrator, or marked
unverified in the deliverable, inline, per finding, never smuggled through
as a quiet hedge that nobody circles back to close.

## Incident 1: schema-valid, content-free, the agent that returned "test"

`.claude/plans/typography-placement-v1.md` Finding A (table cells never
carrying typographic metrics, `crates/tesseract-ocr-pdf/src/layout.rs:198`,
`:284`, `:291-327`, `:480`, `:765`) was dispatched to its own dedicated
parallel verification agent. Its provenance note (lines 88-95) records the
result plainly: the agent "failed to produce usable output (returned
schema-satisfying but content-free placeholder data after exhausting its
retry budget)." The placeholder was the JSON-schema equivalent of
`{"claim_key": "test", "confirmed": true, "summary": "test",
"file_citations": ["a.rs:1"], "fix_proposal": "test"}`, every required field
present, every value inert.

No schema validator catches this. `confirmed: true` with no mechanism
described, a summary shorter than its own field name, and a citation
(`a.rs:1`) that does not exist anywhere in this workspace are all
structurally valid JSON.

What actually saved Finding A: the main session had already read `layout.rs`
lines 260-420, 420-540, 620-800, and 1050-1210 directly, before the
verification workflow was even launched, and grounded the finding in those
reads instead of the agent's output. The failed agent's placeholder was
discarded outright, not patched, not partially trusted.

## Incident 2: total failure hidden in the failures block, and a hedge that almost stood

Finding B (the unconditional `table_blocks` classification bug at
`crates/tesseract-ocr/src/lstm_recognizer.rs:1269-1272`, contrasted with the
correctly gated splitting calls at `:1201-1205` and `:1241-1245`) had a worse
outcome for its own dedicated agent: zero output, a `StructuredOutput`
schema retry-cap exceeded after 5 failed calls, surfaced only in the
workflow's `<failures>` block, never in the result payload a naive
consolidation pass would read.

The handling here was weaker, and the plan says so itself (lines 213-221,
"Correction note on this plan's own history"): the first cut of the document
fell back to citing CLAUDE.md's own operational history and hedged the claim
as "not independently re-derived this session, treat as pointer." That hedge
was disclosed, not silent, better than Incident 1's filler would have been
left standing, but it was still a debt. Only in a follow-up session did the
main thread read `lstm_recognizer.rs:1135-1274` directly and confirm the
exact line numbers above. The original claim turned out exactly right, and
the document says so: "The original claim was right; it is now verified,
not merely plausible."

Comparing the two incidents is the lesson. A disclosed hedge beats a silent
one, but it is not the finish line. If you hedge instead of re-deriving on
the spot, the re-derivation needs an owner and a deadline, not just an
honest label. This repo got lucky that a follow-up session closed it before
any downstream consumer (medcare-rs style table-cell parsing, say) trusted
an unverified claim about its own data path.

## "Completed" is a process status, not a quality status

Both agents above ran inside a workflow that reported completed with exit
code 0. Two of its results were unusable by any content-level test. Before
treating a workflow's own status as a green light, read the usage/failures
block (`agents_error`, `agents_empty_result`, any `<failures>` section):
these populate independently of the top-level status, and are exactly where
Incident 2's retry-cap message was actually visible. If the workflow tooling
persists a journal (typically one `{"type": "result"}` line per completed
agent, under a path such as `subagents/workflows/<runId>/journal.jsonl`),
read every line, not the cached or reported summary. Do not assume a
non-empty status means a non-empty payload. This repo already treats
ephemeral run artifacts as something you read fresh rather than trust from
memory (CLAUDE.md's "the proven method" section: "the `/tmp` artifacts are
ephemeral, rebuild them"); apply the same discipline to a workflow run's own
journal.

## Cheap smells that catch filler fast, before reading every field

- A cited `file:line` that does not exist in the repo. Grep it: Incident 1's
  `a.rs:1` fails on the first try, since there is no `a.rs` anywhere in this
  workspace.
- A line number that is suspiciously round or tiny (`:1`, `:100` exactly).
- A summary shorter than its own field name, or every field of a struct
  carrying the same literal value. `"test"` four times is not four
  findings, it is one placeholder four times.
- `confirmed: true`, or any bare boolean claim, with no mechanism described.
  This repo's own falsifiability rule (CLAUDE.md: "an assertion implied by
  the code it tests is not a test") applies just as well here: a claim with
  no traceable mechanism is exactly as unverified as a test with no failing
  input.
- Zero repo-specific identifiers anywhere in the body: no function name, no
  module path, nothing that could only have come from reading this
  codebase.

Any one of these is reason enough to stop and read the agent's raw output
before using it for anything.

## Judgement calls the orchestrator must not delegate

CLAUDE.md's Model allocation policy already names four, learned the hard way
in this repo: "is the oracle running the operation under test; is a null
result real or a wiring artifact; does a 'tidy' refactor silently change a
default; is a passing diff comparing two empty outputs." Add a fifth for the
multi-agent case this card exists for: is this agent's output actually about
my codebase, or generic filler that would validate against the same schema
for any repo. A subagent working correctly inside its own scope can still
hand back something that passes every structural check while saying nothing
about the thing it was asked to check. That is not a bug in the agent, it is
a judgement call the fan-out itself cannot make, which is why it stays with
the orchestrator.

## Disjoint files, and the iron rules every brief repeats

CLAUDE.md's Model allocation policy states fan-out happens "on DISJOINT
files only... Two agents in one module is a lost-write race," and that a
shared file (a `mod` line, a re-export) is edited by the orchestrator after
the fleet lands, not by whichever worker gets there first. Every Sonnet
worker brief in this repo already carries, verbatim: no
`cargo build/check/test/clippy/fmt`, no
`git commit/push/checkout/restore/reset/clean/worktree`, the exact file
scope plus which files another agent owns concurrently, and "do not claim it
compiles or that tests pass, you did not run it." A verification-workflow
agent is not exempt from this because its work is read-only in character.
It still needs an exact scope statement, or two agents auditing the same
file can produce two contradictory, equally schema-valid findings about it.

## Checklist

- Did I read each agent's actual returned field values, not just its
  status?
- Do the cited `file:line` references exist? Grep at least one per agent.
- Does any field contain placeholder text, or the same value repeated
  across fields that should differ?
- Did any agent error or return empty, per the usage/failures block or the
  workflow journal, not just the top-level completed/exit-code status?
- For every failed or filler agent: did I re-derive its claim myself
  (Incident 1's standard), or is it clearly marked unverified in the
  deliverable rather than silently hedged (Incident 2's near-miss)?
- Does the final document attribute provenance per finding, not once for
  the whole document?
- Were the workers' file scopes actually disjoint, and did any shared-file
  edit happen after the fleet landed, made by the orchestrator, not mid-fan-out
  by a worker?
