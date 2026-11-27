# Klickwege parity: paperless-ngx vs. tesseract-paperless-web

Status 2026-11-26. This document lists the click paths of the paperless-ngx web UI and what
`tesseract-paperless-web` offers for each one. It also records what the Charon → ruff_charon →
Codegraph pipeline (lance-graph PR #2000) could prove about our side.

## Sources

- **paperless-ngx:** `AdaWorldAPI/paperless-ngx` at `6ce43be`, read from source.
  - Navigation: `src-ui/src/app/app-routing.module.ts`. It has 22 routed views plus redirects.
  - Sidebar links: `components/app-frame/app-frame.component.html`.
  - Detail tabs: `DocumentDetailNavIDs`.
  - Actions: the `(click)` handlers of the document list, the bulk editor and the detail page.
- **tesseract-paperless-web:** this branch. `routes::router()` is the complete list of entry
  points, with 27 routes. It was read in full, so "missing" below means no route exists.
- **Codegraph:**
  - Charon `c69e231` (the commit the PR #2000 probe used) → `ruff_charon import` →
    `cg_query` (lance-graph `ba04c26`).
  - The commands and the measurements are in § Codegraph below.

## Legend

| Mark | Meaning |
|---|---|
| ✅ | Same click path, same result. |
| ◐ | Reachable, but narrower than paperless-ngx (named in the row). |
| ✗ | No route. |
| ＋ | Ours only. |

The **Codegraph** column gives what the compiler facts show for our handler:
- `H` means the handler is CONFIRMED in the store, as axum's `FnOnce::call_once` for it.
- `body?` means its callees are not observed: the body is not read (see § Codegraph).

## Navigation (sidebar and routes)

| paperless-ngx | Ours | Parity | Codegraph |
|---|---|---|---|
| `/dashboard` | `/dashboard` | ◐ counts and recent documents; no configurable widgets or saved-view widgets | H, body? |
| `/documents` (list, filter, sort) | `/documents?q=` | ◐ full-text search (Tantivy, ranked, snippets); no filter bar, sort or column choice | H, body? |
| `/view/:id` (saved view) | — | ✗ no saved views. The desktop folders are the nearest thing (one fixed view per definition). | — |
| `/documents/:id` | `/documents/:hash` | ◐ see "Document detail" | H, body? |
| `/documents/:id/:section` | — | ✗ one page, no tabs | — |
| `/asn/:id` | — | ✗ no archive serial number | — |
| `/attributes/tags` | `/definitions?kind=tag` | ◐ create, rename, matching rule, retire; no colour, no inbox tag | H, body? |
| `/attributes/correspondents` | `/definitions?kind=correspondent` | ◐ as tags | H, body? |
| `/attributes/documenttypes` | `/definitions?kind=document_type` | ◐ as tags | H, body? |
| `/attributes/storagepaths` | — | ✗ | — |
| `/attributes/customfields` | — | ✗ | — |
| `/savedviews` | — | ✗ | — |
| `/mail` (mail accounts and rules) | — | ✗ | — |
| `/workflows` | — | ✗ | — |
| `/share-links` | — | ✗ | — |
| `/usersgroups` | — | ◐ sign-in through OIDC (ZITADEL); users and groups are managed at the identity provider. Customer scope comes from `PAPERLESS_CUSTOMER_POLICY`. | — |
| `/settings` | — | ◐ branding through Railway variables only; no per-user UI settings | — |
| `/config` | — | ✗ | — |
| `/tasks` | — | ✗ OCR runs inline in the upload request; there is no task queue | — |
| `/trash` | — | ✗ delete is final (it also removes the S3 copy) | — |
| `/logs` | — | ✗ (stdout, Railway logs) | — |
| Upload (drag and drop in the frame) | `/` + `POST /upload` | ◐ file or image URL, optional customer; one file per request; the S-2 dedup gate | H, body? |
| — | `/inbox` | ＋ unreviewed documents, paged (paperless uses an inbox tag) | H, body? |
| — | `/desktop`, `/desktop/folders/:source/:key` | ＋ folders as windows | H, body? |
| — | `/api/v1/customers/:code/documents` | ＋ for report-rs | H, body? |

## Document detail

| paperless-ngx tab / action | Ours | Parity |
|---|---|---|
| Details (title, ASN, created date, correspondent, type, storage path, tags) | assign / unassign per kind | ◐ correspondent, type and tags only; no title edit, no ASN, no storage path |
| AUTO suggestions | `POST …/accept` | ✅ matching algorithms transcribed (S-8), accept one suggestion |
| Content | text and regions on the page | ✅ read only |
| Metadata | MIME, source, pages, confidence, hash | ◐ no original/archive checksums or PDF metadata |
| Preview (PDF viewer, zoom) | — | ✗ (`Download original` only) |
| Notes | — | ✗ |
| Permissions | — | ◐ customer field (`POST …/customer`), no per-user permissions |
| History | — | ✗ |
| Duplicates | — | ◐ the S-2 gate stops an exact duplicate at upload (same SHA-256) and opens the existing document; no near-duplicate tab |
| Download | `GET …/original` | ✅ |
| Delete | `POST …/delete` | ◐ no trash, no undo |
| Mark reviewed (inbox) | `POST …/review` | ✅ ("next" goes to the next unreviewed document) |
| Previous / next document | — | ✗ |
| Print, email, share link, more like this | — | ✗ |
| Edit PDF, rotate, split, merge, remove password, reprocess | — | ✗ |

## Document list actions

| paperless-ngx | Ours | Parity |
|---|---|---|
| Full-text search | `/documents?q=` | ✅ ranked, snippets |
| Filter by correspondent / type / tag (`clickTag` …) | desktop folder of that definition | ◐ one definition at a time, no combination |
| Bulk select and bulk edit (tags, correspondent, type, permissions, delete, download, merge, rotate, reprocess, email) | — | ✗ |
| Save view / save view as | — | ✗ |

## Codegraph: what the compiler facts can show about these click paths

### The run

From `crates/tesseract-paperless-web`, Charon on nightly-2026-09-17:

```sh
charon cargo --no-dedup-serialized-ast --opaque '_' --include 'crate' --include 'crate::_' \
  --hide-marker-traits --start-from 'crate::routes::router' --start-from 'crate::desktop' \
  --start-from 'crate::glass' --dest-file $T/tesseract_paperless_web.llbc
```

Then, from `crates/`:

```sh
ruff_charon import $STORE $T/tesseract_paperless_web.llbc tesseract_paperless_web
cg_query $STORE . sweep
cg_query $STORE . symbols ::call_once
cg_query $STORE . callees tesseract_paperless_web::routes::router
cg_query $STORE . refs tesseract_paperless_web::routes::router
cg_query $STORE . refs tesseract_paperless_web::desktop::folder
cg_query $STORE . callees tesseract_paperless_web::desktop::folder
cg_query $STORE . callers tesseract_paperless_web::desktop::valid_key
```

| Measurement | Value |
|---|---|
| Charon on the whole crate (default start `crate`) | **killed (SIGKILL, out of memory)** after reaching ~10 GB; the machine has 15 GB |
| Whole crate, every foreign crate `--opaque` | killed again: 13.5 GB peak after 28 s |
| Started from `routes::router`, `desktop`, `glass` | 14 s, 0.88 GB peak, 30 MB LLBC |
| `ruff_charon import` | harvested 1,615, represented 1,584, residual 31, dangling 0 |
| `cg_query sweep` | 384 positions, 236 answered, 532 empty, 3 ms (with lance-graph #2215: 497 answered, 271 empty) |
| Charon warnings | 255, all `Coroutine types are not supported yet` / `Coroutines are not supported` |

### What it shows

- **Every route's handler is in the store, and so is the router's edge to it.**
  - Each of the 27 handlers has a CONFIRMED caller row:
    `{impl FnOnce<…> for <handler>}::call_once`. That is axum's `Handler` impl.
  - `router()` has a `REFERENCES` row to each of the 27 handlers (27 of 27), with the
    call-site offset. `refs <handler>` shows it, for example `refs desktop::folder` →
    `REFERENCES tesseract_paperless_web::routes::router`. An earlier version of this document
    said "no route facts"; it had queried `refs router`, which asks the opposite direction.
  - `callees routes::router` lists the axum builder calls (`Router::route`, `get`, `post`,
    `layer`, `with_state`, `DefaultBodyLimit::max`, `RequestBodyLimitLayer::new`) over 60 call
    sites.
- **Synchronous helpers are fully covered.** For example:
  - `desktop::entry_of → routes::format_unix_ms` (CONFIRMED);
  - `desktop::valid_key → str::bytes, RangeInclusive::contains, …`.
- **The harvest names every body it did not read.** `ruff_charon` writes one
  `BODY_NOT_STRUCTURED` residual per function whose body Charon did not translate: 31 of 31
  here (the 27 handlers and 4 async helpers).

### What it cannot show yet: intake gaps, not a reason to replace Codegraph

1. **Handler bodies are not read.**
   - Every axum handler is an `async fn`. Its body is a coroutine, and Charon `c69e231` does
     not translate coroutines (the 255 warnings).
   - Before lance-graph #2215 the answer was an empty set with no UNKNOWN line:
     `callees desktop::folder` returned 0 sites, and `callers desktop::valid_key` returned no
     caller, although its callers are the handlers `desktop` and `folder`.
   - With #2215 (draft) the residual names become per-function UNKNOWN. Measured on the same
     store:
     - `callees desktop::folder` → `UNKNOWN body-not-read … functions=280`.
     - `callers desktop::valid_key` lists the 31 unread bodies as UNKNOWN, `desktop` and
       `folder` among them.
     - `sweep`: 532 → 271 empty answers.
   - This makes the gap honest. It does not close it: the calls inside a handler are still
     not facts.
2. **The URL is not a fact.** The `REFERENCES router → handler` edge exists, but the string
   passed to `.route("…")` at that call site is not stored. The URL ↔ handler column above
   comes from reading `router()`.
3. **Whole-crate extraction does not fit in memory** on this crate (lancedb, arrow, datafusion
   in the dependency tree), even with every foreign crate opaque. Only a narrow
   `--start-from` run completes.

### Verdict

- **Proven by Codegraph:** the handler inventory (27 of 27), the router → handler edges
  (27 of 27, with offsets), the synchronous helper call graph, and the list of unread bodies.
- **Not yet proven:** the URL of each edge, and what each handler calls.

Both gaps belong in Codegraph's intake, in lance-graph, and neither needs this crate to change
shape:

1. **Route literals:** store the string argument of the `.route(…)` call next to the
   `REFERENCES` edge that already has its offset. Route declarations are the source; no
   hand-written route table.
2. **Async bodies:** investigate a targeted intake of `async fn` bodies (in Charon or in
   ruff_charon) before asking for handlers to be rewritten.

The goal is one graph, `View → Action → Route → Handler → Effect → Next View`, with each edge
graded as source-observed, inferred, runtime-confirmed or unknown. For our app the view and
action side comes from the a2ui-rs screen contracts and the Askama templates. For paperless-ngx
it comes from its Angular routes, guards, template `(click)` actions and component
relationships. Playwright then confirms a few selected real task paths at runtime; it is
not a replacement for the graph.

None of this is implemented here. The UNKNOWN half is in lance-graph #2215.
