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
- `body?` means its callees are not observed (see § Codegraph).

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
cg_query $STORE . sweep | callees SYM | callers SYM | refs SYM | symbols SUBSTR
```

| Measurement | Value |
|---|---|
| Charon on the whole crate (default start `crate`) | **killed (SIGKILL, out of memory)** after reaching ~10 GB; the machine has 15 GB |
| Whole crate, every foreign crate `--opaque` | killed again: 13.5 GB peak after 28 s |
| Started from `routes::router`, `desktop`, `glass` | 14 s, 0.88 GB peak, 30 MB LLBC |
| `ruff_charon import` | harvested 1,615, represented 1,584, residual 31, dangling 0 |
| `cg_query sweep` | 384 positions, 236 answered, 532 empty, 3 ms |
| Charon warnings | 255, all `Coroutine types are not supported yet` / `Coroutines are not supported` |

### What it shows

- **Every route's handler is in the store.**
  - Each of the 27 handlers has a CONFIRMED caller row:
    `{impl FnOnce<…> for <handler>}::call_once`. That is axum's `Handler` impl, the same 27
    names `router()` registers.
  - `callees routes::router` lists the axum builder calls (`Router::route`, `get`, `post`,
    `layer`, `with_state`, `DefaultBodyLimit::max`, `RequestBodyLimitLayer::new`) over 60 call
    sites.
- **Synchronous helpers are fully covered.** For example:
  - `desktop::entry_of → routes::format_unix_ms` (CONFIRMED);
  - `desktop::valid_key → str::bytes, RangeInclusive::contains, …`.

### What it cannot show, and how it fails

1. **No click path goes past the handler.**
   - Every axum handler is an `async fn`. Its body is a coroutine, and Charon `c69e231` does
     not translate coroutines (the 255 warnings).
   - So `callees` of every handler returns **0 sites**. For example, `callees desktop::folder`
     is empty, although the source calls `scope_of`, `registry`, `valid_key` and
     `FolderSource::listing`.
   - The empty set comes with **no UNKNOWN line**. This is the "false unaffected" defect PR
     #2000 recorded for public symbols, now for async bodies, and it is worse here: the bodies
     are in the observed files.
2. **False "no callers".**
   - `callers desktop::valid_key` reports 0 sites. Its real callers are the async handlers
     `desktop` and `folder`.
   - The line that does appear (`UNKNOWN external-callers-unobserved`) comes from the symbol
     being `pub`, not from the coroutine gap. A private function called only from a handler
     would get no UNKNOWN line at all.
3. **No route-path facts.**
   - The URL strings (`"/desktop/folders/:source/:key"`) are constants passed to
     `Router::route`. `refs routes::router` returns `confirmed=0 possible=0`.
   - The store holds *that* a handler exists, never *which URL* reaches it. A Klickweg (URL →
     handler → effect) therefore cannot be read from Codegraph today. The URL ↔ handler column
     above comes from reading `router()`.
4. **Whole-crate extraction does not fit in memory** on this crate (lancedb, arrow, datafusion
   in the dependency tree), even with every foreign crate opaque. Only a narrow
   `--start-from` run completes.

### Verdict

- **Proven by Codegraph:** the handler inventory (27 of 27 routes) and the synchronous helper
  call graph.
- **Not proven, read from source instead:** the click path from a URL to what the handler
  does, for all 27 routes.

Klickwege parity via Codegraph needs one of these:
- **(a)** coroutine support in the Charon fork, or a lowering of `async fn` bodies in ruff_charon;
- **(b)** handlers written as thin async shells over named sync functions (the archive and
  desktop logic would then be visible);
- **(c)** a route table as data (`const ROUTES: &[(&str, Method, Handler)]`), so the URL ↔
  handler edge is a fact the compiler sees.

Before (a), Codegraph should report an async body as `UNKNOWN coroutine-body-untranslated`, not
as an empty set. These are proposals; none is implemented in this PR.
