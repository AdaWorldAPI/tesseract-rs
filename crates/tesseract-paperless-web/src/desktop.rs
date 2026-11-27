//! The desktop view: folders as glass tiles on the Rosé wallpaper, each opening
//! as a window (an iframe of the folder's own page) with minimise, close,
//! "open as page" and a taskbar entry. A document in a folder window opens its
//! detail page as another window. Without script every tile and entry is a
//! plain link to the same page. Modelled on report-rs's desktop view
//! (`frame.rs::desktop_page`, `static/desktop.js`).
//!
//! # Folder sources
//!
//! The shell never names a backend. It asks every [`FolderSource`] in
//! [`registry`] for its folders and, for one folder, its entries. The archive
//! itself is the first source ([`ArchiveSource`]: "All documents" plus one
//! folder per correspondent, document type and tag). A later source (an S3
//! bucket on Tigris, a SharePoint folder through Microsoft Graph) is one more
//! `impl FolderSource` and one more line in [`registry`]; the routes, the
//! templates and the window script stay as they are.
//!
//! # Scope
//!
//! A source answers with the caller's [`Scope`] in hand and lists only what
//! the caller may see. For the archive that is the same filter the document
//! list applies, so a scoped caller's folders count and list its own documents
//! only, and an entry links to a detail page that answers 404 for anything
//! else anyway.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use askama::Template;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::Extension;
use tesseract_paperless::archive_meta::MetaKind;

use crate::auth::Caller;
use crate::customers::Scope;
use crate::state::AppState;

/// The desktop route, linked from every page's navigation.
pub const DESKTOP_ROUTE: &str = "/desktop";
/// Entries shown in one folder window; the window says when there are more.
pub const FOLDER_LIMIT: usize = 200;

/// What a source sees of the request.
#[derive(Clone, Copy)]
pub struct FolderCtx<'a> {
    pub state: &'a AppState,
    pub scope: &'a Scope,
}

/// The picture on a folder tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FolderIcon {
    /// Every document of a source.
    All,
    Correspondent,
    DocumentType,
    Tag,
}

impl FolderIcon {
    /// Inner SVG markup, 24×24 viewBox, stroked.
    fn svg(self) -> &'static str {
        match self {
            Self::All => "<path d=\"M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z\"/>",
            Self::Correspondent => "<circle cx=\"12\" cy=\"8\" r=\"4\"/><path d=\"M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6\"/>",
            Self::DocumentType => "<path d=\"M7 3h7l5 5v13H7z\"/><path d=\"M14 3v5h5M10 13h6M10 17h6\"/>",
            Self::Tag => "<path d=\"M3 12V4h8l10 10-8 8z\"/><circle cx=\"7.5\" cy=\"7.5\" r=\"1.5\"/>",
        }
    }
}

/// One folder a source offers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Folder {
    /// Unique within its source; `[a-z0-9_-]`, 1..=64 characters.
    pub key: String,
    pub label: String,
    pub icon: FolderIcon,
    /// Entries the caller would see in it.
    pub count: usize,
}

/// One entry in a folder window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// Window id when it opens on the desktop; unique per entry.
    pub id: String,
    pub label: String,
    /// One line under the label (date, pages, size).
    pub detail: String,
    /// The page the entry opens, same-origin.
    pub href: String,
}

/// The contents of one folder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Listing {
    pub label: String,
    /// At most [`FOLDER_LIMIT`].
    pub entries: Vec<Entry>,
    /// Entries in the folder, the shown ones included.
    pub total: usize,
}

/// A boxed future, so sources can sit behind `dyn`.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Anything the desktop can show folders of.
pub trait FolderSource: Send + Sync {
    /// The source's segment in folder URLs; `[a-z0-9-]`, unique in [`registry`].
    fn id(&self) -> &'static str;
    /// The source's name, for a tile's tooltip.
    fn label(&self) -> &'static str;
    /// The folders the caller may open, in display order.
    fn folders<'a>(&'a self, ctx: FolderCtx<'a>) -> BoxFuture<'a, Result<Vec<Folder>, String>>;
    /// One folder's entries; `Ok(None)` for a key this source does not offer
    /// to this caller (the shell answers 404, the same as an unknown key).
    fn listing<'a>(
        &'a self,
        ctx: FolderCtx<'a>,
        key: &'a str,
    ) -> BoxFuture<'a, Result<Option<Listing>, String>>;
}

/// Every folder source, in the order their tiles appear. The one place a new
/// source is added.
#[must_use]
pub fn registry() -> Vec<Arc<dyn FolderSource>> {
    vec![Arc::new(ArchiveSource)]
}

/// `[a-z0-9_-]`, 1..=64 characters: what a source id or folder key may be.
/// Checked before any source is asked, so a key never reaches a backend raw.
#[must_use]
pub fn valid_key(s: &str) -> bool {
    (1..=64).contains(&s.len())
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

// ---- the archive as a folder source ---------------------------------------

/// The archive: "All documents", then one folder per active correspondent,
/// document type and tag.
pub struct ArchiveSource;

/// Folder key of a definition: `<kind>-<id>`.
fn definition_key(kind: MetaKind, id: u32) -> String {
    format!("{}-{id}", kind.as_str())
}

/// Inverse of [`definition_key`].
fn parse_definition_key(key: &str) -> Option<(MetaKind, u32)> {
    let (kind, id) = key.rsplit_once('-')?;
    let id: u32 = id.parse().ok()?;
    // `parse` accepts "+1" and leading zeros; only the canonical spelling is a key.
    (id.to_string() == key[kind.len() + 1..]).then_some(())?;
    Some((MetaKind::parse(kind)?, id))
}

fn kind_icon(kind: MetaKind) -> FolderIcon {
    match kind {
        MetaKind::Correspondent => FolderIcon::Correspondent,
        MetaKind::DocumentType => FolderIcon::DocumentType,
        MetaKind::Tag => FolderIcon::Tag,
    }
}

/// Display order of the kinds on the desktop.
const KINDS: [MetaKind; 3] = [
    MetaKind::Correspondent,
    MetaKind::DocumentType,
    MetaKind::Tag,
];

impl ArchiveSource {
    /// Every document the caller may see, newest first.
    async fn visible(
        ctx: FolderCtx<'_>,
    ) -> Result<Vec<tesseract_paperless::store::DocumentRow>, String> {
        let store = &ctx.state.store;
        let n = store.count().await.map_err(|e| e.to_string())?.max(1);
        let rows = store.list(n).await.map_err(|e| e.to_string())?;
        crate::routes::filter_visible(ctx.state, ctx.scope, rows)
            .await
            .map_err(|e| e.to_string())
    }

    /// The hashes the caller may see, read from the hash column alone (no
    /// document bodies): what the tile counts need. Same rule as
    /// `routes::filter_visible`: everything for `Scope::All`, otherwise only
    /// documents whose customer the scope covers.
    async fn visible_hashes(ctx: FolderCtx<'_>) -> Result<Vec<String>, String> {
        let store = &ctx.state.store;
        let hashes = store.hashes().await.map_err(|e| e.to_string())?;
        if *ctx.scope == Scope::All {
            return Ok(hashes);
        }
        let customers = store.meta().customers().await.map_err(|e| e.to_string())?;
        Ok(hashes
            .into_iter()
            .filter(|h| ctx.scope.covers(customers.get(h).map(String::as_str)))
            .collect())
    }

    /// Visible document hashes per active definition.
    async fn members(
        ctx: FolderCtx<'_>,
        visible: &HashSet<&str>,
    ) -> Result<HashMap<(MetaKind, u32), HashSet<String>>, String> {
        let meta = ctx.state.store.meta();
        let mut out: HashMap<(MetaKind, u32), HashSet<String>> = HashMap::new();
        for a in meta.all_assignments().await.map_err(|e| e.to_string())? {
            if visible.contains(a.content_sha256_hex.as_str()) {
                out.entry((a.kind, a.definition_id))
                    .or_default()
                    .insert(a.content_sha256_hex);
            }
        }
        Ok(out)
    }
}

fn entry_of(r: &tesseract_paperless::store::DocumentRow) -> Entry {
    let label = r
        .filename
        .clone()
        .unwrap_or_else(|| "(untitled)".to_string());
    let pages = if r.page_count == 1 {
        "1 page".to_string()
    } else {
        format!("{} pages", r.page_count)
    };
    Entry {
        id: format!("doc-{}", r.content_sha256_hex),
        label,
        detail: format!(
            "{} \u{b7} {pages}",
            crate::routes::format_unix_ms(r.ingested_at_unix_ms)
        ),
        href: format!("/documents/{}", r.content_sha256_hex),
    }
}

fn listing_of(label: String, rows: &[&tesseract_paperless::store::DocumentRow]) -> Listing {
    Listing {
        label,
        total: rows.len(),
        entries: rows
            .iter()
            .take(FOLDER_LIMIT)
            .map(|r| entry_of(r))
            .collect(),
    }
}

impl FolderSource for ArchiveSource {
    fn id(&self) -> &'static str {
        "archive"
    }

    fn label(&self) -> &'static str {
        "Archive"
    }

    fn folders<'a>(&'a self, ctx: FolderCtx<'a>) -> BoxFuture<'a, Result<Vec<Folder>, String>> {
        Box::pin(async move {
            let hashes = Self::visible_hashes(ctx).await?;
            let visible: HashSet<&str> = hashes.iter().map(String::as_str).collect();
            let members = Self::members(ctx, &visible).await?;
            let defs = ctx
                .state
                .store
                .meta()
                .definitions()
                .await
                .map_err(|e| e.to_string())?;
            let mut folders = vec![Folder {
                key: "all".to_string(),
                label: "All documents".to_string(),
                icon: FolderIcon::All,
                count: visible.len(),
            }];
            for kind in KINDS {
                let mut of_kind: Vec<_> = defs
                    .iter()
                    .filter(|d| d.kind == kind && !d.retired)
                    .collect();
                of_kind.sort_by_key(|d| d.name.to_lowercase());
                folders.extend(of_kind.into_iter().map(|d| {
                    Folder {
                        key: definition_key(kind, d.definition_id),
                        label: d.name.clone(),
                        icon: kind_icon(kind),
                        count: members
                            .get(&(kind, d.definition_id))
                            .map_or(0, HashSet::len),
                    }
                }));
            }
            Ok(folders)
        })
    }

    fn listing<'a>(
        &'a self,
        ctx: FolderCtx<'a>,
        key: &'a str,
    ) -> BoxFuture<'a, Result<Option<Listing>, String>> {
        Box::pin(async move {
            if key == "all" {
                let rows = Self::visible(ctx).await?;
                let refs: Vec<_> = rows.iter().collect();
                return Ok(Some(listing_of("All documents".to_string(), &refs)));
            }
            let Some((kind, id)) = parse_definition_key(key) else {
                return Ok(None);
            };
            let defs = ctx
                .state
                .store
                .meta()
                .definitions()
                .await
                .map_err(|e| e.to_string())?;
            let Some(def) = defs
                .iter()
                .find(|d| d.kind == kind && d.definition_id == id && !d.retired)
            else {
                return Ok(None);
            };
            let rows = Self::visible(ctx).await?;
            let visible: HashSet<&str> =
                rows.iter().map(|r| r.content_sha256_hex.as_str()).collect();
            let members = Self::members(ctx, &visible).await?;
            let empty = HashSet::new();
            let these = members.get(&(kind, id)).unwrap_or(&empty);
            let refs: Vec<_> = rows
                .iter()
                .filter(|r| these.contains(&r.content_sha256_hex))
                .collect();
            Ok(Some(listing_of(def.name.clone(), &refs)))
        })
    }
}

// ---- the shell -------------------------------------------------------------

/// A tile on the desktop.
struct TileView {
    id: String,
    label: String,
    hint: String,
    href: String,
    icon: &'static str,
    count: usize,
}

#[derive(Template)]
#[template(path = "desktop.html")]
struct DesktopTemplate {
    brand_name: String,
    sign_in: bool,
    tiles: Vec<TileView>,
    /// A source that failed to answer, with why; the others still show.
    errors: Vec<String>,
}

#[derive(Template)]
#[template(path = "folder.html")]
struct FolderTemplate {
    label: String,
    source: &'static str,
    entries: Vec<Entry>,
    total: usize,
}

fn page<T: Template>(status: StatusCode, t: &T) -> Response {
    match t.render() {
        Ok(s) => (status, Html(s)).into_response(),
        Err(e) => {
            eprintln!("template render error: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, "internal template error").into_response()
        }
    }
}

/// `GET /desktop`.
pub async fn desktop(
    State(state): State<Arc<AppState>>,
    caller: Option<Extension<Caller>>,
) -> Response {
    let scope = crate::routes::scope_of(&state, caller.as_ref());
    let ctx = FolderCtx {
        state: &state,
        scope: &scope,
    };
    let mut tiles = Vec::new();
    let mut errors = Vec::new();
    for src in registry() {
        match src.folders(ctx).await {
            Ok(folders) => tiles.extend(folders.into_iter().filter(|f| valid_key(&f.key)).map(
                |f| TileView {
                    id: format!("{}-{}", src.id(), f.key),
                    hint: format!("{} \u{b7} {} \u{b7} {}", src.label(), f.label, f.count),
                    href: format!("{DESKTOP_ROUTE}/folders/{}/{}", src.id(), f.key),
                    icon: f.icon.svg(),
                    count: f.count,
                    label: f.label,
                },
            )),
            Err(e) => errors.push(format!("{}: {e}", src.label())),
        }
    }
    page(
        StatusCode::OK,
        &DesktopTemplate {
            brand_name: crate::branding::get().name.clone(),
            sign_in: crate::auth::enabled(),
            tiles,
            errors,
        },
    )
}

/// `GET /desktop/folders/:source/:key`: one folder, as the page its window
/// shows. An unknown source, an invalid key, and a folder the source does not
/// offer this caller are the same 404.
pub async fn folder(
    State(state): State<Arc<AppState>>,
    caller: Option<Extension<Caller>>,
    Path((source, key)): Path<(String, String)>,
) -> Response {
    let not_found = || (StatusCode::NOT_FOUND, "no such folder").into_response();
    if !valid_key(&source) || !valid_key(&key) {
        return not_found();
    }
    let Some(src) = registry().into_iter().find(|s| s.id() == source) else {
        return not_found();
    };
    let scope = crate::routes::scope_of(&state, caller.as_ref());
    let ctx = FolderCtx {
        state: &state,
        scope: &scope,
    };
    match src.listing(ctx, &key).await {
        Ok(Some(l)) => page(
            StatusCode::OK,
            &FolderTemplate {
                label: l.label,
                source: src.label(),
                entries: l.entries,
                total: l.total,
            },
        ),
        Ok(None) => not_found(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("folder read failed: {e}"),
        )
            .into_response(),
    }
}

/// `GET /static/desktop.js`: the window script.
pub async fn script() -> Response {
    (
        [
            (
                axum::http::header::CONTENT_TYPE,
                "text/javascript; charset=utf-8",
            ),
            (axum::http::header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        include_str!("../static/desktop.js"),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_keys_round_trip_and_reject_non_canonical_spellings() {
        for kind in KINDS {
            for id in [0u32, 1, 42, u32::MAX] {
                assert_eq!(
                    parse_definition_key(&definition_key(kind, id)),
                    Some((kind, id))
                );
            }
        }
        for bad in [
            "tag-", "tag-01", "tag-+1", "tag--1", "-1", "label-1", "tag", "all", "tag-1x",
        ] {
            assert_eq!(parse_definition_key(bad), None, "{bad}");
        }
    }

    /// Disable: return `true` from `valid_key`; the uppercase, slash, dot and
    /// over-long cases then pass.
    #[test]
    fn valid_key_admits_the_url_alphabet_only() {
        for ok in ["all", "tag-3", "document_type-12", "s3-tigris", "a"] {
            assert!(valid_key(ok), "{ok}");
        }
        let long = "a".repeat(65);
        for bad in ["", "Tag-1", "a/b", "..", "a.b", "a b", "ä", long.as_str()] {
            assert!(!valid_key(bad), "{bad:?}");
        }
    }

    #[test]
    fn source_ids_are_unique_and_valid() {
        let ids: Vec<_> = registry().iter().map(|s| s.id()).collect();
        let unique: HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len());
        assert!(ids.iter().all(|i| valid_key(i) && !i.contains('_')));
    }
}
