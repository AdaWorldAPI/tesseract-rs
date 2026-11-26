//! Physikalisches Glas: the shared `a2ui-theme` material `physical-glass`, the
//! background every AdaWorldAPI portal shows (Rosé unless the viewer picks
//! another in a `<select data-pg-bg-choice>`). The stylesheet, the script and
//! the wallpapers come from `a2ui_theme::asset`, the same bytes report-rs
//! serves; this module holds no copy of them.
//!
//! The archive uses the wallpaper only: every page loads the script with
//! `data-surfaces=":not(*)"`, so no surface asks for refraction maps and the
//! map endpoint (`/optics/surface/map`, `a2ui_paint::optics::map_service` in
//! report-rs) is not served here. Serving it would pull `a2ui-paint`, and with
//! it OGAR from git beside the OGAR path dependencies this crate already has.

use axum::extract::Path;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};

/// Where the pages load the material from. The stylesheet names its
/// wallpapers relative to itself, so these three paths move together.
pub const CSS_ROUTE: &str = "/static/physical-glass.css";
pub const JS_ROUTE: &str = "/static/physical-glass.js";
pub const WALLPAPER_ROUTE: &str = "/static/physical-glass/wallpapers/:file";

fn text(path: &str, media_type: &'static str) -> Response {
    match a2ui_theme::asset(path).and_then(|a| std::str::from_utf8(a.bytes).ok()) {
        Some(t) => (
            [
                (header::CONTENT_TYPE, media_type),
                (header::CACHE_CONTROL, "public, max-age=3600"),
            ],
            t,
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// `/static/physical-glass.css`.
pub async fn css() -> Response {
    text(
        "physical-glass/physical-glass.css",
        "text/css; charset=utf-8",
    )
}

/// `/static/physical-glass.js`.
pub async fn js() -> Response {
    text(
        "physical-glass/physical-glass.js",
        "text/javascript; charset=utf-8",
    )
}

/// `/static/physical-glass/wallpapers/:file`: a wallpaper of the shared
/// material. Only `.webp` names the theme crate holds; the lookup is an exact
/// match in its asset table, so no request reaches the file system.
pub async fn wallpaper(Path(file): Path<String>) -> Response {
    let asset = file
        .ends_with(".webp")
        .then(|| a2ui_theme::asset(&format!("physical-glass/wallpapers/{file}")))
        .flatten();
    match asset {
        Some(a) => (
            [
                (header::CONTENT_TYPE, a.media_type),
                (header::CACHE_CONTROL, "public, max-age=86400"),
            ],
            a.bytes,
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}
