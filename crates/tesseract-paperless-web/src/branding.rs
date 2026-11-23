//! Optional branding: a deployment can show its own name, tagline, logo and
//! colours without a code change.
//!
//! With nothing set the pages are exactly the neutral archive
//! (`tesseract-paperless`, the original tagline, accent `#2563eb`, no logo).
//! No customer's identity is compiled in: the image is neutral, and a
//! deployment brings its own identity through one of two routes.
//!
//! # A brand document: `BRAND_TOML` or `BRAND_FILE`
//!
//! `BRAND_TOML` holds the TOML text itself (for hosts such as Railway that only
//! pass variables); `BRAND_FILE` is the path of a TOML file. Setting both is
//! refused. The document has one `[brand]` table; every key is optional and an
//! unset key keeps its default. Unknown keys are refused, so a typo cannot pass
//! unnoticed.
//!
//! | key | effect |
//! |---|---|
//! | `name` | page titles and the main heading (1 to 60 characters) |
//! | `tagline` | the line under the heading on the upload page (up to 240 characters) |
//! | `logo_url` | an image beside the heading: an `https://` URL or a same-site path starting with a single `/` |
//! | `font_family` | the page font |
//! | `bg` | the page background |
//! | `fg` | the body text |
//! | `h1` | headings |
//! | `muted` | secondary text |
//! | `accent` | links, buttons and the current menu entry |
//! | `on_accent` | text on an accent background; computed from `accent` when unset |
//! | `border` | card, input and table borders |
//! | `card_bg` | card and input background |
//! | `subtle_bg` | table headings and the quick-menu rail |
//! | `side_bg` | the top bar, side menu and footer background |
//! | `side_fg` | text on `side_bg`; computed from `side_bg` when unset |
//! | `side_hover` | the marker beside a hovered side-menu entry (backgrounds never change, so side text keeps its contrast) |
//! | `radius` | card corner radius |
//! | `radius_sm` | input and button corner radius |
//! | `elevation` | card shadow (`none` for flat cards) |
//!
//! The sixteen colour, font and shape tokens reach the stylesheet as CSS custom
//! properties on `:root` (`font_family` becomes `--brand-font-family`, and so
//! on); the stylesheet uses `var(--brand-*)` everywhere instead of fixed
//! colours. The pages are light only (`color-scheme: light`).
//!
//! A token is accepted only when it is not empty and every character is an
//! ASCII letter or digit or one of ``space # ( ) , . % - _ + / " '``. That
//! refuses `;`, braces, `<`, `>`, backslashes and line breaks, so a value can
//! neither close the declaration nor the `<style>` element. `on_accent` and
//! `side_fg` are computed (white or black, whichever contrasts more) only from
//! a `#rgb` / `#rrggbb` colour; with any other colour syntax for `accent` or
//! `side_bg`, set them explicitly.
//!
//! An error in this document is fatal: the deployment asked for this brand, so
//! a refused value stops the start ([`BrandError`]) instead of quietly serving
//! the neutral look.
//!
//! Example: the corporate design the `datagroup` theme of report-rs uses,
//! set through `BRAND_TOML` (documentation only, nothing here is compiled in):
//!
//! ```toml
//! [brand]
//! name = "DATAGROUP Dokumente"
//! accent = "#c8102e"
//! muted = "#4a4a4a"
//! fg = "#1a1a1a"
//! h1 = "#1a1a1a"
//! bg = "#f5f5f5"
//! border = "#e0e6ec"
//! subtle_bg = "#eef1f4"
//! side_bg = "#0d3a64"
//! radius = "14px"
//! radius_sm = "10px"
//! elevation = "0 4px 16px rgba(0,0,0,.08)"
//! ```
//!
//! # Per-setting variables
//!
//! These keep their original behaviour: an invalid value is ignored with one
//! line on stderr and the default is used, never fatal.
//!
//! | variable | effect | accepted |
//! |---|---|---|
//! | `BRAND_NAME` | page titles and the main heading | 1 to 60 characters after trimming |
//! | `BRAND_TAGLINE` | the line under the heading on the upload page | up to 240 characters |
//! | `BRAND_ACCENT` | the `accent` token | `#rgb` or `#rrggbb` |
//! | `BRAND_LOGO_URL` | an image beside the heading | an `https://` URL or a same-site path starting with a single `/` |
//! | `BRAND_HEADER_BG` | the `side_bg` token (top bar, side menu, footer) | `#rgb` or `#rrggbb` |
//!
//! Combining `BRAND_TOML` or `BRAND_FILE` with any of these is refused as
//! ambiguous: two sources would disagree about which one wins.
//!
//! An empty or blank value counts as unset everywhere. All text values are
//! also HTML-escaped by the templates.
//!
//! When a custom name is set the footer adds "powered by tesseract-rs", so a
//! rebranded deployment still names the engine underneath. paperless-rs uses
//! this to ship the archive under its own name.

use std::fmt;
use std::sync::OnceLock;

use serde::Deserialize;

/// The default name, the archive's own.
pub const DEFAULT_NAME: &str = "tesseract-paperless";
/// The default tagline on the upload page.
pub const DEFAULT_TAGLINE: &str = "Upload a scan or photo. It is OCR'd with the pure-Rust recognizer, deduplicated by content hash, and archived \u{2014} searchable, browsable, paperless-ngx-shaped.";
/// The default accent colour.
pub const DEFAULT_ACCENT: &str = "#2563eb";

const DEFAULT_SIDE_BG: &str = "#1f2937";

const MAX_NAME: usize = 60;
const MAX_TAGLINE: usize = 240;
const MAX_URL: usize = 2048;

/// The legacy per-setting variables, which cannot be combined with a brand
/// document.
const LEGACY_VARS: [&str; 5] = [
    "BRAND_NAME",
    "BRAND_TAGLINE",
    "BRAND_ACCENT",
    "BRAND_LOGO_URL",
    "BRAND_HEADER_BG",
];

/// The branding the pages render.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Branding {
    pub name: String,
    pub tagline: String,
    logo_url: Option<String>,
    pub font_family: String,
    pub bg: String,
    pub fg: String,
    pub h1: String,
    pub muted: String,
    pub accent: String,
    pub on_accent: String,
    pub border: String,
    pub card_bg: String,
    pub subtle_bg: String,
    pub side_bg: String,
    pub side_fg: String,
    pub side_hover: String,
    pub radius: String,
    pub radius_sm: String,
    pub elevation: String,
}

impl Default for Branding {
    fn default() -> Self {
        Branding {
            name: DEFAULT_NAME.to_owned(),
            tagline: DEFAULT_TAGLINE.to_owned(),
            logo_url: None,
            font_family: "system-ui, -apple-system, \"Segoe UI\", Roboto, sans-serif".to_owned(),
            bg: "#ffffff".to_owned(),
            fg: "#1f2937".to_owned(),
            h1: "#111827".to_owned(),
            muted: "#6b7280".to_owned(),
            accent: DEFAULT_ACCENT.to_owned(),
            on_accent: on_colour(DEFAULT_ACCENT).to_owned(),
            border: "#e5e7eb".to_owned(),
            card_bg: "#ffffff".to_owned(),
            subtle_bg: "#f3f4f6".to_owned(),
            side_bg: DEFAULT_SIDE_BG.to_owned(),
            side_fg: on_colour(DEFAULT_SIDE_BG).to_owned(),
            side_hover: "rgba(255,255,255,.08)".to_owned(),
            radius: "12px".to_owned(),
            radius_sm: "8px".to_owned(),
            elevation: "none".to_owned(),
        }
    }
}

/// Why the brand document was refused. All of these are fatal at start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BrandError {
    /// `BRAND_FILE` names a file that cannot be read.
    Read(String),
    /// `BRAND_TOML` and `BRAND_FILE` are both set.
    Both,
    /// A brand document is set together with a per-setting variable.
    Mixed,
    /// The document is not valid TOML, or has a key or table that does not exist.
    Parse(String),
    /// A value is not accepted for its key.
    Value { key: &'static str, value: String },
}

impl fmt::Display for BrandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BrandError::Read(e) => write!(f, "BRAND_FILE cannot be read: {e}"),
            BrandError::Both => write!(
                f,
                "BRAND_TOML and BRAND_FILE are both set: use one of them"
            ),
            BrandError::Mixed => write!(
                f,
                "BRAND_TOML / BRAND_FILE cannot be combined with BRAND_NAME, BRAND_TAGLINE, \
                 BRAND_ACCENT, BRAND_LOGO_URL or BRAND_HEADER_BG: put everything in the brand document"
            ),
            BrandError::Parse(e) => write!(f, "the brand document is not valid: {e}"),
            BrandError::Value { key, value } => write!(
                f,
                "brand {key}: {value:?} is not accepted \
                 (see the branding module documentation for the accepted values)"
            ),
        }
    }
}

impl std::error::Error for BrandError {}

/// The `[brand]` table of a brand document. Every key is optional.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct BrandTable {
    name: Option<String>,
    tagline: Option<String>,
    logo_url: Option<String>,
    font_family: Option<String>,
    bg: Option<String>,
    fg: Option<String>,
    h1: Option<String>,
    muted: Option<String>,
    accent: Option<String>,
    on_accent: Option<String>,
    border: Option<String>,
    card_bg: Option<String>,
    subtle_bg: Option<String>,
    side_bg: Option<String>,
    side_fg: Option<String>,
    side_hover: Option<String>,
    radius: Option<String>,
    radius_sm: Option<String>,
    elevation: Option<String>,
}

/// The root of a brand document.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct BrandFile {
    brand: Option<BrandTable>,
}

/// `Some` only for a value with something in it: blank counts as unset.
fn nonblank(v: Option<String>) -> Option<String> {
    v.filter(|v| !v.trim().is_empty())
}

impl Branding {
    /// Reads the settings from the environment and, for `BRAND_FILE`, the file
    /// system. A brand document (`BRAND_TOML` / `BRAND_FILE`) is checked
    /// strictly and any problem is an error; the per-setting variables only
    /// warn (see the module documentation).
    ///
    /// `get` returns a variable's value or `None`; `read_file` reads the file
    /// named by `BRAND_FILE`; `warn` receives each ignored per-setting value.
    pub fn from_sources(
        get: impl Fn(&str) -> Option<String>,
        read_file: impl Fn(&str) -> std::io::Result<String>,
        warn: impl FnMut(String),
    ) -> Result<Branding, BrandError> {
        let set = |k: &str| nonblank(get(k));
        let legacy_set = || LEGACY_VARS.iter().any(|k| set(k).is_some());
        match (set("BRAND_TOML"), set("BRAND_FILE")) {
            (Some(_), Some(_)) => Err(BrandError::Both),
            (None, None) => Ok(Self::from_vars(&get, warn)),
            (Some(text), None) => {
                if legacy_set() {
                    return Err(BrandError::Mixed);
                }
                Self::from_toml(&text)
            }
            (None, Some(path)) => {
                if legacy_set() {
                    return Err(BrandError::Mixed);
                }
                let text =
                    read_file(&path).map_err(|e| BrandError::Read(format!("{path}: {e}")))?;
                Self::from_toml(&text)
            }
        }
    }

    /// Reads a brand document. Unset keys keep their defaults; `on_accent` and
    /// `side_fg` are computed from the final `accent` / `side_bg` unless given.
    fn from_toml(text: &str) -> Result<Branding, BrandError> {
        let file: BrandFile = toml::from_str(text).map_err(|e| BrandError::Parse(e.to_string()))?;
        let mut t = file.brand.unwrap_or_default();
        let on_accent_given = t.on_accent.as_deref().is_some_and(|v| !v.trim().is_empty());
        let side_fg_given = t.side_fg.as_deref().is_some_and(|v| !v.trim().is_empty());
        let mut b = Branding::default();

        if let Some(v) = nonblank(t.name.take()) {
            let v = v.trim();
            if !(1..=MAX_NAME).contains(&v.chars().count()) {
                return Err(BrandError::Value {
                    key: "name",
                    value: v.to_owned(),
                });
            }
            b.name = v.to_owned();
        }
        if let Some(v) = nonblank(t.tagline.take()) {
            let v = v.trim();
            if v.chars().count() > MAX_TAGLINE {
                return Err(BrandError::Value {
                    key: "tagline",
                    value: v.to_owned(),
                });
            }
            b.tagline = v.to_owned();
        }
        if let Some(v) = nonblank(t.logo_url.take()) {
            let v = v.trim();
            if !is_logo_url(v) {
                return Err(BrandError::Value {
                    key: "logo_url",
                    value: v.to_owned(),
                });
            }
            b.logo_url = Some(v.to_owned());
        }

        macro_rules! tokens {
            ($($f:ident),+ $(,)?) => {$(
                if let Some(v) = nonblank(t.$f.take()) {
                    b.$f = css_value(stringify!($f), &v)?;
                }
            )+};
        }
        tokens!(
            font_family,
            bg,
            fg,
            h1,
            muted,
            accent,
            on_accent,
            border,
            card_bg,
            subtle_bg,
            side_bg,
            side_fg,
            side_hover,
            radius,
            radius_sm,
            elevation,
        );

        // The text colour on the accent / side background is computed only from a
        // `#rgb` / `#rrggbb` value. For any other colour syntax the contrast is
        // unknown, so the brand must name the text colour itself rather than get a
        // guessed white.
        if !on_accent_given {
            if !is_hex_colour(&b.accent) {
                return Err(BrandError::Value {
                    key: "on_accent",
                    value: "required when accent is not #rgb or #rrggbb".to_owned(),
                });
            }
            b.on_accent = on_colour(&b.accent).to_owned();
        }
        if !side_fg_given {
            if !is_hex_colour(&b.side_bg) {
                return Err(BrandError::Value {
                    key: "side_fg",
                    value: "required when side_bg is not #rgb or #rrggbb".to_owned(),
                });
            }
            b.side_fg = on_colour(&b.side_bg).to_owned();
        }
        Ok(b)
    }

    /// Reads the per-setting variables through `get`, which returns a
    /// variable's value or `None`. Rejected values are reported through `warn`
    /// and replaced by the default.
    pub fn from_vars(get: impl Fn(&str) -> Option<String>, mut warn: impl FnMut(String)) -> Self {
        // An empty value means unset: a platform may define a variable with no
        // value, and that must not read as a rejected setting.
        let get = |k: &str| get(k).filter(|v| !v.trim().is_empty());
        let mut b = Branding::default();
        if let Some(v) = get("BRAND_NAME") {
            let v = v.trim();
            let n = v.chars().count();
            if (1..=MAX_NAME).contains(&n) {
                b.name = v.to_owned();
            } else {
                warn(format!(
                    "BRAND_NAME ignored: must be 1 to {MAX_NAME} characters"
                ));
            }
        }
        if let Some(v) = get("BRAND_TAGLINE") {
            let v = v.trim();
            if v.chars().count() <= MAX_TAGLINE {
                b.tagline = v.to_owned();
            } else {
                warn(format!(
                    "BRAND_TAGLINE ignored: longer than {MAX_TAGLINE} characters"
                ));
            }
        }
        if let Some(v) = get("BRAND_ACCENT") {
            let v = v.trim();
            if is_hex_colour(v) {
                b.accent = v.to_ascii_lowercase();
                b.on_accent = on_colour(&b.accent).to_owned();
            } else {
                warn("BRAND_ACCENT ignored: use #rgb or #rrggbb".to_owned());
            }
        }
        if let Some(v) = get("BRAND_HEADER_BG") {
            let v = v.trim();
            if is_hex_colour(v) {
                b.side_bg = v.to_ascii_lowercase();
                b.side_fg = on_colour(&b.side_bg).to_owned();
            } else {
                warn("BRAND_HEADER_BG ignored: use #rgb or #rrggbb".to_owned());
            }
        }
        if let Some(v) = get("BRAND_LOGO_URL") {
            let v = v.trim();
            if is_logo_url(v) {
                b.logo_url = Some(v.to_owned());
            } else {
                warn(
                    "BRAND_LOGO_URL ignored: use an https:// URL or a path starting with /"
                        .to_owned(),
                );
            }
        }
        b
    }

    /// The logo URL, if one was accepted.
    #[must_use]
    pub fn logo_url(&self) -> Option<&str> {
        self.logo_url.as_deref()
    }

    /// Whether a custom name is set, which adds the "powered by tesseract-rs"
    /// credit.
    #[must_use]
    pub fn is_custom(&self) -> bool {
        self.name != DEFAULT_NAME
    }

    /// The sixteen tokens in stylesheet order, by their underscore names.
    fn tokens(&self) -> [(&'static str, &str); 16] {
        [
            ("font_family", &self.font_family),
            ("bg", &self.bg),
            ("fg", &self.fg),
            ("h1", &self.h1),
            ("muted", &self.muted),
            ("accent", &self.accent),
            ("on_accent", &self.on_accent),
            ("border", &self.border),
            ("card_bg", &self.card_bg),
            ("subtle_bg", &self.subtle_bg),
            ("side_bg", &self.side_bg),
            ("side_fg", &self.side_fg),
            ("side_hover", &self.side_hover),
            ("radius", &self.radius),
            ("radius_sm", &self.radius_sm),
            ("elevation", &self.elevation),
        ]
    }

    /// One `:root { --brand-…: value; … }` rule with all sixteen tokens. Every
    /// value passed [`is_css_value`] (or is a built-in default), so the result
    /// is safe to place unescaped inside a `<style>` element.
    #[must_use]
    pub fn css_vars(&self) -> String {
        let declarations: Vec<String> = self
            .tokens()
            .iter()
            .map(|(k, v)| format!("--brand-{}: {v};", k.replace('_', "-")))
            .collect();
        format!(":root {{ {} }}", declarations.join(" "))
    }
}

/// White or black, whichever contrasts more with `colour` (WCAG relative
/// luminance). The better of the two always reaches at least 4.58:1. A colour
/// that is not `#rgb` / `#rrggbb` has no luminance here and gets white.
fn on_colour(colour: &str) -> &'static str {
    let l = relative_luminance(colour).unwrap_or(0.0);
    let with_white = 1.05 / (l + 0.05);
    let with_black = (l + 0.05) / 0.05;
    if with_white >= with_black {
        "#fff"
    } else {
        "#000"
    }
}

/// WCAG relative luminance of an accepted `#rgb` / `#rrggbb` colour.
fn relative_luminance(hex: &str) -> Option<f64> {
    let h = hex.strip_prefix('#')?;
    let full: String = match h.len() {
        3 => h.chars().flat_map(|c| [c, c]).collect(),
        6 => h.to_owned(),
        _ => return None,
    };
    let channel = |i: usize| -> Option<f64> {
        let v = f64::from(u8::from_str_radix(full.get(i..i + 2)?, 16).ok()?) / 255.0;
        Some(if v <= 0.040_45 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        })
    };
    Some(0.2126 * channel(0)? + 0.7152 * channel(2)? + 0.0722 * channel(4)?)
}

fn is_hex_colour(v: &str) -> bool {
    let Some(hex) = v.strip_prefix('#') else {
        return false;
    };
    matches!(hex.len(), 3 | 6) && hex.bytes().all(|c| c.is_ascii_hexdigit())
}

/// Whether `v` may be written into the stylesheet as a CSS value: something in
/// it, and only ASCII letters, digits and ``space # ( ) , . % - _ + / " '``.
/// The refused characters (`;`, braces, `<`, `>`, backslash, line breaks) are
/// exactly the ones that could end the declaration, the rule or the `<style>`
/// element. It is checked on the raw value, so a trailing line break is refused
/// instead of being trimmed away.
fn is_css_value(v: &str) -> bool {
    !v.trim().is_empty()
        && v.chars()
            .all(|c| c.is_ascii_alphanumeric() || " #(),.%-_+/\"'".contains(c))
        && is_closed(v)
}

/// Every parenthesis is closed and every quote ends. An unclosed `calc(` or
/// `"Font` would make the CSS parser swallow the declarations after it, so
/// later tokens such as `--brand-accent` would silently be undefined.
fn is_closed(v: &str) -> bool {
    let mut depth = 0usize;
    let mut quote: Option<char> = None;
    for c in v.chars() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), _) => {}
            (None, '"' | '\'') => quote = Some(c),
            (None, '(') => depth += 1,
            (None, ')') => match depth.checked_sub(1) {
                Some(d) => depth = d,
                None => return false,
            },
            (None, _) => {}
        }
    }
    depth == 0 && quote.is_none()
}

/// The trimmed value when [`is_css_value`] accepts `raw`, else the error for `key`.
fn css_value(key: &'static str, raw: &str) -> Result<String, BrandError> {
    if is_css_value(raw) {
        Ok(raw.trim().to_owned())
    } else {
        Err(BrandError::Value {
            key,
            value: raw.to_owned(),
        })
    }
}

fn is_logo_url(v: &str) -> bool {
    // A backslash is rejected anywhere: browsers read `/\host/x` as
    // `//host/x`, a different site, and no ordinary path needs one.
    if v.is_empty()
        || v.len() > MAX_URL
        || v.chars()
            .any(|c| c.is_whitespace() || c.is_control() || c == '\\')
    {
        return false;
    }
    let same_site = v.starts_with('/') && !v.starts_with("//");
    // An absolute address needs a host, and no userinfo that could make the
    // visible host a different one (`https://good@evil/`).
    let https_host = v.strip_prefix("https://").is_some_and(|rest| {
        let host = rest.split(['/', '?', '#']).next().unwrap_or("");
        !host.is_empty() && !host.contains('@')
    });
    same_site || https_host
}

/// The branding chosen at start by [`install`].
static INSTALLED: OnceLock<Branding> = OnceLock::new();
/// The neutral branding, used until (or unless) one is installed.
static NEUTRAL: OnceLock<Branding> = OnceLock::new();

/// Makes `b` the process-wide branding. The first call wins; `main` calls it
/// once, before the router is built.
pub fn install(b: Branding) {
    // A second call is ignored by design: the brand never changes while the
    // process runs.
    let _ = INSTALLED.set(b);
}

/// The process-wide branding: the installed one, else the neutral default.
///
/// Test builds always return the neutral default and never read the
/// environment, so a developer's `BRAND_*` variables cannot change what the
/// template tests see; the parsing itself is tested through
/// [`Branding::from_sources`] and [`Branding::from_vars`].
#[cfg(test)]
pub fn get() -> &'static Branding {
    NEUTRAL.get_or_init(Branding::default)
}

/// The process-wide branding: the installed one, else the neutral default.
#[cfg(not(test))]
pub fn get() -> &'static Branding {
    INSTALLED
        .get()
        .unwrap_or_else(|| NEUTRAL.get_or_init(Branding::default))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unclosed functions and quotes would swallow the later declarations;
    /// balanced ones, including quoted font names, pass.
    #[test]
    fn unclosed_functions_and_quotes_are_refused() {
        for bad in [
            "calc(",
            "rgba(0,0,0,.1",
            "\"Inter, sans-serif",
            "'x",
            "a)b(",
            "rgba(0,0,0))",
        ] {
            assert!(!is_css_value(bad), "{bad:?} must be refused");
            assert!(
                matches!(
                    Branding::from_toml(&format!("[brand]\nfont_family = {bad:?}\n")),
                    Err(BrandError::Value {
                        key: "font_family",
                        ..
                    })
                ),
                "{bad:?} must stop the start"
            );
        }
        for ok in [
            "rgba(0,0,0,.08)",
            "\"Segoe UI\", sans-serif",
            "'a(b'",
            "calc(1px + 2px)",
        ] {
            assert!(is_css_value(ok), "{ok:?} must pass");
        }
    }

    /// An absolute logo address needs a real host.
    #[test]
    fn a_logo_address_needs_a_host() {
        for bad in [
            "https://",
            "https:///logo.svg",
            "https://user@evil.example/l.svg",
            "https://?x",
        ] {
            assert!(!is_logo_url(bad), "{bad:?}");
        }
        assert!(is_logo_url("https://cdn.example.org/l.svg"));
        assert!(is_logo_url("/static/logo.svg"));
    }

    /// A non-hex accent or side background needs its text colour named: the
    /// contrast cannot be computed, and a guessed white could be unreadable.
    /// Two-sided: the same colours pass once the text colour is given.
    #[test]
    fn a_non_hex_colour_needs_its_text_colour_named() {
        let toml = |t: &str| Branding::from_toml(t);
        assert!(matches!(
            toml("[brand]\naccent = \"rgb(255,255,0)\"\n"),
            Err(BrandError::Value {
                key: "on_accent",
                ..
            })
        ));
        assert!(matches!(
            toml("[brand]\nside_bg = \"rgb(13,58,100)\"\n"),
            Err(BrandError::Value { key: "side_fg", .. })
        ));
        let ok = toml(
            "[brand]\naccent = \"rgb(255,255,0)\"\non_accent = \"#000\"\nside_bg = \"rgb(13,58,100)\"\nside_fg = \"#fff\"\n",
        )
        .expect("named text colours are accepted");
        assert_eq!(ok.on_accent, "#000");
        assert_eq!(ok.side_fg, "#fff");
    }

    fn from(pairs: &[(&str, &str)]) -> (Branding, Vec<String>) {
        let mut warnings = Vec::new();
        let b = Branding::from_vars(
            |k| {
                pairs
                    .iter()
                    .find(|(n, _)| *n == k)
                    .map(|(_, v)| (*v).to_owned())
            },
            |w| warnings.push(w),
        );
        (b, warnings)
    }

    /// `from_sources` over fake variables and a fake file system.
    fn sources(
        pairs: &[(&str, &str)],
        files: &[(&str, &str)],
    ) -> (Result<Branding, BrandError>, Vec<String>) {
        let mut warnings = Vec::new();
        let r = Branding::from_sources(
            |k| {
                pairs
                    .iter()
                    .find(|(n, _)| *n == k)
                    .map(|(_, v)| (*v).to_owned())
            },
            |p| {
                files
                    .iter()
                    .find(|(n, _)| *n == p)
                    .map(|(_, c)| (*c).to_owned())
                    .ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::NotFound, "no such file")
                    })
            },
            |w| warnings.push(w),
        );
        (r, warnings)
    }

    fn toml_brand(body: &str) -> Result<Branding, BrandError> {
        sources(&[("BRAND_TOML", body)], &[]).0
    }

    #[test]
    fn nothing_set_is_the_unbranded_archive() {
        let (b, w) = from(&[]);
        assert_eq!(b, Branding::default());
        assert!(!b.is_custom());
        assert!(w.is_empty());
    }

    /// The defaults keep today's light look, every default token passes the
    /// check a configured token must pass, and the computed pairs are white.
    #[test]
    fn default_tokens_are_the_neutral_values_and_pass_their_own_check() {
        let b = Branding::default();
        let expected = [
            (
                "font_family",
                "system-ui, -apple-system, \"Segoe UI\", Roboto, sans-serif",
            ),
            ("bg", "#ffffff"),
            ("fg", "#1f2937"),
            ("h1", "#111827"),
            ("muted", "#6b7280"),
            ("accent", "#2563eb"),
            ("on_accent", "#fff"),
            ("border", "#e5e7eb"),
            ("card_bg", "#ffffff"),
            ("subtle_bg", "#f3f4f6"),
            ("side_bg", "#1f2937"),
            ("side_fg", "#fff"),
            ("side_hover", "rgba(255,255,255,.08)"),
            ("radius", "12px"),
            ("radius_sm", "8px"),
            ("elevation", "none"),
        ];
        let tokens = b.tokens();
        assert_eq!(tokens.len(), expected.len());
        for ((k, v), (ek, ev)) in tokens.iter().zip(expected) {
            assert_eq!((*k, *v), (ek, ev));
            assert!(is_css_value(v), "default {k} must pass its own check: {v}");
        }
    }

    #[test]
    fn accepted_values_are_used() {
        let (b, w) = from(&[
            ("BRAND_NAME", "  paperless-rs "),
            ("BRAND_TAGLINE", "Documents in, answers out."),
            ("BRAND_ACCENT", "#0F766E"),
            ("BRAND_LOGO_URL", "https://example.org/logo.svg"),
        ]);
        assert_eq!(b.name, "paperless-rs");
        assert_eq!(b.tagline, "Documents in, answers out.");
        assert_eq!(b.accent, "#0f766e");
        assert_eq!(b.logo_url(), Some("https://example.org/logo.svg"));
        assert!(b.is_custom());
        assert!(w.is_empty());
    }

    #[test]
    fn empty_values_count_as_unset() {
        let (b, w) = from(&[
            ("BRAND_NAME", "  "),
            ("BRAND_ACCENT", ""),
            ("BRAND_LOGO_URL", ""),
        ]);
        assert_eq!(b, Branding::default());
        assert!(w.is_empty(), "{w:?}");
    }

    /// A blank `BRAND_TOML` / `BRAND_FILE` is unset: it neither selects the
    /// brand-document route nor makes a legacy variable "mixed" with it.
    #[test]
    fn blank_document_variables_are_unset() {
        let (r, w) = sources(
            &[
                ("BRAND_TOML", "  "),
                ("BRAND_FILE", ""),
                ("BRAND_NAME", "Legacy Name"),
            ],
            &[],
        );
        assert_eq!(r.expect("legacy route").name, "Legacy Name");
        assert!(w.is_empty());
    }

    /// Each rejected value falls back to its default and is reported once.
    #[test]
    fn rejected_values_fall_back_and_warn() {
        let long = "x".repeat(MAX_NAME + 1);
        let (b, w) = from(&[
            ("BRAND_NAME", &long),
            ("BRAND_ACCENT", "red;}body{display:none"),
            ("BRAND_LOGO_URL", "javascript:alert(1)"),
        ]);
        assert_eq!(b.name, DEFAULT_NAME);
        assert_eq!(b.accent, DEFAULT_ACCENT);
        assert_eq!(b.logo_url(), None);
        assert_eq!(w.len(), 3, "{w:?}");
    }

    #[test]
    fn logo_urls() {
        for ok in ["/static/logo.png", "https://cdn.example.org/l.svg"] {
            assert!(is_logo_url(ok), "{ok}");
        }
        for bad in [
            "//evil.example/x.png",
            "http://example.org/l.png",
            "data:image/svg+xml,<svg/>",
            "javascript:alert(1)",
            "/a b.png",
            "/\\other.example/logo.png",
            "https://example.org\\@evil.example/l.png",
            "",
        ] {
            assert!(!is_logo_url(bad), "{bad}");
        }
    }

    /// White text on the default and other dark accents, near-black on light
    /// ones; two-sided, so a constant answer cannot pass.
    #[test]
    fn button_text_contrasts_with_the_accent() {
        assert_eq!(Branding::default().on_accent, "#fff");
        assert_eq!(on_colour("#0f766e"), "#fff");
        assert_eq!(on_colour("#000"), "#fff");
        assert_eq!(on_colour("#fff"), "#000");
        assert_eq!(on_colour("#ffff00"), "#000");
    }

    /// Contrast is measured against the colour actually returned, at the
    /// accents nearest the switch point, where a wrong foreground would fall
    /// below WCAG AA (4.5:1).
    #[test]
    fn button_text_meets_aa_contrast_near_the_switch_point() {
        let ratio = |a: &str, b: &str| {
            let (la, lb) = (
                relative_luminance(a).expect("valid"),
                relative_luminance(b).expect("valid"),
            );
            (la.max(lb) + 0.05) / (la.min(lb) + 0.05)
        };
        for accent in ["#757575", "#767676", "#777777", "#787878", "#7f7f7f"] {
            let fg = on_colour(accent);
            assert!(
                ratio(accent, fg) >= 4.5,
                "{accent} with {fg}: {}",
                ratio(accent, fg)
            );
        }
    }

    #[test]
    fn hex_colours() {
        for ok in ["#abc", "#A1B2C3"] {
            assert!(is_hex_colour(ok), "{ok}");
        }
        for bad in ["abc", "#abcd", "#ggg", "#12345", "blue"] {
            assert!(!is_hex_colour(bad), "{bad}");
        }
    }

    /// No customer's colours are compiled in: the bar colour is set only by
    /// `BRAND_HEADER_BG`, its text contrasts with it, and a bad value is
    /// ignored. Disable: give `side_bg` a customer's colour as its default.
    #[test]
    fn the_side_bar_colour_comes_only_from_the_variable() {
        let (b, w) = from(&[]);
        assert!(w.is_empty());
        assert_eq!(b.side_bg, DEFAULT_SIDE_BG);
        assert_eq!(b.side_fg, "#fff");
        let (b, w) = from(&[("BRAND_HEADER_BG", "#0D3A64")]);
        assert!(w.is_empty());
        assert_eq!(b.side_bg, "#0d3a64");
        assert_eq!(b.side_fg, "#fff");
        let (b, _) = from(&[("BRAND_HEADER_BG", "#fafafa")]);
        assert_eq!(b.side_bg, "#fafafa");
        assert_eq!(b.side_fg, "#000");
        let (b, w) = from(&[("BRAND_HEADER_BG", "navy")]);
        assert_eq!(b.side_bg, DEFAULT_SIDE_BG);
        assert_eq!(b.side_fg, "#fff");
        assert_eq!(w.len(), 1);
    }

    /// The legacy accent variable moves `accent` and the computed text colour
    /// together; an invalid one warns and never becomes an error, even through
    /// `from_sources`.
    #[test]
    fn legacy_variables_map_onto_the_tokens_and_never_fail() {
        let (r, w) = sources(&[("BRAND_ACCENT", "#FFFF00")], &[]);
        let b = r.expect("legacy");
        assert_eq!(
            (b.accent.as_str(), b.on_accent.as_str()),
            ("#ffff00", "#000")
        );
        assert!(w.is_empty());

        let (r, w) = sources(
            &[("BRAND_ACCENT", "red;}"), ("BRAND_HEADER_BG", "navy")],
            &[],
        );
        assert_eq!(r.expect("legacy never fails"), Branding::default());
        assert_eq!(w.len(), 2, "{w:?}");
    }

    /// A document overrides only the keys it sets, and the computed text colours
    /// follow the overridden colours. Two-sided: compared against the default
    /// with exactly those fields changed.
    #[test]
    fn toml_overrides_only_the_keys_it_sets() {
        let b = toml_brand("[brand]\naccent = \"#ffff00\"\nradius = \"4px\"\n").expect("valid");
        let expected = Branding {
            accent: "#ffff00".to_owned(),
            on_accent: "#000".to_owned(),
            radius: "4px".to_owned(),
            ..Branding::default()
        };
        assert_eq!(b, expected);
        assert_ne!(b, Branding::default());
    }

    #[test]
    fn toml_computed_pairs_follow_the_colour_unless_given() {
        let b = toml_brand("[brand]\nside_bg = \"#fafafa\"\n").expect("valid");
        assert_eq!(
            (b.side_bg.as_str(), b.side_fg.as_str()),
            ("#fafafa", "#000")
        );
        // An explicit value wins over the computed one.
        let b = toml_brand("[brand]\naccent = \"#ffff00\"\non_accent = \"#123\"\n").expect("valid");
        assert_eq!(b.on_accent, "#123");
        // An empty document is the default.
        assert_eq!(toml_brand("[brand]\n").expect("valid"), Branding::default());
        assert_eq!(
            toml_brand("# nothing\n").expect("valid"),
            Branding::default()
        );
    }

    #[test]
    fn toml_sets_name_tagline_and_logo() {
        let b = toml_brand(
            "[brand]\nname = \"  Archiv  \"\ntagline = \"Alles an einem Ort.\"\nlogo_url = \"https://example.org/l.svg\"\n",
        )
        .expect("valid");
        assert_eq!(b.name, "Archiv");
        assert_eq!(b.tagline, "Alles an einem Ort.");
        assert_eq!(b.logo_url(), Some("https://example.org/l.svg"));
        assert!(b.is_custom());
    }

    /// `BRAND_FILE` is read through the injected reader, from exactly the path
    /// the variable names.
    #[test]
    fn brand_file_is_read_from_the_named_path() {
        let files = [
            ("/etc/brand.toml", "[brand]\nname = \"From File\"\n"),
            ("/etc/other.toml", "[brand]\nname = \"Other\"\n"),
        ];
        let (r, _) = sources(&[("BRAND_FILE", "/etc/brand.toml")], &files);
        assert_eq!(r.expect("read").name, "From File");
        let (r, _) = sources(&[("BRAND_FILE", "/etc/other.toml")], &files);
        assert_eq!(r.expect("read").name, "Other");
    }

    #[test]
    fn an_unreadable_brand_file_is_fatal() {
        let (r, _) = sources(&[("BRAND_FILE", "/missing.toml")], &[]);
        match r {
            Err(BrandError::Read(e)) => assert!(e.contains("/missing.toml"), "{e}"),
            other => panic!("expected Read, got {other:?}"),
        }
    }

    #[test]
    fn both_document_variables_are_refused() {
        let (r, _) = sources(
            &[("BRAND_TOML", "[brand]\n"), ("BRAND_FILE", "/b.toml")],
            &[("/b.toml", "[brand]\n")],
        );
        assert_eq!(r, Err(BrandError::Both));
    }

    /// A document together with any legacy variable is ambiguous, whichever way
    /// the document is supplied.
    #[test]
    fn a_document_mixed_with_a_legacy_variable_is_refused() {
        for legacy in LEGACY_VARS {
            let (r, _) = sources(&[("BRAND_TOML", "[brand]\n"), (legacy, "x")], &[]);
            assert_eq!(r, Err(BrandError::Mixed), "BRAND_TOML + {legacy}");
            let (r, _) = sources(
                &[("BRAND_FILE", "/b.toml"), (legacy, "x")],
                &[("/b.toml", "[brand]\n")],
            );
            assert_eq!(r, Err(BrandError::Mixed), "BRAND_FILE + {legacy}");
        }
    }

    #[test]
    fn unknown_keys_tables_and_bad_syntax_are_parse_errors() {
        for bad in [
            "[brand]\ncolour = \"#fff\"\n",
            "[brand]\naccnt = \"#fff\"\n",
            "[theme]\naccent = \"#fff\"\n",
            "stray = 1\n",
            "[brand\naccent = ",
            "[brand]\naccent = 5\n",
        ] {
            assert!(
                matches!(toml_brand(bad), Err(BrandError::Parse(_))),
                "{bad:?}"
            );
        }
    }

    /// Values that could close the declaration, the rule or the `<style>`
    /// element are refused, and the error names the key.
    #[test]
    fn stylesheet_breaking_values_are_refused() {
        for bad in [
            "red; color: blue",
            "#fff}",
            "#fff</style><script>",
            "url(x)\\",
            "#fff\n",
        ] {
            for key in ["accent", "side_bg", "font_family", "elevation"] {
                let doc = format!("[brand]\n{key} = {bad:?}\n");
                match toml_brand(&doc) {
                    Err(BrandError::Value { key: k, .. }) => assert_eq!(k, key, "{bad:?}"),
                    other => panic!("{key} = {bad:?}: expected Value, got {other:?}"),
                }
            }
        }
    }

    #[test]
    fn stylesheet_safe_values_are_accepted() {
        for (key, good) in [
            ("accent", "#c8102e"),
            ("side_hover", "rgba(255,255,255,.08)"),
            ("elevation", "0 4px 16px rgba(0,0,0,.08)"),
            ("radius", "14px"),
            ("elevation", "none"),
            ("font_family", "Inter, \"Helvetica Neue\", sans-serif"),
        ] {
            let b = toml_brand(&format!("[brand]\n{key} = {good:?}\n")).expect(good);
            let stored = b
                .tokens()
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| (*v).to_owned());
            assert_eq!(stored.as_deref(), Some(good), "{key}");
        }
    }

    #[test]
    fn name_tagline_and_logo_are_checked_in_a_document() {
        let long_name = "x".repeat(MAX_NAME + 1);
        let long_tagline = "x".repeat(MAX_TAGLINE + 1);
        let cases = [
            (format!("name = {long_name:?}"), "name"),
            (format!("tagline = {long_tagline:?}"), "tagline"),
            (
                "logo_url = \"http://example.org/l.png\"".to_owned(),
                "logo_url",
            ),
            ("logo_url = \"//evil.example/x.png\"".to_owned(), "logo_url"),
            ("logo_url = \"javascript:alert(1)\"".to_owned(), "logo_url"),
            ("logo_url = \"/a b.png\"".to_owned(), "logo_url"),
        ];
        for (line, key) in cases {
            match toml_brand(&format!("[brand]\n{line}\n")) {
                Err(BrandError::Value { key: k, .. }) => assert_eq!(k, key, "{line}"),
                other => panic!("{line}: expected Value, got {other:?}"),
            }
        }
        // The boundary itself is accepted.
        let edge = "x".repeat(MAX_NAME);
        assert!(toml_brand(&format!("[brand]\nname = {edge:?}\n")).is_ok());
    }

    /// Every token is emitted exactly once, under its hyphenated name, from the
    /// resolved values (not the defaults).
    #[test]
    fn css_vars_emit_all_sixteen_tokens_once() {
        let b = toml_brand("[brand]\naccent = \"#ffff00\"\n").expect("valid");
        let css = b.css_vars();
        assert!(css.starts_with(":root { ") && css.ends_with(" }"), "{css}");
        assert!(!css.contains('\n'), "one line: {css}");
        assert_eq!(css.matches("--brand-").count(), 16, "{css}");
        for name in [
            "font-family",
            "bg",
            "fg",
            "h1",
            "muted",
            "accent",
            "on-accent",
            "border",
            "card-bg",
            "subtle-bg",
            "side-bg",
            "side-fg",
            "side-hover",
            "radius",
            "radius-sm",
            "elevation",
        ] {
            assert_eq!(
                css.matches(&format!("--brand-{name}: ")).count(),
                1,
                "{name}"
            );
        }
        assert!(css.contains("--brand-accent: #ffff00;"), "{css}");
        assert!(css.contains("--brand-on-accent: #000;"), "{css}");
    }

    /// The corporate-design example in the module documentation parses and
    /// yields its stated values.
    #[test]
    fn the_documented_datagroup_example_parses() {
        let doc = r##"
[brand]
name = "DATAGROUP Dokumente"
accent = "#c8102e"
muted = "#4a4a4a"
fg = "#1a1a1a"
h1 = "#1a1a1a"
bg = "#f5f5f5"
border = "#e0e6ec"
subtle_bg = "#eef1f4"
side_bg = "#0d3a64"
radius = "14px"
radius_sm = "10px"
elevation = "0 4px 16px rgba(0,0,0,.08)"
"##;
        let b = toml_brand(doc).expect("the documented example is valid");
        assert_eq!(b.name, "DATAGROUP Dokumente");
        assert_eq!(b.accent, "#c8102e");
        assert_eq!(b.side_bg, "#0d3a64");
        assert_eq!(b.side_fg, "#fff");
        assert_eq!(b.elevation, "0 4px 16px rgba(0,0,0,.08)");
        assert!(b.is_custom());
    }

    #[test]
    fn errors_describe_themselves() {
        for e in [
            BrandError::Read("/x: no such file".to_owned()),
            BrandError::Both,
            BrandError::Mixed,
            BrandError::Parse("bad".to_owned()),
            BrandError::Value {
                key: "accent",
                value: "red;".to_owned(),
            },
        ] {
            assert!(!e.to_string().is_empty());
        }
        let v = BrandError::Value {
            key: "accent",
            value: "red;".to_owned(),
        }
        .to_string();
        assert!(v.contains("accent") && v.contains("red;"), "{v}");
    }
}
