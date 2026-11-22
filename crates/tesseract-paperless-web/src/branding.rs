//! Optional branding: a deployment can show its own name, tagline, accent
//! colour and logo without a code change.
//!
//! Every setting comes from an environment variable and is optional. With none
//! set the pages are exactly the unbranded archive (`tesseract-paperless`, the
//! original tagline, accent `#2563eb`, no logo).
//!
//! | variable | effect | accepted |
//! |---|---|---|
//! | `BRAND_NAME` | page titles and the main heading | 1 to 60 characters after trimming |
//! | `BRAND_TAGLINE` | the line under the heading on the upload page | up to 240 characters |
//! | `BRAND_ACCENT` | links and buttons | `#rgb` or `#rrggbb` |
//! | `BRAND_LOGO_URL` | an image beside the heading | an `https://` URL or a same-site path starting with a single `/` |
//!
//! An empty value counts as unset. A value that is not accepted is ignored
//! with one line on stderr and the
//! default is used, so a typo cannot break the pages or inject markup. All
//! values are also HTML-escaped by the templates.
//!
//! When a custom name is set the upload page's footer adds "powered by
//! tesseract-rs", so a rebranded deployment still names the engine underneath.
//! paperless-rs uses this to ship the archive under its own name.

use std::sync::OnceLock;

/// The default name, the archive's own.
pub const DEFAULT_NAME: &str = "tesseract-paperless";
/// The default tagline on the upload page.
pub const DEFAULT_TAGLINE: &str = "Upload a scan or photo. It is OCR'd with the pure-Rust recognizer, deduplicated by content hash, and archived \u{2014} searchable, browsable, paperless-ngx-shaped.";
/// The default accent colour.
pub const DEFAULT_ACCENT: &str = "#2563eb";

const MAX_NAME: usize = 60;
const MAX_TAGLINE: usize = 240;
const MAX_URL: usize = 2048;

/// The branding the pages render.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Branding {
    pub name: String,
    pub tagline: String,
    pub accent: String,
    logo_url: Option<String>,
}

impl Default for Branding {
    fn default() -> Self {
        Branding {
            name: DEFAULT_NAME.to_owned(),
            tagline: DEFAULT_TAGLINE.to_owned(),
            accent: DEFAULT_ACCENT.to_owned(),
            logo_url: None,
        }
    }
}

impl Branding {
    /// Reads the settings through `get`, which returns a variable's value or
    /// `None`. Rejected values are reported through `warn` and replaced by the
    /// default.
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
            } else {
                warn("BRAND_ACCENT ignored: use #rgb or #rrggbb".to_owned());
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
}

fn is_hex_colour(v: &str) -> bool {
    let Some(hex) = v.strip_prefix('#') else {
        return false;
    };
    matches!(hex.len(), 3 | 6) && hex.bytes().all(|c| c.is_ascii_hexdigit())
}

fn is_logo_url(v: &str) -> bool {
    if v.is_empty() || v.len() > MAX_URL || v.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return false;
    }
    let same_site = v.starts_with('/') && !v.starts_with("//");
    same_site || v.starts_with("https://")
}

/// The process-wide branding, read from the environment on first use.
pub fn get() -> &'static Branding {
    static BRANDING: OnceLock<Branding> = OnceLock::new();
    BRANDING.get_or_init(|| {
        Branding::from_vars(
            |k| std::env::var(k).ok(),
            |w| eprintln!("tesseract-paperless-web: {w}"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn nothing_set_is_the_unbranded_archive() {
        let (b, w) = from(&[]);
        assert_eq!(b, Branding::default());
        assert!(!b.is_custom());
        assert!(w.is_empty());
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
            "",
        ] {
            assert!(!is_logo_url(bad), "{bad}");
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
}
