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
//! | `BRAND_THEME` | the page theme | `default`, or `datagroup` (report-rs's default corporate theme) |
//!
//! `datagroup` matches report-rs's `Theme::Datagroup` (`templates/themes.css`
//! there): a navy header bar, a red accent, Segoe UI, white cards with a soft
//! shadow on a light grey page. It sets the accent to `#c8102e` unless
//! `BRAND_ACCENT` names another, and pins the light colour scheme, as
//! report-rs does.
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

/// A page theme: colours and type layered over the base styles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Theme {
    /// The archive's own look: system colours, light and dark.
    #[default]
    Default,
    /// report-rs's default corporate theme.
    Datagroup,
}

/// The accent the `datagroup` theme uses unless `BRAND_ACCENT` is set.
pub const DATAGROUP_ACCENT: &str = "#c8102e";

/// report-rs `themes.css`, `:root[data-theme="datagroup"]`, applied to this
/// archive's markup. Values are copied, not derived: the two must be compared
/// by eye when report-rs changes its theme.
const DATAGROUP_CSS: &str = r#"
  :root { color-scheme: light; }
  body { font-family: "Segoe UI", system-ui, -apple-system, Arial, sans-serif; font-size: 14px;
         background: #f5f5f5; color: #1a1a1a; }
  header.shell { background: #0d3a64; border-bottom: 0; }
  header.shell .brand, header.shell .end, form.signout button { color: #fff; }
  nav.main a { color: rgba(255,255,255,.75); }
  nav.main a:hover { color: #fff; }
  nav.main a[aria-current="page"] { color: #fff; border-bottom-color: #c8102e; }
  .card { background: #fff; border: 1px solid #e0e6ec; border-radius: 14px;
          box-shadow: 0 4px 16px rgba(0,0,0,.08); }
  h1 { color: #1a1a1a; }
  th { text-transform: uppercase; font-size: 11px; color: #4a4a4a; }
  button:hover, :focus-visible { outline: 2px solid #c8102e; outline-offset: 2px; }
  footer { color: #4a4a4a; }
"#;

/// The branding the pages render.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Branding {
    pub name: String,
    pub tagline: String,
    pub accent: String,
    logo_url: Option<String>,
    theme: Theme,
}

impl Default for Branding {
    fn default() -> Self {
        Branding {
            name: DEFAULT_NAME.to_owned(),
            tagline: DEFAULT_TAGLINE.to_owned(),
            accent: DEFAULT_ACCENT.to_owned(),
            logo_url: None,
            theme: Theme::Default,
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
        if let Some(v) = get("BRAND_THEME") {
            match v.trim().to_ascii_lowercase().as_str() {
                "default" => {}
                "datagroup" => {
                    b.theme = Theme::Datagroup;
                    b.accent = DATAGROUP_ACCENT.to_owned();
                }
                _ => warn("BRAND_THEME ignored: use default or datagroup".to_owned()),
            }
        }
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

    /// The theme's CSS, layered after the base styles; empty for the default.
    #[must_use]
    pub fn theme_css(&self) -> &'static str {
        match self.theme {
            Theme::Default => "",
            Theme::Datagroup => DATAGROUP_CSS,
        }
    }

    /// The logo URL, if one was accepted.
    #[must_use]
    pub fn logo_url(&self) -> Option<&str> {
        self.logo_url.as_deref()
    }

    /// The text colour for a button whose background is the accent: white or
    /// black, whichever contrasts more with the accent (WCAG relative
    /// luminance). The better of the two always reaches at least 4.58:1. White
    /// for the default accent, so default pages are unchanged; black for a
    /// light accent such as `#fff`.
    #[must_use]
    pub fn on_accent(&self) -> &'static str {
        let l = relative_luminance(&self.accent).unwrap_or(0.0);
        let with_white = 1.05 / (l + 0.05);
        let with_black = (l + 0.05) / 0.05;
        if with_white >= with_black {
            "#fff"
        } else {
            "#000"
        }
    }

    /// Whether a custom name is set, which adds the "powered by tesseract-rs"
    /// credit.
    #[must_use]
    pub fn is_custom(&self) -> bool {
        self.name != DEFAULT_NAME
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
    same_site || v.starts_with("https://")
}

/// The process-wide branding, read from the environment on first use.
///
/// Test builds never read the environment, so a developer's `BRAND_*`
/// variables cannot change what the template tests see; the parsing itself is
/// tested through [`Branding::from_vars`].
#[cfg(test)]
pub fn get() -> &'static Branding {
    static BRANDING: OnceLock<Branding> = OnceLock::new();
    BRANDING.get_or_init(Branding::default)
}

/// The process-wide branding, read from the environment on first use.
#[cfg(not(test))]
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
        let with = |accent: &str| Branding {
            accent: accent.to_owned(),
            ..Branding::default()
        };
        assert_eq!(Branding::default().on_accent(), "#fff");
        assert_eq!(with("#0f766e").on_accent(), "#fff");
        assert_eq!(with("#000").on_accent(), "#fff");
        assert_eq!(with("#fff").on_accent(), "#000");
        assert_eq!(with("#ffff00").on_accent(), "#000");
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
            let b = Branding {
                accent: accent.to_owned(),
                ..Branding::default()
            };
            let fg = b.on_accent();
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

    /// Disable: leave `theme` at Default for `datagroup`, or let the preset
    /// accent override an explicit `BRAND_ACCENT`.
    #[test]
    fn the_datagroup_theme_brings_report_rs_colours_and_yields_to_an_explicit_accent() {
        let (b, w) = from(&[("BRAND_THEME", "Datagroup")]);
        assert!(w.is_empty());
        assert_eq!(b.accent, DATAGROUP_ACCENT);
        assert!(b.theme_css().contains("#0d3a64"), "navy header bar");
        assert!(b.theme_css().contains("Segoe UI"));
        let (b, _) = from(&[("BRAND_THEME", "datagroup"), ("BRAND_ACCENT", "#123456")]);
        assert_eq!(b.accent, "#123456");
        assert!(!b.theme_css().is_empty());
        let (b, w) = from(&[("BRAND_THEME", "neon")]);
        assert_eq!(b.theme_css(), "");
        assert_eq!(w.len(), 1);
        assert_eq!(Branding::default().theme_css(), "");
    }
}
