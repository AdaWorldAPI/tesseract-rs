//! The interface language: German by default, English on request.
//!
//! The viewer's choice is the `lang` cookie (`de` or `en`), set by
//! `GET /lang?set=de|en&back=<path>` (the switch in the top bar). [`negotiate`]
//! reads it once per request and puts it in scope; a template asks for its text
//! with `self.t("English text")`, and [`t`] answers in the request's language.
//! English is the key, so a text without a German entry shows in English rather
//! than as a blank (the test `every_template_text_has_a_german_entry` keeps that
//! from happening unnoticed).
//!
//! The interface language is a presentation choice only: it never changes what a
//! caller may see, and it is independent of the OCR language of a document.

use axum::extract::{Query, Request};
use axum::http::{header, HeaderMap, HeaderValue};
use axum::middleware::Next;
use axum::response::{IntoResponse, Redirect, Response};
use serde::Deserialize;

/// The interface language.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Lang {
    /// German, the default.
    #[default]
    De,
    /// English.
    En,
}

impl Lang {
    /// The `lang` attribute and cookie value.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Lang::De => "de",
            Lang::En => "en",
        }
    }

    /// From a cookie or query value; anything but `en` is German.
    #[must_use]
    pub fn parse(v: &str) -> Self {
        if v.trim() == "en" {
            Lang::En
        } else {
            Lang::De
        }
    }
}

tokio::task_local! {
    static LANG: Lang;
}

/// The current request's interface language; German outside a request.
#[must_use]
pub fn current() -> Lang {
    LANG.try_with(|l| *l).unwrap_or_default()
}

/// The cookie that remembers the choice.
pub const COOKIE: &str = "lang";

/// The language a request's cookie asks for.
#[must_use]
pub fn from_headers(headers: &HeaderMap) -> Lang {
    headers
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(';'))
        .filter_map(|kv| kv.trim().split_once('='))
        .find(|(k, _)| *k == COOKIE)
        .map(|(_, v)| Lang::parse(v))
        .unwrap_or_default()
}

/// Middleware: put the request's interface language in scope.
pub async fn negotiate(req: Request, next: Next) -> Response {
    let lang = from_headers(req.headers());
    let mut resp = LANG.scope(lang, next.run(req)).await;
    resp.headers_mut()
        .append(header::VARY, HeaderValue::from_static("Cookie"));
    resp
}

/// `GET /lang?set=de|en&back=/path`.
#[derive(Deserialize)]
pub struct Switch {
    set: Option<String>,
    back: Option<String>,
}

/// Remember the language and go back: to `back`, else to the page the switch was
/// clicked on (`Referer`). Only a path on this site is followed, so the switch is
/// no open redirect.
pub async fn switch(headers: HeaderMap, Query(q): Query<Switch>) -> Response {
    let lang = Lang::parse(q.set.as_deref().unwrap_or(""));
    let local = |b: &str| b.starts_with('/') && !b.starts_with("//") && !b.contains('\\');
    let from_referer = headers
        .get(header::REFERER)
        .and_then(|v| v.to_str().ok())
        .and_then(|r| {
            let rest = r.split_once("://")?.1;
            let host = headers.get(header::HOST)?.to_str().ok()?;
            rest.strip_prefix(host).map(str::to_string)
        })
        .filter(|p| local(p) && !p.starts_with("/lang"));
    let back = q
        .back
        .filter(|b| local(b))
        .or(from_referer)
        .unwrap_or_else(|| "/dashboard".to_string());
    let mut r = Redirect::to(&back).into_response();
    if let Ok(v) = HeaderValue::from_str(&format!(
        "{COOKIE}={}; Path=/; Max-Age=31536000; SameSite=Lax",
        lang.code()
    )) {
        r.headers_mut().insert(header::SET_COOKIE, v);
    }
    r
}

/// `en` in the request's language.
#[must_use]
pub fn t(en: &'static str) -> &'static str {
    match current() {
        Lang::En => en,
        Lang::De => de(en).unwrap_or(en),
    }
}

/// The German text for an English key.
#[must_use]
pub fn de(en: &str) -> Option<&'static str> {
    Some(match en {
        // shell
        "Desktop" => "Desktop",
        "Background" => "Hintergrund",
        "Sign out" => "Abmelden",
        "Main" => "Hauptmenü",
        "Main (compact)" => "Hauptmenü (kompakt)",
        "Dashboard" => "Übersicht",
        "Inbox" => "Eingang",
        "Documents" => "Dokumente",
        "Correspondents" => "Korrespondenten",
        "Tags" => "Schlagwörter",
        "Document types" => "Dokumenttypen",
        "Upload" => "Hochladen",
        "Quick actions" => "Schnellzugriff",
        "Language" => "Sprache",
        "Pure-Rust OCR (no libtesseract/leptonica at runtime)" => {
            "Texterkennung in reinem Rust (ohne libtesseract/leptonica zur Laufzeit)"
        }
        "archive" => "Archiv",
        "powered by tesseract-rs" => "mit tesseract-rs",
        // upload
        "document archive" => "Dokumentenarchiv",
        "Upload a document image" => "Dokumentbild hochladen",
        "OR" => "ODER",
        "Fetch an image URL" => "Bild von einer Adresse laden",
        "Private, loopback, and link-local addresses are refused (SSRF guard); 10 MB / 10 s cap." => {
            "Private, Loopback- und Link-Local-Adressen werden abgelehnt (SSRF-Schutz); höchstens 10 MB in 10 s."
        }
        "Document language" => "Dokumentsprache",
        "German" => "Deutsch",
        "English" => "Englisch",
        "Customer" => "Kunde",
        "No customer has been assigned to your account, so you cannot file documents yet." => {
            "Ihrem Konto ist noch kein Kunde zugeordnet, deshalb können Sie noch keine Dokumente ablegen."
        }
        "optional, e.g. K01" => "optional, z. B. K01",
        "Ingest document" => "Dokument erfassen",
        "A document already in the archive (byte-identical content) is recognized as a duplicate and skipped — no repeat OCR spend, and you land straight on the existing record." => {
            "Ein Dokument, das schon im Archiv liegt (byte-gleicher Inhalt), wird als Duplikat erkannt und übersprungen: keine erneute Texterkennung, Sie landen direkt beim vorhandenen Eintrag."
        }
        // list
        "Documents not yet marked reviewed, newest first. Open one, file it, and mark it reviewed to take it off this list." => {
            "Noch nicht geprüfte Dokumente, die neuesten zuerst. Öffnen, ablegen und als geprüft markieren nimmt es von dieser Liste."
        }
        "Search" => "Suchen",
        "Search recognized text…" => "Erkannten Text durchsuchen …",
        "document" => "Dokument",
        "documents" => "Dokumente",
        "matching" => "zu",
        "to review" => "zu prüfen",
        "Nothing to review." => "Nichts zu prüfen.",
        "No documents yet." => "Noch keine Dokumente.",
        "Ingest one" => "Eines erfassen",
        "page" => "Seite",
        "pages" => "Seiten",
        "confidence" => "Konfidenz",
        "low confidence" => "niedrige Konfidenz",
        "Inbox pages" => "Seiten des Eingangs",
        "Newer" => "Neuere",
        "Older" => "Ältere",
        "Page" => "Seite",
        // dashboard
        "All in the inbox" => "Alle im Eingang",
        "Upload a document" => "Dokument hochladen",
        // document
        "All documents" => "Alle Dokumente",
        "low" => "niedrig",
        "ingested" => "erfasst",
        "Download original" => "Original herunterladen",
        "The original is not available: this document was stored before originals were kept. Upload the same file again to attach it." => {
            "Das Original fehlt: Dieses Dokument wurde gespeichert, bevor Originale aufbewahrt wurden. Laden Sie dieselbe Datei erneut hoch, um es anzuhängen."
        }
        "Metadata" => "Metadaten",
        "not filed under a customer" => "keinem Kunden zugeordnet",
        "Customer code" => "Kundencode",
        "e.g. K01, empty to clear" => "z. B. K01, leer zum Entfernen",
        "Set customer" => "Kunde setzen",
        "Reviewed" => "Geprüft",
        "Mark unreviewed" => "Als ungeprüft markieren",
        "Not reviewed" => "Nicht geprüft",
        "Mark reviewed" => "Als geprüft markieren",
        "Mark reviewed, next" => "Als geprüft markieren, weiter",
        "None assigned." => "Nichts zugeordnet.",
        "Unassign" => "Entfernen",
        "No definitions yet" => "Noch keine Definitionen",
        "create one" => "eine anlegen",
        "Assign" => "Zuordnen",
        "Suggestions" => "Vorschläge",
        "expected" => "erwartet",
        "evidence" => "Belege",
        "Accept" => "Übernehmen",
        "because" => "weil",
        "Ignoring a suggestion stores nothing." => "Einen Vorschlag zu übergehen, speichert nichts.",
        "Harvested fields" => "Erkannte Felder",
        "Recognized text" => "Erkannter Text",
        "Regions" => "Bereiche",
        "Extracted facts" => "Extrahierte Aussagen",
        "Subject" => "Subjekt",
        "Predicate" => "Prädikat",
        "Object" => "Objekt",
        "Sentence" => "Satz",
        "Delete this document from the archive?" => "Dieses Dokument aus dem Archiv löschen?",
        "Delete document" => "Dokument löschen",
        // definitions
        "Correspondents, document types and tags in one place. Each also has its own page in the menu." => {
            "Korrespondenten, Dokumenttypen und Schlagwörter an einem Ort. Jede Art hat auch eine eigene Seite im Menü."
        }
        "All definitions" => "Alle Definitionen",
        "AUTO model" => "AUTO-Modell",
        "Re-mine now" => "Jetzt neu lernen",
        "A new definition is AUTO: it is suggested from reviewed documents, never assigned without a person accepting it." => {
            "Eine neue Definition ist AUTO: Sie wird aus geprüften Dokumenten vorgeschlagen und nie ohne Zustimmung eines Menschen zugeordnet."
        }
        "None yet." => "Noch keine.",
        "retired" => "stillgelegt",
        "pattern" => "Muster",
        "case-insensitive" => "Groß-/Kleinschreibung egal",
        "yes" => "ja",
        "no" => "nein",
        "Name" => "Name",
        "Rename" => "Umbenennen",
        "Matching algorithm" => "Zuordnungsverfahren",
        "Pattern" => "Muster",
        "ignore case" => "Groß-/Kleinschreibung ignorieren",
        "Save rule" => "Regel speichern",
        "Retire this definition? Its id is never reused." => {
            "Diese Definition stilllegen? Ihre Kennung wird nie wiederverwendet."
        }
        "Retire" => "Stilllegen",
        "New name" => "Neuer Name",
        "Create" => "Anlegen",
        "Correspondent" => "Korrespondent",
        "Document type" => "Dokumenttyp",
        "Tag" => "Schlagwort",
        // desktop
        "Classic view" => "Klassische Ansicht",
        "Folders" => "Ordner",
        "This folder is empty." => "Dieser Ordner ist leer.",
        "Showing" => "Angezeigt",
        "of" => "von",
        "Use the document search for the rest." => "Den Rest finden Sie über die Dokumentsuche.",
        "Archive" => "Archiv",
        // labels the code passes to templates (`DYNAMIC` lists them)
        "header" => "Kopfzeile",
        "footer" => "Fußzeile",
        "main" => "Hauptteil",
        "nav" => "Navigation",
        "table" => "Tabelle",
        "figure" => "Abbildung",
        "text" => "Text",
        "None" => "Keine",
        "Any" => "Eines der Wörter",
        "All" => "Alle Wörter",
        "Literal" => "Wörtlich",
        "Regex" => "Regulärer Ausdruck",
        "Fuzzy" => "Unscharf",
        "Auto" => "Auto",
        "Unknown" => "Unbekannt",
        _ => return None,
    })
}

/// The labels the code passes to templates instead of literal text: kinds,
/// region kinds, matching algorithms, folder labels.
#[cfg(test)]
const DYNAMIC: &[&str] = &[
    "Correspondent",
    "Document type",
    "Tags",
    "Correspondents",
    "Document types",
    "header",
    "footer",
    "main",
    "nav",
    "table",
    "figure",
    "text",
    "None",
    "Any",
    "All",
    "Literal",
    "Regex",
    "Fuzzy",
    "Auto",
    "Unknown",
    "Archive",
    "All documents",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_dynamic_label_has_a_german_entry() {
        for k in DYNAMIC {
            assert!(de(k).is_some(), "no German for {k:?}");
        }
    }

    /// Every `self.t("…")` in a template has a German entry, so the German interface
    /// never falls back to English unnoticed. CAN FIRE: add a `self.t("x")` with an
    /// unknown text to any template.
    #[test]
    fn every_template_text_has_a_german_entry() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/templates");
        let mut seen = 0;
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let text = std::fs::read_to_string(&path).unwrap();
            for part in text.split("self.t(\"").skip(1) {
                let key = &part[..part.find("\")").expect("closed t(\"…\")")];
                assert!(
                    de(key).is_some(),
                    "{}: no German for {key:?}",
                    path.display()
                );
                seen += 1;
            }
        }
        assert!(seen > 80, "the templates go through t(): {seen}");
    }

    #[tokio::test]
    async fn german_is_the_default_and_english_is_chosen() {
        assert_eq!(current(), Lang::De);
        assert_eq!(
            LANG.scope(Lang::De, async { t("Documents") }).await,
            "Dokumente"
        );
        assert_eq!(
            LANG.scope(Lang::En, async { t("Documents") }).await,
            "Documents"
        );
        let mut h = HeaderMap::new();
        assert_eq!(from_headers(&h), Lang::De);
        h.insert(header::COOKIE, HeaderValue::from_static("a=1; lang=en"));
        assert_eq!(from_headers(&h), Lang::En);
        h.insert(header::COOKIE, HeaderValue::from_static("lang=fr"));
        assert_eq!(from_headers(&h), Lang::De);
    }

    #[tokio::test]
    async fn the_switch_goes_back_only_to_this_site() {
        for (back, want) in [
            (Some("/documents?q=x"), "/documents?q=x"),
            (Some("//evil.example/"), "/dashboard"),
            (Some("https://evil.example/"), "/dashboard"),
            (Some("/\\evil"), "/dashboard"),
            (None, "/dashboard"),
        ] {
            let r = switch(
                HeaderMap::new(),
                Query(Switch {
                    set: Some("en".into()),
                    back: back.map(str::to_string),
                }),
            )
            .await;
            assert_eq!(r.headers()[header::LOCATION], want, "{back:?}");
            assert!(r.headers()[header::SET_COOKIE]
                .to_str()
                .unwrap()
                .starts_with("lang=en;"));
        }
    }
}
