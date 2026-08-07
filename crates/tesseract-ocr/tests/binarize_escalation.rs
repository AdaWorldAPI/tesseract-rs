//! Integration falsifier for [`binarize_page_escalating`] — which rung a
//! REAL degraded page lands on. Lives in `tests/` (not the lib test module)
//! because it reads `corpus/quality/*.pgm`, matching this crate's convention
//! that corpus-dependent tests live here, not in the lib module (which stays
//! self-contained procedural fixtures).
//!
//! Every fixture below is measured, not assumed — the exact ink-fraction
//! numbers this file's own `escalate_probe.rs` diagnostic produced (now
//! deleted, per this crate's no-permanent-diagnostics convention) are quoted
//! in [`binarize_page_escalating`]'s doc comment. Rungs 3 and 4 are reached
//! via a compression transform applied to REAL corpus pixels here, not a
//! contrived synthetic image — see each test's own doc for the honest scope
//! of what it proves.

use std::path::Path;

use tesseract_ocr::xy_cut::{binarize_page_escalating, EscalatedMode};

fn corpus() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/quality")
}

fn load(name: &str) -> (Vec<u8>, usize, usize) {
    let bytes = std::fs::read(corpus().join(name))
        .unwrap_or_else(|e| panic!("missing corpus fixture {name}: {e}"));
    tesseract_ocr::image_input::parse_pgm(&bytes)
        .unwrap_or_else(|e| panic!("failed to parse {name}: {e}"))
}

/// Compress `grey`'s dynamic range toward mid-grey by factor `b` (`b=1.0` is
/// a no-op; `b→0` flattens toward uniform 128) — the SAME transform
/// `corpus/gen/gen_faded_contrast.py` uses to build the committed
/// `faded_*.pgm` fixtures, applied here to a DIFFERENT source image so a
/// single fixture can carry two independent degradations at once
/// (illumination unevenness from the source page, contrast compression from
/// this transform).
fn compress_contrast(grey: &[u8], b: f32) -> Vec<u8> {
    grey.iter()
        .map(|&p| {
            (128.0 + b * (f32::from(p) - 128.0))
                .round()
                .clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Deterministic LCG dither — cheap additive noise with no external
/// dependency, used only to push the rung-4 fixture into a genuinely
/// harder regime (see that test's own doc for why).
fn add_noise(grey: &[u8], amplitude: f32, seed: u32) -> Vec<u8> {
    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 24) as i32 - 128
    };
    grey.iter()
        .map(|&p| {
            (f32::from(p) + f32::from(next() as i16) * amplitude)
                .round()
                .clamp(0.0, 255.0) as u8
        })
        .collect()
}

const WHSIZE: usize = 16;

#[test]
fn otsu_succeeds_directly_on_a_clean_page() {
    let (grey, w, h) = load("uneven_clean.pgm");
    let r = binarize_page_escalating(&grey, w, h, WHSIZE);
    assert_eq!(
        r.mode,
        EscalatedMode::Otsu,
        "a clean page must stop at rung 1 — no escalation cost on the common case"
    );
}

/// Otsu ALSO succeeds directly on faded (but not illumination-uneven)
/// contrast — proving the ladder does not waste escalation when Otsu
/// already handles the degradation. This is the mirror finding to
/// `CLAUDE.md`'s Sauvola-vs-Otsu faded-contrast section: Otsu is the one
/// that is robust here, Sauvola is the one that fails (on the SEVERE
/// variant — see `sauvola_recovers_where_otsu_floods` for why this test
/// uses the MODERATE `faded_060.pgm`, not `faded_085.pgm`).
#[test]
fn otsu_succeeds_directly_on_moderate_faded_contrast() {
    let (grey, w, h) = load("faded_060.pgm");
    let r = binarize_page_escalating(&grey, w, h, WHSIZE);
    assert_eq!(
        r.mode,
        EscalatedMode::Otsu,
        "moderate faded contrast does not defeat Otsu; the ladder must not \
         escalate past rung 1 here"
    );
}

/// Rung 2, the can-fire half: Otsu's illumination flood, Sauvola recovers.
///
/// Disable-verified: forcing `is_plausible_ink_frac` to always return `true`
/// makes this land on `Otsu` instead (Otsu's own flood output is silently
/// accepted); forcing it to always return `false` makes this exhaust all
/// four rungs and land on `Singh` instead of `Sauvola`.
#[test]
fn sauvola_recovers_where_otsu_floods() {
    let (grey, w, h) = load("uneven_linear_085.pgm");
    let r = binarize_page_escalating(&grey, w, h, WHSIZE);
    assert_eq!(
        r.mode,
        EscalatedMode::Sauvola,
        "Otsu floods under illumination unevenness; Sauvola must recover at rung 2"
    );
    assert!(
        r.ink_frac < 0.05,
        "the accepted rung's own ink fraction must be a healthy reading, not \
         a partially-recovered one: got {}",
        r.ink_frac
    );
}

/// Rung 3, the can-fire half: a REAL page carrying BOTH degradations at
/// once — illumination unevenness (from `uneven_linear_085.pgm`'s own
/// source) AND contrast compression (`compress_contrast`, `b=0.15`) —
/// defeats Otsu AND Sauvola simultaneously; Wolf recovers.
///
/// Anti-vacuity: asserts BOTH upstream rungs are genuinely implausible on
/// this exact fixture (not assumed from the doc comment's quoted numbers,
/// which were measured on a throwaway diagnostic that no longer exists) —
/// recomputed here directly via `binarize_page_with`, so a change to either
/// method's behaviour that accidentally makes this fixture easy again is
/// caught as a fixture-precondition failure, not a silent pass.
#[test]
fn wolf_recovers_where_otsu_and_sauvola_both_fail() {
    let (grey, w, h) = load("uneven_linear_085.pgm");
    let combined = compress_contrast(&grey, 0.15);

    use tesseract_ocr::xy_cut::{binarize_page_with, BinarizeMode};
    let otsu_frac = {
        let b = binarize_page_with(&combined, w, h, BinarizeMode::Otsu);
        b.iter().filter(|&&p| p == 0).count() as f32 / b.len() as f32
    };
    let sauvola_frac = {
        let b = binarize_page_with(
            &combined,
            w,
            h,
            BinarizeMode::Sauvola {
                whsize: WHSIZE,
                k: 0.34,
            },
        );
        b.iter().filter(|&&p| p == 0).count() as f32 / b.len() as f32
    };
    assert!(
        !(0.005..=0.15).contains(&otsu_frac),
        "fixture must genuinely defeat Otsu, or this test proves nothing \
         about rung 3: otsu_frac={otsu_frac}"
    );
    assert!(
        !(0.005..=0.15).contains(&sauvola_frac),
        "fixture must genuinely defeat Sauvola too, or escalation would stop \
         at rung 2: sauvola_frac={sauvola_frac}"
    );

    let r = binarize_page_escalating(&combined, w, h, WHSIZE);
    assert_eq!(
        r.mode,
        EscalatedMode::Wolf,
        "a page defeating both Otsu and Sauvola must recover at rung 3 (Wolf)"
    );
}

/// Rung 4 — the terminal rung, reached and returned without panicking when
/// Otsu, Sauvola AND Wolf all read implausible.
///
/// **Honest scope, matching [`binarize_page_escalating`]'s own doc pin: this
/// test does NOT claim Singh recovers anything.** At this fixture's severity
/// (illumination unevenness + `b=0.01` contrast compression + additive
/// noise) Wolf and Singh BOTH read `ink_frac ≈ 0` — measured, not assumed,
/// via the same direct recomputation `wolf_recovers_where_otsu_and_sauvola_
/// both_fail` uses. What this test proves is narrower and still real: the
/// cascade mechanism itself terminates at rung 4 and returns a value (never
/// loops, never panics) when every earlier rung has failed — a genuine
/// falsifier for the CONTROL FLOW, not for Singh's recovery power.
#[test]
fn singh_is_the_terminal_rung_when_every_earlier_rung_fails() {
    let (grey, w, h) = load("uneven_vignette_085.pgm");
    let combined = add_noise(&compress_contrast(&grey, 0.01), 0.15, 0x1234_5678);

    use tesseract_ocr::xy_cut::{binarize_page_with, BinarizeMode};
    let frac_of = |mode: BinarizeMode| {
        let b = binarize_page_with(&combined, w, h, mode);
        b.iter().filter(|&&p| p == 0).count() as f32 / b.len() as f32
    };
    let otsu_frac = frac_of(BinarizeMode::Otsu);
    let sauvola_frac = frac_of(BinarizeMode::Sauvola {
        whsize: WHSIZE,
        k: 0.34,
    });
    let wolf_frac = frac_of(BinarizeMode::Wolf {
        whsize: WHSIZE,
        k: 0.5,
    });
    for (label, frac) in [
        ("otsu", otsu_frac),
        ("sauvola", sauvola_frac),
        ("wolf", wolf_frac),
    ] {
        assert!(
            !(0.005..=0.15).contains(&frac),
            "fixture must genuinely defeat {label} too, or this test does not \
             reach rung 4: {label}_frac={frac}"
        );
    }

    let r = binarize_page_escalating(&combined, w, h, WHSIZE);
    assert_eq!(
        r.mode,
        EscalatedMode::Singh,
        "when every earlier rung fails, the ladder must terminate at Singh \
         rather than panicking or looping"
    );
}
