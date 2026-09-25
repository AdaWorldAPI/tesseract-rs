//! Wave A of `.claude/plans/paperless-archive-integration-v1.md`: the token
//! lane persists, receipts are addressed by content rather than position, and
//! alphabet saturation is a reported condition instead of a silent empty
//! stream.
//!
//! Every claim here has the input that would make it fail written next to it.

use tesseract_paperless::token::contract::{
    query_refusals, source_refusals, ContractDecodeError, NormRule, TokenizerContract, VOCAB_CAP,
};
use tesseract_paperless::token::docir::SpanKey;
use tesseract_paperless::token::lane::{LaneDecodeError, TokenLane};
use tesseract_paperless::token::seam_tantivy::{handle_for, resolve_handle, SeamStore};

const DOC_A: [u8; 32] = [0xA1; 32];
const DOC_B: [u8; 32] = [0xB2; 32];

/// `(document, page, reading order, canonical text)` — enough spans that a
/// run crosses particle boundaries and one run is an exact multiple of 12.
const SPANS: &[([u8; 32], u16, u16, &str)] = &[
    (DOC_A, 1, 0, "the invoice total is due on receipt"),
    (DOC_A, 1, 1, "payment terms net thirty days"),
    (DOC_A, 2, 0, "abcdefghijkl"),
    (DOC_B, 1, 0, "the receipt for the invoice is attached"),
    (DOC_B, 1, 3, "thirty days net"),
];

fn corpus() -> Vec<u8> {
    SPANS.iter().flat_map(|s| s.3.bytes()).collect()
}

fn build(order: &[usize], contract: &TokenizerContract) -> TokenLane {
    let mut lane = TokenLane::new();
    for &i in order {
        let (sha, page, reading_order, text) = SPANS[i];
        let doc = lane.intern_document(sha);
        let (ids, _) = contract.try_encode(text.as_bytes()).expect("in alphabet");
        lane.append(
            SpanKey {
                doc,
                page,
                reading_order,
            },
            0,
            contract,
            &ids,
        );
    }
    lane
}

fn decoded(lane: &TokenLane, contract: &TokenizerContract, handle: &str) -> Option<Vec<u8>> {
    let r = resolve_handle(lane, handle)?;
    Some(lane.view(r, contract)?.decode())
}

/// FAILS IF: the contract or lane loses anything on a save/load cycle — a
/// restored receipt that decodes to different bytes, or a restored contract
/// with a different identity.
#[test]
fn a_persisted_lane_resolves_every_handle_to_the_same_bytes_after_reload() {
    let contract = TokenizerContract::train(&corpus(), NormRule::Identity);
    let lane = build(&[0, 1, 2, 3, 4], &contract);
    let handles: Vec<String> = lane
        .receipts()
        .iter()
        .map(|r| handle_for(&lane, r).expect("interned"))
        .collect();

    let contract2 = TokenizerContract::from_bytes(&contract.to_bytes()).expect("contract loads");
    let lane2 = TokenLane::from_bytes(&lane.to_bytes()).expect("lane loads");

    assert_eq!(contract2.contract_id(), contract.contract_id());
    assert_eq!(lane2.receipts(), lane.receipts());
    for (h, (_, _, _, text)) in handles.iter().zip(SPANS) {
        assert_eq!(
            decoded(&lane2, &contract2, h).as_deref(),
            Some(text.as_bytes()),
            "{h}"
        );
    }
    // The restored contract encodes exactly as the original did.
    for (_, _, _, text) in SPANS {
        assert_eq!(
            contract2.try_encode(text.as_bytes()),
            contract.try_encode(text.as_bytes())
        );
    }
}

/// FAILS IF: handles are positional. The same spans appended in a different
/// order sit at different lane indexes; a content address must still resolve
/// every one of them to the same text.
#[test]
fn a_handle_survives_reordering_the_lane() {
    let contract = TokenizerContract::train(&corpus(), NormRule::Identity);
    let forward = build(&[0, 1, 2, 3, 4], &contract);
    let reversed = build(&[4, 3, 2, 1, 0], &contract);

    let mut moved = 0;
    for (i, r) in forward.receipts().iter().enumerate() {
        let h = handle_for(&forward, r).expect("interned");
        let there = resolve_handle(&reversed, &h).expect("resolves in the other lane");
        if reversed.receipts().iter().position(|x| x == there) != Some(i) {
            moved += 1;
        }
        assert_eq!(
            decoded(&reversed, &contract, &h),
            decoded(&forward, &contract, &h),
            "{h}"
        );
    }
    // Anti-vacuity: the reorder really moved receipts, so a positional handle
    // would have pointed at the wrong span.
    assert!(moved >= 4, "only {moved} receipts moved");
}

/// FAILS IF: a malformed or foreign handle resolves to anything.
#[test]
fn a_handle_for_an_unknown_span_resolves_to_nothing() {
    let contract = TokenizerContract::train(&corpus(), NormRule::Identity);
    let lane = build(&[0, 1, 2], &contract);
    let good = handle_for(&lane, &lane.receipts()[0]).expect("interned");
    assert!(
        resolve_handle(&lane, &good).is_some(),
        "the control must resolve"
    );

    let wrong_page = good.replacen(":1:0", ":9:0", 1);
    let foreign_doc = good.replacen(&"a1".repeat(32), &"b2".repeat(32), 1);
    for bad in [
        wrong_page.as_str(),
        foreign_doc.as_str(),
        "rcpt:0",
        "rcpt:",
        "invoice",
        &good[..good.len() - 2],
        &format!("{good}:7"),
    ] {
        assert!(resolve_handle(&lane, bad).is_none(), "{bad} resolved");
    }
}

/// FAILS IF: a 256-byte alphabet assigns a byte the PAD id (it would then
/// vanish on decode), or saturation goes unreported.
#[test]
fn a_full_byte_alphabet_is_reported_and_never_collides_with_pad() {
    let corpus: Vec<u8> = (0..=255u8).collect();
    let (contract, report) = TokenizerContract::train_reported(&corpus, NormRule::Identity);

    assert_eq!(report.distinct_bytes, 256);
    assert_eq!(report.excluded_bytes, vec![255]);
    assert!(report.vocab_full && report.saturated());
    assert_eq!(contract.vocab_len(), VOCAB_CAP);

    // Every byte that DID fit round-trips exactly — no id aliases PAD.
    let fit: Vec<u8> = (0..255u8).collect();
    let (ids, _) = contract.try_encode(&fit).expect("all in alphabet");
    assert!(!ids.contains(&0xFF));
    assert_eq!(contract.decode(&ids).0, fit);

    // The excluded byte is refused and counted, never silently dropped.
    let before = source_refusals();
    assert!(contract.try_encode(&[1, 2, 255]).is_none());
    assert!(source_refusals() > before);
}

/// FAILS IF: an ordinary corpus is reported as saturated — a signal that
/// fires on every archive carries no information.
#[test]
fn an_ordinary_corpus_is_not_reported_saturated() {
    let (_, report) = TokenizerContract::train_reported(&corpus(), NormRule::Identity);
    assert!(report.excluded_bytes.is_empty());
    assert!(!report.saturated(), "{report:?}");
    assert!(report.distinct_bytes > 10, "anti-vacuity: a real alphabet");
}

/// FAILS IF: a query containing an untrained byte looks the same as a query
/// that is merely absent from the archive.
#[test]
fn an_unencodable_query_is_distinguished_from_a_miss() {
    let contract = TokenizerContract::train(&corpus(), NormRule::Identity);
    let store = SeamStore {
        contract,
        lane: TokenLane::new(),
    };
    assert_eq!(store.covers("invoice"), Ok(()));
    assert_eq!(store.covers("net zebra"), Err(4), "'z' was never trained");
    assert_eq!(store.covers("€"), Err(0));

    let before = query_refusals();
    assert!(store
        .contract
        .try_encode_query("zebra".as_bytes())
        .is_none());
    assert!(query_refusals() > before);
}

/// FAILS IF: corrupted persisted bytes load. A lane that loads must frame
/// every receipt inside its own particles.
#[test]
fn corrupted_persisted_bytes_are_refused() {
    let contract = TokenizerContract::train(&corpus(), NormRule::Identity);
    let lane = build(&[0, 1, 2, 3, 4], &contract);
    let good = lane.to_bytes();
    assert!(
        TokenLane::from_bytes(&good).is_ok(),
        "the control must load"
    );

    assert_eq!(
        TokenLane::from_bytes(&good[..good.len() - 1]).unwrap_err(),
        LaneDecodeError::BadLength
    );
    let mut magic = good.clone();
    magic[0] ^= 1;
    assert_eq!(
        TokenLane::from_bytes(&magic).unwrap_err(),
        LaneDecodeError::BadMagic
    );

    // The last receipt's `particle_count` is its last 4 bytes; inflate it past
    // the particles the lane holds.
    let mut framing = good.clone();
    let n = framing.len();
    framing[n - 4..].copy_from_slice(&999u32.to_le_bytes());
    assert_eq!(
        TokenLane::from_bytes(&framing).unwrap_err(),
        LaneDecodeError::BadReceipt(4)
    );

    let cbytes = contract.to_bytes();
    let mut forward = cbytes.clone();
    // Header is 14 bytes; each expansion is (tag, l, r). Make entry 0 a pair
    // that refers to itself.
    forward[14..17].copy_from_slice(&[1, 0, 0]);
    assert_eq!(
        TokenizerContract::from_bytes(&forward).unwrap_err(),
        ContractDecodeError::Invalid("pair refers forward")
    );
    assert_eq!(
        TokenizerContract::from_bytes(&cbytes[..cbytes.len() - 1]).unwrap_err(),
        ContractDecodeError::BadLength
    );
}

/// FAILS IF: two lawful sub-region receipts in the same region share a handle.
/// They differ only by `byte_from`, so a handle that omits it resolves both to
/// whichever was appended last.
#[test]
fn sub_region_receipts_in_one_region_get_distinct_handles() {
    let contract = TokenizerContract::train(&corpus(), NormRule::Identity);
    let mut lane = TokenLane::new();
    let doc = lane.intern_document(DOC_A);
    let key = SpanKey {
        doc,
        page: 1,
        reading_order: 0,
    };
    let (first, _) = contract.try_encode(b"the invoice").expect("in alphabet");
    let (second, _) = contract.try_encode(b"is due").expect("in alphabet");
    let r1 = lane.append(key, 0, &contract, &first);
    let r2 = lane.append(key, 12, &contract, &second);

    let h1 = handle_for(&lane, &r1).expect("interned");
    let h2 = handle_for(&lane, &r2).expect("interned");
    assert_ne!(h1, h2);
    assert_eq!(
        decoded(&lane, &contract, &h1).as_deref(),
        Some(&b"the invoice"[..])
    );
    assert_eq!(
        decoded(&lane, &contract, &h2).as_deref(),
        Some(&b"is due"[..])
    );
}

/// FAILS IF: a loaded lane whose ids are framed correctly but not assigned by
/// the contract (PAD, or past its vocabulary) reaches a consumer. Every
/// consumer indexes the contract's tables by id, so it must panic there.
#[test]
fn a_loaded_lane_with_unassigned_ids_yields_no_view_instead_of_panicking() {
    let contract = TokenizerContract::train(&corpus(), NormRule::Identity);
    assert!(
        contract.vocab_len() < 0xFE,
        "anti-vacuity: 0xFE must be unassigned"
    );
    let lane = build(&[0], &contract);
    let good = lane.to_bytes();
    let control = TokenLane::from_bytes(&good).expect("control loads");
    assert!(control.view(&control.receipts()[0], &contract).is_some());

    // Header 20 B + one 32 B document; the first particle's first id follows.
    let particles_at = 20 + 32;
    for bad in [0xFE_u8, 0xFF] {
        let mut bytes = good.clone();
        bytes[particles_at] = bad;
        let lane = TokenLane::from_bytes(&bytes).expect("framing is still valid");
        let r = lane.receipts()[0];
        assert!(
            lane.view(&r, &contract).is_none(),
            "id {bad:#04x} was viewed"
        );
        let h = handle_for(&lane, &r).expect("interned");
        assert!(decoded(&lane, &contract, &h).is_none());
    }
}

/// FAILS IF: a small crafted contract blob whose pairs keep doubling their
/// surface is materialised instead of refused — 40 entries would ask for
/// 2^39 bytes.
#[test]
fn a_doubling_expansion_chain_is_refused_before_it_allocates() {
    let n: u32 = 40;
    let mut bytes = b"PLTOKC01".to_vec();
    bytes.push(0); // Identity
    bytes.push(255); // VOCAB_CAP
    bytes.extend_from_slice(&n.to_le_bytes());
    bytes.extend_from_slice(&[0, b'a', 0]);
    for i in 1..n {
        let prev = u8::try_from(i - 1).expect("< 255");
        bytes.extend_from_slice(&[1, prev, prev]);
    }
    assert_eq!(
        TokenizerContract::from_bytes(&bytes).unwrap_err(),
        ContractDecodeError::Invalid("surface too long")
    );

    // Silence twin: the same chain kept short enough is a lawful contract.
    let short = 8u32;
    let mut ok = bytes[..10].to_vec();
    ok.extend_from_slice(&short.to_le_bytes());
    ok.extend_from_slice(&bytes[14..14 + 3 * short as usize]);
    let c = TokenizerContract::from_bytes(&ok).expect("a short chain loads");
    assert_eq!(c.surface(7).len(), 128);
}
