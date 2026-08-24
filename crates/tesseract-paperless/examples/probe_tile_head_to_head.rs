//! `PROBE-TILE-H2H-1` — head-to-head: the flat 8-bit token id (255-cap, the
//! first cut's invention) vs the **256:256 rail token** (`u16`, 65 536-cap,
//! one token per `(hi:lo)` rail, six per 12-byte particle — the native V3
//! reading, and byte-for-byte the same SHAPE as `DeepNSM`-v2's `WordId`).
//!
//! Separate tiles, same shape (operator ruling): a token id lives in its OWN
//! 256×256 tile; `WordId` lives in `DeepNSM`'s. Nothing here touches
//! `PaletteVocab` — the shape is shared, the space is not.
//!
//! Same trainer, same corpora, same span population. ONLY the cap and the
//! packing differ, so any difference in the table is the carrier's.
//!
//! # Honesty box
//! - The 8-bit side's "saturation at 75 KB" was an artifact of a cap the first
//!   cut invented. This probe measures what the substrate's own width does on
//!   the same bytes.
//! - Cost is reported as merge/apply operation counts, never wall time.
//! - The trainer here is a positions-list BPE (linked token list + lazy heap),
//!   NOT the naive full-pass one carried from #1012 — the naive form is
//!   O(merges × stream) and unusable at a 65 536 cap. Both widths use THIS
//!   trainer, so the comparison stays fair; its output is validated by
//!   byte-exact reconstruction, not assumed.

#![allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::cast_precision_loss
)]

use std::collections::{BinaryHeap, HashMap};

/// Dead-node marker in the linked token list.
const DEAD: u32 = u32::MAX;

#[derive(Clone, Copy)]
enum Exp {
    Base(u8),
    Pair(u32, u32),
}

/// One trained table, id width = u32 internally, capped by `cap` (255 for the
/// u8 carrier, `65_535` for the u16 rail carrier — one id reserved as PAD in
/// each width).
struct Bpe {
    expand: Vec<Exp>,
    base_of: HashMap<u8, u32>,
    /// (left,right) -> merged id, in training order (rank = id).
    ranks: HashMap<(u32, u32), u32>,
    strings: Vec<Vec<u8>>,
    hit_cap: bool,
    apply_ops: usize,
}

impl Bpe {
    fn train(corpus: &[u8], cap: usize) -> Self {
        // Base alphabet.
        let mut base_of: HashMap<u8, u32> = HashMap::new();
        let mut expand: Vec<Exp> = Vec::new();
        let mut strings: Vec<Vec<u8>> = Vec::new();
        for &b in corpus {
            base_of.entry(b).or_insert_with(|| {
                expand.push(Exp::Base(b));
                strings.push(vec![b]);
                u32::try_from(expand.len() - 1).expect("alphabet fits")
            });
        }
        // Linked token list.
        let n = corpus.len();
        let mut id: Vec<u32> = corpus.iter().map(|b| base_of[b]).collect();
        let n32 = u32::try_from(n).expect("corpus fits u32");
        let mut next: Vec<u32> = (1..=n32).collect();
        let mut prev: Vec<u32> = (0..n32).map(|i| i.wrapping_sub(1)).collect();
        if n > 0 {
            next[n - 1] = DEAD;
        }
        // Pair counts, occurrence positions (left-node index), lazy max-heap.
        let mut counts: HashMap<(u32, u32), i64> = HashMap::new();
        let mut occ: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        let mut heap: BinaryHeap<(i64, (u32, u32))> = BinaryHeap::new();
        for i in 0..n.saturating_sub(1) {
            let p = (id[i], id[i + 1]);
            *counts.entry(p).or_default() += 1;
            occ.entry(p)
                .or_default()
                .push(u32::try_from(i).expect("fits"));
        }
        for (&p, &c) in &counts {
            heap.push((c, p));
        }
        let mut ranks: HashMap<(u32, u32), u32> = HashMap::new();
        let mut apply_ops = 0usize;
        let mut hit_cap = false;
        loop {
            if expand.len() >= cap {
                hit_cap = true;
                break;
            }
            // Pop until a live entry.
            let Some((c, p)) = heap.pop() else { break };
            if counts.get(&p).copied().unwrap_or(0) != c {
                continue; // stale
            }
            if c < 2 {
                break;
            }
            let new_id = u32::try_from(expand.len()).expect("bounded by cap");
            expand.push(Exp::Pair(p.0, p.1));
            let mut s = strings[p.0 as usize].clone();
            s.extend_from_slice(&strings[p.1 as usize]);
            strings.push(s);
            ranks.insert(p, new_id);
            counts.insert(p, 0);
            let positions = occ.remove(&p).unwrap_or_default();
            for i in positions {
                apply_ops += 1;
                let i = i as usize;
                // Validate: node alive, still this pair.
                if id[i] == DEAD || id[i] != p.0 {
                    continue;
                }
                let j = next[i];
                if j == DEAD || id[j as usize] != p.1 {
                    continue;
                }
                let j = j as usize;
                // Merge j into i.
                let l = prev[i];
                let r = next[j];
                id[i] = new_id;
                id[j] = DEAD;
                next[i] = r;
                if r != DEAD {
                    prev[r as usize] = u32::try_from(i).expect("fits");
                }
                // Left neighbour pair updates.
                if l != DEAD {
                    let ol = (id[l as usize], p.0);
                    let e = counts.entry(ol).or_default();
                    *e -= 1;
                    heap.push((*e, ol));
                    let nl = (id[l as usize], new_id);
                    let e = counts.entry(nl).or_default();
                    *e += 1;
                    heap.push((*e, nl));
                    occ.entry(nl).or_default().push(l);
                }
                // Right neighbour pair updates.
                if r != DEAD {
                    let or_ = (p.1, id[r as usize]);
                    let e = counts.entry(or_).or_default();
                    *e -= 1;
                    heap.push((*e, or_));
                    let nr = (new_id, id[r as usize]);
                    let e = counts.entry(nr).or_default();
                    *e += 1;
                    heap.push((*e, nr));
                    occ.entry(nr)
                        .or_default()
                        .push(u32::try_from(i).expect("fits"));
                }
            }
        }
        Self {
            expand,
            base_of,
            ranks,
            strings,
            hit_cap,
            apply_ops,
        }
    }

    /// Encode a span: repeatedly merge the lowest-rank adjacent pair present.
    /// Returns `None` on a byte outside the trained alphabet.
    fn encode(&self, src: &[u8], ops: &mut usize) -> Option<Vec<u32>> {
        let mut s: Vec<u32> = Vec::with_capacity(src.len());
        for b in src {
            s.push(*self.base_of.get(b)?);
        }
        loop {
            let mut best: Option<(u32, usize)> = None;
            for k in 0..s.len().saturating_sub(1) {
                *ops += 1;
                if let Some(&r) = self.ranks.get(&(s[k], s[k + 1])) {
                    if best.is_none_or(|(br, _)| r < br) {
                        best = Some((r, k));
                    }
                }
            }
            let Some((rank, _)) = best else { break };
            let (a, b) = match self.expand[rank as usize] {
                Exp::Pair(a, b) => (a, b),
                Exp::Base(_) => unreachable!("ranks map only holds pairs"),
            };
            let mut out = Vec::with_capacity(s.len());
            let mut k = 0;
            while k < s.len() {
                if k + 1 < s.len() && s[k] == a && s[k + 1] == b {
                    out.push(rank);
                    k += 2;
                } else {
                    out.push(s[k]);
                    k += 1;
                }
            }
            s = out;
        }
        Some(s)
    }

    fn decode(&self, tokens: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut stack = Vec::new();
        for &t in tokens {
            stack.push(t);
            while let Some(x) = stack.pop() {
                match self.expand[x as usize] {
                    Exp::Base(b) => out.push(b),
                    Exp::Pair(a, b) => {
                        stack.push(b);
                        stack.push(a);
                    }
                }
            }
        }
        out
    }
}

/// One measured row.
struct Row {
    corpus: &'static str,
    width: &'static str,
    cap: usize,
    table: usize,
    hit_cap: bool,
    tokens: usize,
    ratio: f64,
    particles: usize,
    particle_bytes: usize,
    uniq: usize,
    max_surface: usize,
    train_ops: usize,
    enc_ops: usize,
}

fn paragraphs(raw: &str, max: usize) -> Vec<String> {
    let raw = raw.trim_start_matches('\u{feff}').replace("\r\n", "\n");
    let mut out = Vec::new();
    for para in raw.split("\n\n") {
        let p = para.split_whitespace().collect::<Vec<_>>().join(" ");
        if p.len() < 40 {
            continue;
        }
        out.push(p);
        if out.len() >= max {
            break;
        }
    }
    out
}

fn run(corpus: &'static str, spans: &[String], width: &'static str, cap: usize, per: usize) -> Row {
    let joined = spans.join("\n").into_bytes();
    let t = Bpe::train(&joined, cap);
    let mut tokens = 0usize;
    let mut particles = 0usize;
    let mut enc_ops = 0usize;
    let mut uniq = std::collections::HashSet::new();
    for s in spans {
        let ids = t
            .encode(s.as_bytes(), &mut enc_ops)
            .expect("trained on this corpus");
        assert_eq!(
            t.decode(&ids),
            s.as_bytes(),
            "[FAIL] H2H-RECON {corpus}/{width}: span does not decode byte-exact"
        );
        tokens += ids.len();
        particles += ids.len().div_ceil(per);
        uniq.extend(ids.iter().copied());
    }
    let src: usize = spans.iter().map(String::len).sum();
    Row {
        corpus,
        width,
        cap,
        table: t.expand.len(),
        hit_cap: t.hit_cap,
        tokens,
        ratio: src as f64 / tokens as f64,
        particles,
        particle_bytes: particles * 12,
        uniq: uniq.len(),
        max_surface: uniq
            .iter()
            .map(|&i| t.strings[i as usize].len())
            .max()
            .unwrap_or(0),
        train_ops: t.apply_ops,
        enc_ops,
    }
}

fn main() {
    println!("PROBE-TILE-H2H-1 — flat u8 (255-cap) vs 256:256 rail token (u16, 65 536-cap)\n");

    // ---- H2H-SHAPE: the rail token IS WordId's shape, in a separate tile ----
    // Same split/join arithmetic as deepnsm_v2::vocab — proven by calling both
    // on a sweep, not by asserting prose. Separate tiles: nothing in this probe
    // constructs or consults a PaletteVocab.
    for id in [0u16, 1, 0x00FF, 0x0100, 0x50AA, 0xFFFE] {
        let (hi, lo) = ((id >> 8) as u8, (id & 0xFF) as u8);
        assert_eq!(deepnsm_v2::vocab::split(id), (hi, lo), "shape drift");
        assert_eq!(deepnsm_v2::vocab::join(hi, lo), id, "shape drift");
    }
    println!(
        "  [PASS] H2H-SHAPE a rail token (hi:lo) and a DeepNSM WordId (basin:identity) are the \
         SAME split/join arithmetic on a u16 — verified against deepnsm_v2::vocab::split/join \
         on a sweep incl. both byte boundaries. Separate tiles: this probe never touches \
         PaletteVocab, so the shape is shared and the space is not"
    );

    let kjv: Vec<String> = super_scene();
    let alice300 = paragraphs(include_str!("../corpus/alice.txt"), 300);
    let alice_full = paragraphs(include_str!("../corpus/alice.txt"), usize::MAX);
    println!(
        "\n  corpora: kjv-scene {} B / {} spans; alice-300 {} B / {} spans; alice-full {} B / {} spans\n",
        kjv.iter().map(String::len).sum::<usize>(),
        kjv.len(),
        alice300.iter().map(String::len).sum::<usize>(),
        alice300.len(),
        alice_full.iter().map(String::len).sum::<usize>(),
        alice_full.len()
    );

    let mut rows = Vec::new();
    for (name, spans) in [
        ("kjv-scene", &kjv),
        ("alice-300", &alice300),
        ("alice-full", &alice_full),
    ] {
        rows.push(run(name, spans, "u8/12per", 255, 12));
        rows.push(run(name, spans, "u16/6per", 65_535, 6));
    }

    println!(
        "{:<12} {:<9} {:>6} {:>7} {:>8} {:>8} {:>7} {:>6} {:>10} {:>7} {:>6} {:>11} {:>11}",
        "corpus",
        "carrier",
        "cap",
        "table",
        "capped?",
        "tokens",
        "ratio",
        "uniq",
        "maxB/tok",
        "part.",
        "res.KB",
        "train ops",
        "enc ops"
    );
    for r in &rows {
        println!(
            "{:<12} {:<9} {:>6} {:>7} {:>8} {:>8} {:>7.2} {:>6} {:>10} {:>7} {:>6.1} {:>11} {:>11}",
            r.corpus,
            r.width,
            r.cap,
            r.table,
            if r.hit_cap { "CAP" } else { "corpus" },
            r.tokens,
            r.ratio,
            r.uniq,
            r.max_surface,
            r.particles,
            r.particle_bytes as f64 / 1024.0,
            r.train_ops,
            r.enc_ops
        );
    }

    // ---- H2H-SAT: who set the table size, the cap or the corpus? ----
    let u8_alice = rows
        .iter()
        .find(|r| r.corpus == "alice-full" && r.width == "u8/12per")
        .expect("row");
    let u16_alice = rows
        .iter()
        .find(|r| r.corpus == "alice-full" && r.width == "u16/6per")
        .expect("row");
    assert!(
        u8_alice.hit_cap,
        "[FAIL] H2H-SAT: expected the u8 table to be cap-bound on alice-full"
    );
    assert!(
        !u16_alice.hit_cap,
        "[FAIL] H2H-SAT: expected the u16 table to be corpus-bound on alice-full"
    );
    println!(
        "\n  [PASS] H2H-SAT on alice-full the u8 table is CAP-bound at {} while the u16 table \
         stops at {} of 65 535 — the CORPUS set it (no adjacent pair repeats), with {:.1}% of \
         the tile still free. The old \"saturates at 75 KB\" was the cap's property, not the \
         corpus's",
        u8_alice.table,
        u16_alice.table,
        100.0 * (65_535.0 - u16_alice.table as f64) / 65_535.0
    );
    let bytes_u8 = u8_alice.particle_bytes;
    let bytes_u16 = u16_alice.particle_bytes;
    println!(
        "\n  resident particle bytes, alice-full: u8 {} vs u16 {} ({}) — six wider tokens per \
         particle vs twelve narrower ones; the RATIO of surface covered per particle is what \
         decides, and it is measured above, not assumed",
        bytes_u8,
        bytes_u16,
        if bytes_u16 < bytes_u8 {
            format!(
                "u16 smaller by {:.1}%",
                100.0 * (bytes_u8 - bytes_u16) as f64 / bytes_u8 as f64
            )
        } else {
            format!(
                "u16 LARGER by {:.1}%",
                100.0 * (bytes_u16 - bytes_u8) as f64 / bytes_u8 as f64
            )
        }
    );
    println!("\nPROBE-TILE-H2H-1: measured; reconstruction byte-exact on every span, both widths");
}

/// The same KJV scene the seam probe uses, as spans.
fn super_scene() -> Vec<String> {
    [
        "But of the tree of the knowledge of good and evil, thou shalt not eat of it: for in \
         the day that thou eatest thereof thou shalt surely die.",
        "And they were both naked, the man and his wife, and were not ashamed.",
        "Now the serpent was more subtil than any beast of the field which the LORD God had \
         made. And he said unto the woman, Yea, hath God said, Ye shall not eat of every tree \
         of the garden?",
        "And the serpent said unto the woman, Ye shall not surely die:",
        "And when the woman saw that the tree was good for food, and that it was pleasant to \
         the eyes, and a tree to be desired to make one wise, she took of the fruit thereof, \
         and did eat, and gave also unto her husband with her; and he did eat.",
        "And the eyes of them both were opened, and they knew that they were naked; and they \
         sewed fig leaves together, and made themselves aprons.",
        "And they heard the voice of the LORD God walking in the garden in the cool of the \
         day: and Adam and his wife hid themselves from the presence of the LORD God amongst \
         the trees of the garden.",
        "And he said, I heard thy voice in the garden, and I was afraid, because I was naked; \
         and I hid myself.",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect()
}
