//! Probe: the S-7 axes on the report substrate, with a Tantivy match entering
//! as a MASK.
//!
//! One synthetic archive (20 000 documents) carries paperless-ngx's organizing
//! metadata: correspondent, document type, storage path (each optional), a
//! created month, tags (M2M), an owner, per-user and per-group shares, and a
//! trash flag. A Tantivy index over the same rows is queried with
//! `SearchMaskCollector`, and the counts every paperless-ngx summary screen
//! shows are folded by `lance-graph-report` from lanes and masks.
//!
//! The oracle is a per-row loop written against paperless-ngx's own rules,
//! not against this crate:
//! - visibility: `documents/search/_backend.py::build_permission_filter`
//!   (unowned, owned, shared with the user, shared with one of their groups);
//!   superuser: `documents/permissions.py::get_document_count_filter_for_user`
//!   (everything not in the trash);
//! - a per-tag count is the number of permitted documents carrying the tag
//!   (`annotate_document_count_by_ids`), optionally intersected with a search.
//!
//! It is NOT paperless-ngx's ORM executed on the same fixture — that would
//! need a Django database here. The rules above are cited so the oracle can
//! be read against them line by line.

use std::sync::OnceLock;

use lance_graph_report::{
    AbiBatch, AxisRole, CellValue, CoordSpec, Measure, PlannerPolicy, ReportPlan, Selection,
    SourceId, SourceRef,
};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, FAST, INDEXED, TEXT};
use tantivy::{doc, Index, IndexReader};
use tesseract_paperless::axes::{
    self, build_batch, superuser, tag_mask, visible_to, with_search, ArchiveDoc, AxisDomains,
    SearchMaskCollector, CORRESPONDENT, CREATED_MONTH, DOCUMENT_TYPE, SEARCH, STORAGE_PATH,
};

const N: usize = 20_000;
const SOURCE: SourceId = SourceId(7);
const DOMAINS: AxisDomains = AxisDomains {
    correspondents: 12,
    document_types: 6,
    storage_paths: 4,
    months: 24,
    users: 5,
    groups: 3,
    tags: 16,
};
/// Group membership per user — the user's CURRENT groups, as paperless-ngx
/// passes `viewer_group_ids`.
const MEMBERSHIP: [&[u32]; 5] = [&[0], &[0, 1], &[], &[2], &[1]];
const WORDS: [&str; 12] = [
    "invoice",
    "contract",
    "receipt",
    "reminder",
    "insurance",
    "tax",
    "salary",
    "rent",
    "offer",
    "delivery",
    "warranty",
    "statement",
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u32 {
        (self.next() % n) as u32
    }
    fn chance(&mut self, pct: u64) -> bool {
        self.next() % 100 < pct
    }
}

struct Fixture {
    docs: Vec<ArchiveDoc>,
    words: Vec<Vec<&'static str>>,
    batch: AbiBatch,
    index: Index,
    reader: IndexReader,
    text: Field,
}

fn fixture() -> &'static Fixture {
    static F: OnceLock<Fixture> = OnceLock::new();
    F.get_or_init(|| {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut docs = Vec::with_capacity(N);
        let mut words: Vec<Vec<&'static str>> = Vec::with_capacity(N);
        for _ in 0..N {
            let mut tags: Vec<u32> = (0..rng.below(4)).map(|_| rng.below(16)).collect();
            tags.sort_unstable();
            tags.dedup();
            docs.push(ArchiveDoc {
                correspondent: rng.chance(85).then(|| rng.below(12)),
                document_type: rng.chance(80).then(|| rng.below(6)),
                storage_path: rng.chance(50).then(|| rng.below(4)),
                created_month: rng.below(24),
                owner: rng.chance(70).then(|| rng.below(5)),
                tags,
                viewers: if rng.chance(10) {
                    vec![rng.below(5)]
                } else {
                    vec![]
                },
                viewer_groups: if rng.chance(10) {
                    vec![rng.below(3)]
                } else {
                    vec![]
                },
                deleted: rng.chance(3),
            });
            words.push((0..4).map(|_| WORDS[rng.below(12) as usize]).collect());
        }
        let batch = build_batch(SOURCE, 1, DOMAINS, &docs).expect("axes lay out");

        let mut sb = Schema::builder();
        let row = sb.add_u64_field("row", FAST | INDEXED);
        let text = sb.add_text_field("text", TEXT);
        let index = Index::create_in_ram(sb.build());
        let mut w = index.writer(50_000_000).expect("writer");
        for (i, ws) in words.iter().enumerate() {
            w.add_document(doc!(row => i as u64, text => ws.join(" ")))
                .expect("add");
        }
        w.commit().expect("commit");
        let reader = index.reader().expect("reader");
        Fixture {
            docs,
            words,
            batch,
            index,
            reader,
            text,
        }
    })
}

#[derive(Clone, Copy, Debug)]
enum Who {
    Super,
    User(u32),
}

/// The oracle's visibility — paperless-ngx's rule, per row.
fn visible(d: &ArchiveDoc, who: Who) -> bool {
    if d.deleted {
        return false;
    }
    match who {
        Who::Super => true,
        Who::User(u) => {
            d.owner.is_none()
                || d.owner == Some(u)
                || d.viewers.contains(&u)
                || d.viewer_groups
                    .iter()
                    .any(|g| MEMBERSHIP[u as usize].contains(g))
        }
    }
}

fn selection(who: Who) -> Selection {
    match who {
        Who::Super => superuser(),
        Who::User(u) => visible_to(u, MEMBERSHIP[u as usize]),
    }
}

/// A search the oracle can evaluate: every listed word must occur (`all`),
/// or at least one must (`!all`). The Tantivy query string says the same.
struct Search {
    query: &'static str,
    words: &'static [&'static str],
    all: bool,
}

const SEARCHES: [Search; 3] = [
    Search {
        query: "invoice",
        words: &["invoice"],
        all: true,
    },
    Search {
        query: "invoice AND reminder",
        words: &["invoice", "reminder"],
        all: true,
    },
    Search {
        query: "tax OR rent",
        words: &["tax", "rent"],
        all: false,
    },
];

fn matches(ws: &[&str], s: &Search) -> bool {
    let has = |w: &&str| ws.contains(w);
    if s.all {
        s.words.iter().all(has)
    } else {
        s.words.iter().any(has)
    }
}

fn search_mask(f: &Fixture, s: &Search) -> axes::SearchMask {
    let q = QueryParser::for_index(&f.index, vec![f.text])
        .parse_query(s.query)
        .expect("query parses");
    f.reader
        .searcher()
        .search(&q, &SearchMaskCollector::new("row", N))
        .expect("search")
}

/// The batch a query folds over: the archive, plus the search mask if any.
fn batch_for(f: &Fixture, s: Option<&Search>) -> AbiBatch {
    match s {
        None => f.batch.clone(),
        Some(s) => {
            let m = search_mask(f, s);
            assert_eq!(m.stray, 0, "index and archive disagree on rows");
            with_search(&f.batch, m.words.into()).expect("attach search")
        }
    }
}

fn plan(sel: Selection) -> ReportPlan {
    ReportPlan::over(SourceRef {
        id: SOURCE,
        generation: 1,
    })
    .filter(sel)
    .measure(Measure::count())
}

fn as_count(v: CellValue) -> i64 {
    match v {
        CellValue::Int(n) => n,
        CellValue::Null => 0,
        CellValue::Real(r) => panic!("a count folded to a real: {r}"),
    }
}

fn fold_count(batch: &AbiBatch, sel: Selection) -> i64 {
    let (r, _) = plan(sel)
        .execute(batch, &PlannerPolicy::default())
        .expect("fold");
    as_count(r.grand_total(&Measure::count()))
}

fn base_selection(who: Who, s: Option<&Search>) -> Selection {
    let sel = selection(who);
    match s {
        Some(_) => sel.and(Selection::Mask(SEARCH)),
        None => sel,
    }
}

fn oracle_rows<'f>(
    f: &'f Fixture,
    who: Who,
    s: Option<&'f Search>,
) -> impl Iterator<Item = &'f ArchiveDoc> + 'f {
    f.docs
        .iter()
        .zip(&f.words)
        .filter(move |(d, ws)| visible(d, who) && s.is_none_or(|s| matches(ws, s)))
        .map(|(d, _)| d)
}

#[test]
fn a_search_mask_is_exactly_the_ranked_hit_set() {
    let f = fixture();
    let searcher = f.reader.searcher();
    for s in &SEARCHES {
        let m = search_mask(f, s);
        assert_eq!(m.stray, 0);
        // The ranked path, collected in full, maps each hit back to its row.
        let q = QueryParser::for_index(&f.index, vec![f.text])
            .parse_query(s.query)
            .unwrap();
        let hits = searcher
            .search(&q, &TopDocs::with_limit(N).order_by_score())
            .unwrap();
        let mut ranked = vec![0u64; N.div_ceil(64)];
        for (_, addr) in &hits {
            let row = searcher
                .segment_reader(addr.segment_ord)
                .fast_fields()
                .u64("row")
                .unwrap()
                .first(addr.doc_id)
                .unwrap() as usize;
            ranked[row >> 6] |= 1 << (row & 63);
        }
        assert_eq!(m.words, ranked, "{}", s.query);
        // …and both equal the oracle's reading of the query.
        let oracle = f.words.iter().filter(|ws| matches(ws, s)).count() as u64;
        assert_eq!(m.count(), oracle, "{}", s.query);
        // Anti-vacuity: a real, partial selection — neither empty-ish nor
        // near-total (`tax OR rent` is ~52% by construction: 4 words of 12).
        let n = oracle as usize;
        assert!(n * 100 > N && n * 10 < N * 9, "{}: {oracle}", s.query);
    }
}

#[test]
fn a_stray_row_is_reported_not_folded() {
    let mut sb = Schema::builder();
    let row = sb.add_u64_field("row", FAST | INDEXED);
    let text = sb.add_text_field("text", TEXT);
    let index = Index::create_in_ram(sb.build());
    let mut w = index.writer(15_000_000).unwrap();
    w.add_document(doc!(row => 1u64, text => "invoice"))
        .unwrap();
    w.add_document(doc!(row => 64u64, text => "invoice"))
        .unwrap(); // outside 0..10
    w.add_document(doc!(text => "invoice")).unwrap(); // no row at all
    w.commit().unwrap();
    let q = QueryParser::for_index(&index, vec![text])
        .parse_query("invoice")
        .unwrap();
    let m = index
        .reader()
        .unwrap()
        .searcher()
        .search(&q, &SearchMaskCollector::new("row", 10))
        .unwrap();
    assert_eq!(m.stray, 2, "can fire: out-of-range and missing rows");
    assert_eq!(m.words, vec![0b10], "only the in-range row is set");
}

#[test]
fn per_tag_counts_match_paperless_ngx_semantics() {
    let f = fixture();
    let whos = [
        Who::Super,
        Who::User(0),
        Who::User(1),
        Who::User(2),
        Who::User(3),
        Who::User(4),
    ];
    let searches: [Option<&Search>; 4] = [
        None,
        Some(&SEARCHES[0]),
        Some(&SEARCHES[1]),
        Some(&SEARCHES[2]),
    ];
    for s in searches {
        let batch = batch_for(f, s);
        for who in whos {
            for t in 0..DOMAINS.tags {
                let got = fold_count(
                    &batch,
                    base_selection(who, s).and(Selection::Mask(tag_mask(t))),
                );
                let want = oracle_rows(f, who, s)
                    .filter(|d| d.tags.contains(&t))
                    .count() as i64;
                assert_eq!(got, want, "tag {t}, {who:?}, {:?}", s.map(|s| s.query));
            }
        }
    }
}

#[test]
fn correspondent_by_month_pivot_matches_the_oracle_cell_for_cell() {
    let f = fixture();
    let s = &SEARCHES[2];
    let batch = batch_for(f, Some(s));
    let who = Who::User(1);
    let (r, _) = plan(base_selection(who, Some(s)))
        .axis(CoordSpec::Field(CORRESPONDENT), AxisRole::Row)
        .axis(CoordSpec::Field(CREATED_MONTH), AxisRole::Column)
        .execute(&batch, &PlannerPolicy::default())
        .expect("fold");
    let mut want = vec![vec![0i64; DOMAINS.months as usize]; DOMAINS.correspondents as usize + 1];
    for d in oracle_rows(f, who, Some(s)) {
        let c = d.correspondent.map_or(0, |c| c + 1) as usize;
        want[c][d.created_month as usize] += 1;
    }
    let count = Measure::count();
    let mut total = 0;
    for (c, row) in want.iter().enumerate() {
        for (m, &w) in row.iter().enumerate() {
            let got = as_count(r.value(&count, &[], &[c as u32], &[m as u32]));
            assert_eq!(got, w, "correspondent ordinal {c}, month {m}");
            total += w;
        }
    }
    assert_eq!(as_count(r.grand_total(&count)), total);
    // Anti-vacuity: the "no correspondent" row is populated and the pivot is
    // not the whole archive.
    assert!(want[0].iter().sum::<i64>() > 0);
    assert!((total as usize) * 3 < N);
}

#[test]
fn every_single_valued_axis_facets_like_the_oracle_including_none() {
    let f = fixture();
    let who = Who::User(3);
    let batch = batch_for(f, None);
    type Get = fn(&ArchiveDoc) -> Option<u32>;
    let axes: [(_, u32, Get); 3] = [
        (DOCUMENT_TYPE, DOMAINS.document_types, |d| d.document_type),
        (STORAGE_PATH, DOMAINS.storage_paths, |d| d.storage_path),
        (CORRESPONDENT, DOMAINS.correspondents, |d| d.correspondent),
    ];
    for (field, domain, get) in axes {
        let (r, _) = plan(selection(who))
            .axis(CoordSpec::Field(field), AxisRole::Row)
            .execute(&batch, &PlannerPolicy::default())
            .expect("fold");
        let mut want = vec![0i64; domain as usize + 1];
        for d in oracle_rows(f, who, None) {
            want[get(d).map_or(0, |v| v + 1) as usize] += 1;
        }
        for (o, &w) in want.iter().enumerate() {
            let got = as_count(r.value(&Measure::count(), &[], &[o as u32], &[]));
            assert_eq!(got, w, "{field:?} ordinal {o}");
        }
        assert!(want[0] > 0, "{field:?}: the none bucket is exercised");
    }
}

#[test]
fn the_permission_filter_fires_for_users_and_stays_silent_for_the_superuser() {
    let f = fixture();
    let batch = &f.batch;
    let live = f.docs.iter().filter(|d| !d.deleted).count() as i64;
    let trashed = N as i64 - live;
    assert!(trashed > 0, "the trash is exercised");
    // Stay silent: a superuser sees everything outside the trash.
    assert_eq!(fold_count(batch, superuser()), live);
    for u in 0..DOMAINS.users {
        let got = fold_count(batch, selection(Who::User(u)));
        let want = oracle_rows(f, Who::User(u), None).count() as i64;
        assert_eq!(got, want, "user {u}");
        // Can fire: every user is denied a real share of the live archive.
        assert!(got < live && got > 0, "user {u}: {got} of {live}");
    }
    // The share and group masks each matter: user 2 has no groups, user 1 has
    // two — their visible sets differ, and user 1's is not a subset of what
    // the owner rule alone would give.
    let owner_only = |u: u32| {
        f.docs
            .iter()
            .filter(|d| !d.deleted && (d.owner.is_none() || d.owner == Some(u)))
            .count() as i64
    };
    assert!(fold_count(batch, selection(Who::User(1))) > owner_only(1));
}
