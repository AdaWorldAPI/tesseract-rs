//! The S3 copy of the archive: originals, document records and a metadata
//! snapshot (paperless-rs `docs/RECOVERY.md`).
//!
//! Local first, then S3: the volume stays the primary copy and S3 is the copy
//! that survives losing it. Every function here takes the local stores
//! explicitly, so restore can run inside `AppState::load` before the rest of
//! the state exists.
//!
//! Layout under the configured prefix:
//!
//! - `originals/<sha256>`: the exact uploaded bytes;
//! - `documents/<sha256>.json`: a [`DocumentRecord`], written after its original;
//! - `metadata/current.json`: a [`SnapshotFile`], replaced after every change.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tesseract_paperless::archive_meta::{
    AssignmentRow, DefinitionRow, MetaKind, MetaSnapshot, ReviewRow, Source,
};
use tesseract_paperless::blobs::BlobStore;
use tesseract_paperless::kv::ContentSha256;
use tesseract_paperless::store::LanceStore;

use crate::s3::{ObjectError, ObjectStore};

/// Schema tag of a [`DocumentRecord`].
pub const RECORD_SCHEMA: &str = "paperless-rs/document-record/v1";
/// Schema tag of a [`SnapshotFile`].
pub const SNAPSHOT_SCHEMA: &str = "paperless-rs/metadata-snapshot/v1";
/// File name of the restore marker in the original-file directory.
pub const RESTORE_MARKER: &str = ".restore-in-progress";

/// Why a replica operation failed.
#[derive(Debug)]
pub enum ReplicaError {
    /// The bucket refused or could not be reached.
    Object(ObjectError),
    /// The local archive failed.
    Local(String),
    /// A stored object is not what it should be.
    Invalid(String),
}

impl std::fmt::Display for ReplicaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Object(e) => write!(f, "{e}"),
            Self::Local(e) => write!(f, "local archive: {e}"),
            Self::Invalid(e) => write!(f, "invalid object: {e}"),
        }
    }
}

impl From<ObjectError> for ReplicaError {
    fn from(e: ObjectError) -> Self {
        Self::Object(e)
    }
}

fn local(e: impl std::fmt::Display) -> ReplicaError {
    ReplicaError::Local(e.to_string())
}

/// One archived document as kept in S3.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentRecord {
    pub schema: String,
    pub content_sha256_hex: String,
    pub filename: Option<String>,
    pub ingested_at_unix_ms: i64,
    pub mean_confidence: u32,
    pub low_confidence: bool,
    pub doc_ir_json: String,
    pub spo_json: Option<String>,
    pub has_original: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct DefinitionJson {
    kind: String,
    definition_id: u32,
    name: String,
    match_algorithm: u8,
    match_pattern: String,
    case_insensitive: bool,
    retired: bool,
    created_at_unix_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct AssignmentJson {
    content_sha256_hex: String,
    kind: String,
    definition_id: u32,
    source: String,
    assigned_at_unix_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ReviewJson {
    content_sha256_hex: String,
    reviewed_at_unix_ms: i64,
}

/// The metadata snapshot as kept in S3.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotFile {
    schema: String,
    taken_at_unix_ms: i64,
    definitions: Vec<DefinitionJson>,
    assignments: Vec<AssignmentJson>,
    reviews: Vec<ReviewJson>,
}

impl SnapshotFile {
    fn from_snapshot(s: &MetaSnapshot, now_ms: i64) -> Self {
        Self {
            schema: SNAPSHOT_SCHEMA.to_string(),
            taken_at_unix_ms: now_ms,
            definitions: s
                .definitions
                .iter()
                .map(|d| DefinitionJson {
                    kind: d.kind.as_str().to_string(),
                    definition_id: d.definition_id,
                    name: d.name.clone(),
                    match_algorithm: d.match_algorithm,
                    match_pattern: d.match_pattern.clone(),
                    case_insensitive: d.case_insensitive,
                    retired: d.retired,
                    created_at_unix_ms: d.created_at_unix_ms,
                })
                .collect(),
            assignments: s
                .assignments
                .iter()
                .map(|a| AssignmentJson {
                    content_sha256_hex: a.content_sha256_hex.clone(),
                    kind: a.kind.as_str().to_string(),
                    definition_id: a.definition_id,
                    source: a.source.as_str().to_string(),
                    assigned_at_unix_ms: a.assigned_at_unix_ms,
                })
                .collect(),
            reviews: s
                .reviews
                .iter()
                .map(|r| ReviewJson {
                    content_sha256_hex: r.content_sha256_hex.clone(),
                    reviewed_at_unix_ms: r.reviewed_at_unix_ms,
                })
                .collect(),
        }
    }

    fn to_snapshot(&self) -> Result<MetaSnapshot, ReplicaError> {
        if self.schema != SNAPSHOT_SCHEMA {
            return Err(ReplicaError::Invalid(format!(
                "metadata snapshot schema {}",
                self.schema
            )));
        }
        let kind = |k: &str| {
            MetaKind::parse(k).ok_or_else(|| ReplicaError::Invalid(format!("kind {k}")))
        };
        Ok(MetaSnapshot {
            definitions: self
                .definitions
                .iter()
                .map(|d| {
                    Ok(DefinitionRow {
                        kind: kind(&d.kind)?,
                        definition_id: d.definition_id,
                        name: d.name.clone(),
                        match_algorithm: d.match_algorithm,
                        match_pattern: d.match_pattern.clone(),
                        case_insensitive: d.case_insensitive,
                        retired: d.retired,
                        created_at_unix_ms: d.created_at_unix_ms,
                    })
                })
                .collect::<Result<_, ReplicaError>>()?,
            assignments: self
                .assignments
                .iter()
                .map(|a| {
                    Ok(AssignmentRow {
                        content_sha256_hex: a.content_sha256_hex.clone(),
                        kind: kind(&a.kind)?,
                        definition_id: a.definition_id,
                        source: Source::parse(&a.source)
                            .ok_or_else(|| ReplicaError::Invalid(format!("source {}", a.source)))?,
                        assigned_at_unix_ms: a.assigned_at_unix_ms,
                    })
                })
                .collect::<Result<_, ReplicaError>>()?,
            reviews: self
                .reviews
                .iter()
                .map(|r| ReviewRow {
                    content_sha256_hex: r.content_sha256_hex.clone(),
                    reviewed_at_unix_ms: r.reviewed_at_unix_ms,
                })
                .collect(),
        })
    }
}

/// What [`Replica::replicate`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReplicateReport {
    /// Documents whose record was missing and has now been written.
    pub uploaded: usize,
    /// Documents that could not be copied this time.
    pub failed: usize,
}

/// What [`restore`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RestoreReport {
    /// Documents written locally from S3.
    pub restored: usize,
    /// Documents already present locally (an interrupted restore resuming).
    pub already_local: usize,
    /// Records skipped because they failed a check.
    pub skipped: usize,
    /// Documents restored without their original, because it was missing in S3
    /// or failed its hash check.
    pub without_original: usize,
    /// Whether a metadata snapshot was imported.
    pub metadata: bool,
}

/// The S3 side of the archive.
pub struct Replica {
    store: Arc<dyn ObjectStore>,
    prefix: String,
    /// Serializes snapshot uploads; each reads the state while holding it, so
    /// an older snapshot never lands after a newer one.
    snapshot: tokio::sync::Mutex<()>,
}

async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> Result<T, ObjectError> + Send + 'static,
) -> Result<T, ReplicaError> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| ReplicaError::Local(format!("task: {e}")))?
        .map_err(ReplicaError::Object)
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|d| i64::try_from(d.as_millis()).ok())
        .unwrap_or(0)
}

impl Replica {
    pub fn new(store: Arc<dyn ObjectStore>, prefix: &str) -> Self {
        Self {
            store,
            prefix: prefix.trim_matches('/').to_string(),
            snapshot: tokio::sync::Mutex::new(()),
        }
    }

    fn key(&self, rest: &str) -> String {
        if self.prefix.is_empty() {
            rest.to_string()
        } else {
            format!("{}/{rest}", self.prefix)
        }
    }

    pub fn original_key(&self, hash: &str) -> String {
        self.key(&format!("originals/{hash}"))
    }

    pub fn record_key(&self, hash: &str) -> String {
        self.key(&format!("documents/{hash}.json"))
    }

    fn records_prefix(&self) -> String {
        self.key("documents/")
    }

    pub fn snapshot_key(&self) -> String {
        self.key("metadata/current.json")
    }

    /// Hashes that have a record in S3.
    ///
    /// # Errors
    /// [`ReplicaError::Object`] when the listing fails.
    pub async fn recorded_hashes(&self) -> Result<HashSet<String>, ReplicaError> {
        let store = self.store.clone();
        let prefix = self.records_prefix();
        let keys = blocking(move || store.list(&prefix)).await?;
        let prefix = self.records_prefix();
        Ok(keys
            .iter()
            .filter_map(|k| k.strip_prefix(prefix.as_str()))
            .filter_map(|k| k.strip_suffix(".json"))
            .filter(|h| ContentSha256::from_hex(h).is_some())
            .map(str::to_string)
            .collect())
    }

    /// Copy one local document: its original first (when there is one), then
    /// its record.
    ///
    /// # Errors
    /// [`ReplicaError`] when the document cannot be read locally or written to S3.
    pub async fn upload_document(
        &self,
        store: &LanceStore,
        blobs: &Arc<BlobStore>,
        hash: &str,
    ) -> Result<(), ReplicaError> {
        let key = ContentSha256::from_hex(hash)
            .ok_or_else(|| ReplicaError::Invalid(format!("hash {hash}")))?;
        let row = store
            .get(hash)
            .await
            .map_err(local)?
            .ok_or_else(|| ReplicaError::Local(format!("no document {hash}")))?;
        let b = blobs.clone();
        let original = tokio::task::spawn_blocking(move || b.get(&key))
            .await
            .map_err(|e| ReplicaError::Local(format!("task: {e}")))?
            .map_err(local)?;
        let has_original = original.is_some();
        if let Some(bytes) = original {
            let s = self.store.clone();
            let k = self.original_key(hash);
            blocking(move || s.put(&k, &bytes, "application/octet-stream")).await?;
        }
        let record = DocumentRecord {
            schema: RECORD_SCHEMA.to_string(),
            content_sha256_hex: hash.to_string(),
            filename: row.filename,
            ingested_at_unix_ms: row.ingested_at_unix_ms,
            mean_confidence: row.mean_confidence,
            low_confidence: row.low_confidence,
            doc_ir_json: row.doc_ir_json,
            spo_json: row.spo_json,
            has_original,
        };
        let body = serde_json::to_vec(&record).map_err(local)?;
        let s = self.store.clone();
        let k = self.record_key(hash);
        blocking(move || s.put(&k, &body, "application/json")).await
    }

    /// Remove one document from S3: the record first, then the original, so a
    /// failure leaves an unused original, never a record without its original.
    ///
    /// # Errors
    /// [`ReplicaError::Object`] when a delete fails.
    pub async fn delete_document(&self, hash: &str) -> Result<(), ReplicaError> {
        let s = self.store.clone();
        let k = self.record_key(hash);
        blocking(move || s.delete(&k)).await?;
        let s = self.store.clone();
        let k = self.original_key(hash);
        blocking(move || s.delete(&k)).await
    }

    /// Upload the current metadata as the snapshot.
    ///
    /// # Errors
    /// [`ReplicaError`] when the metadata cannot be read or written.
    pub async fn upload_snapshot(&self, store: &LanceStore) -> Result<(), ReplicaError> {
        let _serial = self.snapshot.lock().await;
        let snap = store.meta().export().await.map_err(local)?;
        let body =
            serde_json::to_vec(&SnapshotFile::from_snapshot(&snap, now_ms())).map_err(local)?;
        let s = self.store.clone();
        let k = self.snapshot_key();
        blocking(move || s.put(&k, &body, "application/json")).await
    }

    /// Copy every local document that has no record in S3, then upload the
    /// snapshot. Never deletes anything in S3.
    ///
    /// # Errors
    /// [`ReplicaError`] when the listing, the local read or the snapshot fails.
    /// A single document that fails to copy is counted, not returned.
    pub async fn replicate(
        &self,
        store: &LanceStore,
        blobs: &Arc<BlobStore>,
    ) -> Result<ReplicateReport, ReplicaError> {
        let recorded = self.recorded_hashes().await?;
        let mut report = ReplicateReport::default();
        for hash in store.hashes().await.map_err(local)? {
            if recorded.contains(&hash) {
                continue;
            }
            match self.upload_document(store, blobs, &hash).await {
                Ok(()) => report.uploaded += 1,
                Err(e) => {
                    eprintln!("replica: copying {hash} failed: {e}");
                    report.failed += 1;
                }
            }
        }
        self.upload_snapshot(store).await?;
        Ok(report)
    }

    /// Local documents with no record in S3.
    ///
    /// # Errors
    /// [`ReplicaError`] when the listing or the local read fails.
    pub async fn backlog(&self, store: &LanceStore) -> Result<usize, ReplicaError> {
        let recorded = self.recorded_hashes().await?;
        Ok(store
            .hashes()
            .await
            .map_err(local)?
            .into_iter()
            .filter(|h| !recorded.contains(h))
            .count())
    }
}

/// Whether a start must restore: the local archive is empty, or an earlier
/// restore did not finish.
///
/// # Errors
/// [`ReplicaError::Local`] when the archive cannot be counted.
pub async fn needs_restore(store: &LanceStore, blob_dir: &Path) -> Result<bool, ReplicaError> {
    if blob_dir.join(RESTORE_MARKER).exists() {
        return Ok(true);
    }
    Ok(store.count().await.map_err(local)? == 0)
}

/// Restore the local archive from S3 (`docs/RECOVERY.md` §6). Idempotent: an
/// interrupted restore resumes, skipping documents already local.
///
/// # Errors
/// [`ReplicaError::Object`] when the bucket cannot be listed or a record cannot
/// be read; the marker then stays and the next start resumes.
/// [`ReplicaError::Local`] when the local archive cannot be written.
pub async fn restore(
    replica: &Replica,
    store: &LanceStore,
    blobs: &Arc<BlobStore>,
) -> Result<RestoreReport, ReplicaError> {
    let marker = blobs.root().join(RESTORE_MARKER);
    std::fs::write(&marker, b"restore in progress\n").map_err(local)?;

    let mut report = RestoreReport::default();
    let local_hashes: HashSet<String> = store.hashes().await.map_err(local)?.into_iter().collect();
    let mut recorded: Vec<String> = replica.recorded_hashes().await?.into_iter().collect();
    recorded.sort();
    for hash in recorded {
        if local_hashes.contains(&hash) {
            report.already_local += 1;
            continue;
        }
        let s = replica.store.clone();
        let k = replica.record_key(&hash);
        let body = blocking(move || s.get(&k)).await?;
        let record: DocumentRecord = match serde_json::from_slice(&body) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("restore: record {hash} unreadable ({e}); skipped");
                report.skipped += 1;
                continue;
            }
        };
        if record.schema != RECORD_SCHEMA || record.content_sha256_hex != hash {
            eprintln!("restore: record {hash} has the wrong schema or hash; skipped");
            report.skipped += 1;
            continue;
        }
        let Some(key) = ContentSha256::from_hex(&hash) else {
            report.skipped += 1;
            continue;
        };
        let ir = match ogar_doc_ir::from_json(&record.doc_ir_json) {
            Ok(ir) => ir,
            Err(e) => {
                eprintln!("restore: record {hash} has an unreadable DocIr ({e:?}); skipped");
                report.skipped += 1;
                continue;
            }
        };
        if record.has_original {
            let s = replica.store.clone();
            let k = replica.original_key(&hash);
            match blocking(move || s.get(&k)).await {
                Ok(bytes) if ContentSha256::of(&bytes) == key => {
                    let b = blobs.clone();
                    tokio::task::spawn_blocking(move || b.put(&key, &bytes))
                        .await
                        .map_err(|e| ReplicaError::Local(format!("task: {e}")))?
                        .map_err(local)?;
                }
                Ok(_) => {
                    eprintln!("restore: original of {hash} fails its hash check; restored without it");
                    report.without_original += 1;
                }
                Err(ReplicaError::Object(e)) if e.is_not_found() => {
                    eprintln!("restore: original of {hash} is missing; restored without it");
                    report.without_original += 1;
                }
                Err(e) => return Err(e),
            }
        }
        store
            .put(
                &key,
                record.filename.as_deref(),
                record.mean_confidence,
                record.low_confidence,
                &ir,
                record.ingested_at_unix_ms,
                record.spo_json.as_deref(),
            )
            .await
            .map_err(local)?;
        report.restored += 1;
    }

    let s = replica.store.clone();
    let k = replica.snapshot_key();
    match blocking(move || s.get(&k)).await {
        Ok(body) => {
            let file: SnapshotFile = serde_json::from_slice(&body)
                .map_err(|e| ReplicaError::Invalid(format!("metadata snapshot: {e}")))?;
            let snapshot = file.to_snapshot()?;
            let live: HashSet<String> = store.hashes().await.map_err(local)?.into_iter().collect();
            store
                .meta()
                .import(&snapshot, &live)
                .await
                .map_err(local)?;
            report.metadata = true;
        }
        Err(ReplicaError::Object(e)) if e.is_not_found() => {}
        Err(e) => return Err(e),
    }

    std::fs::remove_file(&marker).map_err(local)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use tesseract_paperless::archive_meta::{MetaKind, Source};

    use super::*;
    use crate::s3::memory::MemoryStore;

    struct Local {
        store: LanceStore,
        blobs: Arc<BlobStore>,
        _dirs: (tempfile::TempDir, tempfile::TempDir),
    }

    async fn local() -> Local {
        let archive = tempfile::tempdir().expect("archive");
        let blobs = tempfile::tempdir().expect("blobs");
        Local {
            store: LanceStore::connect(&archive.path().to_string_lossy())
                .await
                .expect("connect"),
            blobs: Arc::new(BlobStore::open(blobs.path()).expect("blobs")),
            _dirs: (archive, blobs),
        }
    }

    fn ir(text: &str) -> ogar_doc_ir::DocIr {
        ogar_doc_ir::DocIr {
            version: ogar_doc_ir::DOC_IR_VERSION.to_string(),
            source: ogar_doc_ir::Provenance::Ocr,
            geometry: ogar_doc_ir::Geometry::DomOrder,
            content_sha256: [0u8; 32],
            mime: "image/png".to_string(),
            pages: vec![ogar_doc_ir::DocPage {
                number: 0,
                width: 100,
                height: 100,
                regions: vec![ogar_doc_ir::Region {
                    kind: ogar_doc_ir::RegionKind::Text,
                    bbox: ogar_doc_ir::BBoxRail {
                        tl: ogar_doc_ir::Rail { x: 0, y: 0 },
                        br: ogar_doc_ir::Rail { x: 10, y: 10 },
                    },
                    reading_order: 0,
                    text: Some(text.to_string()),
                    cells: Vec::new(),
                    children: Vec::new(),
                }],
            }],
            fields: Vec::new(),
        }
    }

    /// Store one document locally, with or without its original.
    async fn add(l: &Local, bytes: &[u8], name: &str, at: i64, with_original: bool) -> String {
        let key = ContentSha256::of(bytes);
        if with_original {
            l.blobs.put(&key, bytes).expect("blob");
        }
        l.store
            .put(&key, Some(name), 88, false, &ir(name), at, Some("[]"))
            .await
            .expect("row");
        format!("{key:?}")
    }

    fn replica(bucket: &Arc<MemoryStore>) -> Replica {
        Replica::new(bucket.clone(), "pl")
    }

    /// Slice C acceptance: an empty volume gets back every document with its
    /// original (byte-identical), filename and upload time, every definition
    /// with its id, every assignment and review flag. A document stored
    /// before originals were kept comes back without one.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_empty_volume_restores_documents_originals_and_metadata() {
        let bucket = Arc::new(MemoryStore::default());
        let r = replica(&bucket);
        let src = local().await;
        let a = add(&src, b"original A", "a.png", 100, true).await;
        let b = add(&src, b"legacy B", "b.png", 200, false).await;
        let meta = src.store.meta();
        let tag = meta
            .create_definition(MetaKind::Tag, "Strom", 1)
            .await
            .expect("tag");
        meta.assign(&a, MetaKind::Tag, tag.definition_id, Source::Manual, 2)
            .await
            .expect("assign");
        meta.mark_reviewed(&b, 3).await.expect("review");

        let report = r.replicate(&src.store, &src.blobs).await.expect("replicate");
        assert_eq!(report, ReplicateReport { uploaded: 2, failed: 0 });
        assert_eq!(r.backlog(&src.store).await.expect("backlog"), 0);
        assert!(bucket.keys().contains(&r.original_key(&a)));
        assert!(!bucket.keys().contains(&r.original_key(&b)));

        let dst = local().await;
        assert!(needs_restore(&dst.store, dst.blobs.root()).await.expect("check"));
        let report = restore(&r, &dst.store, &dst.blobs).await.expect("restore");
        assert_eq!(report.restored, 2);
        assert_eq!(report.skipped, 0);
        assert_eq!(report.without_original, 0);
        assert!(report.metadata);
        assert!(!dst.blobs.root().join(RESTORE_MARKER).exists());
        assert!(!needs_restore(&dst.store, dst.blobs.root()).await.expect("check"));

        for (hash, name, at) in [(&a, "a.png", 100), (&b, "b.png", 200)] {
            let row = dst.store.get(hash).await.expect("get").expect("restored row");
            assert_eq!(row.filename.as_deref(), Some(name));
            assert_eq!(row.ingested_at_unix_ms, at);
            assert_eq!(row.mean_confidence, 88);
            assert_eq!(row.spo_json.as_deref(), Some("[]"));
            assert_eq!(row.text(), name);
        }
        let key_a = ContentSha256::from_hex(&a).expect("hex");
        assert_eq!(dst.blobs.get(&key_a).expect("get"), Some(b"original A".to_vec()));
        let key_b = ContentSha256::from_hex(&b).expect("hex");
        assert_eq!(dst.blobs.get(&key_b).expect("get"), None);
        assert_eq!(
            dst.store.meta().export().await.expect("export"),
            src.store.meta().export().await.expect("export")
        );
    }

    /// An S3 copy that cannot be reached stops the restore and keeps the
    /// marker, so the start fails and the next one tries again.
    #[tokio::test(flavor = "multi_thread")]
    async fn restore_fails_while_s3_is_unreachable_and_keeps_its_marker() {
        let bucket = Arc::new(MemoryStore::default());
        bucket.down.store(true, Ordering::SeqCst);
        let dst = local().await;
        let r = replica(&bucket);
        assert!(matches!(
            restore(&r, &dst.store, &dst.blobs).await,
            Err(ReplicaError::Object(_))
        ));
        assert!(dst.blobs.root().join(RESTORE_MARKER).exists());
        assert_eq!(dst.store.count().await.expect("count"), 0);
    }

    /// An empty, reachable bucket is a new archive: the restore finds nothing
    /// and removes its marker.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_empty_bucket_is_a_new_archive() {
        let bucket = Arc::new(MemoryStore::default());
        let dst = local().await;
        let report = restore(&replica(&bucket), &dst.store, &dst.blobs)
            .await
            .expect("restore");
        assert_eq!(report, RestoreReport::default());
        assert!(!dst.blobs.root().join(RESTORE_MARKER).exists());
    }

    /// A restore interrupted after some documents resumes: the marker makes a
    /// non-empty archive restore again, and documents already local are
    /// skipped rather than rewritten.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_interrupted_restore_resumes() {
        let bucket = Arc::new(MemoryStore::default());
        let r = replica(&bucket);
        let src = local().await;
        let a = add(&src, b"doc A", "a.png", 1, true).await;
        add(&src, b"doc B", "b.png", 2, true).await;
        r.replicate(&src.store, &src.blobs).await.expect("replicate");

        let dst = local().await;
        // As if the first attempt restored A and stopped.
        add(&dst, b"doc A", "a.png", 1, true).await;
        std::fs::write(dst.blobs.root().join(RESTORE_MARKER), b"x").expect("marker");
        assert!(needs_restore(&dst.store, dst.blobs.root()).await.expect("check"));

        let report = restore(&r, &dst.store, &dst.blobs).await.expect("resume");
        assert_eq!(report.already_local, 1);
        assert_eq!(report.restored, 1);
        assert_eq!(dst.store.count().await.expect("count"), 2);
        assert!(dst.store.get(&a).await.expect("get").is_some());
        assert!(!needs_restore(&dst.store, dst.blobs.root()).await.expect("check"));
    }

    /// An original whose bytes no longer match its key is not restored; the
    /// document comes back without it. A record that is not a record is
    /// skipped.
    #[tokio::test(flavor = "multi_thread")]
    async fn damaged_objects_are_skipped_not_trusted() {
        let bucket = Arc::new(MemoryStore::default());
        let r = replica(&bucket);
        let src = local().await;
        let a = add(&src, b"doc A", "a.png", 1, true).await;
        let b = add(&src, b"doc B", "b.png", 2, true).await;
        r.replicate(&src.store, &src.blobs).await.expect("replicate");
        bucket
            .objects
            .lock()
            .expect("lock")
            .insert(r.original_key(&a), b"tampered".to_vec());
        bucket
            .objects
            .lock()
            .expect("lock")
            .insert(r.record_key(&b), b"not json".to_vec());

        let dst = local().await;
        let report = restore(&r, &dst.store, &dst.blobs).await.expect("restore");
        assert_eq!(report.restored, 1);
        assert_eq!(report.without_original, 1);
        assert_eq!(report.skipped, 1);
        assert!(dst.store.get(&a).await.expect("get").is_some());
        let key_a = ContentSha256::from_hex(&a).expect("hex");
        assert_eq!(dst.blobs.get(&key_a).expect("get"), None);
        assert!(dst.store.get(&b).await.expect("get").is_none());
    }

    /// The original goes up before its record, and a delete removes the
    /// record before the original.
    #[tokio::test(flavor = "multi_thread")]
    async fn originals_precede_records_and_records_go_first_on_delete() {
        let bucket = Arc::new(MemoryStore::default());
        let r = replica(&bucket);
        let src = local().await;
        let a = add(&src, b"ordered", "a.png", 1, true).await;
        r.upload_document(&src.store, &src.blobs, &a)
            .await
            .expect("upload");
        r.delete_document(&a).await.expect("delete");
        assert_eq!(
            *bucket.log.lock().expect("lock"),
            vec![
                format!("put {}", r.original_key(&a)),
                format!("put {}", r.record_key(&a)),
                format!("delete {}", r.record_key(&a)),
                format!("delete {}", r.original_key(&a)),
            ]
        );
    }

    /// The replicator only adds: a record whose document is not local (a
    /// failed S3 delete, or a damaged local archive) stays in S3.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_replicator_never_deletes_from_s3() {
        let bucket = Arc::new(MemoryStore::default());
        let r = replica(&bucket);
        let src = local().await;
        let a = add(&src, b"kept in s3", "a.png", 1, true).await;
        r.replicate(&src.store, &src.blobs).await.expect("replicate");

        let other = local().await;
        add(&other, b"another", "b.png", 2, true).await;
        let report = r.replicate(&other.store, &other.blobs).await.expect("replicate");
        assert_eq!(report.uploaded, 1);
        assert_eq!(bucket.deletes.load(Ordering::SeqCst), 0);
        assert!(bucket.keys().contains(&r.record_key(&a)));
        assert!(bucket.keys().contains(&r.original_key(&a)));
    }

    /// A failed copy is reported in the backlog and copied by the next pass.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_copy_that_failed_is_made_by_the_next_pass() {
        let bucket = Arc::new(MemoryStore::default());
        let r = replica(&bucket);
        let src = local().await;
        let a = add(&src, b"late copy", "a.png", 1, true).await;
        bucket.down.store(true, Ordering::SeqCst);
        assert!(r.upload_document(&src.store, &src.blobs, &a).await.is_err());
        bucket.down.store(false, Ordering::SeqCst);
        assert_eq!(r.backlog(&src.store).await.expect("backlog"), 1);
        r.replicate(&src.store, &src.blobs).await.expect("replicate");
        assert_eq!(r.backlog(&src.store).await.expect("backlog"), 0);
    }
}
