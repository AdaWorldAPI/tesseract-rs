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

/// The per-document lock a delete holds (`AppState::hash_lock`). A copy takes
/// the same lock, so it can never write a document's S3 objects after a
/// delete has removed them.
pub type DocLock = Arc<tokio::sync::Mutex<()>>;

/// Schema tag of a [`DocumentRecord`].
pub const RECORD_SCHEMA: &str = "paperless-rs/document-record/v1";
/// Schema tag of a [`SnapshotFile`].
pub const SNAPSHOT_SCHEMA: &str = "paperless-rs/metadata-snapshot/v1";
/// File name of the restore marker in the original-file directory.
pub const RESTORE_MARKER: &str = ".restore-in-progress";

/// The metadata snapshot a restore imports, kept beside the marker until the
/// restore finishes. A restore that resumes after a crash imports this same
/// snapshot, never a newer one from S3, so it cannot mix two snapshots.
pub const RESTORE_SNAPSHOT: &str = ".restore-snapshot.json";

/// Written by a restore that had to skip damaged records. While it exists the
/// metadata snapshot in S3 is not replaced: the local metadata lacks the
/// skipped documents' assignments and reviews (start-up reconciliation
/// removes rows whose document is missing), and replacing the snapshot would
/// lose them from S3 as well. It lists the skipped hashes. Every start retries
/// the restore of the documents it lists ([`retry_held`]); once their records
/// are repaired in S3, they come back with their metadata and the hold is
/// removed. Remove it by hand only to give those documents up.
pub const SNAPSHOT_HOLD: &str = ".snapshot-hold";

/// Marker contents: which kind of restore is running.
const MARKER_FIRST: &[u8] = b"restore in progress\n";
const MARKER_RETRY: &[u8] = b"retry of held records in progress\n";

/// Whether a degraded restore left a [`SNAPSHOT_HOLD`] in `blob_dir`.
#[must_use]
pub fn has_snapshot_hold(blob_dir: &Path) -> bool {
    blob_dir.join(SNAPSHOT_HOLD).exists()
}

/// Retry the documents a degraded restore skipped (see [`SNAPSHOT_HOLD`]).
/// It is a restore over a non-empty archive: only documents missing locally
/// are restored, and only *their* assignments and reviews are imported from
/// the held snapshot, so metadata changed locally since then is untouched.
/// The hold is rewritten with whatever is still damaged, or removed.
///
/// # Errors
/// As [`restore`].
pub async fn retry_held(
    replica: &Replica,
    store: &LanceStore,
    blobs: &Arc<BlobStore>,
) -> Result<RestoreReport, ReplicaError> {
    restore(replica, store, blobs).await
}

/// Directory in `BLOB_DIR` holding one empty file per pending S3 delete.
pub const DELETE_INTENTS: &str = ".s3-deletes";

/// How many times a copy starts over when the document changed under it.
const COPY_ROUNDS: usize = 3;

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
    /// Each document's customer. Absent in a snapshot written before
    /// customers existed, which then restores with none.
    #[serde(default)]
    customers: Vec<CustomerJson>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CustomerJson {
    content_sha256_hex: String,
    customer: String,
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
            customers: s
                .customers
                .iter()
                .map(|(h, c)| CustomerJson {
                    content_sha256_hex: h.clone(),
                    customer: c.clone(),
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
        let kind =
            |k: &str| MetaKind::parse(k).ok_or_else(|| ReplicaError::Invalid(format!("kind {k}")));
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
            customers: self
                .customers
                .iter()
                .map(|c| (c.content_sha256_hex.clone(), c.customer.clone()))
                .collect(),
        })
    }
}

/// What [`Replica::replicate`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReplicateReport {
    /// Documents whose record was missing and has now been written.
    pub uploaded: usize,
    /// Documents (copies or pending deletes) that failed this time.
    pub failed: usize,
    /// Pending S3 deletes carried out.
    pub deleted: usize,
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
    snapshot: Arc<tokio::sync::Mutex<()>>,
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
            snapshot: Arc::new(tokio::sync::Mutex::new(())),
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
        self.listed_hashes(self.records_prefix(), ".json").await
    }

    /// Hashes that have an original in S3.
    ///
    /// # Errors
    /// [`ReplicaError::Object`] when the listing fails.
    pub async fn original_hashes(&self) -> Result<HashSet<String>, ReplicaError> {
        self.listed_hashes(self.key("originals/"), "").await
    }

    async fn listed_hashes(
        &self,
        prefix: String,
        suffix: &'static str,
    ) -> Result<HashSet<String>, ReplicaError> {
        let store = self.store.clone();
        let p = prefix.clone();
        let keys = blocking(move || store.list(&p)).await?;
        Ok(keys
            .iter()
            .filter_map(|k| k.strip_prefix(prefix.as_str()))
            .filter_map(|k| k.strip_suffix(suffix))
            .filter(|h| ContentSha256::from_hex(h).is_some())
            .map(str::to_string)
            .collect())
    }

    /// Local documents whose S3 copy is missing or incomplete: no record, or
    /// a local original that S3 does not hold. The second case is a document
    /// stored before originals were kept, whose original was attached later by
    /// a re-upload; its record then still says `has_original: false`, and
    /// copying it again uploads the original and rewrites the record.
    async fn pending(
        &self,
        store: &LanceStore,
        blobs: &Arc<BlobStore>,
    ) -> Result<Vec<String>, ReplicaError> {
        let recorded = self.recorded_hashes().await?;
        let originals = self.original_hashes().await?;
        let local_hashes = store.hashes().await.map_err(local)?;
        let b = blobs.clone();
        tokio::task::spawn_blocking(move || {
            local_hashes
                .into_iter()
                .filter(|h| {
                    if !recorded.contains(h) {
                        return true;
                    }
                    !originals.contains(h)
                        && ContentSha256::from_hex(h).is_some_and(|k| b.contains(&k))
                })
                .collect()
        })
        .await
        .map_err(|e| ReplicaError::Local(format!("task: {e}")))
    }

    /// Copy one local document: its original first (when there is one), then
    /// its record. `lock` is the lock a delete of the same document holds. It
    /// is taken only for the two local checks, never across S3 I/O, so a slow
    /// S3 never holds up an upload or delete:
    ///
    /// 1. under the lock, read the row; a document already deleted is not
    ///    copied (`Ok(false)`);
    /// 2. without the lock, upload the original and the record;
    /// 3. under the lock, check the row again. If a delete removed it while
    ///    the copy ran, the copy removes what it just wrote (record first) and
    ///    returns `Ok(false)`, so a deleted document never stays in S3.
    ///
    /// If that compensating delete fails, the record stays in S3 and a restore
    /// would bring the document back; it is logged with the hash.
    ///
    /// # Errors
    /// [`ReplicaError`] when the document cannot be read locally or written to S3.
    pub async fn upload_document(
        &self,
        store: &LanceStore,
        blobs: &Arc<BlobStore>,
        hash: &str,
        lock: DocLock,
    ) -> Result<bool, ReplicaError> {
        let key = ContentSha256::from_hex(hash)
            .ok_or_else(|| ReplicaError::Invalid(format!("hash {hash}")))?;
        // A few rounds: if the document was deleted and uploaded again while a
        // round's PUTs ran, its row changed and the record just written is the
        // old one, so the next round copies the new row over it.
        for _ in 0..COPY_ROUNDS {
            let row = {
                let _held = lock.clone().lock_owned().await;
                match store.get(hash).await.map_err(local)? {
                    Some(row) => row,
                    None => return Ok(false),
                }
            };
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
            let copied_at = row.ingested_at_unix_ms;
            let copied_name = row.filename.clone();
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
            blocking(move || s.put(&k, &body, "application/json")).await?;

            let now = {
                let _held = lock.clone().lock_owned().await;
                store.get(hash).await.map_err(local)?
            };
            match now {
                Some(r) if r.ingested_at_unix_ms == copied_at && r.filename == copied_name => {
                    return Ok(true);
                }
                // Deleted and uploaded again meanwhile: copy the new row.
                Some(_) => continue,
                None => {
                    if let Err(e) = self.delete_document(hash).await {
                        eprintln!(
                            "replica: {hash} was deleted while being copied and removing the \
                             copy failed: {e}; the S3 copy still holds it"
                        );
                        return Err(e);
                    }
                    return Ok(false);
                }
            }
        }
        Err(ReplicaError::Invalid(format!(
            "{hash} changed during {COPY_ROUNDS} copy rounds; the replicator will retry"
        )))
    }

    /// Record that `hash` was deleted locally and must be removed from S3.
    /// The intent is a file under [`DELETE_INTENTS`] in `blobs`, so it survives
    /// a failed S3 delete and a restart; the replicator retries it on every
    /// pass until it succeeds. Unlike a missing copy, a missing delete could
    /// not otherwise be found again: the replicator never deletes on its own.
    ///
    /// # Errors
    /// [`ReplicaError::Local`] when the intent cannot be written.
    pub fn record_delete_intent(blobs: &BlobStore, hash: &str) -> Result<(), ReplicaError> {
        if ContentSha256::from_hex(hash).is_none() {
            return Err(ReplicaError::Invalid(format!("hash {hash}")));
        }
        let dir = blobs.root().join(DELETE_INTENTS);
        std::fs::create_dir_all(&dir).map_err(local)?;
        std::fs::write(dir.join(hash), b"").map_err(local)
    }

    /// The hashes with a pending S3 delete.
    fn delete_intents(blobs: &BlobStore) -> Vec<String> {
        let Ok(entries) = std::fs::read_dir(blobs.root().join(DELETE_INTENTS)) else {
            return Vec::new();
        };
        let mut out: Vec<String> = entries
            .filter_map(Result::ok)
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|h| ContentSha256::from_hex(h).is_some())
            .collect();
        out.sort();
        out
    }

    /// Carry out one pending S3 delete. If the document is local again (it
    /// was uploaded again after the delete), the intent is dropped and nothing
    /// is deleted: the copy of the new upload owns those keys. Otherwise the
    /// S3 objects are removed (record first) and the intent with them.
    /// `Ok(true)` when S3 objects were removed.
    ///
    /// # Errors
    /// [`ReplicaError`] when the S3 delete fails; the intent then stays.
    pub async fn apply_delete_intent(
        &self,
        store: &LanceStore,
        blobs: &Arc<BlobStore>,
        hash: &str,
        lock: DocLock,
    ) -> Result<bool, ReplicaError> {
        let intent = blobs.root().join(DELETE_INTENTS).join(hash);
        let back = {
            let _held = lock.lock_owned().await;
            store.get(hash).await.map_err(local)?.is_some()
        };
        if back {
            remove_if_present(&intent)?;
            return Ok(false);
        }
        self.delete_document(hash).await?;
        remove_if_present(&intent)?;
        Ok(true)
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

    /// Upload the current metadata as the snapshot, unless a degraded restore
    /// left a [`SNAPSHOT_HOLD`] in `blobs`.
    ///
    /// # Errors
    /// [`ReplicaError`] when the metadata cannot be read or written, or
    /// [`ReplicaError::Invalid`] while the hold is in place.
    pub async fn upload_snapshot(
        &self,
        store: &LanceStore,
        blobs: &Arc<BlobStore>,
    ) -> Result<(), ReplicaError> {
        if blobs.root().join(SNAPSHOT_HOLD).exists() {
            return Err(ReplicaError::Invalid(format!(
                "metadata snapshot not replaced: a restore skipped damaged records \
                 (listed in {SNAPSHOT_HOLD}); remove that file once they are dealt with"
            )));
        }
        let serial = self.snapshot.clone().lock_owned().await;
        let snap = store.meta().export().await.map_err(local)?;
        let body =
            serde_json::to_vec(&SnapshotFile::from_snapshot(&snap, now_ms())).map_err(local)?;
        let s = self.store.clone();
        let k = self.snapshot_key();
        // Released only when the PUT has returned, even if this future is
        // dropped: an older snapshot can never land after a newer one.
        blocking(move || {
            let _serial = serial;
            s.put(&k, &body, "application/json")
        })
        .await
    }

    /// Copy every local document whose S3 copy is missing or lacks its
    /// original, then upload the snapshot. Never deletes anything in S3.
    ///
    /// # Errors
    /// [`ReplicaError`] when the listing, the local read or the snapshot fails.
    /// A single document that fails to copy is counted, not returned.
    pub async fn replicate(
        &self,
        store: &LanceStore,
        blobs: &Arc<BlobStore>,
        lock_for: &(dyn Fn(&ContentSha256) -> DocLock + Sync),
    ) -> Result<ReplicateReport, ReplicaError> {
        let mut report = ReplicateReport::default();
        for hash in Self::delete_intents(blobs) {
            let Some(key) = ContentSha256::from_hex(&hash) else {
                continue;
            };
            match self
                .apply_delete_intent(store, blobs, &hash, lock_for(&key))
                .await
            {
                Ok(true) => report.deleted += 1,
                Ok(false) => {}
                Err(e) => {
                    eprintln!("replica: deleting {hash} from S3 failed: {e}; will retry");
                    report.failed += 1;
                }
            }
        }
        for hash in self.pending(store, blobs).await? {
            let Some(key) = ContentSha256::from_hex(&hash) else {
                continue;
            };
            match self
                .upload_document(store, blobs, &hash, lock_for(&key))
                .await
            {
                Ok(true) => report.uploaded += 1,
                Ok(false) => {}
                Err(e) => {
                    eprintln!("replica: copying {hash} failed: {e}");
                    report.failed += 1;
                }
            }
        }
        self.upload_snapshot(store, blobs).await?;
        Ok(report)
    }

    /// Local documents whose S3 copy is missing or lacks its original.
    ///
    /// # Errors
    /// [`ReplicaError`] when the listing or the local read fails.
    pub async fn backlog(
        &self,
        store: &LanceStore,
        blobs: &Arc<BlobStore>,
    ) -> Result<usize, ReplicaError> {
        Ok(self.pending(store, blobs).await?.len() + Self::delete_intents(blobs).len())
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
    let pinned = blobs.root().join(RESTORE_SNAPSHOT);
    // A first restore (empty archive) imports the metadata of every local
    // document. A restore over a live archive is the retry of a snapshot
    // hold: it imports only the held documents' metadata, so it never
    // overwrites metadata changed locally since the snapshot. The marker says
    // which kind is running, so an interrupted one resumes as the same kind.
    let fresh = match std::fs::read(&marker) {
        Ok(body) => body.starts_with(MARKER_FIRST),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            store.count().await.map_err(local)? == 0
        }
        Err(e) => return Err(local(e)),
    };
    let held: HashSet<String> = std::fs::read_to_string(blobs.root().join(SNAPSHOT_HOLD))
        .map(|t| t.lines().map(str::to_string).collect())
        .unwrap_or_default();
    // A pin belongs to the restore its marker announces. A pin without a
    // marker is left over from a restore that finished (marker removed, pin
    // not yet), so a new restore drops it and reads the current snapshot.
    if !marker.exists() {
        remove_if_present(&pinned)?;
    }
    std::fs::write(&marker, if fresh { MARKER_FIRST } else { MARKER_RETRY }).map_err(local)?;

    let mut report = RestoreReport::default();
    // Documents whose record was damaged and skipped. They are listed in the
    // snapshot hold (below); their metadata is not imported locally. It stays
    // in the S3 snapshot, which the hold keeps from being replaced, so a later
    // restore after the record is repaired brings it back.
    let mut skipped_hashes: HashSet<String> = HashSet::new();
    let local_hashes: HashSet<String> = store.hashes().await.map_err(local)?.into_iter().collect();
    // An original present in S3 is restored even when its record still says
    // `has_original: false`: the record is rewritten only after the original
    // is uploaded, and a failed rewrite must not cost the original.
    let originals = replica.original_hashes().await?;
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
                skipped_hashes.insert(hash.clone());
                continue;
            }
        };
        if record.schema != RECORD_SCHEMA || record.content_sha256_hex != hash {
            eprintln!("restore: record {hash} has the wrong schema or hash; skipped");
            report.skipped += 1;
            skipped_hashes.insert(hash.clone());
            continue;
        }
        let Some(key) = ContentSha256::from_hex(&hash) else {
            report.skipped += 1;
            skipped_hashes.insert(hash.clone());
            continue;
        };
        let ir = match ogar_doc_ir::from_json(&record.doc_ir_json) {
            Ok(ir) => ir,
            Err(e) => {
                eprintln!("restore: record {hash} has an unreadable DocIr ({e:?}); skipped");
                report.skipped += 1;
                skipped_hashes.insert(hash.clone());
                continue;
            }
        };
        if record.has_original || originals.contains(&hash) {
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
                    eprintln!(
                        "restore: original of {hash} fails its hash check; restored without it"
                    );
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

    let fetched = match std::fs::read(&pinned) {
        Ok(body) => Ok(body),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let s = replica.store.clone();
            let k = replica.snapshot_key();
            let got = blocking(move || s.get(&k)).await;
            if let Ok(body) = &got {
                std::fs::write(&pinned, body).map_err(local)?;
            }
            got
        }
        Err(e) => return Err(local(e)),
    };
    match fetched {
        Ok(body) => {
            let file: SnapshotFile = serde_json::from_slice(&body)
                .map_err(|e| ReplicaError::Invalid(format!("metadata snapshot: {e}")))?;
            let snapshot = file.to_snapshot()?;
            // Only documents that are local get their assignments and reviews.
            // Importing rows for skipped documents too would leave rows that a
            // later restore, from a different snapshot, could not replace.
            let local_now: HashSet<String> =
                store.hashes().await.map_err(local)?.into_iter().collect();
            let live: HashSet<String> = if fresh {
                local_now
            } else {
                local_now.intersection(&held).cloned().collect()
            };
            store.meta().import(&snapshot, &live).await.map_err(local)?;
            report.metadata = true;
        }
        Err(ReplicaError::Object(e)) if e.is_not_found() => {}
        Err(e) => return Err(e),
    }

    let hold = blobs.root().join(SNAPSHOT_HOLD);
    if skipped_hashes.is_empty() {
        remove_if_present(&hold)?;
    } else {
        let mut list: Vec<&String> = skipped_hashes.iter().collect();
        list.sort();
        let body: String = list.iter().map(|h| format!("{h}\n")).collect();
        std::fs::write(&hold, body).map_err(local)?;
    }
    // Marker first, then the pin: while the marker exists the pin stays
    // authoritative, so an interruption between the two can never make a
    // resumed restore read a newer snapshot.
    std::fs::remove_file(&marker).map_err(local)?;
    remove_if_present(&pinned)?;
    Ok(report)
}

fn remove_if_present(path: &Path) -> Result<(), ReplicaError> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(local(e)),
    }
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

    /// A fresh lock per call: these tests run no delete concurrently.
    fn no_lock(_: &ContentSha256) -> DocLock {
        Arc::new(tokio::sync::Mutex::new(()))
    }

    fn lock() -> DocLock {
        Arc::new(tokio::sync::Mutex::new(()))
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

        let report = r
            .replicate(&src.store, &src.blobs, &no_lock)
            .await
            .expect("replicate");
        assert_eq!(
            report,
            ReplicateReport {
                uploaded: 2,
                failed: 0,
                deleted: 0
            }
        );
        assert_eq!(r.backlog(&src.store, &src.blobs).await.expect("backlog"), 0);
        assert!(bucket.keys().contains(&r.original_key(&a)));
        assert!(!bucket.keys().contains(&r.original_key(&b)));

        let dst = local().await;
        assert!(needs_restore(&dst.store, dst.blobs.root())
            .await
            .expect("check"));
        let report = restore(&r, &dst.store, &dst.blobs).await.expect("restore");
        assert_eq!(report.restored, 2);
        assert_eq!(report.skipped, 0);
        assert_eq!(report.without_original, 0);
        assert!(report.metadata);
        assert!(!dst.blobs.root().join(RESTORE_MARKER).exists());
        assert!(!needs_restore(&dst.store, dst.blobs.root())
            .await
            .expect("check"));

        for (hash, name, at) in [(&a, "a.png", 100), (&b, "b.png", 200)] {
            let row = dst
                .store
                .get(hash)
                .await
                .expect("get")
                .expect("restored row");
            assert_eq!(row.filename.as_deref(), Some(name));
            assert_eq!(row.ingested_at_unix_ms, at);
            assert_eq!(row.mean_confidence, 88);
            assert_eq!(row.spo_json.as_deref(), Some("[]"));
            assert_eq!(row.text().ok().as_deref(), Some(name));
        }
        let key_a = ContentSha256::from_hex(&a).expect("hex");
        assert_eq!(
            dst.blobs.get(&key_a).expect("get"),
            Some(b"original A".to_vec())
        );
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
        r.replicate(&src.store, &src.blobs, &no_lock)
            .await
            .expect("replicate");

        let dst = local().await;
        // As if the first attempt restored A and stopped.
        add(&dst, b"doc A", "a.png", 1, true).await;
        std::fs::write(dst.blobs.root().join(RESTORE_MARKER), b"x").expect("marker");
        assert!(needs_restore(&dst.store, dst.blobs.root())
            .await
            .expect("check"));

        let report = restore(&r, &dst.store, &dst.blobs).await.expect("resume");
        assert_eq!(report.already_local, 1);
        assert_eq!(report.restored, 1);
        assert_eq!(dst.store.count().await.expect("count"), 2);
        assert!(dst.store.get(&a).await.expect("get").is_some());
        assert!(!needs_restore(&dst.store, dst.blobs.root())
            .await
            .expect("check"));
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
        src.store.meta().mark_reviewed(&b, 3).await.expect("review");
        r.replicate(&src.store, &src.blobs, &no_lock)
            .await
            .expect("replicate");
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
        // The snapshot in S3 is held, so it keeps B's review even after the
        // local metadata loses it (start-up reconciliation drops rows of
        // missing documents) and the next replication pass runs.
        let before = bucket.get(&r.snapshot_key()).expect("snapshot");
        assert!(dst.blobs.root().join(SNAPSHOT_HOLD).exists());
        let live: HashSet<String> = dst
            .store
            .hashes()
            .await
            .expect("hashes")
            .into_iter()
            .collect();
        dst.store.meta().reconcile(&live).await.expect("reconcile");
        assert!(r.replicate(&dst.store, &dst.blobs, &no_lock).await.is_err());
        let after = bucket.get(&r.snapshot_key()).expect("snapshot");
        assert_eq!(after, before);
        let held = SnapshotFile::to_snapshot(&serde_json::from_slice(&after).expect("json"))
            .expect("snapshot");
        assert!(held.reviews.iter().any(|v| v.content_sha256_hex == b));
        // A complete restore leaves no hold.
        let clean = local().await;
        bucket
            .objects
            .lock()
            .expect("lock")
            .remove(&r.record_key(&b));
        restore(&r, &clean.store, &clean.blobs)
            .await
            .expect("restore");
        assert!(!clean.blobs.root().join(SNAPSHOT_HOLD).exists());
    }

    /// A restore in which every record is damaged leaves the archive empty
    /// and imports no assignment or review rows for the skipped documents, so
    /// a later restore from a newer snapshot cannot merge with stale rows.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_restore_that_skips_every_record_imports_no_rows_for_them() {
        let bucket = Arc::new(MemoryStore::default());
        let r = replica(&bucket);
        let src = local().await;
        let a = add(&src, b"only doc", "a.png", 1, true).await;
        let meta = src.store.meta();
        let tag = meta
            .create_definition(MetaKind::Tag, "Alt", 1)
            .await
            .expect("tag");
        meta.assign(&a, MetaKind::Tag, tag.definition_id, Source::Manual, 2)
            .await
            .expect("assign");
        meta.mark_reviewed(&a, 3).await.expect("review");
        r.replicate(&src.store, &src.blobs, &no_lock)
            .await
            .expect("replicate");
        bucket
            .objects
            .lock()
            .expect("lock")
            .insert(r.record_key(&a), b"not json".to_vec());

        let dst = local().await;
        let report = restore(&r, &dst.store, &dst.blobs).await.expect("restore");
        assert_eq!((report.restored, report.skipped), (0, 1));
        let snap = dst.store.meta().export().await.expect("export");
        assert!(snap.assignments.is_empty(), "{:?}", snap.assignments);
        assert!(snap.reviews.is_empty(), "{:?}", snap.reviews);
        // The definitions come back with their ids; the hold keeps S3 intact.
        assert!(snap
            .definitions
            .iter()
            .any(|d| d.definition_id == tag.definition_id));
        assert!(dst.blobs.root().join(SNAPSHOT_HOLD).exists());
    }

    /// A copy whose document was deleted and uploaded again (same bytes, new
    /// name and time) while its PUTs ran does not leave the old record: it
    /// copies the new row over it.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_delayed_copy_does_not_leave_a_stale_record_after_re_upload() {
        let bucket = Arc::new(MemoryStore::default());
        let r = replica(&bucket);
        let src = local().await;
        let bytes: &[u8] = b"same bytes";
        let a = add(&src, bytes, "old.png", 1, true).await;
        bucket.put_delay_ms.store(200, Ordering::SeqCst);
        let l = lock();
        let copy = r.upload_document(&src.store, &src.blobs, &a, l.clone());
        let reupload = async {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let held = l.clone().lock_owned().await;
            src.store.delete(&a).await.expect("delete");
            add(&src, bytes, "new.png", 2, true).await;
            drop(held);
        };
        let (copied, ()) = tokio::join!(copy, reupload);
        assert!(copied.expect("copy"));
        let rec: DocumentRecord =
            serde_json::from_slice(&bucket.get(&r.record_key(&a)).expect("record")).expect("json");
        assert_eq!(rec.filename.as_deref(), Some("new.png"));
        assert_eq!(rec.ingested_at_unix_ms, 2);
    }

    /// A pending S3 delete survives a failure (and so a restart: it is a file)
    /// and is carried out by the next replication pass.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_failed_s3_delete_is_retried_by_the_replicator() {
        let bucket = Arc::new(MemoryStore::default());
        let r = replica(&bucket);
        let src = local().await;
        let a = add(&src, b"to delete", "a.png", 1, true).await;
        r.replicate(&src.store, &src.blobs, &no_lock)
            .await
            .expect("replicate");
        src.store.delete(&a).await.expect("local delete");
        Replica::record_delete_intent(&src.blobs, &a).expect("intent");
        bucket.deletes_fail.store(true, Ordering::SeqCst);
        assert!(r
            .apply_delete_intent(&src.store, &src.blobs, &a, lock())
            .await
            .is_err());
        let intent = src.blobs.root().join(DELETE_INTENTS).join(&a);
        assert!(intent.exists(), "a failed delete keeps its intent");
        assert_eq!(r.backlog(&src.store, &src.blobs).await.expect("backlog"), 1);

        bucket.deletes_fail.store(false, Ordering::SeqCst);
        let report = r
            .replicate(&src.store, &src.blobs, &no_lock)
            .await
            .expect("replicate");
        assert_eq!(report.deleted, 1);
        assert!(!bucket.keys().contains(&r.record_key(&a)));
        assert!(!bucket.keys().contains(&r.original_key(&a)));
        assert!(!intent.exists());
    }

    /// A pending delete for a document that was uploaded again is dropped
    /// without touching S3.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_pending_delete_of_a_re_uploaded_document_is_dropped() {
        let bucket = Arc::new(MemoryStore::default());
        let r = replica(&bucket);
        let src = local().await;
        let a = add(&src, b"back again", "a.png", 1, true).await;
        r.replicate(&src.store, &src.blobs, &no_lock)
            .await
            .expect("replicate");
        Replica::record_delete_intent(&src.blobs, &a).expect("intent");
        let report = r
            .replicate(&src.store, &src.blobs, &no_lock)
            .await
            .expect("replicate");
        assert_eq!(report.deleted, 0);
        assert!(bucket.keys().contains(&r.record_key(&a)));
        assert!(!src.blobs.root().join(DELETE_INTENTS).join(&a).exists());
    }

    /// The documents a degraded restore skipped come back once their records
    /// are repaired: the next start retries them, imports only their
    /// assignments and reviews (leaving metadata changed locally since
    /// alone), removes the hold, and the next snapshot carries everything.
    #[tokio::test(flavor = "multi_thread")]
    async fn held_documents_come_back_with_their_metadata_after_repair() {
        let bucket = Arc::new(MemoryStore::default());
        let r = replica(&bucket);
        let src = local().await;
        let a = add(&src, b"fine", "a.png", 1, true).await;
        let b = add(&src, b"damaged then repaired", "b.png", 2, true).await;
        let meta = src.store.meta();
        let tag = meta
            .create_definition(MetaKind::Tag, "Wichtig", 1)
            .await
            .expect("tag");
        meta.assign(&b, MetaKind::Tag, tag.definition_id, Source::Manual, 2)
            .await
            .expect("assign");
        meta.mark_reviewed(&b, 3).await.expect("review b");
        r.replicate(&src.store, &src.blobs, &no_lock)
            .await
            .expect("replicate");
        let good_record = bucket.get(&r.record_key(&b)).expect("record");
        bucket
            .objects
            .lock()
            .expect("lock")
            .insert(r.record_key(&b), b"not json".to_vec());

        // Degraded restore: a only, hold written.
        let dst = local().await;
        restore(&r, &dst.store, &dst.blobs).await.expect("restore");
        assert!(has_snapshot_hold(dst.blobs.root()));
        let live: HashSet<String> = dst
            .store
            .hashes()
            .await
            .expect("hashes")
            .into_iter()
            .collect();
        dst.store.meta().reconcile(&live).await.expect("reconcile");
        // A local change made while held must survive the retry.
        dst.store
            .meta()
            .mark_reviewed(&a, 9)
            .await
            .expect("review a locally");

        // Repair, then the retry a start performs.
        bucket
            .objects
            .lock()
            .expect("lock")
            .insert(r.record_key(&b), good_record);
        let report = retry_held(&r, &dst.store, &dst.blobs).await.expect("retry");
        assert_eq!((report.restored, report.skipped), (1, 0));
        assert!(!has_snapshot_hold(dst.blobs.root()));
        let snap = dst.store.meta().export().await.expect("export");
        assert!(snap.reviews.iter().any(|v| v.content_sha256_hex == b));
        assert!(snap
            .assignments
            .iter()
            .any(|x| x.content_sha256_hex == b && x.definition_id == tag.definition_id));
        assert!(snap
            .reviews
            .iter()
            .any(|v| v.content_sha256_hex == a && v.reviewed_at_unix_ms == 9));

        // The hold is gone, so the next pass uploads a snapshot with all of it.
        r.replicate(&dst.store, &dst.blobs, &no_lock)
            .await
            .expect("replicate");
        let up = SnapshotFile::to_snapshot(
            &serde_json::from_slice(&bucket.get(&r.snapshot_key()).expect("snapshot"))
                .expect("json"),
        )
        .expect("snapshot");
        assert!(up.reviews.iter().any(|v| v.content_sha256_hex == b));
        assert!(up.assignments.iter().any(|x| x.content_sha256_hex == b));
    }

    /// A slow S3 does not hold the document's lock: an upload or delete of the
    /// same document proceeds while the copy's PUT is running. A delete that
    /// lands meanwhile wins: the copy removes what it wrote.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_slow_copy_does_not_hold_the_lock_and_a_delete_meanwhile_wins() {
        let bucket = Arc::new(MemoryStore::default());
        let r = replica(&bucket);
        let src = local().await;
        let a = add(&src, b"slow copy", "a.png", 1, true).await;
        bucket.put_delay_ms.store(300, Ordering::SeqCst);
        let l = lock();
        let copy = r.upload_document(&src.store, &src.blobs, &a, l.clone());
        let delete = async {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            let held = l.clone().try_lock_owned().expect("free during the S3 PUT");
            src.store.delete(&a).await.expect("delete");
            drop(held);
        };
        let (copied, ()) = tokio::join!(copy, delete);
        assert!(!copied.expect("copy"));
        assert!(!bucket.keys().contains(&r.record_key(&a)));
        assert!(!bucket.keys().contains(&r.original_key(&a)));
    }

    /// The original goes up before its record, and a delete removes the
    /// record before the original.
    #[tokio::test(flavor = "multi_thread")]
    async fn originals_precede_records_and_records_go_first_on_delete() {
        let bucket = Arc::new(MemoryStore::default());
        let r = replica(&bucket);
        let src = local().await;
        let a = add(&src, b"ordered", "a.png", 1, true).await;
        r.upload_document(&src.store, &src.blobs, &a, lock())
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
        r.replicate(&src.store, &src.blobs, &no_lock)
            .await
            .expect("replicate");

        let other = local().await;
        add(&other, b"another", "b.png", 2, true).await;
        let report = r
            .replicate(&other.store, &other.blobs, &no_lock)
            .await
            .expect("replicate");
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
        assert!(r
            .upload_document(&src.store, &src.blobs, &a, lock())
            .await
            .is_err());
        bucket.down.store(false, Ordering::SeqCst);
        assert_eq!(r.backlog(&src.store, &src.blobs).await.expect("backlog"), 1);
        r.replicate(&src.store, &src.blobs, &no_lock)
            .await
            .expect("replicate");
        assert_eq!(r.backlog(&src.store, &src.blobs).await.expect("backlog"), 0);
    }

    /// A restore that resumes imports the snapshot it pinned on its first
    /// attempt, not a newer one from S3, and removes the pin when done.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_resumed_restore_imports_the_pinned_snapshot() {
        let bucket = Arc::new(MemoryStore::default());
        let r = replica(&bucket);
        let src = local().await;
        let a = add(&src, b"pinned", "a.png", 1, true).await;
        src.store.meta().mark_reviewed(&a, 2).await.expect("review");
        r.replicate(&src.store, &src.blobs, &no_lock)
            .await
            .expect("replicate");
        let first = bucket.get(&r.snapshot_key()).expect("snapshot");

        src.store
            .meta()
            .mark_unreviewed(&a)
            .await
            .expect("unreview");
        r.upload_snapshot(&src.store, &src.blobs)
            .await
            .expect("snapshot");
        assert_ne!(bucket.get(&r.snapshot_key()).expect("snapshot"), first);

        let dst = local().await;
        std::fs::write(dst.blobs.root().join(RESTORE_MARKER), b"x").expect("marker");
        std::fs::write(dst.blobs.root().join(RESTORE_SNAPSHOT), &first).expect("pin");
        restore(&r, &dst.store, &dst.blobs).await.expect("restore");
        let snap = dst.store.meta().export().await.expect("export");
        assert!(snap.reviews.iter().any(|v| v.content_sha256_hex == a));
        assert!(!dst.blobs.root().join(RESTORE_SNAPSHOT).exists());
        assert!(!dst.blobs.root().join(RESTORE_MARKER).exists());
    }

    /// A pin left behind by a finished restore (marker already removed) is
    /// not reused: a new restore drops it and imports the current snapshot.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_leftover_pin_without_a_marker_is_not_reused() {
        let bucket = Arc::new(MemoryStore::default());
        let r = replica(&bucket);
        let src = local().await;
        let a = add(&src, b"stale pin", "a.png", 1, true).await;
        src.store.meta().mark_reviewed(&a, 2).await.expect("review");
        r.replicate(&src.store, &src.blobs, &no_lock)
            .await
            .expect("replicate");
        let old = bucket.get(&r.snapshot_key()).expect("snapshot");
        src.store
            .meta()
            .mark_unreviewed(&a)
            .await
            .expect("unreview");
        r.upload_snapshot(&src.store, &src.blobs)
            .await
            .expect("snapshot");

        let dst = local().await;
        std::fs::write(dst.blobs.root().join(RESTORE_SNAPSHOT), &old).expect("stale pin");
        restore(&r, &dst.store, &dst.blobs).await.expect("restore");
        let snap = dst.store.meta().export().await.expect("export");
        assert!(!snap.reviews.iter().any(|v| v.content_sha256_hex == a));
        assert!(!dst.blobs.root().join(RESTORE_SNAPSHOT).exists());
    }

    /// A rewrite of the record that fails after the original reached S3
    /// leaves `has_original: false` beside an original in S3. Restore still
    /// brings the original back.
    #[tokio::test(flavor = "multi_thread")]
    async fn restore_uses_an_original_whose_record_was_not_rewritten() {
        let bucket = Arc::new(MemoryStore::default());
        let r = replica(&bucket);
        let src = local().await;
        let bytes: &[u8] = b"original without rewritten record";
        let a = add(&src, bytes, "a.png", 1, false).await;
        r.replicate(&src.store, &src.blobs, &no_lock)
            .await
            .expect("replicate");
        let key = ContentSha256::of(bytes);
        bucket
            .put(&r.original_key(&a), bytes, "application/octet-stream")
            .expect("put");
        src.blobs.put(&key, bytes).expect("backfill");
        assert_eq!(r.backlog(&src.store, &src.blobs).await.expect("backlog"), 0);

        let dst = local().await;
        let report = restore(&r, &dst.store, &dst.blobs).await.expect("restore");
        assert_eq!(report.without_original, 0);
        assert_eq!(dst.blobs.get(&key).expect("get").as_deref(), Some(bytes));
    }

    /// A copy waits for a delete holding the document's lock, and then finds
    /// the document gone and writes nothing.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_copy_after_a_delete_writes_nothing() {
        let bucket = Arc::new(MemoryStore::default());
        let r = replica(&bucket);
        let src = local().await;
        let a = add(&src, b"deleted while copying", "a.png", 1, true).await;
        let l = lock();
        let held = l.clone().lock_owned().await;
        let copy = r.upload_document(&src.store, &src.blobs, &a, l.clone());
        let delete = async {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            src.store.delete(&a).await.expect("delete");
            drop(held);
        };
        let (copied, ()) = tokio::join!(copy, delete);
        // Had the copy not waited for the lock, it would have read the row
        // before the delete and written it to S3.
        assert!(!copied.expect("copy"));
        assert!(bucket.keys().is_empty());
    }

    /// A legacy document already copied without an original gets its original
    /// later (re-upload backfill). If the immediate copy of the backfill
    /// fails, the replicator still uploads the original and rewrites the
    /// record, and a restore then brings the original back.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_backfilled_original_is_copied_and_the_record_rewritten() {
        let bucket = Arc::new(MemoryStore::default());
        let r = replica(&bucket);
        let src = local().await;
        let bytes: &[u8] = b"legacy then backfilled";
        let a = add(&src, bytes, "a.png", 1, false).await;
        r.replicate(&src.store, &src.blobs, &no_lock)
            .await
            .expect("replicate");
        assert!(!bucket.keys().contains(&r.original_key(&a)));
        assert_eq!(r.backlog(&src.store, &src.blobs).await.expect("backlog"), 0);

        src.blobs
            .put(&ContentSha256::of(bytes), bytes)
            .expect("backfill");
        assert_eq!(r.backlog(&src.store, &src.blobs).await.expect("backlog"), 1);
        let report = r
            .replicate(&src.store, &src.blobs, &no_lock)
            .await
            .expect("replicate");
        assert_eq!(report.uploaded, 1);
        assert_eq!(r.backlog(&src.store, &src.blobs).await.expect("backlog"), 0);
        let log = bucket.log.lock().expect("log").clone();
        let orig = log
            .iter()
            .rposition(|l| l == &format!("put {}", r.original_key(&a)));
        let rec = log
            .iter()
            .rposition(|l| l == &format!("put {}", r.record_key(&a)));
        assert!(orig.expect("original put") < rec.expect("record put"));

        let dst = local().await;
        restore(&r, &dst.store, &dst.blobs).await.expect("restore");
        let key = ContentSha256::of(bytes);
        assert_eq!(dst.blobs.get(&key).expect("get").as_deref(), Some(bytes));
    }
}
