//! The original-file store: each uploaded file's exact bytes, kept on disk under
//! the SHA-256 the S-2 gate already computes.
//!
//! # Order (S-5): bytes before address
//!
//! A caller writes the blob first and the document row second. A crash between
//! the two leaves a blob no row points at, which is collectable garbage. The
//! other order would leave a row whose original does not exist, which nothing
//! notices until someone asks for the file.
//!
//! # Durability
//!
//! [`BlobStore::put`] writes to a temporary file in the target directory, syncs
//! it, renames it into place and syncs the directory. A reader therefore sees
//! either no blob or the whole blob, never a partial one.
//!
//! # Integrity
//!
//! The key is the hash of the content, so every read re-hashes the bytes and
//! refuses a mismatch ([`BlobError::Tampered`]) instead of serving them. A put
//! whose bytes do not hash to the key is refused the same way, so a caller can
//! never file bytes under someone else's address.
//!
//! # Layout
//!
//! `<root>/<first two hex digits>/<64 hex digits>`. The fan-out keeps any one
//! directory small. Paths are built only from a [`ContentSha256`], never from
//! caller text, so a request cannot name a path outside the root.
//!
//! Synchronous `std::fs` on purpose: callers on an async runtime wrap calls in
//! `spawn_blocking`, as they already do for the search index.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::kv::ContentSha256;

/// Why a blob operation failed.
#[derive(Debug)]
pub enum BlobError {
    /// The filesystem refused (permissions, full disk, a file where a
    /// directory should be).
    Io(io::Error),
    /// Stored bytes no longer hash to their key. They are not served.
    Tampered {
        /// The key that was asked for.
        expected: ContentSha256,
        /// What the stored bytes hash to now.
        actual: ContentSha256,
    },
    /// A put whose bytes do not hash to the key it was given.
    KeyMismatch {
        /// The key the caller named.
        expected: ContentSha256,
        /// What the supplied bytes hash to.
        actual: ContentSha256,
    },
}

impl std::fmt::Display for BlobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "blob store I/O: {e}"),
            Self::Tampered { expected, actual } => write!(
                f,
                "stored original for {expected:?} hashes to {actual:?}; refusing to serve it"
            ),
            Self::KeyMismatch { expected, actual } => write!(
                f,
                "bytes hash to {actual:?}, not to the key {expected:?} they were filed under"
            ),
        }
    }
}

impl std::error::Error for BlobError {}

impl From<io::Error> for BlobError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// What [`BlobStore::put`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PutOutcome {
    /// The bytes were written.
    Written,
    /// An intact blob for this key was already present; nothing was written.
    AlreadyPresent,
}

/// A content-addressed store of original files under one directory.
#[derive(Debug)]
pub struct BlobStore {
    root: PathBuf,
}

/// Distinguishes temporary files written concurrently by one process.
static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

impl BlobStore {
    /// Open the store at `root`, creating the directory if it is missing.
    ///
    /// # Errors
    /// [`BlobError::Io`] when the directory cannot be created.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, BlobError> {
        let root = root.into();
        create_dir_durable(&root)?;
        // Sync every existing ancestor's entry too: a start whose directory
        // sync failed exits, and the next start finds the directories present
        // but never synced.
        let mut dir = Some(root.as_path());
        while let Some(d) = dir.filter(|d| !d.as_os_str().is_empty()) {
            sync_dir(d)?;
            dir = d.parent();
        }
        Ok(Self { root })
    }

    /// The directory this store writes under.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn path_of(&self, key: &ContentSha256) -> PathBuf {
        let hex = format!("{key:?}");
        self.root.join(&hex[..2]).join(hex)
    }

    /// Whether a blob file exists for `key`. Says nothing about its integrity;
    /// [`BlobStore::get`] checks that.
    #[must_use]
    pub fn contains(&self, key: &ContentSha256) -> bool {
        self.path_of(key).is_file()
    }

    /// Store `bytes` under `key`.
    ///
    /// When an intact blob is already present nothing is written. A present
    /// blob that no longer matches its key is replaced, because the supplied
    /// bytes provably are the original.
    ///
    /// # Errors
    /// [`BlobError::KeyMismatch`] when `bytes` do not hash to `key` (nothing is
    /// written); [`BlobError::Io`] when the write fails.
    pub fn put(&self, key: &ContentSha256, bytes: &[u8]) -> Result<PutOutcome, BlobError> {
        let actual = ContentSha256::of(bytes);
        if actual != *key {
            return Err(BlobError::KeyMismatch {
                expected: *key,
                actual,
            });
        }
        match self.get(key) {
            Ok(Some(_)) => {
                // A readable blob is not necessarily a durable one: an earlier
                // put may have renamed it into place and then failed to sync
                // the directory. Sync before reporting it as kept.
                let dir = self.path_of(key);
                let dir = dir
                    .parent()
                    .ok_or_else(|| io::Error::other("blob path has no parent"))?;
                sync_dir(dir)?;
                sync_dir(&self.root)?;
                return Ok(PutOutcome::AlreadyPresent);
            }
            Ok(None) | Err(BlobError::Tampered { .. }) => {}
            Err(e) => return Err(e),
        }

        let target = self.path_of(key);
        let dir = target
            .parent()
            .ok_or_else(|| io::Error::other("blob path has no parent"))?;
        create_dir_durable(dir)?;
        let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let temp = dir.join(format!(
            ".{}.{}.{seq}.tmp",
            target
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("blob"),
            std::process::id()
        ));
        let written = (|| -> io::Result<()> {
            let mut f = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp)?;
            f.write_all(bytes)?;
            f.sync_all()?;
            drop(f);
            fs::rename(&temp, &target)?;
            // The rename is only durable once the directory entry is.
            sync_dir(dir)?;
            // The fan-out directory's own entry lives in the root. Synced on
            // every write, not only when this put created the directory: an
            // earlier put may have created it and then failed to sync.
            sync_dir(&self.root)
        })();
        if let Err(e) = written {
            // Best effort: a leftover temporary file is garbage, not a blob,
            // because readers only ever open the final name.
            let _ = fs::remove_file(&temp);
            return Err(BlobError::Io(e));
        }
        Ok(PutOutcome::Written)
    }

    /// The original bytes for `key`, or `None` when no blob is stored.
    ///
    /// # Errors
    /// [`BlobError::Tampered`] when the stored bytes no longer hash to `key`;
    /// [`BlobError::Io`] when the read fails.
    pub fn get(&self, key: &ContentSha256) -> Result<Option<Vec<u8>>, BlobError> {
        let mut f = match File::open(self.path_of(key)) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(BlobError::Io(e)),
        };
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes)?;
        let actual = ContentSha256::of(&bytes);
        if actual != *key {
            return Err(BlobError::Tampered {
                expected: *key,
                actual,
            });
        }
        Ok(Some(bytes))
    }

    /// Remove the blob for `key`. Removing a blob that is not there succeeds.
    ///
    /// # Errors
    /// [`BlobError::Io`] when the file exists but cannot be removed.
    pub fn delete(&self, key: &ContentSha256) -> Result<(), BlobError> {
        match fs::remove_file(self.path_of(key)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(BlobError::Io(e)),
        }
    }
}

/// Create `dir` and any missing ancestors, syncing each new directory's
/// parent so the new entry survives a power loss. A directory that already
/// exists is left alone.
fn create_dir_durable(dir: &Path) -> io::Result<()> {
    if dir.as_os_str().is_empty() || dir.is_dir() {
        return Ok(());
    }
    let parent = dir.parent().filter(|p| !p.as_os_str().is_empty());
    if let Some(p) = parent {
        create_dir_durable(p)?;
    }
    match fs::create_dir(dir) {
        Ok(()) => {}
        // Created concurrently: someone else made it, and synced its parent.
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists && dir.is_dir() => return Ok(()),
        Err(e) => return Err(e),
    }
    sync_dir(parent.unwrap_or_else(|| Path::new(".")))
}

/// Make a directory's entries durable.
fn sync_dir(dir: &Path) -> io::Result<()> {
    #[cfg(test)]
    {
        tests::DIR_SYNCS.with(|n| n.set(n.get() + 1));
        tests::SYNCED.with(|v| v.borrow_mut().push(dir.to_path_buf()));
        if tests::FAIL_DIR_SYNC.with(|f| f.replace(false)) {
            return Err(io::Error::other("injected directory sync failure"));
        }
    }
    File::open(dir)?.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    thread_local! {
        /// Directory syncs performed on this thread.
        pub(super) static DIR_SYNCS: Cell<usize> = const { Cell::new(0) };
        /// Directories synced on this thread, in order.
        pub(super) static SYNCED: std::cell::RefCell<Vec<PathBuf>> =
            const { std::cell::RefCell::new(Vec::new()) };
        /// Fail the next directory sync on this thread.
        pub(super) static FAIL_DIR_SYNC: Cell<bool> = const { Cell::new(false) };
    }

    /// A put that created the fan-out directory and then failed to sync its
    /// parent leaves the directory present. The retry finds it and must still
    /// sync the root before reporting the blob written.
    #[test]
    fn a_retry_after_a_failed_directory_creation_syncs_the_root() {
        let (_d, s) = store();
        let bytes = b"directory created, entry not synced".to_vec();
        let key = ContentSha256::of(&bytes);
        let hex = format!("{key:?}");
        FAIL_DIR_SYNC.with(|f| f.set(true));
        assert!(matches!(s.put(&key, &bytes), Err(BlobError::Io(_))));
        assert!(
            s.root().join(&hex[..2]).is_dir(),
            "the directory was created"
        );
        assert!(!s.contains(&key));

        SYNCED.with(|v| v.borrow_mut().clear());
        assert_eq!(s.put(&key, &bytes).expect("retry"), PutOutcome::Written);
        let synced = SYNCED.with(|v| v.borrow().clone());
        assert!(
            synced.iter().any(|p| p == s.root()),
            "the retry did not sync the root: {synced:?}"
        );
    }

    #[test]
    fn opening_an_existing_store_syncs_its_directories() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("blobs");
        fs::create_dir(&root).expect("pre-existing root");
        SYNCED.with(|v| v.borrow_mut().clear());
        BlobStore::open(&root).expect("open");
        let synced = SYNCED.with(|v| v.borrow().clone());
        assert!(synced.iter().any(|p| p == &root), "{synced:?}");
        assert!(synced.iter().any(|p| p == dir.path()), "{synced:?}");
    }

    #[test]
    fn a_retry_after_a_failed_directory_sync_syncs_before_succeeding() {
        let (_d, s) = store();
        let bytes = b"renamed but not synced".to_vec();
        let key = ContentSha256::of(&bytes);
        // First put: create the fan-out directory, then fail the sync that
        // follows the rename.
        let hex = format!("{key:?}");
        create_dir_durable(&s.root().join(&hex[..2])).expect("fan-out dir");
        FAIL_DIR_SYNC.with(|f| f.set(true));
        assert!(matches!(s.put(&key, &bytes), Err(BlobError::Io(_))));
        assert!(
            s.contains(&key),
            "the rename happened before the failed sync"
        );

        let before = DIR_SYNCS.with(Cell::get);
        assert_eq!(
            s.put(&key, &bytes).expect("retry"),
            PutOutcome::AlreadyPresent
        );
        assert!(
            DIR_SYNCS.with(Cell::get) >= before + 2,
            "the retry reported the blob kept without syncing its directory"
        );

        // A sync failure on the reuse path is reported, not hidden.
        FAIL_DIR_SYNC.with(|f| f.set(true));
        assert!(matches!(s.put(&key, &bytes), Err(BlobError::Io(_))));
    }

    fn store() -> (tempfile::TempDir, BlobStore) {
        let dir = tempfile::tempdir().expect("tempdir");
        let s = BlobStore::open(dir.path().join("blobs")).expect("open");
        (dir, s)
    }

    #[test]
    fn put_then_get_returns_the_exact_bytes() {
        let (_d, s) = store();
        let bytes = b"\x89PNG original bytes\x00\xff".to_vec();
        let key = ContentSha256::of(&bytes);
        assert!(!s.contains(&key));
        assert_eq!(s.put(&key, &bytes).expect("put"), PutOutcome::Written);
        assert!(s.contains(&key));
        assert_eq!(s.get(&key).expect("get"), Some(bytes));
    }

    #[test]
    fn a_missing_blob_is_none_not_an_error() {
        let (_d, s) = store();
        assert_eq!(
            s.get(&ContentSha256::of(b"never stored")).expect("get"),
            None
        );
    }

    #[test]
    fn a_second_put_of_the_same_bytes_writes_nothing() {
        let (_d, s) = store();
        let bytes = b"same".to_vec();
        let key = ContentSha256::of(&bytes);
        s.put(&key, &bytes).expect("put");
        let path = s.path_of(&key);
        let before = fs::metadata(&path)
            .expect("meta")
            .modified()
            .expect("mtime");
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(
            s.put(&key, &bytes).expect("put"),
            PutOutcome::AlreadyPresent
        );
        let after = fs::metadata(&path)
            .expect("meta")
            .modified()
            .expect("mtime");
        assert_eq!(before, after, "the blob was rewritten");
        // One blob file, no leftover temporary files.
        let entries: Vec<_> = fs::read_dir(path.parent().expect("dir"))
            .expect("read_dir")
            .collect();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn a_tampered_blob_is_refused_on_read() {
        let (_d, s) = store();
        let bytes = b"genuine".to_vec();
        let key = ContentSha256::of(&bytes);
        s.put(&key, &bytes).expect("put");
        fs::write(s.path_of(&key), b"altered").expect("tamper");
        match s.get(&key) {
            Err(BlobError::Tampered { expected, actual }) => {
                assert_eq!(expected, key);
                assert_eq!(actual, ContentSha256::of(b"altered"));
            }
            other => panic!("expected Tampered, got {other:?}"),
        }
        // The genuine bytes repair it.
        assert_eq!(s.put(&key, &bytes).expect("put"), PutOutcome::Written);
        assert_eq!(s.get(&key).expect("get"), Some(bytes));
    }

    #[test]
    fn bytes_cannot_be_filed_under_another_key() {
        let (_d, s) = store();
        let key = ContentSha256::of(b"the real document");
        match s.put(&key, b"something else") {
            Err(BlobError::KeyMismatch { expected, .. }) => assert_eq!(expected, key),
            other => panic!("expected KeyMismatch, got {other:?}"),
        }
        assert!(!s.contains(&key));
    }

    #[test]
    fn a_failed_write_leaves_no_blob() {
        let dir = tempfile::tempdir().expect("tempdir");
        let s = BlobStore::open(dir.path().join("blobs")).expect("open");
        let bytes = b"cannot land".to_vec();
        let key = ContentSha256::of(&bytes);
        // A file where the fan-out directory should be makes the write fail.
        let hex = format!("{key:?}");
        fs::write(s.root().join(&hex[..2]), b"not a directory").expect("block");
        assert!(matches!(s.put(&key, &bytes), Err(BlobError::Io(_))));
        assert!(!s.contains(&key));
        assert!(matches!(s.get(&key), Err(BlobError::Io(_)) | Ok(None)));
    }

    #[test]
    fn missing_ancestors_are_created() {
        let dir = tempfile::tempdir().expect("tempdir");
        let s = BlobStore::open(dir.path().join("a").join("b").join("blobs")).expect("open");
        let bytes = b"deep".to_vec();
        let key = ContentSha256::of(&bytes);
        s.put(&key, &bytes).expect("put");
        assert_eq!(s.get(&key).expect("get"), Some(bytes));
    }

    #[test]
    fn delete_removes_the_blob_and_tolerates_absence() {
        let (_d, s) = store();
        let bytes = b"to delete".to_vec();
        let key = ContentSha256::of(&bytes);
        s.put(&key, &bytes).expect("put");
        s.delete(&key).expect("delete");
        assert!(!s.contains(&key));
        s.delete(&key).expect("delete again");
    }
}
