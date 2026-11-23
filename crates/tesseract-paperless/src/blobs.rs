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
        fs::create_dir_all(&root)?;
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
            Ok(Some(_)) => return Ok(PutOutcome::AlreadyPresent),
            Ok(None) | Err(BlobError::Tampered { .. }) => {}
            Err(e) => return Err(e),
        }

        let target = self.path_of(key);
        let dir = target
            .parent()
            .ok_or_else(|| io::Error::other("blob path has no parent"))?;
        fs::create_dir_all(dir)?;
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
            File::open(dir)?.sync_all()
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

#[cfg(test)]
mod tests {
    use super::*;

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
