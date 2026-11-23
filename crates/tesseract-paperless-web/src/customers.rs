//! Who may see which customers' documents.
//!
//! The archive decides this itself, from the caller's identity-provider grants;
//! a service calling on a user's behalf (report-rs) adds no authority. A grant
//! is a `(organization id, role key)` pair from the ID token's roles claim, the
//! same pairs report-rs's own policy keys on, so one Zitadel setup drives both.
//!
//! The policy is JSON, from `PAPERLESS_CUSTOMER_POLICY` or the file named by
//! `PAPERLESS_CUSTOMER_POLICY_FILE`:
//!
//! ```json
//! {"version": 1, "grants": [
//!   {"org": "300000000000000001", "role_key": "paperless.user", "customers": ["K01", "K02"]},
//!   {"org": "300000000000000001", "role_key": "paperless.admin", "customers": "*"}
//! ]}
//! ```
//!
//! A caller's scope is the union of every matching grant. A caller no grant
//! matches sees nothing. `"*"` covers every customer and the documents that
//! have none; a list covers only documents filed under one of its codes.
//!
//! Without a policy the archive keeps its earlier behaviour (every admitted
//! user sees every document) and logs that it does. A policy with sign-in off
//! is refused at start: there would be no grants to read it with.

use std::collections::BTreeSet;

use serde::Deserialize;
use tesseract_paperless::archive_meta::valid_customer_code;

/// What a caller may see.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scope {
    /// Every document, with or without a customer.
    All,
    /// Only documents filed under one of these customers. Empty: nothing.
    Customers(BTreeSet<String>),
}

impl Scope {
    /// Whether a document filed under `customer` (`None`: no customer) is
    /// visible. A document without a customer is visible only to [`Scope::All`].
    #[must_use]
    pub fn covers(&self, customer: Option<&str>) -> bool {
        match (self, customer) {
            (Scope::All, _) => true,
            (Scope::Customers(set), Some(c)) => set.contains(c),
            (Scope::Customers(_), None) => false,
        }
    }

    /// The customers a caller may file a document under, for the upload and
    /// detail forms. `None` for [`Scope::All`] (any valid code, or none).
    #[must_use]
    pub fn choices(&self) -> Option<Vec<String>> {
        match self {
            Scope::All => None,
            Scope::Customers(set) => Some(set.iter().cloned().collect()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Customers {
    All,
    List(BTreeSet<String>),
}

#[derive(Debug, Clone)]
struct Grant {
    org: String,
    role_key: String,
    customers: Customers,
}

/// The parsed policy.
#[derive(Debug, Clone)]
pub struct CustomerPolicy {
    grants: Vec<Grant>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPolicy {
    version: u32,
    grants: Vec<RawGrant>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGrant {
    org: String,
    role_key: String,
    customers: serde_json::Value,
}

impl CustomerPolicy {
    /// Parse a policy. Every mistake is an error, never a skipped entry: a
    /// skipped grant would silently narrow access, a misread one widen it.
    ///
    /// # Errors
    /// A message naming the problem.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let raw: RawPolicy = serde_json::from_str(json)
            .map_err(|e| format!("customer policy is not valid JSON of the expected shape: {e}"))?;
        if raw.version != 1 {
            return Err(format!(
                "customer policy version {} is not supported (expected 1)",
                raw.version
            ));
        }
        if raw.grants.is_empty() {
            return Err("customer policy lists no grants".into());
        }
        let mut grants = Vec::with_capacity(raw.grants.len());
        for (i, g) in raw.grants.into_iter().enumerate() {
            let org = g.org.trim();
            let role_key = g.role_key.trim();
            if org.is_empty() || role_key.is_empty() {
                return Err(format!(
                    "customer policy grant {i}: org and role_key are required"
                ));
            }
            let customers = match &g.customers {
                serde_json::Value::String(s) if s == "*" => Customers::All,
                serde_json::Value::Array(items) => {
                    let mut set = BTreeSet::new();
                    for item in items {
                        let code = item
                            .as_str()
                            .filter(|c| valid_customer_code(c))
                            .ok_or_else(|| {
                                format!("customer policy grant {i}: {item} is not a customer code")
                            })?;
                        set.insert(code.to_string());
                    }
                    if set.is_empty() {
                        return Err(format!(
                            "customer policy grant {i}: an empty customer list admits nothing; \
                             remove the grant instead"
                        ));
                    }
                    Customers::List(set)
                }
                _ => {
                    return Err(format!(
                        "customer policy grant {i}: customers must be \"*\" or a list of codes"
                    ))
                }
            };
            grants.push(Grant {
                org: org.to_string(),
                role_key: role_key.to_string(),
                customers,
            });
        }
        Ok(Self { grants })
    }

    /// Read `PAPERLESS_CUSTOMER_POLICY` / `PAPERLESS_CUSTOMER_POLICY_FILE`.
    /// `Ok(None)` when neither is set; both set is an error.
    ///
    /// # Errors
    /// A message naming the problem.
    pub fn from_vars(get: impl Fn(&str) -> Option<String>) -> Result<Option<Self>, String> {
        let present = |k: &str| get(k).filter(|v| !v.trim().is_empty());
        match (
            present("PAPERLESS_CUSTOMER_POLICY"),
            present("PAPERLESS_CUSTOMER_POLICY_FILE"),
        ) {
            (None, None) => Ok(None),
            (Some(_), Some(_)) => Err(
                "set PAPERLESS_CUSTOMER_POLICY or PAPERLESS_CUSTOMER_POLICY_FILE, not both".into(),
            ),
            (Some(json), None) => Self::from_json(&json).map(Some),
            (None, Some(path)) => {
                let json = std::fs::read_to_string(path.trim())
                    .map_err(|e| format!("PAPERLESS_CUSTOMER_POLICY_FILE: {e}"))?;
                Self::from_json(&json).map(Some)
            }
        }
    }

    /// The process environment version of [`Self::from_vars`].
    ///
    /// # Errors
    /// As [`Self::from_vars`], and for a value that is not valid Unicode.
    pub fn from_env() -> Result<Option<Self>, String> {
        for name in [
            "PAPERLESS_CUSTOMER_POLICY",
            "PAPERLESS_CUSTOMER_POLICY_FILE",
        ] {
            if std::env::var_os(name).is_some_and(|v| v.to_str().is_none()) {
                return Err(format!("{name} is not valid Unicode"));
            }
        }
        Self::from_vars(|k| std::env::var(k).ok())
    }

    /// The scope of a caller holding `grants` (`(org, role_key)` pairs).
    #[must_use]
    pub fn scope_for(&self, grants: &[(String, String)]) -> Scope {
        let mut set = BTreeSet::new();
        for g in &self.grants {
            if !grants.iter().any(|(o, r)| *o == g.org && *r == g.role_key) {
                continue;
            }
            match &g.customers {
                Customers::All => return Scope::All,
                Customers::List(codes) => set.extend(codes.iter().cloned()),
            }
        }
        Scope::Customers(set)
    }

    /// One log line, naming counts only.
    #[must_use]
    pub fn summary(&self) -> String {
        format!("customer policy with {} grant(s)", self.grants.len())
    }
}

/// The start rule: a policy needs sign-in to have grants to read.
///
/// # Errors
/// When a policy is set but sign-in is off.
pub fn check_policy_needs_sign_in(policy: bool, sign_in: bool) -> Result<(), String> {
    if policy && !sign_in {
        return Err("a customer policy is set but sign-in is not configured \
                    (PAPERLESS_OIDC_ISSUER unset): refusing to start"
            .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const POLICY: &str = r#"{"version": 1, "grants": [
        {"org": "o1", "role_key": "paperless.user", "customers": ["K01", "K02"]},
        {"org": "o1", "role_key": "paperless.extra", "customers": ["K03"]},
        {"org": "o1", "role_key": "paperless.admin", "customers": "*"}
    ]}"#;

    fn g(org: &str, role: &str) -> (String, String) {
        (org.to_string(), role.to_string())
    }

    fn codes(c: &[&str]) -> Scope {
        Scope::Customers(c.iter().map(|s| (*s).to_string()).collect())
    }

    /// Disable: return `Scope::All` from `scope_for` when nothing matched, or
    /// match on the role key alone.
    #[test]
    fn scope_is_the_union_of_matching_grants_and_empty_otherwise() {
        let p = CustomerPolicy::from_json(POLICY).expect("parse");
        assert_eq!(
            p.scope_for(&[g("o1", "paperless.user")]),
            codes(&["K01", "K02"])
        );
        assert_eq!(
            p.scope_for(&[g("o1", "paperless.user"), g("o1", "paperless.extra")]),
            codes(&["K01", "K02", "K03"])
        );
        assert_eq!(p.scope_for(&[g("o1", "paperless.admin")]), Scope::All);
        // Same role key under another organization is a different grant.
        assert_eq!(p.scope_for(&[g("o2", "paperless.user")]), codes(&[]));
        assert_eq!(p.scope_for(&[]), codes(&[]));
    }

    /// Disable: make `covers(None)` true for a customer list.
    #[test]
    fn a_document_without_a_customer_is_visible_only_to_all() {
        let s = codes(&["K01"]);
        assert!(s.covers(Some("K01")));
        assert!(!s.covers(Some("K02")));
        assert!(
            !s.covers(None),
            "an unfiled document must not leak to a scoped caller"
        );
        assert!(Scope::All.covers(None));
        assert!(!codes(&[]).covers(Some("K01")));
    }

    #[test]
    fn malformed_policies_are_refused_not_skipped() {
        for bad in [
            "{}",
            r#"{"version": 2, "grants": [{"org":"o","role_key":"r","customers":"*"}]}"#,
            r#"{"version": 1, "grants": []}"#,
            r#"{"version": 1, "grants": [{"org":"","role_key":"r","customers":"*"}]}"#,
            r#"{"version": 1, "grants": [{"org":"o","role_key":"r","customers":"all"}]}"#,
            r#"{"version": 1, "grants": [{"org":"o","role_key":"r","customers":[]}]}"#,
            r#"{"version": 1, "grants": [{"org":"o","role_key":"r","customers":["K 01"]}]}"#,
            r#"{"version": 1, "grants": [{"org":"o","role_key":"r","customers":[1]}]}"#,
            r#"{"version": 1, "grants": [{"org":"o","role_key":"r","customers":"*","x":1}]}"#,
        ] {
            assert!(
                CustomerPolicy::from_json(bad).is_err(),
                "{bad} must be refused"
            );
        }
    }

    #[test]
    fn from_vars_reads_one_source_and_refuses_both() {
        let none = CustomerPolicy::from_vars(|_| None).expect("unset");
        assert!(none.is_none());
        let inline = CustomerPolicy::from_vars(|k| {
            (k == "PAPERLESS_CUSTOMER_POLICY").then(|| POLICY.to_string())
        })
        .expect("inline");
        assert!(inline.is_some());
        assert!(CustomerPolicy::from_vars(|_| Some(POLICY.to_string())).is_err());
    }

    #[test]
    fn a_policy_without_sign_in_refuses_to_start() {
        assert!(check_policy_needs_sign_in(true, false).is_err());
        assert!(check_policy_needs_sign_in(true, true).is_ok());
        assert!(check_policy_needs_sign_in(false, false).is_ok());
    }
}
