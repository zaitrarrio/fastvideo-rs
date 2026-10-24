//! Model names (ADR-0005): `namespace/model[@rev]` for a vendor release and
//! `namespace/model/variant[@rev]` for a fork, where `variant` is
//! `distillation:steps:encoder` and `rev` is the first 12 hex digits of the
//! manifest's canonical digest.
//!
//! The grammar is the registry's WO-001 contract, verbatim:
//!
//! ```text
//! name        = vendor-ref / fork-ref
//! vendor-ref  = namespace "/" model [ "@" rev ]
//! fork-ref    = namespace "/" model "/" variant [ "@" rev ]
//! namespace   = seg            ; 2..39 chars; reserved: gia, admin, api, registry, official
//! model       = seg-dot        ; 1..64 chars, may contain "."
//! seg         = lower-alnum *( lower-alnum / "-" )        ; no leading/trailing "-", no "--"
//! seg-dot     = lower-alnum *( lower-alnum / "-" / "." )  ; no "..", no leading/trailing "." or "-"
//! variant     = distillation ":" steps ":" encoder
//! distillation= "base" / "dmd" / "lcm" / "cm" / "sf" / "adv"
//! steps       = 1*5DIGIT       ; 1..=1000, no leading zeros
//! encoder     = "bf16" / "fp16" / "fp8" / "int8" / "nf4"
//! rev         = 12LOWERHEX
//! ```
//!
//! [`ModelName`] parses with [`FromStr`] and prints its canonical form with
//! [`fmt::Display`]; `parse(display(n)) == n` for every valid name (property
//! test in `tests/names.rs`). Every failure has its own [`NameError`]
//! variant, named in `fixtures/names/invalid.txt` by [`NameError::kind`].

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Namespaces nobody can publish under.
pub const RESERVED_NAMESPACES: &[&str] = &["gia", "admin", "api", "registry", "official"];
/// Shortest namespace.
pub const NAMESPACE_MIN_LEN: usize = 2;
/// Longest namespace.
pub const NAMESPACE_MAX_LEN: usize = 39;
/// Longest model id.
pub const MODEL_MAX_LEN: usize = 64;
/// Largest step count a variant may name.
pub const STEPS_MAX: u16 = 1000;
/// Length of a `rev` (hex digits of the manifest digest).
pub const REV_LEN: usize = 12;

/// Why a string is not a model name. One variant per failure kind; the kind
/// names ([`NameError::kind`]) are the vocabulary of `fixtures/names/invalid.txt`.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum NameError {
    #[error("empty name")]
    Empty,
    #[error("empty segment (`//`, or a leading `/`)")]
    EmptySegment,
    #[error("trailing slash")]
    TrailingSlash,
    #[error("too few segments: a name is `namespace/model` or `namespace/model/variant`")]
    TooFewSegments,
    #[error("too many segments: a name is `namespace/model` or `namespace/model/variant`")]
    TooManySegments,
    #[error("uppercase letter {0:?}: names are lowercase")]
    Uppercase(char),
    #[error("invalid character {0:?}")]
    InvalidChar(char),
    #[error("`-` may not start or end a segment or repeat (`--`)")]
    BadHyphen,
    #[error("`..` in the model id")]
    DoubleDot,
    #[error("`.` may not start or end the model id")]
    BadDot,
    #[error("namespace shorter than {NAMESPACE_MIN_LEN} characters")]
    NamespaceTooShort,
    #[error("{segment} longer than {max} characters")]
    OverLength { segment: &'static str, max: usize },
    #[error("reserved namespace {0:?}")]
    ReservedNamespace(String),
    #[error("variant must be `distillation:steps:encoder`")]
    BadVariant,
    #[error("unknown distillation {0:?}")]
    UnknownDistillation(String),
    #[error("bad steps {0:?}: 1..=1000, digits only, no leading zeros")]
    BadSteps(String),
    #[error("unknown encoder precision {0:?}")]
    UnknownEncoder(String),
    #[error("bad rev {0:?}: exactly {REV_LEN} lowercase hex digits after one `@`")]
    BadRev(String),
}

impl NameError {
    /// The variant's name without its payload, as the fixtures spell it.
    pub fn kind(&self) -> &'static str {
        match self {
            NameError::Empty => "Empty",
            NameError::EmptySegment => "EmptySegment",
            NameError::TrailingSlash => "TrailingSlash",
            NameError::TooFewSegments => "TooFewSegments",
            NameError::TooManySegments => "TooManySegments",
            NameError::Uppercase(_) => "Uppercase",
            NameError::InvalidChar(_) => "InvalidChar",
            NameError::BadHyphen => "BadHyphen",
            NameError::DoubleDot => "DoubleDot",
            NameError::BadDot => "BadDot",
            NameError::NamespaceTooShort => "NamespaceTooShort",
            NameError::OverLength { .. } => "OverLength",
            NameError::ReservedNamespace(_) => "ReservedNamespace",
            NameError::BadVariant => "BadVariant",
            NameError::UnknownDistillation(_) => "UnknownDistillation",
            NameError::BadSteps(_) => "BadSteps",
            NameError::UnknownEncoder(_) => "UnknownEncoder",
            NameError::BadRev(_) => "BadRev",
        }
    }
}

/// Character and hyphen rules shared by `seg` and `seg-dot`.
fn check_segment_chars(s: &str, allow_dot: bool) -> Result<(), NameError> {
    for c in s.chars() {
        if c.is_ascii_uppercase() {
            return Err(NameError::Uppercase(c));
        }
        let ok =
            c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || (allow_dot && c == '.');
        if !ok {
            return Err(NameError::InvalidChar(c));
        }
    }
    if s.starts_with('-') || s.ends_with('-') {
        return Err(NameError::BadHyphen);
    }
    if !allow_dot && s.contains("--") {
        return Err(NameError::BadHyphen);
    }
    if allow_dot {
        if s.contains("..") {
            return Err(NameError::DoubleDot);
        }
        if s.starts_with('.') || s.ends_with('.') {
            return Err(NameError::BadDot);
        }
    }
    Ok(())
}

/// The first segment: who publishes (`wan-ai`, `zay`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Namespace(String);

impl Namespace {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for Namespace {
    type Err = NameError;
    fn from_str(s: &str) -> Result<Self, NameError> {
        if s.is_empty() {
            return Err(NameError::EmptySegment);
        }
        check_segment_chars(s, false)?;
        if s.len() < NAMESPACE_MIN_LEN {
            return Err(NameError::NamespaceTooShort);
        }
        if s.len() > NAMESPACE_MAX_LEN {
            return Err(NameError::OverLength {
                segment: "namespace",
                max: NAMESPACE_MAX_LEN,
            });
        }
        if RESERVED_NAMESPACES.contains(&s) {
            return Err(NameError::ReservedNamespace(s.to_string()));
        }
        Ok(Namespace(s.to_string()))
    }
}

impl fmt::Display for Namespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The second segment: which model (`wan2.2-t2v`, `h3`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelId(String);

impl ModelId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for ModelId {
    type Err = NameError;
    fn from_str(s: &str) -> Result<Self, NameError> {
        if s.is_empty() {
            return Err(NameError::EmptySegment);
        }
        check_segment_chars(s, true)?;
        if s.len() > MODEL_MAX_LEN {
            return Err(NameError::OverLength {
                segment: "model",
                max: MODEL_MAX_LEN,
            });
        }
        Ok(ModelId(s.to_string()))
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Distillation vocabulary v1 (extensible by ADR only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Distillation {
    Base,
    Dmd,
    Lcm,
    Cm,
    Sf,
    Adv,
}

impl Distillation {
    pub const ALL: [Distillation; 6] = [
        Distillation::Base,
        Distillation::Dmd,
        Distillation::Lcm,
        Distillation::Cm,
        Distillation::Sf,
        Distillation::Adv,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            Distillation::Base => "base",
            Distillation::Dmd => "dmd",
            Distillation::Lcm => "lcm",
            Distillation::Cm => "cm",
            Distillation::Sf => "sf",
            Distillation::Adv => "adv",
        }
    }
}

impl FromStr for Distillation {
    type Err = NameError;
    fn from_str(s: &str) -> Result<Self, NameError> {
        Self::ALL
            .into_iter()
            .find(|d| d.as_str() == s)
            .ok_or_else(|| NameError::UnknownDistillation(s.to_string()))
    }
}

impl fmt::Display for Distillation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Text-encoder precision vocabulary v1 (extensible by ADR only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EncoderPrecision {
    Bf16,
    Fp16,
    Fp8,
    Int8,
    Nf4,
}

impl EncoderPrecision {
    pub const ALL: [EncoderPrecision; 5] = [
        EncoderPrecision::Bf16,
        EncoderPrecision::Fp16,
        EncoderPrecision::Fp8,
        EncoderPrecision::Int8,
        EncoderPrecision::Nf4,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            EncoderPrecision::Bf16 => "bf16",
            EncoderPrecision::Fp16 => "fp16",
            EncoderPrecision::Fp8 => "fp8",
            EncoderPrecision::Int8 => "int8",
            EncoderPrecision::Nf4 => "nf4",
        }
    }
}

impl FromStr for EncoderPrecision {
    type Err = NameError;
    fn from_str(s: &str) -> Result<Self, NameError> {
        Self::ALL
            .into_iter()
            .find(|e| e.as_str() == s)
            .ok_or_else(|| NameError::UnknownEncoder(s.to_string()))
    }
}

impl fmt::Display for EncoderPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How a fork differs from its parent: `distillation:steps:encoder`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Variant {
    pub distillation: Distillation,
    pub steps: u16,
    pub encoder: EncoderPrecision,
}

impl Variant {
    /// `Err(BadSteps)` unless `1 <= steps <= 1000`.
    pub fn new(
        distillation: Distillation,
        steps: u16,
        encoder: EncoderPrecision,
    ) -> Result<Self, NameError> {
        if steps == 0 || steps > STEPS_MAX {
            return Err(NameError::BadSteps(steps.to_string()));
        }
        Ok(Variant {
            distillation,
            steps,
            encoder,
        })
    }
}

fn parse_steps(s: &str) -> Result<u16, NameError> {
    let bad = || NameError::BadSteps(s.to_string());
    if s.is_empty() || s.len() > 5 || !s.bytes().all(|b| b.is_ascii_digit()) {
        return Err(bad());
    }
    if s.len() > 1 && s.starts_with('0') {
        return Err(bad());
    }
    let n: u32 = s.parse().map_err(|_| bad())?;
    if n == 0 || n > u32::from(STEPS_MAX) {
        return Err(bad());
    }
    u16::try_from(n).map_err(|_| bad())
}

impl FromStr for Variant {
    type Err = NameError;
    fn from_str(s: &str) -> Result<Self, NameError> {
        if s.is_empty() {
            return Err(NameError::EmptySegment);
        }
        if let Some(c) = s.chars().find(|c| c.is_ascii_uppercase()) {
            return Err(NameError::Uppercase(c));
        }
        let mut parts = s.split(':');
        let (Some(d), Some(st), Some(e), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return Err(NameError::BadVariant);
        };
        let distillation = d.parse()?;
        let steps = parse_steps(st)?;
        let encoder = e.parse()?;
        Ok(Variant {
            distillation,
            steps,
            encoder,
        })
    }
}

impl fmt::Display for Variant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.distillation, self.steps, self.encoder)
    }
}

/// The first 12 hex digits of a manifest's canonical digest.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Rev(String);

impl Rev {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The rev of a manifest with this canonical digest.
    pub fn from_digest(digest: &[u8; 32]) -> Rev {
        let hex = blake3::Hash::from_bytes(*digest).to_hex();
        Rev(hex[..REV_LEN].to_string())
    }

    /// Whether this rev names the manifest with the given full digest (hex).
    pub fn matches_digest_hex(&self, digest_hex: &str) -> bool {
        digest_hex.len() >= REV_LEN
            && digest_hex.is_char_boundary(REV_LEN)
            && digest_hex[..REV_LEN] == self.0
    }
}

impl FromStr for Rev {
    type Err = NameError;
    fn from_str(s: &str) -> Result<Self, NameError> {
        let ok = s.len() == REV_LEN
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        if ok {
            Ok(Rev(s.to_string()))
        } else {
            Err(NameError::BadRev(s.to_string()))
        }
    }
}

impl fmt::Display for Rev {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A model name: a vendor release or a fork, optionally pinned to a `rev`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelName {
    Vendor {
        namespace: Namespace,
        model: ModelId,
        rev: Option<Rev>,
    },
    Fork {
        namespace: Namespace,
        model: ModelId,
        variant: Variant,
        rev: Option<Rev>,
    },
}

impl ModelName {
    pub fn namespace(&self) -> &Namespace {
        match self {
            ModelName::Vendor { namespace, .. } | ModelName::Fork { namespace, .. } => namespace,
        }
    }

    pub fn model(&self) -> &ModelId {
        match self {
            ModelName::Vendor { model, .. } | ModelName::Fork { model, .. } => model,
        }
    }

    /// `Some` for a fork.
    pub fn variant(&self) -> Option<&Variant> {
        match self {
            ModelName::Vendor { .. } => None,
            ModelName::Fork { variant, .. } => Some(variant),
        }
    }

    pub fn rev(&self) -> Option<&Rev> {
        match self {
            ModelName::Vendor { rev, .. } | ModelName::Fork { rev, .. } => rev.as_ref(),
        }
    }

    pub fn is_fork(&self) -> bool {
        matches!(self, ModelName::Fork { .. })
    }

    /// The same name without its `@rev`.
    pub fn without_rev(&self) -> ModelName {
        self.clone().with_rev(None)
    }

    /// The same name with `rev` replaced.
    pub fn with_rev(mut self, new_rev: Option<Rev>) -> ModelName {
        match &mut self {
            ModelName::Vendor { rev, .. } | ModelName::Fork { rev, .. } => *rev = new_rev,
        }
        self
    }

    /// Whether two names are the same model (namespace, model, variant),
    /// ignoring `@rev`.
    pub fn same_model(&self, other: &ModelName) -> bool {
        self.without_rev() == other.without_rev()
    }
}

impl FromStr for ModelName {
    type Err = NameError;

    fn from_str(s: &str) -> Result<Self, NameError> {
        if s.is_empty() {
            return Err(NameError::Empty);
        }
        let (body, rev) = match s.split_once('@') {
            Some((body, rev)) => (body, Some(rev)),
            None => (s, None),
        };
        let rev = match rev {
            Some(r) => Some(r.parse::<Rev>()?),
            None => None,
        };
        if body.is_empty() {
            return Err(NameError::Empty);
        }
        if body.ends_with('/') {
            return Err(NameError::TrailingSlash);
        }
        let segments: Vec<&str> = body.split('/').collect();
        if segments.iter().any(|seg| seg.is_empty()) {
            return Err(NameError::EmptySegment);
        }
        match segments.as_slice() {
            [_] => Err(NameError::TooFewSegments),
            [ns, model] => Ok(ModelName::Vendor {
                namespace: ns.parse()?,
                model: model.parse()?,
                rev,
            }),
            [ns, model, variant] => Ok(ModelName::Fork {
                namespace: ns.parse()?,
                model: model.parse()?,
                variant: variant.parse()?,
                rev,
            }),
            _ => Err(NameError::TooManySegments),
        }
    }
}

impl fmt::Display for ModelName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelName::Vendor {
                namespace,
                model,
                rev,
            } => {
                write!(f, "{namespace}/{model}")?;
                if let Some(rev) = rev {
                    write!(f, "@{rev}")?;
                }
            }
            ModelName::Fork {
                namespace,
                model,
                variant,
                rev,
            } => {
                write!(f, "{namespace}/{model}/{variant}")?;
                if let Some(rev) = rev {
                    write!(f, "@{rev}")?;
                }
            }
        }
        Ok(())
    }
}

impl Serialize for ModelName {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for ModelName {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_and_fork_round_trip() {
        for s in [
            "wan-ai/wan2.2-t2v",
            "zay/wan2.2-t2v/dmd:4:fp8@0123456789ab",
            "minimax/h3@abcdef012345",
        ] {
            let n: ModelName = s.parse().unwrap();
            assert_eq!(n.to_string(), s);
        }
    }

    #[test]
    fn kinds() {
        assert_eq!(
            "Zay/wan".parse::<ModelName>().unwrap_err().kind(),
            "Uppercase"
        );
        assert_eq!(
            "gia/wan".parse::<ModelName>().unwrap_err().kind(),
            "ReservedNamespace"
        );
        assert_eq!(
            "zay/wan/dmd:0:fp8".parse::<ModelName>().unwrap_err().kind(),
            "BadSteps"
        );
        assert_eq!(
            "zay/wan/dmd:04:fp8"
                .parse::<ModelName>()
                .unwrap_err()
                .kind(),
            "BadSteps"
        );
        assert_eq!(
            "zay/wan/dmd:1001:fp8"
                .parse::<ModelName>()
                .unwrap_err()
                .kind(),
            "BadSteps"
        );
        assert_eq!(
            "zay/wan@0123456789AB"
                .parse::<ModelName>()
                .unwrap_err()
                .kind(),
            "BadRev"
        );
        assert_eq!(
            "zay/wan/".parse::<ModelName>().unwrap_err().kind(),
            "TrailingSlash"
        );
    }

    #[test]
    fn rev_from_digest() {
        let d = *blake3::hash(b"x").as_bytes();
        let rev = Rev::from_digest(&d);
        assert_eq!(rev.as_str().len(), REV_LEN);
        assert!(rev.matches_digest_hex(blake3::hash(b"x").to_hex().as_str()));
    }
}
