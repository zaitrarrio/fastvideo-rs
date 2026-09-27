//! LTX model ids, their engine targets, and the T2V/I2V support matrix
//! (design §0.3, §0.6, §4.5; ltx §3).
//!
//! Model ids map to engine tiers (owner decisions override design §4.5):
//!
//! | LTX `model` | Matrix | Engine target |
//! |---|---|---|
//! | `ltx-2-5-pro`, `ltx-2-3-pro` | pro | tier **max** (`ltx-pro` alias) |
//! | `ltx-2-5-fast`, `ltx-2-3-fast` | fast | tier **turbo** (`ltx-turbo`) |
//! | `ltx-turbo` | fast | tier **turbo** |
//! | `ltx-draft` | fast | tier **draft** (`ltx-draft`); below the quality gate |
//! | `ltx-2-fast`, `ltx-2-pro` | — | removed upstream: 400 |
//!
//! Every target is configurable ([`LtxModels::set`]), e.g. pinning
//! `ltx-2-3-pro` to a specific engine model id. A known id whose target is not
//! served answers `403 permission_error`; an unknown id is `400`.
//!
//! The support matrix (ltx §3) is enforced for every id; `ltx-turbo` and
//! `ltx-draft` are ours and follow the `*-fast` rows.
//!
//! | Class | Resolution | FPS | Duration (s) |
//! |---|---|---|---|
//! | fast | 720p, 1080p | 24, 25 | 6, 8, …, 20 |
//! | fast | 720p, 1080p | 48, 50 | 6, 8, 10 |
//! | fast | 1440p, 4K | 24, 25, 48, 50 | 6, 8, 10 |
//! | pro | 720p … 4K | 24, 25, 48, 50 | 6, 8, 10 |

use std::collections::BTreeMap;

use fastvideo_protocol::{ApiError, Tier};

/// Which support-matrix rows a model follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelClass {
    Fast,
    Pro,
}

/// What a model id runs on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    /// The LTX-family model bound to a tier.
    Tier(Tier),
    /// A specific engine model id (or alias).
    Model(String),
}

impl Target {
    /// The engine-side name placed in `GenerationRequest::model`: the
    /// canonical tier alias (`ltx-pro`, `ltx-turbo`, `ltx-draft`) or the id.
    pub fn engine_name(&self) -> String {
        match self {
            Target::Tier(t) => tier_alias(*t).to_owned(),
            Target::Model(m) => m.clone(),
        }
    }
}

/// The engine's canonical LTX tier alias (matches
/// `fastvideo_engine_service::tier_alias(Family::Ltx2, tier)`).
pub fn tier_alias(t: Tier) -> &'static str {
    match t {
        Tier::Max => "ltx-pro",
        Tier::Turbo => "ltx-turbo",
        Tier::Draft => "ltx-draft",
    }
}

/// The tier a canonical alias names.
pub fn parse_tier_alias(s: &str) -> Option<Tier> {
    [Tier::Max, Tier::Turbo, Tier::Draft]
        .into_iter()
        .find(|t| tier_alias(*t) == s)
}

/// One accepted `model` id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LtxModel {
    pub class: ModelClass,
    pub target: Target,
    /// `duration: null` is valid for the id upstream (2.5 ids and ours); it
    /// still answers `Unsupported(LtxAutoDuration)` here.
    pub auto_duration: bool,
}

/// Ids removed upstream on 2026-08-16 (ltx §3).
pub const REMOVED_MODELS: [&str; 2] = ["ltx-2-fast", "ltx-2-pro"];

/// The `model` table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LtxModels {
    map: BTreeMap<String, LtxModel>,
}

impl Default for LtxModels {
    fn default() -> Self {
        let m = |class, tier, auto| LtxModel {
            class,
            target: Target::Tier(tier),
            auto_duration: auto,
        };
        let map = [
            ("ltx-2-5-pro", m(ModelClass::Pro, Tier::Max, true)),
            ("ltx-2-3-pro", m(ModelClass::Pro, Tier::Max, false)),
            ("ltx-2-5-fast", m(ModelClass::Fast, Tier::Turbo, true)),
            ("ltx-2-3-fast", m(ModelClass::Fast, Tier::Turbo, false)),
            ("ltx-turbo", m(ModelClass::Fast, Tier::Turbo, true)),
            ("ltx-draft", m(ModelClass::Fast, Tier::Draft, true)),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect();
        Self { map }
    }
}

impl LtxModels {
    /// Rebinds (or adds) `id`.
    pub fn set(&mut self, id: impl Into<String>, model: LtxModel) -> &mut Self {
        self.map.insert(id.into(), model);
        self
    }
    /// Rebinds the target of an existing id; `false` if unknown.
    pub fn retarget(&mut self, id: &str, target: Target) -> bool {
        match self.map.get_mut(id) {
            Some(m) => {
                m.target = target;
                true
            }
            None => false,
        }
    }
    pub fn get(&self, id: &str) -> Option<&LtxModel> {
        self.map.get(id)
    }
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(String::as_str)
    }

    /// Looks up the `model` field. Removed and unknown ids are
    /// `invalid_request_error`.
    pub fn lookup(&self, id: &str) -> Result<&LtxModel, ApiError> {
        if REMOVED_MODELS.contains(&id) {
            return Err(ApiError::invalid_param(
                "model",
                format!("model `{id}` has been removed; use one of: {}", self.list()),
            ));
        }
        self.get(id).ok_or_else(|| {
            ApiError::invalid_param(
                "model",
                format!("model must be one of: {}", self.list()),
            )
        })
    }

    fn list(&self) -> String {
        self.ids().collect::<Vec<_>>().join(", ")
    }
}

/// A resolution tier (ltx §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ResTier {
    P720,
    P1080,
    P1440,
    K4,
}

impl ResTier {
    /// The landscape `(width, height)`.
    pub fn landscape(&self) -> (u32, u32) {
        match self {
            ResTier::P720 => (1280, 720),
            ResTier::P1080 => (1920, 1080),
            ResTier::P1440 => (2560, 1440),
            ResTier::K4 => (3840, 2160),
        }
    }
    pub const ALL: [ResTier; 4] = [ResTier::P720, ResTier::P1080, ResTier::P1440, ResTier::K4];
}

/// A validated `resolution` string.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
    pub tier: ResTier,
}

/// Every accepted `resolution` string: the 16:9 and 9:16 form of each tier.
pub fn resolution_strings() -> Vec<String> {
    ResTier::ALL
        .iter()
        .flat_map(|t| {
            let (w, h) = t.landscape();
            [format!("{w}x{h}"), format!("{h}x{w}")]
        })
        .collect()
}

impl Resolution {
    /// Parses `WIDTHxHEIGHT`; only the eight documented strings pass.
    pub fn parse(s: &str) -> Result<Self, ApiError> {
        let bad = || {
            ApiError::invalid_param(
                "resolution",
                format!(
                    "resolution must be one of: {}",
                    resolution_strings().join(", ")
                ),
            )
        };
        let (w, h) = s.split_once('x').ok_or_else(bad)?;
        let (w, h): (u32, u32) = (w.parse().map_err(|_| bad())?, h.parse().map_err(|_| bad())?);
        ResTier::ALL
            .into_iter()
            .find(|t| {
                let (lw, lh) = t.landscape();
                (w, h) == (lw, lh) || (w, h) == (lh, lw)
            })
            .map(|tier| Resolution {
                width: w,
                height: h,
                tier,
            })
            .ok_or_else(bad)
    }
}

/// Frame rates the API documents.
pub const API_FPS: [u32; 4] = [24, 25, 48, 50];
/// The default `fps`.
pub const DEFAULT_FPS: u32 = 24;

/// The durations (seconds) the matrix allows for `class` at `res` and `fps`.
pub fn allowed_durations(class: ModelClass, res: ResTier, fps: u32) -> Vec<u32> {
    let long = class == ModelClass::Fast
        && matches!(res, ResTier::P720 | ResTier::P1080)
        && matches!(fps, 24 | 25);
    let max = if long { 20 } else { 10 };
    (6..=max).step_by(2).collect()
}

/// Checks `fps` against the documented list.
pub fn check_fps(fps: u32) -> Result<(), ApiError> {
    if API_FPS.contains(&fps) {
        Ok(())
    } else {
        Err(ApiError::invalid_param(
            "fps",
            format!("fps must be one of: 24, 25, 48, 50 (got {fps})"),
        ))
    }
}

/// Checks `duration` against the matrix.
pub fn check_duration(
    model_id: &str,
    m: &LtxModel,
    res: Resolution,
    fps: u32,
    duration: u32,
) -> Result<(), ApiError> {
    let allowed = allowed_durations(m.class, res.tier, fps);
    if allowed.contains(&duration) {
        return Ok(());
    }
    let list: Vec<String> = allowed.iter().map(u32::to_string).collect();
    Err(ApiError::invalid_param(
        "duration",
        format!(
            "duration {duration} is not supported for {model_id} at {}x{} and {fps} fps; allowed: {}",
            res.width,
            res.height,
            list.join(", ")
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_table() {
        use ModelClass::*;
        use ResTier::*;
        let long: Vec<u32> = (6..=20).step_by(2).collect();
        let short = vec![6, 8, 10];
        for fps in API_FPS {
            for res in ResTier::ALL {
                let want_fast = if matches!(res, P720 | P1080) && matches!(fps, 24 | 25) {
                    &long
                } else {
                    &short
                };
                assert_eq!(&allowed_durations(Fast, res, fps), want_fast, "{res:?} {fps}");
                assert_eq!(allowed_durations(Pro, res, fps), short, "{res:?} {fps}");
            }
        }
    }

    #[test]
    fn resolutions() {
        assert_eq!(resolution_strings().len(), 8);
        for s in resolution_strings() {
            let r = Resolution::parse(&s).unwrap();
            assert_eq!(format!("{}x{}", r.width, r.height), s);
        }
        for bad in ["1920X1080", "1920x1088", "768x512", "x", "", "1280x720x1", "-1x720"] {
            assert!(Resolution::parse(bad).is_err(), "{bad}");
        }
        assert_eq!(Resolution::parse("2160x3840").unwrap().tier, ResTier::K4);
    }

    #[test]
    fn default_table() {
        let t = LtxModels::default();
        let tier = |id: &str| t.get(id).unwrap().target.clone();
        assert_eq!(tier("ltx-2-5-pro"), Target::Tier(Tier::Max));
        assert_eq!(tier("ltx-2-3-pro"), Target::Tier(Tier::Max));
        assert_eq!(tier("ltx-2-5-fast"), Target::Tier(Tier::Turbo));
        assert_eq!(tier("ltx-turbo"), Target::Tier(Tier::Turbo));
        assert_eq!(tier("ltx-draft"), Target::Tier(Tier::Draft));
        assert_eq!(Target::Tier(Tier::Max).engine_name(), "ltx-pro");
        for id in REMOVED_MODELS {
            let e = t.lookup(id).unwrap_err();
            assert!(e.message.contains("removed"), "{}", e.message);
        }
        assert!(t.lookup("ltx-3").is_err());
        for tr in [Tier::Max, Tier::Turbo, Tier::Draft] {
            assert_eq!(parse_tier_alias(tier_alias(tr)), Some(tr));
        }
    }
}
