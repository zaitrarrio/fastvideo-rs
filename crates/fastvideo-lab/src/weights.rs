//! Weights by tree for a `weights = "volume+r2"` tier (docs/adr/0019-fv-lab.md
//! §14): every tree an arm reads comes from the EU volume when the volume
//! holds it (`scripts/gpu/volumes.sh` `FV_EU_VOLUME_TREES`), else from the R2
//! weights store when it is published there (`scripts/gpu/weights-r2.tsv`),
//! fetched onto the container disk before the arm by the repository's own
//! tools (`r2-weights.py plan`, then `r2-fetch.py`, the fetcher fv-boot runs);
//! a tree in neither is not fetched, and the arm's gate records it as skipped
//! (`weights incomplete`), as on a `volume` tier.
//!
//! `{weights}` is one root on the container disk, `<scratch>/weights`
//! ([`WEIGHTS_DIR`]): a symlink per volume tree into `<work>/weights` (the
//! volume, never written) and the fetched trees as plain directories. An arm
//! keeps the trees it fetched while later arms read them; an arm that must
//! fetch first removes every fetched tree it does not read ([`step`]), so the
//! container disk holds at most the largest single arm's R2 trees
//! ([`WeightsPlan::peak_bytes`]).
//!
//! The R2 key reaches the pod as Runpod secret references ([`r2_env_refs`]:
//! the values live in Runpod and are filled in inside the container, never in
//! a launch, a file, a log or an argv), and the runner hands it to the plan
//! step alone ([`R2Key`]).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use fastvideo_deploy::launch::parse_eu_trees;
use fastvideo_manifest::family::tree_covers;

/// The published trees (`dest`, `rev`, files, bytes, manifest SHA-256, source).
pub const R2_TABLE: &str = "scripts/gpu/weights-r2.tsv";
/// `FV_EU_VOLUME_TREES`: the trees on the EU volume.
pub const VOLUMES_SH: &str = "scripts/gpu/volumes.sh";
/// The merged weight root under the scratch disk (`{weights}`).
pub const WEIGHTS_DIR: &str = "weights";
/// Where the plan of presigned URLs goes, under the scratch disk: outside
/// `<scratch>/runs`, which the pod serves over HTTP.
pub const R2_STATE_DIR: &str = "r2";
/// What a `volume+r2` tier's container disk keeps besides the fetched trees
/// (GB): the 120 GB every `volume` tier has (runs, frames, the TAE files, the
/// library links), so a tier's `container_disk_gb` is at least its largest
/// arm's R2 trees plus this (tests/gold_tier.rs).
pub const DISK_RESERVE_GB: u64 = 120;

/// The env names the R2 key goes by on the pod (what `r2-weights.py` reads),
/// and the Runpod secret each is filled from.
pub const R2_KEY_ENV: [(&str, &str); 2] = [
    ("FV_WEIGHTS_R2_ACCESS_KEY_ID", "fv_r2_access_key_id"),
    ("FV_WEIGHTS_R2_SECRET_ACCESS_KEY", "fv_r2_secret_access_key"),
];

/// A `volume+r2` pod's env: the R2 key as Runpod secret references (the
/// mechanism of the serve templates and `segments-pod.sh`'s R2 mode), never
/// values. The endpoint and bucket are `r2-weights.py`'s defaults (the
/// account's endpoint, `fv-weights`).
pub fn r2_env_refs() -> BTreeMap<String, String> {
    R2_KEY_ENV
        .iter()
        .map(|(k, s)| ((*k).to_owned(), format!("{{{{ RUNPOD_SECRET_{s} }}}}")))
        .collect()
}

/// The R2 key in the runner's memory. Never printed: `Debug` shows only
/// whether it is there.
#[derive(Clone, PartialEq, Eq)]
pub struct R2Key {
    pub id: String,
    pub secret: String,
}

impl std::fmt::Debug for R2Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("R2Key(<redacted>)")
    }
}

impl R2Key {
    /// The key from `get` ([`R2_KEY_ENV`]); `None` when a part is missing,
    /// empty, or still a secret reference Runpod did not fill in.
    pub fn from_env(get: impl Fn(&str) -> Option<String>) -> Option<R2Key> {
        let read = |k: &str| {
            get(k)
                .map(|v| v.trim().to_owned())
                .filter(|v| !v.is_empty() && !v.contains("{{"))
        };
        Some(R2Key {
            id: read(R2_KEY_ENV[0].0)?,
            secret: read(R2_KEY_ENV[1].0)?,
        })
    }

    /// `(name, value)` for the plan step's environment.
    pub fn env(&self) -> [(&'static str, &str); 2] {
        [
            (R2_KEY_ENV[0].0, self.id.as_str()),
            (R2_KEY_ENV[1].0, self.secret.as_str()),
        ]
    }
}

/// One published tree (a `weights-r2.tsv` row).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct R2Tree {
    pub dest: String,
    pub rev: String,
    pub bytes: u64,
}

impl R2Tree {
    /// `dest@rev`, what `r2-weights.py plan` takes.
    pub fn spec(&self) -> String {
        format!("{}@{}", self.dest, self.rev)
    }
}

/// Bytes as `r2-weights.py` prints them: decimal GB, two places.
pub fn gb(bytes: u64) -> String {
    format!("{:.2} GB", bytes as f64 / 1e9)
}

/// `weights-r2.tsv`: dest → tree. A dest published twice is an error (the
/// runner would not know which revision to fetch).
pub fn parse_r2_table(text: &str) -> Result<BTreeMap<String, R2Tree>, String> {
    let mut out = BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let (Some(dest), Some(rev), Some(bytes)) = (f.first(), f.get(1), f.get(3)) else {
            return Err(format!(
                "{R2_TABLE}:{}: not dest<TAB>rev<TAB>files<TAB>bytes…",
                i + 1
            ));
        };
        let bytes: u64 = bytes
            .parse()
            .map_err(|_| format!("{R2_TABLE}:{}: bytes `{bytes}`", i + 1))?;
        let ok = |s: &str| {
            !s.is_empty()
                && !s.starts_with('/')
                && s.split('/').all(|p| !p.is_empty() && p != "..")
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b'/'))
        };
        if !ok(dest) || !ok(rev) || rev.contains('/') {
            return Err(format!("{R2_TABLE}:{}: `{dest}` `{rev}`", i + 1));
        }
        let t = R2Tree {
            dest: (*dest).to_owned(),
            rev: (*rev).to_owned(),
            bytes,
        };
        if out.insert(t.dest.clone(), t).is_some() {
            return Err(format!("{R2_TABLE}:{}: `{dest}` twice", i + 1));
        }
    }
    Ok(out)
}

/// Where an arm's tree comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source<'a> {
    /// The EU volume tree that holds it.
    Volume(&'a str),
    /// The published tree that holds it (the longest dest covering it).
    R2(&'a R2Tree),
    /// Neither: the arm's gate decides (skipped, `weights incomplete`).
    Missing,
}

/// The trees of the EU volume and of the R2 weights store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeSources {
    pub volume: Vec<String>,
    pub r2: BTreeMap<String, R2Tree>,
}

impl TreeSources {
    /// `<root>/scripts/gpu/volumes.sh` and `<root>/scripts/gpu/weights-r2.tsv`.
    pub fn load(root: &Path) -> Result<TreeSources, String> {
        let read = |rel: &str| {
            let p = root.join(rel);
            std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))
        };
        let volume = parse_eu_trees(&read(VOLUMES_SH)?);
        if volume.is_empty() {
            return Err(format!("{VOLUMES_SH}: no FV_EU_VOLUME_TREES"));
        }
        Ok(TreeSources {
            volume,
            r2: parse_r2_table(&read(R2_TABLE)?)?,
        })
    }

    /// The source of one tree: the volume first (what it holds is never
    /// fetched), then R2.
    pub fn source(&self, tree: &str) -> Source<'_> {
        if let Some(v) = self.volume.iter().find(|v| tree_covers(v, tree)) {
            return Source::Volume(v);
        }
        self.r2
            .values()
            .filter(|t| tree_covers(&t.dest, tree))
            .max_by_key(|t| t.dest.len())
            .map_or(Source::Missing, Source::R2)
    }

    /// The volume trees no other volume tree holds: one symlink each in the
    /// merged root (`auxiliary/tae`, not `auxiliary`, so a fetched
    /// `auxiliary/<x>` lands on the container disk, never in the volume).
    pub fn link_roots(&self) -> Vec<&str> {
        let mut out: Vec<&str> = self
            .volume
            .iter()
            .filter(|v| !self.volume.iter().any(|o| o != *v && tree_covers(o, v)))
            .map(String::as_str)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The R2 trees an arm reads, checked: none may hold a volume tree (it
    /// would sit where the volume's symlink is), and a nested tree comes with
    /// its parent (`r2-fetch.py` lands it inside the parent).
    pub fn arm_r2(&self, arm: &str, trees: &[String]) -> Result<BTreeSet<R2Tree>, String> {
        let mut need = BTreeSet::new();
        for t in trees {
            if let Source::R2(r) = self.source(t) {
                need.insert(r.clone());
            }
        }
        for r in &need {
            if let Some(v) = self.volume.iter().find(|v| tree_covers(&r.dest, v)) {
                return Err(format!(
                    "arm {arm}: the R2 tree {} holds the volume tree {v}",
                    r.dest
                ));
            }
            if let Some(p) = self
                .r2
                .values()
                .find(|p| p.dest != r.dest && tree_covers(&p.dest, &r.dest) && !need.contains(*p))
            {
                return Err(format!(
                    "arm {arm}: the R2 tree {} is nested in {}, which the arm does not read (list it too)",
                    r.dest, p.dest
                ));
            }
        }
        Ok(need)
    }
}

/// What one arm does to the fetched trees before its gate.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ArmWeights {
    /// Fetched trees removed first (fetched for an earlier arm, not read by
    /// this one).
    pub drop: Vec<R2Tree>,
    /// Trees fetched from R2.
    pub fetch: Vec<R2Tree>,
    /// Every R2 tree the arm reads (on the disk while it runs).
    pub reads: Vec<R2Tree>,
}

/// One arm's step from the fetched trees `held`: nothing when every tree it
/// reads is already there; else drop every held tree it does not read (and
/// that holds none it reads), then fetch the rest.
pub fn step(held: &BTreeSet<R2Tree>, need: &BTreeSet<R2Tree>) -> ArmWeights {
    let fetch: Vec<R2Tree> = need.difference(held).cloned().collect();
    let drop = if fetch.is_empty() {
        Vec::new()
    } else {
        held.iter()
            .filter(|h| !need.contains(*h))
            .filter(|h| !need.iter().any(|n| tree_covers(&h.dest, &n.dest)))
            .cloned()
            .collect()
    };
    ArmWeights {
        drop,
        fetch,
        reads: need.iter().cloned().collect(),
    }
}

/// The R2 side of a run, every fetch assumed to succeed: what `--dry-run`
/// prints and what sizes the container disk.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WeightsPlan {
    /// `(arm, its step)`, in run order, for the arms given.
    pub arms: Vec<(String, ArmWeights)>,
    /// The most bytes of fetched trees on the disk at once.
    pub peak_bytes: u64,
    /// Every byte fetched.
    pub fetch_bytes: u64,
}

impl WeightsPlan {
    /// The plan of `arms` (`(name, trees)` in run order).
    pub fn new(sources: &TreeSources, arms: &[(&str, &[String])]) -> Result<WeightsPlan, String> {
        let mut held: BTreeSet<R2Tree> = BTreeSet::new();
        let mut plan = WeightsPlan::default();
        for (name, trees) in arms {
            let need = sources.arm_r2(name, trees)?;
            let s = step(&held, &need);
            for d in &s.drop {
                held.remove(d);
            }
            held.extend(s.fetch.iter().cloned());
            plan.peak_bytes = plan
                .peak_bytes
                .max(held.iter().map(|t| t.bytes).sum::<u64>());
            plan.fetch_bytes += s.fetch.iter().map(|t| t.bytes).sum::<u64>();
            plan.arms.push(((*name).to_owned(), s));
        }
        Ok(plan)
    }

    /// The step of one arm.
    pub fn arm(&self, name: &str) -> Option<&ArmWeights> {
        self.arms.iter().find(|(a, _)| a == name).map(|(_, w)| w)
    }

    /// The distinct trees fetched.
    pub fn fetched_trees(&self) -> BTreeSet<&R2Tree> {
        self.arms.iter().flat_map(|(_, w)| &w.fetch).collect()
    }
}

/// A `volume+r2` run's settings (`RunConfig::r2`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct R2Setup {
    pub sources: TreeSources,
    /// `{weights}`: the merged root (`<scratch>/weights`).
    pub root: PathBuf,
    /// `python3` (a test points it at a fake).
    pub python: String,
    /// `apt-get`, run when `python3` is missing (the runtime image has none).
    pub apt_get: String,
    /// The key for the plan step; `None`: R2 arms are skipped (`r2 fetch failed`).
    pub key: Option<R2Key>,
}

impl R2Setup {
    /// The pod's settings: the root under `scratch`, `python3`, `apt-get`.
    pub fn new(sources: TreeSources, scratch: &Path, key: Option<R2Key>) -> R2Setup {
        R2Setup {
            sources,
            root: scratch.join(WEIGHTS_DIR),
            python: "python3".into(),
            apt_get: "apt-get".into(),
            key,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLE: &str = "# comment\n\
        big\tr1\t3\t100000000000\tsha\thub\n\
        small\tr2\t1\t1000000000\tsha\tdir\n\
        loras/a\tr3\t1\t500000000\tsha\thub\n\
        vol\teu-1\t1\t5\tsha\tdir\n\
        big/nested\tr4\t1\t7\tsha\tdir\n";

    fn sources() -> TreeSources {
        TreeSources {
            volume: vec![
                "vol".into(),
                "vol/inner".into(),
                "aux/tae".into(),
                "aux/lpips".into(),
            ],
            r2: parse_r2_table(TABLE).unwrap(),
        }
    }

    fn v(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| (*x).to_owned()).collect()
    }

    #[test]
    fn the_table_parses_and_refuses_twice_and_bad_rows() {
        let t = parse_r2_table(TABLE).unwrap();
        assert_eq!(t["big"].spec(), "big@r1");
        assert_eq!(t["loras/a"].bytes, 500_000_000);
        assert!(parse_r2_table(" a\tr\t1\t2\n").is_err());
        assert!(parse_r2_table("a\tr\t1\t2\na\tr\t1\t2\n")
            .unwrap_err()
            .contains("twice"));
        assert!(parse_r2_table("a\tr\t1\n").is_err());
        assert!(parse_r2_table("a\tr\t1\tx\n").is_err());
        assert!(parse_r2_table("../a\tr\t1\t2\n").is_err());
        assert!(parse_r2_table("a\tr/x\t1\t2\n").is_err());
    }

    #[test]
    fn the_volume_wins_then_the_longest_r2_dest() {
        let s = sources();
        assert_eq!(s.source("vol"), Source::Volume("vol"));
        assert_eq!(s.source("vol/x.safetensors"), Source::Volume("vol"));
        assert_eq!(s.source("aux/tae/t.safetensors"), Source::Volume("aux/tae"));
        assert!(matches!(s.source("big"), Source::R2(t) if t.dest == "big"));
        assert!(matches!(s.source("big/nested/f"), Source::R2(t) if t.dest == "big/nested"));
        assert!(matches!(s.source("loras/a"), Source::R2(t) if t.dest == "loras/a"));
        assert_eq!(s.source("loras"), Source::Missing);
        assert_eq!(s.source("bigger"), Source::Missing);
        assert_eq!(s.link_roots(), ["aux/lpips", "aux/tae", "vol"]);
    }

    #[test]
    fn an_r2_tree_may_not_hold_a_volume_tree_or_come_without_its_parent() {
        let mut s = sources();
        s.r2.insert(
            "aux".into(),
            R2Tree {
                dest: "aux".into(),
                rev: "r".into(),
                bytes: 1,
            },
        );
        assert!(s
            .arm_r2("x", &v(&["aux/other"]))
            .unwrap_err()
            .contains("holds the volume tree"));
        let s = sources();
        assert!(s
            .arm_r2("x", &v(&["big/nested"]))
            .unwrap_err()
            .contains("nested in big"));
        assert_eq!(s.arm_r2("x", &v(&["big", "big/nested"])).unwrap().len(), 2);
    }

    #[test]
    fn an_arm_that_fetches_drops_what_it_does_not_read_and_others_keep_it() {
        let s = sources();
        let arms = [
            ("a", v(&["big", "vol"])),
            ("b", v(&["vol"])),
            ("c", v(&["big"])),
            ("d", v(&["small", "missing-tree"])),
            ("e", v(&["small", "loras/a"])),
            ("f", v(&["big", "big/nested"])),
        ];
        let arms: Vec<(&str, &[String])> = arms.iter().map(|(n, t)| (*n, t.as_slice())).collect();
        let p = WeightsPlan::new(&s, &arms).unwrap();
        fn names(w: &[R2Tree]) -> Vec<&str> {
            w.iter().map(|t| t.dest.as_str()).collect()
        }
        let at = |a: &str| p.arm(a).unwrap();
        assert_eq!(names(&at("a").fetch), ["big"]);
        assert!(at("b").fetch.is_empty() && at("b").drop.is_empty());
        assert!(at("c").fetch.is_empty() && at("c").drop.is_empty());
        assert_eq!(names(&at("d").drop), ["big"]);
        assert_eq!(names(&at("d").fetch), ["small"]);
        assert_eq!(names(&at("e").fetch), ["loras/a"]);
        assert!(at("e").drop.is_empty());
        assert_eq!(names(&at("f").drop), ["loras/a", "small"]);
        assert_eq!(names(&at("f").fetch), ["big", "big/nested"]);
        assert_eq!(p.peak_bytes, 100_000_000_007);
        assert_eq!(
            p.fetch_bytes,
            2 * 100_000_000_000 + 1_000_000_000 + 500_000_000 + 7
        );
        assert_eq!(p.fetched_trees().len(), 4);
    }

    #[test]
    fn a_held_parent_of_a_needed_tree_is_kept() {
        let s = sources();
        let held: BTreeSet<R2Tree> = [s.r2["big"].clone(), s.r2["small"].clone()].into();
        // Reading only the nested tree is refused by arm_r2; step itself
        // never drops a held parent of what it reads.
        let need: BTreeSet<R2Tree> = [s.r2["big/nested"].clone()].into();
        let st = step(&held, &need);
        assert_eq!(st.drop, [s.r2["small"].clone()]);
    }

    #[test]
    fn the_key_is_read_whole_never_shown_and_refs_are_references() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| {
                pairs
                    .iter()
                    .find(|(a, _)| *a == k)
                    .map(|(_, b)| (*b).to_owned())
            }
        };
        let k = R2Key::from_env(env(&[
            ("FV_WEIGHTS_R2_ACCESS_KEY_ID", "id1"),
            ("FV_WEIGHTS_R2_SECRET_ACCESS_KEY", "sec1"),
        ]))
        .unwrap();
        assert_eq!(format!("{k:?}"), "R2Key(<redacted>)");
        assert_eq!(k.env()[1], ("FV_WEIGHTS_R2_SECRET_ACCESS_KEY", "sec1"));
        assert!(R2Key::from_env(env(&[("FV_WEIGHTS_R2_ACCESS_KEY_ID", "id1")])).is_none());
        assert!(R2Key::from_env(env(&[
            (
                "FV_WEIGHTS_R2_ACCESS_KEY_ID",
                "{{ RUNPOD_SECRET_fv_r2_access_key_id }}"
            ),
            ("FV_WEIGHTS_R2_SECRET_ACCESS_KEY", "sec1"),
        ]))
        .is_none());
        assert!(R2Key::from_env(env(&[
            ("FV_WEIGHTS_R2_ACCESS_KEY_ID", " "),
            ("FV_WEIGHTS_R2_SECRET_ACCESS_KEY", "sec1"),
        ]))
        .is_none());
        let refs = r2_env_refs();
        assert_eq!(
            refs["FV_WEIGHTS_R2_ACCESS_KEY_ID"],
            "{{ RUNPOD_SECRET_fv_r2_access_key_id }}"
        );
        assert!(refs.values().all(|v| v.starts_with("{{ RUNPOD_SECRET_")));
        assert_eq!(gb(147_857_050_891), "147.86 GB");
    }
}
