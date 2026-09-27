//! LTX-2 on the technique layer: the stage-2 attention route (dense, the
//! LTX-2.5 Sol route, the LTX-2.3 PISA route) from the command line, the
//! active profile and the recipe default, plus the load-time techniques
//! (precision, TAEHV, offload) through their settings.
//!
//! | seam | command line (wins) | profile | default |
//! |---|---|---|---|
//! | stage-2 attention_backend | `--sol-stage2` / `--dense-stage2` / `--pisa-stage2` | `sol_attn` (`ltx25_stage2`) / `dense_attention` / `pisa` | Sol on the 2.5 distilled two-stage 3-forward refine (`default_sol_stage2`) |
//! | ffn_precision | `FASTVIDEO_FP8`, `FASTVIDEO_NVFP4` | `fp8` / `nvfp4` (video FFN) / `bf16_linears` | bf16 |
//! | video_decoder | `--ltx-tae-weights`, `FASTVIDEO_LTX2_TAE_WEIGHTS` | `taehv` | conv VAE |
//! | residency | `--offload`, `--dit-offload`, `FASTVIDEO_LTX_OFFLOAD`, `FASTVIDEO_DIT_OFFLOAD` | `offload.placement` / `.dit` | none / auto |
//!
//! The stage-2 kernels implement exactly the published routes
//! (`ltx2::sol::route`, `ltx2::pisa::route`), so a profile's `sol_attn` /
//! `pisa` must describe that route; a different one is refused rather than
//! silently run as the published one.

use super::sol::LAYERS_PER_FORWARD;
use crate::techniques::methods::{DenseAttention, Pisa, SinkMode, SolAttn};
use crate::techniques::registry::ltx2_spec;
use crate::techniques::schedule::Route;
use crate::techniques::{compose, Plan, Profile, Schedule, Seam, Technique, HORIZON};

/// Stage-2 video self-attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ltx2Stage2 {
    Dense,
    Sol,
    Pisa,
}

/// The command line's stage-2 flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stage2Flags {
    pub sol: bool,
    pub dense: bool,
    pub pisa: bool,
}

impl Stage2Flags {
    pub fn any(self) -> bool {
        self.sol || self.dense || self.pisa
    }
}

#[derive(Debug)]
pub struct Ltx2Techniques {
    pub stage2: Ltx2Stage2,
    /// `Ltx2Request::sol_stage2` / `pisa_stage2`. Both can be set from the
    /// command line, which the request then refuses, as before.
    sol: bool,
    pisa: bool,
    pub plan: Plan,
    pub sources: Vec<(&'static str, &'static str)>,
}

/// The published stage-2 routes as techniques.
pub fn published_pisa() -> Pisa {
    Pisa {
        enabled: Schedule::Const(true),
        sparsity: Some(super::pisa::SPARSITY),
        dense_layers: Some(crate::techniques::StepSet::first(super::pisa::DENSE_LAYERS.len())),
    }
}

fn same_route(a: &SolAttn, b: &SolAttn) -> bool {
    (0..super::sol::STAGE2_TAUS.len())
        .all(|s| (0..LAYERS_PER_FORWARD).all(|l| a.route.at(s, l) == b.route.at(s, l)))
}

impl Ltx2Techniques {
    /// `recipe_default`: `default_sol_stage2(cfg, two_stage, refine, false,
    /// false)`, i.e. whether this workload's reference runs Sol stage 2.
    pub fn resolve(
        flags: Stage2Flags,
        recipe_default: bool,
        profile: Option<&Profile>,
    ) -> Result<Self, String> {
        if let Some(p) = profile {
            if p.model != "ltx2" {
                return Err(format!(
                    "technique profile {} is for {}, not ltx2",
                    p.name, p.model
                ));
            }
        }
        let mut items: Vec<Box<dyn Technique>> = Vec::new();
        let mut p_attention: Option<Box<dyn Technique>> = None;
        if let Some(p) = profile {
            for t in p.plan()?.techniques {
                if t.writes().contains(&Seam::AttentionBackend) {
                    if t.enabled().as_const().is_none() {
                        return Err(format!(
                            "ltx2: '{}' is enabled on some steps only; the stage-2 route is per request",
                            t.name()
                        ));
                    }
                    p_attention = Some(t);
                } else {
                    items.push(t);
                }
            }
        }
        let mut sources = Vec::new();
        let legacy = (
            flags.sol || (recipe_default && !flags.pisa && !flags.dense),
            flags.pisa,
        );
        let stage2 = if flags.any() {
            // The legacy resolution: `sol || default_sol_stage2(.., pisa, dense)`.
            sources.push(("attention_backend", "command line"));
            match legacy {
                (true, _) => Ltx2Stage2::Sol,
                (false, true) => Ltx2Stage2::Pisa,
                (false, false) => Ltx2Stage2::Dense,
            }
        } else if let Some(t) = p_attention {
            sources.push(("attention_backend", "profile"));
            if let Some(sol) = t.downcast_ref::<SolAttn>() {
                if sol.sink != SinkMode::None || !same_route(sol, &SolAttn::ltx25_stage2()) {
                    return Err(format!(
                        "ltx2: techniques.sol_attn must be the published stage-2 route (preset \"ltx25_stage2\": layer 0 dense, taus 1 / 1.25 / 1.5, no sink); got {} sink {}",
                        sol.route.describe(),
                        sol.sink.as_str()
                    ));
                }
                Ltx2Stage2::Sol
            } else if t.downcast_ref::<DenseAttention>().is_some() {
                Ltx2Stage2::Dense
            } else if let Some(p) = t.downcast_ref::<Pisa>() {
                let want = published_pisa();
                let ok = p.sparsity.is_none_or(|s| Some(s) == want.sparsity)
                    && p.dense_layers.as_ref().is_none_or(|d| Some(d) == want.dense_layers.as_ref());
                if !ok {
                    return Err("ltx2: techniques.pisa must be the published stage-2 route (sparsity 0.9, layers 0-1 dense)".into());
                }
                Ltx2Stage2::Pisa
            } else {
                return Err(format!(
                    "ltx2: attention technique '{}' is not implemented for LTX-2",
                    t.name()
                ));
            }
        } else {
            sources.push(("attention_backend", "recipe"));
            if recipe_default {
                Ltx2Stage2::Sol
            } else {
                Ltx2Stage2::Dense
            }
        };
        match stage2 {
            Ltx2Stage2::Sol => items.push(Box::new(SolAttn::ltx25_stage2())),
            Ltx2Stage2::Pisa => items.push(Box::new(published_pisa())),
            Ltx2Stage2::Dense => {}
        }
        let plan = compose(items, &ltx2_spec(), HORIZON).map_err(|e| e.to_string())?;
        let (sol, pisa) = if flags.any() {
            legacy
        } else {
            (stage2 == Ltx2Stage2::Sol, stage2 == Ltx2Stage2::Pisa)
        };
        Ok(Self {
            stage2,
            sol,
            pisa,
            plan,
            sources,
        })
    }

    /// With the process's active profile.
    pub fn from_process(flags: Stage2Flags, recipe_default: bool) -> Result<Self, String> {
        let active = crate::techniques::settings::active();
        Self::resolve(flags, recipe_default, active.profile.as_ref())
    }

    pub fn sol_stage2(&self) -> bool {
        self.sol
    }

    pub fn pisa_stage2(&self) -> bool {
        self.pisa
    }

    pub fn describe(&self) -> String {
        let src: Vec<String> = self.sources.iter().map(|(s, f)| format!("{s} from {f}")).collect();
        format!("{} [{}]", self.plan.describe(), src.join(", "))
    }
}

/// The stage-2 Sol technique's route equals the published clock.
pub fn technique_route(step: usize, layer: usize) -> Route {
    SolAttn::ltx25_stage2().route.at(step, layer)
}

#[cfg(test)]
mod tests {
    use super::super::sol::{route, Ltx25SolRoute};
    use super::*;

    const ALL: [Stage2Flags; 8] = [
        Stage2Flags { sol: false, dense: false, pisa: false },
        Stage2Flags { sol: true, dense: false, pisa: false },
        Stage2Flags { sol: false, dense: true, pisa: false },
        Stage2Flags { sol: false, dense: false, pisa: true },
        Stage2Flags { sol: true, dense: true, pisa: false },
        Stage2Flags { sol: true, dense: false, pisa: true },
        Stage2Flags { sol: false, dense: true, pisa: true },
        Stage2Flags { sol: true, dense: true, pisa: true },
    ];

    /// No profile: `sol = flag || default_sol_stage2(.., pisa, dense)` and
    /// `pisa = flag`, exactly as the callers computed it.
    #[test]
    fn no_profile_is_the_legacy_resolution() {
        for flags in ALL {
            for default in [false, true] {
                let t = Ltx2Techniques::resolve(flags, default, None).unwrap();
                let legacy_sol = flags.sol || (default && !flags.pisa && !flags.dense);
                assert_eq!(t.sol_stage2(), legacy_sol, "{flags:?} {default}");
                assert_eq!(t.pisa_stage2(), flags.pisa, "{flags:?} {default}");
            }
        }
    }

    #[test]
    fn the_stage2_preset_is_the_published_route() {
        for step in 0..3 {
            for layer in 0..LAYERS_PER_FORWARD {
                let want = match route(step, layer).unwrap() {
                    Ltx25SolRoute::Dense => Route::Dense,
                    Ltx25SolRoute::Sol { tau } => Route::Sparse { tau },
                };
                assert_eq!(technique_route(step, layer), want, "{step} {layer}");
            }
        }
    }

    fn profile(body: &str) -> Profile {
        Profile::parse(&format!("[id]\nname='p'\n[pipeline]\nmodel='ltx2'\n{body}")).unwrap()
    }

    #[test]
    fn profiles_pick_the_route_and_the_command_line_wins() {
        let dense = profile("[techniques.dense_attention]\n");
        let t = Ltx2Techniques::resolve(Stage2Flags::default(), true, Some(&dense)).unwrap();
        assert_eq!(t.stage2, Ltx2Stage2::Dense);
        let flags = Stage2Flags { sol: true, ..Default::default() };
        assert!(Ltx2Techniques::resolve(flags, true, Some(&dense)).unwrap().sol_stage2());
        let sol = profile("[techniques.sol_attn]\npreset='ltx25_stage2'\n");
        assert!(Ltx2Techniques::resolve(Stage2Flags::default(), false, Some(&sol)).unwrap().sol_stage2());
        let pisa = profile("[techniques.pisa]\n");
        assert!(Ltx2Techniques::resolve(Stage2Flags::default(), true, Some(&pisa)).unwrap().pisa_stage2());
        // A route the kernels do not implement is refused.
        let other = profile("[techniques.sol_attn]\npreset='ltx25_stage2'\ndense_layers=2\n");
        assert!(Ltx2Techniques::resolve(Stage2Flags::default(), true, Some(&other)).is_err());
        let h3 = profile("[techniques.sol_attn]\npreset='rtx'\n");
        assert!(Ltx2Techniques::resolve(Stage2Flags::default(), true, Some(&h3)).is_err());
        // No step cache on LTX-2.
        assert!(Profile::parse("[id]\nname='p'\n[pipeline]\nmodel='ltx2'\n[techniques.teacache]\n").is_err());
    }

    #[test]
    fn disabled_stage2_techniques_are_the_baseline() {
        let off = profile("[techniques.sol_attn]\nenabled=false\n[techniques.fp8]\nenabled=false\n[techniques.taehv]\nenabled=false\n");
        for default in [false, true] {
            let a = Ltx2Techniques::resolve(Stage2Flags::default(), default, None).unwrap();
            let b = Ltx2Techniques::resolve(Stage2Flags::default(), default, Some(&off)).unwrap();
            assert_eq!(a.stage2, b.stage2);
            assert_eq!(a.plan.names(), b.plan.names());
        }
        assert!(off.settings().unwrap().is_empty());
    }

    #[test]
    fn ltx_settings_are_the_ltx_flags() {
        let p = profile(
            "[techniques.fp8]\n[techniques.taehv]\nweights='/tae/taeltx2_3_wide.safetensors'\n[techniques.offload]\nplacement='cpu'\n",
        );
        let got: Vec<(String, String)> = p
            .settings()
            .unwrap()
            .iter()
            .map(|(k, v, _)| (k.to_string(), v.to_string()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("FASTVIDEO_FP8".into(), "1".into()),
                ("FASTVIDEO_LTX2_TAE_WEIGHTS".into(), "/tae/taeltx2_3_wide.safetensors".into()),
                ("FASTVIDEO_LTX_OFFLOAD".into(), "cpu".into()),
            ]
        );
        assert!(Profile::parse("[id]\nname='p'\n[pipeline]\nmodel='h3'\n[techniques.offload]\nplacement='cpu'\n")
            .and_then(|p| p.settings())
            .is_err());
        assert!(Profile::parse("[id]\nname='p'\n[pipeline]\nmodel='ltx2'\n[techniques.taeh3]\n")
            .and_then(|p| p.settings())
            .is_err());
    }
}
