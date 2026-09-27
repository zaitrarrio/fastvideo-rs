//! Technique composition layer: acceleration techniques as typed, composable,
//! conflict-checked items instead of per-model env flags and if/else chains.
//!
//! Modelled on sol-engine `techniques/` (NVlabs/Sana, branch `sol-engine`
//! @ 6c2f582, the checkout `scripts/gpu/upstream/setup.sh` pins):
//!
//! | here | sol-engine |
//! |---|---|
//! | [`technique`]: [`Technique`], [`Phase`], [`Seam`], [`Capability`], [`ModelSpec`] | `technique.py`, `transform.py`, `spec.py` |
//! | [`schedule`]: step sets, [`Schedule`], the `(step, layer)` [`SparseRoute`] | `schedule.py` |
//! | [`compose`]: capability + seam-conflict check, ordered [`Plan`] | `compose.py` |
//! | [`registry`]: `[techniques.<name>]` factories, model specs | `registry.py` |
//! | [`profile`]: typed TOML profiles, sol-engine config import | `config_manifest.py`, `config/*/*.toml` |
//! | [`settings`]: env > profile > default for every `FASTVIDEO_*` knob | `TransformContext.env` / `set_env` |
//! | [`kernels`]: [`KernelBackend`]s (nvcc, oxide, cuDNN, cuBLAS) per op | — (sol-engine picks backends by env) |
//!
//! The model adapters (`fastvideo_models::h3::techniques` for H3) turn a
//! profile plus the legacy env flags plus the recipe's defaults into a
//! [`Plan`], and the pipelines read their seams from it. See
//! `docs/techniques.md`.

pub mod builtin;
pub mod compose;
pub mod kernels;
pub mod methods;
pub mod profile;
pub mod registry;
pub mod schedule;
pub mod settings;
pub mod technique;

pub use compose::{compose, CompositionError, Plan, HORIZON};
pub use kernels::{KernelBackend, KernelChoice, KernelOp, Provider};
pub use profile::Profile;
pub use schedule::{Route, Schedule, SparseRoute, StepSet};
pub use technique::{Capability, Kind, ModelSpec, Phase, Seam, Technique, TransformPhase};

#[cfg(test)]
mod tests {
    use super::methods::*;
    use super::registry::{h3_spec, ltx2_spec};
    use super::*;

    fn on() -> Schedule<bool> {
        Schedule::Const(true)
    }

    #[test]
    fn two_attention_backends_are_a_conflict() {
        let items: Vec<Box<dyn Technique>> = vec![
            Box::new(SolAttn::rtx()),
            Box::new(Vsa {
                enabled: on(),
                sparsity: None,
                group: None,
            }),
        ];
        let e = compose(items, &h3_spec(), HORIZON).unwrap_err().to_string();
        assert!(
            e.contains("exclusive seam 'attention_backend' has multiple active writers: [\"sol_attn\", \"vsa\"]"),
            "{e}"
        );
    }

    #[test]
    fn two_precisions_and_two_decoders_conflict() {
        let p = Profile::parse(
            "[id]\nname='x'\n[pipeline]\nmodel='h3'\n[techniques.mxfp8]\n[techniques.w8a8]\n",
        )
        .unwrap_err();
        assert!(p.contains("ffn_precision"), "{p}");
        let p = Profile::parse(
            "[id]\nname='x'\n[pipeline]\nmodel='h3'\n[techniques.taeh3]\n[techniques.taehv]\n",
        )
        .unwrap_err();
        assert!(p.contains("video_decoder"), "{p}");
    }

    #[test]
    fn a_disabled_technique_is_not_in_the_plan_and_conflicts_with_nothing() {
        let p = Profile::parse(
            "[id]\nname='x'\n[pipeline]\nmodel='h3'\n[techniques.sol_attn]\npreset='rtx'\n[techniques.vsa]\nenabled=false\n",
        )
        .unwrap();
        assert_eq!(p.plan().unwrap().names(), vec!["sol_attn"]);
        let off = Profile::parse(
            "[id]\nname='x'\n[pipeline]\nmodel='h3'\n[techniques.teacache]\nenabled=false\n[techniques.bf16_activations]\nenabled=false\n",
        )
        .unwrap();
        assert!(off.plan().unwrap().techniques.is_empty());
        assert!(off.settings().unwrap().is_empty(), "OFF installs nothing");
    }

    #[test]
    fn runtime_techniques_on_disjoint_steps_share_an_exclusive_seam() {
        // compose.py:71-75: two runtime writers clash only when co-active.
        let a = TeaCache {
            enabled: Schedule::Before {
                n: 4,
                value: true,
                then: false,
            },
            ..TeaCache::rtx()
        };
        let b = TeaCache {
            enabled: Schedule::Before {
                n: 4,
                value: false,
                then: true,
            },
            ..TeaCache::rtx()
        };
        assert!(check_ok(vec![Box::new(a.clone()), Box::new(b)]));
        assert!(!check_ok(vec![Box::new(a.clone()), Box::new(a)]));
    }

    fn check_ok(items: Vec<Box<dyn Technique>>) -> bool {
        compose::check_conflicts(&items, HORIZON).is_empty()
    }

    #[test]
    fn capabilities_are_checked() {
        // H3 has no NVFP4 linear path; LTX-2 has no step cache.
        let e = Profile::parse("[id]\nname='x'\n[pipeline]\nmodel='h3'\n[techniques.nvfp4]\n")
            .unwrap_err();
        assert!(e.contains("SupportsNvfp4Linear"), "{e}");
        let items: Vec<Box<dyn Technique>> = vec![Box::new(TeaCache::rtx())];
        assert!(compose(items, &ltx2_spec(), HORIZON).is_err());
    }

    #[test]
    fn plans_order_load_then_build_then_runtime() {
        let p = Profile::parse(
            "[id]\nname='x'\n[pipeline]\nmodel='h3'\n[techniques.teacache]\n[techniques.sol_attn]\n[techniques.mxfp8]\n",
        )
        .unwrap();
        assert_eq!(
            p.plan().unwrap().names(),
            vec!["mxfp8", "sol_attn", "teacache"]
        );
    }

    #[test]
    fn unknown_keys_and_names_are_errors() {
        assert!(
            Profile::parse("[id]\nname='x'\n[pipeline]\nmodel='h3'\n[techniques.sol_atn]\n")
                .is_err()
        );
        assert!(Profile::parse(
            "[id]\nname='x'\n[pipeline]\nmodel='h3'\n[techniques.sol_attn]\ntua=1.0\n"
        )
        .is_err());
        assert!(
            Profile::parse("[id]\nname='x'\n[pipeline]\nmodel='h3'\n[kernels]\nflash='v2'\n")
                .is_err()
        );
        assert!(
            Profile::parse("[id]\nname='x'\n[pipeline]\nmodel='h3'\n[env]\nPATH='x'\n").is_err()
        );
        assert!(Profile::parse("[id]\nname='x'\n[pipeline]\nmodel='h3'\nsteps=4\n").is_err());
    }

    #[test]
    fn settings_are_what_the_legacy_flags_were() {
        let p = Profile::parse(
            "[id]\nname='x'\n[pipeline]\nmodel='h3'\n\
             [techniques.w8a8]\n[techniques.bf16_activations]\n\
             [techniques.offload]\ndit='streamed'\nlookahead=2\n\
             [techniques.kernel_fusion]\nh3=false\n\
             [kernels]\ndense_attention='cudnn'\nsol_attention='nvcc:x4f'\n\
             [env]\nFASTVIDEO_VSA_TMA='0'\n",
        )
        .unwrap();
        let s = p.settings().unwrap();
        let got: Vec<(&str, &str)> = s.iter().map(|(k, v, _)| (k, v)).collect();
        assert_eq!(
            got,
            vec![
                ("FASTVIDEO_BF16_ACT", "1"),
                ("FASTVIDEO_DIT_OFFLOAD", "streamed"),
                ("FASTVIDEO_DIT_OFFLOAD_LOOKAHEAD", "2"),
                ("FASTVIDEO_FLASH_KERNEL", "cudnn"),
                ("FASTVIDEO_H3_FUSE", "0"),
                ("FASTVIDEO_H3_QUANT", "w8a8"),
                ("FASTVIDEO_SOL_KERNEL", "x4f"),
                ("FASTVIDEO_VSA_TMA", "0"),
            ]
        );
        let clash = Profile::parse(
            "[id]\nname='x'\n[pipeline]\nmodel='h3'\n[techniques.w8a8]\n[env]\nFASTVIDEO_H3_QUANT='mxfp8'\n",
        )
        .unwrap();
        assert!(clash.settings().is_err());
    }
}
