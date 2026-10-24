//! The H3 and LTX-2 load lifecycle on the CPU (`docs/adr/0017-loader-stages.md`):
//! plan → open → place → fuse without a GPU.
//!
//! * **plan goldens**: for the served configurations (H3 `4step-vsa`, `sol-h3`,
//!   `8step`, the RTX `fullopt` cell; LTX-2.5 distilled, two-stage, dev
//!   audio-to-video, reference-to-video, the offload placements) on the
//!   devices they run on, the plan's decisions (what loads, the quant mode,
//!   the offload, the adapter and step-cache fusions) are pinned in
//!   `tests/goldens/{h3,ltx2}_load_plan.txt`. A change to a decision is a
//!   change to the golden, reviewed. `FV_GOLDEN_BLESS=1` prints the current
//!   text instead of comparing.
//! * **open**: H3's plan and open on a tiny synthetic snapshot (the reference
//!   fixtures under `src/h3/fixtures/`).
//! * **the full lifecycle**: LTX-2's plan → open → place → fuse on the tiny
//!   fixtures (`src/ltx2/fixtures/`) at a tiny config, to a pipeline. H3's
//!   place needs the production dims (`transformer_config` refuses any
//!   other), so H3's place and fuse run on the GPU cells only; and neither
//!   family's generate runs on the CPU (their text encoders are the
//!   production Qwen3-VL / Gemma configurations).

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Instant;

use fastvideo_cudarc::h3::pipeline::{
    H3LoadPlan, H3PipelineOptions, I2vEncoderChoice, TextEncoderChoice,
};
use fastvideo_cudarc::ltx2::pipeline::{
    Ltx2LoadPlan, Ltx2Paths, LtxOffload, PipelineOptions, TextResidency,
};
use fastvideo_cudarc::rt::load_stages::DeviceFacts;
use fastvideo_models::settings::{Settings, Snapshot};
use fastvideo_models::techniques::settings::Active;
use fastvideo_models::techniques::Profile;

const GIB: u64 = 1 << 30;

/// An RTX PRO 6000 (sm_120) as a load finds it: 95 GiB free.
fn rtx_pro_6000() -> DeviceFacts {
    DeviceFacts::device((12, 0), 95 * GIB)
}

/// A 32 GiB card of the same generation (an RTX 5090).
fn card_32g() -> DeviceFacts {
    DeviceFacts::device((12, 0), 30 * GIB)
}

/// A fresh, empty directory for one test.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fv-b6-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    dir
}

fn snapshot(env: &[(&str, &str)], table: &Settings) -> Snapshot {
    let env: Vec<(String, OsString)> = env
        .iter()
        .map(|(k, v)| (k.to_string(), OsString::from(v)))
        .collect();
    Snapshot::from_parts(
        &|k| env.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone()),
        table,
    )
}

fn compare(name: &str, got: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/goldens")
        .join(name);
    if std::env::var("FV_GOLDEN_BLESS").ok().as_deref() == Some("1") {
        eprintln!("==== {name} ====\n{got}==== end {name} ====");
        return;
    }
    let want = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read golden {}: {e}\n---- got ----\n{got}", path.display()));
    if got != want {
        for (i, (g, w)) in got.lines().zip(want.lines()).enumerate() {
            assert_eq!(g, w, "{name} line {}\n---- got ----\n{got}", i + 1);
        }
        panic!(
            "{name}: {} lines, golden {}\n---- got ----\n{got}",
            got.lines().count(),
            want.lines().count()
        );
    }
}

/// One H3 case: the options a served model (or a `techniques` cell) loads
/// with, its own profile if any, the process environment, the device.
struct H3Case {
    name: &'static str,
    recipe: &'static str,
    profile: Option<&'static str>,
    env: &'static [(&'static str, &'static str)],
    facts: DeviceFacts,
}

fn h3_plan_text(root: &Path, case: &H3Case) -> String {
    let profile = case
        .profile
        .map(|p| Profile::load(Path::new(p)).unwrap_or_else(|e| panic!("{p}: {e}")));
    // fv-serve loads a model under its own profile's settings
    // (`settings::scope_model`): the snapshot's table is the profile's.
    let table = profile
        .as_ref()
        .map(|p| p.settings().unwrap_or_else(|e| panic!("{e}")))
        .unwrap_or_default();
    let options = H3PipelineOptions {
        recipe: Some(case.recipe.to_owned()),
        text_encoder: TextEncoderChoice::Auto,
        i2v_encoder: I2vEncoderChoice::Auto,
        profile: case.profile.map(|_| profile.clone()),
        ..Default::default()
    };
    let plan = H3LoadPlan::new(
        root,
        options,
        &snapshot(case.env, &table),
        &Active::default(),
        case.facts,
    );
    let body = match plan {
        Ok(p) => p.describe(),
        Err(e) => format!("error: {e}\n"),
    };
    format!("== {} ==\n{body}", case.name)
}

#[test]
fn h3_plans_are_the_golden_decisions() {
    let root = scratch("h3-plan");
    let cases = [
        H3Case {
            name: "fasth3-4step-vsa (served: profile h3/fasth3_4step_vsa) on an RTX PRO 6000",
            recipe: "4step-vsa",
            profile: Some("h3/fasth3_4step_vsa"),
            env: &[],
            facts: rtx_pro_6000(),
        },
        H3Case {
            name: "fasth3-4step-vsa-768p (techniques cell, -env arm) on an RTX PRO 6000",
            recipe: "4step-vsa",
            profile: None,
            env: &[],
            facts: rtx_pro_6000(),
        },
        H3Case {
            name: "fasth3-4step-vsa on the CPU",
            recipe: "4step-vsa",
            profile: None,
            env: &[],
            facts: DeviceFacts::cpu(),
        },
        H3Case {
            name: "sol-h3 (served: profile h3/sol_h3_4step_engine_ladder) on an RTX PRO 6000",
            recipe: "sol-h3",
            profile: Some("h3/sol_h3_4step_engine_ladder"),
            env: &[],
            facts: rtx_pro_6000(),
        },
        H3Case {
            name: "fasth3-8step on an RTX PRO 6000",
            recipe: "8step",
            profile: None,
            env: &[],
            facts: rtx_pro_6000(),
        },
        H3Case {
            name: "fasth3-8step on a 32 GiB card",
            recipe: "8step",
            profile: None,
            env: &[],
            facts: card_32g(),
        },
        H3Case {
            name: "fasth3-8step, FASTVIDEO_DIT_OFFLOAD=streamed, FASTVIDEO_H3_QUANT=w8a8",
            recipe: "8step",
            profile: None,
            env: &[
                ("FASTVIDEO_DIT_OFFLOAD", "streamed"),
                ("FASTVIDEO_H3_QUANT", "w8a8"),
            ],
            facts: rtx_pro_6000(),
        },
        H3Case {
            name: "h3-480p-fullopt (techniques cell, -env arm: sol-h3-rtx + FASTVIDEO_H3_SOL_CACHE=teacache)",
            recipe: "sol-h3-rtx",
            profile: None,
            env: &[("FASTVIDEO_H3_SOL_CACHE", "teacache")],
            facts: rtx_pro_6000(),
        },
        H3Case {
            name: "h3-480p-fullopt (techniques cell, -prof arm: profile h3/rtx5090_fullopt + FASTVIDEO_H3_QUANT=mxfp8)",
            recipe: "sol-h3-rtx",
            profile: Some("h3/rtx5090_fullopt"),
            env: &[("FASTVIDEO_H3_QUANT", "mxfp8")],
            facts: rtx_pro_6000(),
        },
    ];
    let got: String = cases.iter().map(|c| h3_plan_text(&root, c)).collect();
    compare("h3_load_plan.txt", &got);
    let _ = std::fs::remove_dir_all(&root);
}

fn fixture(family: &str, name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join(family)
        .join("fixtures")
        .join(name)
}

/// `dir/<sub>/diffusion_pytorch_model.safetensors`, a copy of a fixture.
fn component(dir: &Path, sub: &str, family: &str, fixture_name: &str) {
    let d = dir.join(sub);
    std::fs::create_dir_all(&d).unwrap_or_else(|e| panic!("{}: {e}", d.display()));
    std::fs::copy(
        fixture(family, fixture_name),
        d.join("diffusion_pytorch_model.safetensors"),
    )
    .unwrap_or_else(|e| panic!("{family}/{fixture_name}: {e}"));
}

#[test]
fn h3_plan_and_open_run_on_the_cpu_on_a_tiny_snapshot() {
    let root = scratch("h3-open");
    component(&root, "transformer", "h3", "dit_tiny.st");
    component(&root, "vae", "h3", "vae_tiny.st");
    component(&root, "audio_vae", "h3", "audio_vae_tiny.st");
    let options = H3PipelineOptions {
        recipe: Some("8step".into()),
        text_encoder: TextEncoderChoice::Streamed,
        ..Default::default()
    };
    let settings = snapshot(&[], &Settings::default());
    let _scope = settings.scope();
    let plan = H3LoadPlan::new(
        &root,
        options,
        &settings,
        &Active::default(),
        DeviceFacts::cpu(),
    )
    .expect("plan");
    assert!(!plan.load_text_encoder, "a streamed encoder opens nothing");
    assert!(plan.dit_tree.is_none());
    let opened = plan.open().expect("open");
    assert!(opened.text_map().is_none());
    assert!(!opened.has_dit_tree());
    // The DiT's map is the tiny transformer, the decoders' maps their tiny
    // checkpoints: the place stage reads these mappings.
    assert!(opened
        .dit_map()
        .has_tensor("token_refiner.refiner_blocks.0.attn.to_q.weight"));
    let vae = opened.vae_map().expect("the video decoder's map");
    let audio = opened.audio_vae_map().expect("the audio decoder's map");
    assert!(vae.has_tensor("ref.video") && audio.has_tensor("ref.conv_pre"));
    let _ = std::fs::remove_dir_all(&root);
}

fn ltx_tiny_config() -> fastvideo_models::ltx2::config::Ltx2Config {
    use fastvideo_models::ltx2::config::{
        Ltx2AudioVaeConfig, Ltx2TransformerConfig, Ltx2VideoVaeConfig, Ltx2VocoderConfig,
    };
    // The configs of `src/ltx2/reference_tests.rs` (and
    // `scripts/gpu/ltx2_tiny_reference.py`), which made the fixtures.
    let mut cfg = fastvideo_models::ltx2::config::ltx2_19b_distilled();
    cfg.transformer = Ltx2TransformerConfig {
        in_channels: 6,
        out_channels: 6,
        num_attention_heads: 4,
        attention_head_dim: 8,
        cross_attention_dim: 32,
        audio_in_channels: 5,
        audio_out_channels: 5,
        audio_num_attention_heads: 4,
        audio_attention_head_dim: 4,
        audio_cross_attention_dim: 16,
        num_layers: 2,
        caption_channels: 12,
        ..Ltx2TransformerConfig::ltx2_19b()
    };
    cfg.vae = Ltx2VideoVaeConfig {
        latent_channels: 4,
        decoder_block_out_channels: vec![8, 16, 32],
        decoder_layers_per_block: vec![1, 1, 1, 1],
        patch_size: 2,
        ..Ltx2VideoVaeConfig::ltx2_19b()
    };
    cfg.audio_vae = Ltx2AudioVaeConfig {
        base_channels: 4,
        num_res_blocks: 1,
        latent_channels: 2,
        mel_bins: 8,
        ..Ltx2AudioVaeConfig::ltx2_19b()
    };
    cfg.vocoder = Ltx2VocoderConfig {
        in_channels: 16,
        hidden_channels: 64,
        upsample_kernel_sizes: vec![7, 4, 4, 4, 4],
        upsample_factors: vec![3, 2, 2, 2, 2],
        ..Ltx2VocoderConfig::ltx2_19b()
    };
    cfg
}

#[test]
fn ltx2_lifecycle_plan_open_place_fuse_runs_on_the_cpu() {
    let weights = scratch("ltx2-lifecycle");
    component(&weights, "transformer", "ltx2", "dit_tiny.st");
    component(&weights, "vae", "ltx2", "vae_tiny.st");
    component(&weights, "audio_vae", "ltx2", "audio_vae_tiny.st");
    component(&weights, "vocoder", "ltx2", "vocoder_tiny.st");
    let cfg = ltx_tiny_config();
    let paths = Ltx2Paths {
        weights: weights.clone(),
        dit: weights.clone(),
        text: None,
    };
    let options = PipelineOptions {
        text_residency: TextResidency::Streamed,
        ..Default::default()
    };
    // No Gemma in the tiny snapshot: skip its read-ahead.
    let settings = snapshot(&[("FASTVIDEO_LTX2_TEXT_WARM", "0")], &Settings::default());
    let _scope = settings.scope();
    let plan = Ltx2LoadPlan::new(
        &paths,
        &cfg,
        &options,
        &settings,
        &Active::default(),
        DeviceFacts::cpu(),
        Instant::now(),
    )
    .expect("plan");
    assert!(
        !plan.residency.is_streamed(),
        "no device: nothing to stream around"
    );
    let opened = plan.open().expect("open");
    assert!(opened
        .dit_map()
        .expect("a distilled DiT opens at load")
        .has_tensor("transformer_blocks.0.attn1.to_q.weight"));
    let placed = plan.place(opened).expect("place");
    assert!(placed.has_dit() && placed.has_decoders());
    let pipeline = plan.fuse(placed).expect("fuse");
    assert!(!pipeline.residency().is_streamed());
    assert_eq!(pipeline.ic_lora_downscale(), None);
    let _ = std::fs::remove_dir_all(&weights);
}

/// A safetensors file with one `lora_A` key and `metadata`.
fn tiny_lora(path: &Path, metadata: &str) {
    let header = format!(
        r#"{{"__metadata__":{metadata},"transformer_blocks.0.attn1.to_q.lora_A.weight":{{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]}}}}"#
    );
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&0f32.to_le_bytes());
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

struct LtxCase {
    name: &'static str,
    dev: bool,
    options: PipelineOptions,
    env: &'static [(&'static str, &'static str)],
    facts: DeviceFacts,
}

#[test]
fn ltx2_plans_are_the_golden_decisions() {
    let weights = scratch("ltx2-plan");
    // A two-stage snapshot carries the spatial upsampler; the dev
    // transformer's distilled LoRA sits beside the weights.
    std::fs::create_dir_all(weights.join("latent_upsampler")).expect("upsampler dir");
    std::fs::write(
        weights.join("ltx-2.5-22b-distilled-lora-450-bf16.safetensors"),
        b"",
    )
    .expect("lora");
    let ic = weights.join("ltx-2.5-22b-ic-lora-ingredients.safetensors");
    tiny_lora(&ic, r#"{"reference_downscale_factor":"2"}"#);
    let paths = Ltx2Paths {
        weights: weights.clone(),
        dit: weights.clone(),
        text: None,
    };
    let streamed_text = PipelineOptions {
        text_residency: TextResidency::Streamed,
        ..Default::default()
    };
    let cases = [
        LtxCase {
            name: "ltx25-distill two-stage (served ltx25-distill-sol) on an RTX PRO 6000",
            dev: false,
            options: streamed_text.clone(),
            env: &[],
            facts: rtx_pro_6000(),
        },
        LtxCase {
            name: "ltx25-distill two-stage on a 32 GiB card",
            dev: false,
            options: streamed_text.clone(),
            env: &[],
            facts: card_32g(),
        },
        LtxCase {
            name: "ltx25-distill two-stage, offload cpu (FASTVIDEO_LTX_OFFLOAD=cpu)",
            dev: false,
            options: PipelineOptions {
                offload: Some(LtxOffload::Cpu),
                ..streamed_text.clone()
            },
            env: &[],
            facts: rtx_pro_6000(),
        },
        LtxCase {
            name: "ltx25-distill with the tiny autoencoder (served ltx25-distill-sol-nvfp4-taehv)",
            dev: false,
            options: PipelineOptions {
                tae: Some(weights.join("taeltx2_3_wide.safetensors")),
                ..streamed_text.clone()
            },
            env: &[("FASTVIDEO_NVFP4", "1")],
            facts: rtx_pro_6000(),
        },
        LtxCase {
            name: "ltx25-ref2v (IC-LoRA) on an RTX PRO 6000",
            dev: false,
            options: PipelineOptions {
                ic_lora: Some(ic.clone()),
                ..streamed_text.clone()
            },
            env: &[],
            facts: rtx_pro_6000(),
        },
        LtxCase {
            name: "ltx25-a2v-guided (dev DiT, distilled LoRA, base on the host) on an RTX PRO 6000",
            dev: true,
            options: PipelineOptions {
                lora_base_host: true,
                ..streamed_text.clone()
            },
            env: &[],
            facts: rtx_pro_6000(),
        },
        LtxCase {
            name: "ltx25-distill on the CPU",
            dev: false,
            options: streamed_text,
            env: &[],
            facts: DeviceFacts::cpu(),
        },
    ];
    let mut got = String::new();
    for case in &cases {
        let cfg = if case.dev {
            fastvideo_models::ltx2::config::ltx2_5_22b_dev()
        } else {
            fastvideo_models::ltx2::config::ltx2_5_22b_distilled()
        };
        let plan = Ltx2LoadPlan::new(
            &paths,
            &cfg,
            &case.options,
            &snapshot(case.env, &Settings::default()),
            &Active::default(),
            case.facts,
            Instant::now(),
        );
        got.push_str(&format!("== {} ==\n", case.name));
        got.push_str(&match plan {
            Ok(p) => p.describe(),
            Err(e) => format!("error: {e}\n"),
        });
    }
    compare("ltx2_load_plan.txt", &got);
    let _ = std::fs::remove_dir_all(&weights);
}
