//! The H3 and LTX-2 load lifecycle on the CPU (`docs/adr/0017-loader-stages.md`):
//! plan → open → place → fuse without a GPU.
//!
//! * **plan goldens**: for the served configurations (H3 `4step-vsa`, `sol-h3`,
//!   `8step`, the RTX `fullopt` cell; LTX-2.5 distilled, two-stage, dev
//!   audio-to-video, reference-to-video, the offload placements) on the
//!   devices they run on, the plan's decisions (what loads, the quant mode,
//!   the offload, the adapter and step-cache fusions) are pinned in
//!   `tests/goldens/{h3,ltx2}_load_plan.txt`. A change to a decision is a
//!   change to the golden, reviewed. `FV_GOLDEN_BLESS=1` writes the current
//!   text to the golden instead of comparing. The H3 cases plan on a
//!   synthetic weights directory with the EU volume's layout (the served
//!   pre-quantized trees with their manifests, the Preview LoRAs, a Trim
//!   NVFP4 file, a ModelOpt NVFP4 `transformer/`, a VDN checkpoint) on an
//!   RTX PRO 6000, a B200, an H100, a 32 GiB card and the CPU.
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

/// Compare `got` with `tests/goldens/<name>`. `FV_GOLDEN_BLESS=1` writes it
/// there instead (and a copy under the target directory's `tmp/goldens/`,
/// for a run on a build pod whose source tree is a mirror).
fn compare(name: &str, got: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/goldens")
        .join(name);
    if std::env::var("FV_GOLDEN_BLESS").ok().as_deref() == Some("1") {
        let copy = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("goldens");
        for p in [path.clone(), copy.join(name)] {
            if let Some(dir) = p.parent() {
                std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
            }
            std::fs::write(&p, got).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            eprintln!("blessed {}", p.display());
        }
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

/// An NVIDIA B200 (sm_100): 178 GiB free.
fn b200() -> DeviceFacts {
    DeviceFacts::device((10, 0), 178 * GIB)
}

/// An NVIDIA H100 (sm_90): 79 GiB free.
fn h100() -> DeviceFacts {
    DeviceFacts::device((9, 0), 79 * GIB)
}

/// One H3 case: the snapshot under the weights directory, the options a
/// served model (or a `techniques` cell) loads with, its own profile if any,
/// the process environment, the device.
struct H3Case {
    name: &'static str,
    /// The snapshot, under the synthetic weights directory.
    root: &'static str,
    recipe: &'static str,
    profile: Option<&'static str>,
    /// The VDN checkpoint directory under the weights directory.
    vdn: Option<&'static str>,
    env: &'static [(&'static str, &'static str)],
    facts: DeviceFacts,
}

impl H3Case {
    fn plan(&self, weights: &Path) -> Result<H3LoadPlan, String> {
        let profile = self
            .profile
            .map(|p| Profile::load(Path::new(p)).unwrap_or_else(|e| panic!("{p}: {e}")));
        // fv-serve loads a model under its own profile's settings
        // (`settings::scope_model`): the snapshot's table is the profile's.
        let table = profile
            .as_ref()
            .map(|p| p.settings().unwrap_or_else(|e| panic!("{e}")))
            .unwrap_or_default();
        let options = H3PipelineOptions {
            recipe: Some(self.recipe.to_owned()),
            text_encoder: TextEncoderChoice::Auto,
            i2v_encoder: I2vEncoderChoice::Auto,
            profile: self.profile.map(|_| profile.clone()),
            vdn: self.vdn.map(|d| weights.join(d)),
            ..Default::default()
        };
        H3LoadPlan::new(
            &weights.join(self.root),
            options,
            &snapshot(self.env, &table),
            &Active::default(),
            self.facts,
        )
        .map_err(|e| e.to_string())
    }

    fn text(&self, weights: &Path) -> String {
        let body = match self.plan(weights) {
            Ok(p) => p.describe(),
            Err(e) => format!("error: {e}\n"),
        };
        format!("== {} ==\n{body}", self.name)
    }
}

fn touch(path: &Path, bytes: &[u8]) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    }
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// The EU volume's H3 layout in miniature (names and markers only, no
/// tensor bytes): `h3-base/` with a transformer shard, the Preview LoRAs
/// beside it, a FastH3 Trim snapshot with its NVFP4 tree, a ModelOpt NVFP4
/// `transformer/` and a VDN stage-B checkpoint. The served pre-quantized
/// trees (`transformer_prequant_{4step-vsa,sol-h3}_mxfp8`) are added by
/// [`write_served_trees`].
fn h3_weights() -> PathBuf {
    let w = scratch("h3-plan");
    touch(
        &w.join("h3-base/transformer/diffusion_pytorch_model-00001-of-00001.safetensors"),
        b"",
    );
    for v in ["vsa", "dense"] {
        touch(
            &w.join(format!(
                "FastH3-4-step-Preview-v1-LoRA/{v}-datafree/adapter_model.safetensors"
            )),
            b"",
        );
    }
    std::fs::create_dir_all(w.join("fasth3-trim-8step")).expect("trim dir");
    touch(
        &w.join(format!(
            "{}/{}/{}",
            fastvideo_cudarc::h3::nvfp4::TREE,
            fastvideo_cudarc::h3::nvfp4::SUBDIR,
            fastvideo_cudarc::h3::nvfp4::FILE
        )),
        b"",
    );
    touch(
        &w.join("fasth3-4step-vsa-nvfp4/transformer/config.json"),
        br#"{"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}"#,
    );
    touch(
        &w.join("vdn-h3-stage-b-2000")
            .join(fastvideo_models::h3::vdn::VDN_CONFIG_REL),
        include_bytes!("../src/h3/fixtures/vdn_stage_b_config.json"),
    );
    w
}

/// The pre-quantized trees the EU volume carries beside `h3-base/`
/// (`transformer_prequant_<recipe>_mxfp8`), each with the manifest
/// `fv-gpucheck quantize-dit` writes for the load an RTX PRO 6000 makes
/// (the identity the plan keys the selection on, the source shards' names
/// and sizes; no tensor files, which only the open stage reads).
fn write_served_trees(weights: &Path) {
    use fastvideo_cudarc::h3::dit_tree::{self, Manifest};
    let root = weights.join("h3-base");
    for recipe in ["4step-vsa", "sol-h3"] {
        let case = H3Case {
            name: "",
            root: "h3-base",
            recipe,
            profile: None,
            vdn: None,
            env: &[],
            facts: rtx_pro_6000(),
        };
        let plan = case.plan(weights).expect("plan");
        let identity = plan.dit_tree_identity.expect("the load has a stored form");
        let dir = root.join(format!("transformer_prequant_{recipe}_{}", identity.quant));
        let manifest = Manifest {
            format: dit_tree::FORMAT.into(),
            version: dit_tree::VERSION,
            identity,
            source: dit_tree::source_entries(&root.join("transformer")).expect("source"),
            files: Vec::new(),
            tensors: 0,
            quantized_linears: 0,
            digest: "0000000000000000".into(),
            created_by: "load_stages test".into(),
        };
        touch(
            &dir.join(dit_tree::MANIFEST),
            serde_json::to_string_pretty(&manifest)
                .expect("manifest json")
                .as_bytes(),
        );
    }
}

#[test]
fn h3_plans_are_the_golden_decisions() {
    let weights = h3_weights();
    write_served_trees(&weights);
    fn case(
        name: &'static str,
        root: &'static str,
        recipe: &'static str,
        profile: Option<&'static str>,
        env: &'static [(&'static str, &'static str)],
        facts: DeviceFacts,
    ) -> H3Case {
        H3Case {
            name,
            root,
            recipe,
            profile,
            vdn: None,
            env,
            facts,
        }
    }
    let cases = [
        case(
            "fasth3-4step-vsa (served: profile h3/fasth3_4step_vsa) on an RTX PRO 6000",
            "h3-base",
            "4step-vsa",
            Some("h3/fasth3_4step_vsa"),
            &[],
            rtx_pro_6000(),
        ),
        case(
            "fasth3-4step-vsa-768p (techniques cell, -env arm) on an RTX PRO 6000",
            "h3-base",
            "4step-vsa",
            None,
            &[],
            rtx_pro_6000(),
        ),
        case(
            "fasth3-4step-vsa, FASTVIDEO_H3_DIT_TREE=off (the Preview merge at load)",
            "h3-base",
            "4step-vsa",
            None,
            &[("FASTVIDEO_H3_DIT_TREE", "off")],
            rtx_pro_6000(),
        ),
        case(
            "fasth3-4step-vsa on a B200",
            "h3-base",
            "4step-vsa",
            None,
            &[],
            b200(),
        ),
        case(
            "fasth3-4step-vsa on an H100",
            "h3-base",
            "4step-vsa",
            None,
            &[],
            h100(),
        ),
        case(
            "fasth3-4step-vsa on the CPU",
            "h3-base",
            "4step-vsa",
            None,
            &[],
            DeviceFacts::cpu(),
        ),
        case(
            "sol-h3 (served: profile h3/sol_h3_4step_engine_ladder) on an RTX PRO 6000",
            "h3-base",
            "sol-h3",
            Some("h3/sol_h3_4step_engine_ladder"),
            &[],
            rtx_pro_6000(),
        ),
        case(
            "sol-h3, FASTVIDEO_H3_QUANT=mxfp8 on an H100 (W8A8 below sm_100)",
            "h3-base",
            "sol-h3",
            None,
            &[("FASTVIDEO_H3_QUANT", "mxfp8")],
            h100(),
        ),
        case(
            "fasth3-8step on an RTX PRO 6000",
            "h3-base",
            "8step",
            None,
            &[],
            rtx_pro_6000(),
        ),
        case(
            "fasth3-8step on a 32 GiB card",
            "h3-base",
            "8step",
            None,
            &[],
            card_32g(),
        ),
        case(
            "fasth3-8step, FASTVIDEO_DIT_OFFLOAD=streamed, FASTVIDEO_H3_QUANT=w8a8",
            "h3-base",
            "8step",
            None,
            &[
                ("FASTVIDEO_DIT_OFFLOAD", "streamed"),
                ("FASTVIDEO_H3_QUANT", "w8a8"),
            ],
            rtx_pro_6000(),
        ),
        case(
            "h3-480p-fullopt (techniques cell, -env arm: sol-h3-rtx + FASTVIDEO_H3_SOL_CACHE=teacache)",
            "h3-base",
            "sol-h3-rtx",
            None,
            &[("FASTVIDEO_H3_SOL_CACHE", "teacache")],
            rtx_pro_6000(),
        ),
        case(
            "h3-480p-fullopt (techniques cell, -prof arm: profile h3/rtx5090_fullopt + FASTVIDEO_H3_QUANT=mxfp8)",
            "h3-base",
            "sol-h3-rtx",
            Some("h3/rtx5090_fullopt"),
            &[("FASTVIDEO_H3_QUANT", "mxfp8")],
            rtx_pro_6000(),
        ),
        case(
            "fasth3-trim-nvfp4 (trim-8step-nvfp4: the Trim's ComfyUI NVFP4 file) on an RTX PRO 6000",
            "fasth3-trim-8step",
            "trim-8step-nvfp4",
            None,
            &[],
            rtx_pro_6000(),
        ),
        case(
            "fasth3-trim-nvfp4 on an H100 (no NVFP4 tensor cores)",
            "fasth3-trim-8step",
            "trim-8step-nvfp4",
            None,
            &[],
            h100(),
        ),
        case(
            "fasth3-4step-vsa-nvfp4 (a ModelOpt NVFP4 transformer/) on a B200",
            "fasth3-4step-vsa-nvfp4",
            "4step-vsa-nvfp4",
            None,
            &[],
            b200(),
        ),
        H3Case {
            name: "vdn-h3 (vdn-h3-50step, the stage-B checkpoint) on an RTX PRO 6000",
            root: "h3-base",
            recipe: "vdn-h3-50step",
            profile: None,
            vdn: Some("vdn-h3-stage-b-2000"),
            env: &[],
            facts: rtx_pro_6000(),
        },
    ];
    let got: String = cases.iter().map(|c| c.text(&weights)).collect();
    compare("h3_load_plan.txt", &got);
    let _ = std::fs::remove_dir_all(&weights);
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
            name: "ltx25-distill two-stage, --offload cpu (PipelineOptions::offload)",
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
