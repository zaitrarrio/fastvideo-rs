//! The port against diffusers itself, on the CPU, at toy sizes.
//!
//! The loop references in each module's tests are written from the *spec*; if
//! the spec misreads the reference, they inherit the mistake. These tests do
//! not: `scripts/gpu/ltx2_tiny_reference.py` instantiates diffusers' own LTX-2
//! classes with tiny configs and seeded random weights, runs them once in
//! float32, and stores weights, input and output in `fixtures/*.st` — safetensors
//! files under another extension, because the repository ignores `*.safetensors`.
//! Here the same weights go through the production loaders and the outputs
//! must agree. A swapped shift/scale, a wrong rotary layout, a mis-paired
//! depth-to-space or a dropped bias does not need 19B parameters — or a GPU —
//! to show. The configs mirror the ones in that script; change both together.

use std::path::PathBuf;

use fastvideo_models::ltx2::config::{Ltx2AudioVaeConfig, Ltx2ConnectorsConfig, Ltx2TransformerConfig, Ltx2VideoVaeConfig, Ltx2VocoderConfig};
use fastvideo_models::ltx2::{Ltx2RopeTables, ScalarDivision};

use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

use super::audio_vae::AudioDecoder;
use super::keys::{Keys, Layout};
use super::text::{HiddenStack, TextConnectors};
use super::transformer::{Ltx2Transformer, Ropes};
use super::vae::VideoDecoder;
use super::vocoder::Vocoder;

fn fixture(name: &str) -> WeightMap {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/ltx2/fixtures").join(name);
    WeightMap::open_files(std::slice::from_ref(&path)).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn tensor(map: &WeightMap, key: &str) -> CudaTensor {
    let (shape, data) = map.get_f32(key).unwrap_or_else(|e| panic!("{key}: {e}"));
    CudaTensor::from_vec(data, shape).expect("fixture tensor")
}

/// Relative L2 and the worst element, with the shape checked first.
fn assert_matches(what: &str, got: &CudaTensor, map: &WeightMap, key: &str, max_rel: f64) {
    let (shape, want) = map.get_f32(key).unwrap_or_else(|e| panic!("{key}: {e}"));
    assert_eq!(got.numel(), want.len(), "{what}: ours {:?} vs reference {shape:?}", got.shape);
    let got = got.host_cow().expect("host");
    let (mut err, mut norm, mut worst) = (0f64, 0f64, (0usize, 0f32, 0f32));
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        let d = f64::from(a - b);
        err += d * d;
        norm += f64::from(*b) * f64::from(*b);
        if (a - b).abs() > (worst.1 - worst.2).abs() {
            worst = (i, *a, *b);
        }
    }
    let rel = (err / norm.max(1e-30)).sqrt();
    assert!(rel <= max_rel, "{what}: rel L2 {rel:.3e} > {max_rel:.0e}; worst at {}: ours {} vs reference {}", worst.0, worst.1, worst.2);
}

#[test]
fn dit_matches_diffusers_on_a_tiny_config() {
    let cfg = Ltx2TransformerConfig {
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
    let map = fixture("dit_tiny.st");
    let model = Ltx2Transformer::load(&map, &Keys::transformer(Layout::Diffusers), &cfg).expect("load");
    // The fixture was made on the CPU, where torch divides rather than
    // multiplying by a reciprocal.
    let tables = Ltx2RopeTables::with_division(&cfg, [3, 2, 3], 5, 24.0, ScalarDivision::Exact);
    let cos = CudaTensor::from_vec(tables.video.cos.clone(), vec![tables.video.cos.len()]).expect("cos");
    let sin = CudaTensor::from_vec(tables.video.sin.clone(), vec![tables.video.sin.len()]).expect("sin");
    assert_matches("video rope cos", &cos, &map, "ref.rope.video.cos", 1e-6);
    assert_matches("video rope sin", &sin, &map, "ref.rope.video.sin", 1e-6);

    let ropes = Ropes::upload(&tables).expect("ropes");
    let text = model.project_text(&tensor(&map, "ref.ctx_video"), &tensor(&map, "ref.ctx_audio")).expect("text");
    let mut taps = Vec::new();
    let mut observe = |i: usize, v: &CudaTensor, a: &CudaTensor| -> crate::wan::tensor::Result<()> {
        taps.push((i, v.clone(), a.clone()));
        Ok(())
    };
    // Every sub-layer tap the oracle's hooks produce, under the oracle's names.
    let mut probed = Vec::new();
    let mut probe = |name: &str, t: &CudaTensor| -> crate::wan::tensor::Result<()> {
        probed.push((name.to_string(), t.clone()));
        Ok(())
    };
    let (v, a) = model
        .forward_probed(&tensor(&map, "ref.video_in"), &tensor(&map, "ref.audio_in"), &text, 725.0, &ropes, Some(&mut observe), Some(&mut probe))
        .expect("forward");
    // 2 blocks x 2 streams x (4 sub-layers x in/out + 3 "after") + 2 heads x 2.
    assert_eq!(probed.len(), 2 * 2 * 11 + 4);
    for (name, t) in &probed {
        assert_matches(name, t, &map, &format!("ref.{name}"), 5e-5);
    }
    for (i, tv, ta) in &taps {
        assert_matches(&format!("block {i} video"), tv, &map, &format!("ref.block{i}.video"), 2e-5);
        assert_matches(&format!("block {i} audio"), ta, &map, &format!("ref.block{i}.audio"), 2e-5);
    }
    assert_eq!(taps.len(), 2);
    assert_matches("video velocity", &v, &map, "ref.video_out", 5e-5);
    assert_matches("audio velocity", &a, &map, "ref.audio_out", 5e-5);
}

#[test]
fn connectors_match_diffusers_on_a_tiny_config() {
    let cfg = Ltx2ConnectorsConfig {
        caption_channels: 8,
        text_proj_in_factor: 3,
        video_connector_num_attention_heads: 2,
        video_connector_attention_head_dim: 4,
        video_connector_num_layers: 2,
        video_connector_num_learnable_registers: 4,
        audio_connector_num_attention_heads: 2,
        audio_connector_attention_head_dim: 4,
        audio_connector_num_layers: 1,
        audio_connector_num_learnable_registers: 4,
        connector_rope_base_seq_len: 16,
        ..Ltx2ConnectorsConfig::ltx2_19b()
    };
    let map = fixture("connectors_tiny.st");
    let model = TextConnectors::load(&map, &Keys::connectors(Layout::Diffusers), &cfg).expect("load");
    // Three real tokens, left-padded to eight by the reference.
    let (shape, states) = map.get_f32("ref.hidden_states").expect("states");
    assert_eq!(shape, vec![3, 8, 3]);
    let out = model.forward(&HiddenStack::from_interleaved(&states, 3, 8, 3).expect("stack"), 8).expect("forward");
    assert_matches("video context", &out.video, &map, "ref.video", 2e-5);
    assert_matches("audio context", &out.audio, &map, "ref.audio", 2e-5);
}

#[test]
fn video_vae_matches_diffusers_on_a_tiny_config() {
    let cfg = Ltx2VideoVaeConfig {
        latent_channels: 4,
        decoder_block_out_channels: [8, 16, 32],
        decoder_layers_per_block: [1, 1, 1, 1],
        patch_size: 2,
        ..Ltx2VideoVaeConfig::ltx2_19b()
    };
    let map = fixture("vae_tiny.st");
    let dec = VideoDecoder::load(&map, &cfg).expect("load");
    let z = tensor(&map, "ref.latent");
    assert_matches("decoded video", &dec.decode(&z).expect("decode"), &map, "ref.video", 5e-5);
    // And streamed in the smallest chunks: the same video.
    let mut pieces = Vec::new();
    dec.decode_streaming_chunked(&z, 1, &mut |_, f| {
        pieces.push(f.clone());
        Ok(())
    })
    .expect("streamed decode");
    assert!(pieces.len() > 1);
    let all = CudaTensor::cat(&pieces.iter().collect::<Vec<_>>(), 0).expect("cat").permute(&[1, 0, 2, 3]).expect("permute");
    assert_matches("streamed video", &all, &map, "ref.video", 5e-5);
}

#[test]
fn audio_vae_and_vocoder_match_diffusers_on_tiny_configs() {
    let cfg = Ltx2AudioVaeConfig { base_channels: 4, num_res_blocks: 1, latent_channels: 2, mel_bins: 8, ..Ltx2AudioVaeConfig::ltx2_19b() };
    let map = fixture("audio_vae_tiny.st");
    let dec = AudioDecoder::load(&map, &cfg).expect("load");
    assert_matches("mel", &dec.decode_packed(&tensor(&map, "ref.latent")).expect("decode"), &map, "ref.mel", 5e-5);

    let cfg = Ltx2VocoderConfig {
        in_channels: 16,
        hidden_channels: 64,
        upsample_kernel_sizes: [7, 4, 4, 4, 4],
        upsample_factors: [3, 2, 2, 2, 2],
        ..Ltx2VocoderConfig::ltx2_19b()
    };
    let map = fixture("vocoder_tiny.st");
    let voc = Vocoder::load(&map, &cfg).expect("load");
    assert_matches("waveform", &voc.forward(&tensor(&map, "ref.mel")).expect("forward"), &map, "ref.wave", 5e-5);
}
