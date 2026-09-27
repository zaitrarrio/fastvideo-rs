//! MMAudio modules against MMAudio's own classes (CPU, f32, seeded random
//! weights) from `scripts/gpu/mmaudio_tiny_reference.py`. The fixtures are
//! not committed (Synchformer's is 0.5 GB): point `FV_MMAUDIO_FIXTURES` at the
//! script's output and run
//! `cargo test -p fastvideo-cudarc --release mmaudio::reference_tests -- --ignored`.

use std::path::PathBuf;

use fastvideo_models::mmaudio::{BigVganConfig, MmAudioDiTConfig, MmAudioVaeConfig};

use super::bigvgan::BigVgan;
use super::clip::{ClipText, ClipTowerConfig, ClipVisual};
use super::synchformer::Synchformer;
use super::transformer::MmAudioTransformer;
use super::vae::MmAudioVaeDecoder;
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

fn dir() -> Option<PathBuf> {
    std::env::var_os("FV_MMAUDIO_FIXTURES").map(PathBuf::from)
}

fn map(name: &str) -> Option<WeightMap> {
    let p = dir()?.join(format!("{name}.safetensors"));
    Some(WeightMap::open_files(&[p]).expect("fixture"))
}

fn get(m: &WeightMap, k: &str) -> CudaTensor {
    let (s, v) = m.get_f32(k).expect(k);
    CudaTensor::from_vec(v, s).unwrap()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "lengths");
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        num += (f64::from(*x) - f64::from(*y)).powi(2);
        den += f64::from(*y).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

fn check(name: &str, ours: &CudaTensor, want: &CudaTensor, tol: f64) {
    let e = rel_l2(&ours.host_cow().unwrap(), &want.host_cow().unwrap());
    eprintln!("{name}: rel-L2 {e:.3e}");
    assert!(e < tol, "{name}: rel-L2 {e} >= {tol}");
}

#[test]
#[ignore]
fn dit_matches_reference() {
    let Some(m) = map("dit") else { return };
    let cfg = MmAudioDiTConfig {
        latent_dim: 8,
        clip_dim: 12,
        sync_dim: 10,
        text_dim: 12,
        hidden_dim: 32,
        depth: 3,
        fused_depth: 1,
        num_heads: 2,
        mlp_ratio: 4.0,
        text_seq_len: 5,
        v2: true,
        qk_norm_eps: f32::EPSILON,
    };
    let net = MmAudioTransformer::load(cfg, &m).unwrap();
    let cond = net
        .preprocess(&get(&m, "__in_clip"), &get(&m, "__in_sync"), &get(&m, "__in_text"), 11)
        .unwrap();
    check("clip_f", &cond.clip_f, &get(&m, "__out_clip_f"), 1e-5);
    check("sync_f", &cond.sync_f, &get(&m, "__out_sync_f"), 1e-5);
    check("text_f", &cond.text_f, &get(&m, "__out_text_f"), 1e-5);
    let rot = net.rotations(11, 4).unwrap();
    let flow = net.predict_flow(&get(&m, "__in_latent"), 0.37, &cond, &rot).unwrap();
    check("flow", &flow, &get(&m, "__out_flow"), 1e-4);
}

#[test]
#[ignore]
fn vae_matches_reference() {
    let Some(m) = map("vae") else { return };
    let dec = MmAudioVaeDecoder::load(MmAudioVaeConfig::tiny(), &m).unwrap();
    let mel = dec.decode(&get(&m, "__in_z")).unwrap();
    // The pixel norm drops its 1e-4 (see vae.rs); random weights are not
    // magnitude preserving, so allow a little more than f32 noise.
    check("vae", &mel, &get(&m, "__out_mel"), 1e-3);
}

#[test]
#[ignore]
fn bigvgan_matches_reference() {
    let Some(m) = map("bigvgan") else { return };
    let v = BigVgan::load(BigVganConfig::tiny(), &m).unwrap();
    let wav = v.forward(&get(&m, "__in_mel")).unwrap();
    let want = get(&m, "__out_wav");
    check("bigvgan", &wav, &want.reshape(vec![1, want.numel()]).unwrap(), 1e-4);
}

#[test]
#[ignore]
fn synchformer_matches_reference() {
    let Some(m) = map("synchformer") else { return };
    let s = Synchformer::load(&m).unwrap();
    let f = s.encode(&get(&m, "__in_frames")).unwrap();
    check("synchformer", &f, &get(&m, "__out_feat"), 1e-4);
}

#[test]
#[ignore]
fn clip_matches_reference() {
    let Some(m) = map("clip") else { return };
    let cfg = ClipTowerConfig { width: 32, layers: 2, heads: 2 };
    let vis = ClipVisual::load(cfg.clone(), &m, 14, 2, 16).unwrap();
    check("clip image", &vis.encode(&get(&m, "__in_pixels")).unwrap(), &get(&m, "__out_image"), 1e-4);
    let txt = ClipText::load(cfg, &m, 49408, 77).unwrap();
    let ids = get(&m, "__in_ids").host_cow().unwrap().into_owned();
    let want = get(&m, "__out_text");
    let d = dir().unwrap();
    let prompt = std::fs::read_to_string(d.join("clip_prompt.txt")).unwrap();
    for (b, text) in [prompt.as_str(), ""].iter().enumerate() {
        let ref_ids: Vec<u32> = ids[b * 77..(b + 1) * 77].iter().map(|&v| v as u32).collect();
        let ours = fastvideo_models::mmaudio::tokenize::tokenize(&d.join("tokenizer.json"), text).unwrap();
        assert_eq!(ours, ref_ids, "token ids for {text:?}");
        let got = txt.encode(&ours).unwrap();
        check(&format!("clip text {b}"), &got, &want.narrow(0, b, 1).unwrap(), 1e-4);
    }
}

#[test]
#[ignore]
fn frames_match_torchvision() {
    use fastvideo_models::mmaudio::frames::{clip_pixels, sync_pixels};
    let Some(m) = map("frames") else { return };
    let (_, rgb) = m.get_f32("__in_rgb").unwrap();
    let (h, w) = (480, 832);
    let frames: Vec<Vec<u8>> = rgb.chunks_exact(h * w * 3).map(|f| f.iter().map(|&v| v as u8).collect()).collect();
    let refs: Vec<&[u8]> = frames.iter().map(Vec::as_slice).collect();
    let cp = clip_pixels(&refs, h, w, false);
    let sp = sync_pixels(&refs, h, w);
    for (name, ours, key, lsb) in [("clip", cp, "__out_clip", 1.0 / 255.0), ("sync", sp, "__out_sync", 2.0 / 255.0)] {
        let (_, want) = m.get_f32(key).unwrap();
        let max = ours.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let off = ours.iter().zip(&want).filter(|(a, b)| (*a - *b).abs() > 1e-6).count();
        eprintln!("{name} pixels: max |d| {max:.4} ({:.2} lsb), {off} of {} differ, rel-L2 {:.3e}",
            max / lsb, ours.len(), rel_l2(&ours, &want));
        assert!(max <= lsb * 1.01, "{name}: more than one uint8 step off");
    }
}
