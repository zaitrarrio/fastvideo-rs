//! Old against new (ADR-0021 "Evidence"): the pre-B4c caches (`legacy.rs`,
//! verbatim) and the one implementation must derive the same keys, write the
//! same bytes under the same names, accept and reject exactly the same files,
//! hit and miss on the same calls, and leave the same files behind when a
//! write fails. Plus the properties the format promises: atomic writes and
//! concurrent writers that can never produce a misread.
//!
//! B4c2 (ADR-0021 "B4c2") intentionally moved every key: new domains and
//! more identity fields. The key tests below therefore compare the
//! *pre-B4c2* derivations (`*_KEY_V1`, `cache_key_v1`) with legacy — they
//! pin the names entries already on disk have — and
//! `b4c2_keys_never_name_a_legacy_entry` shows the new keys never name one.
//! The entry format did not change in B4c2, so every format, protocol and
//! failure test below still compares the one implementation with legacy.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use fastvideo_loader::{LazyDType, LazyStore, SafetensorsWriter, TensorSpec};

use super::legacy;
use super::{
    identity, sha256, tokens_width, Gemm, Numerics, PromptCache, WeightsRead, H3_KEY, H3_KEY_V1,
    LTX2, LTX2_KEY, LTX2_KEY_V1, WAN_KEY, WAN_KEY_V1,
};
use crate::h3::text_cache as h3;
use crate::llm::{Act, DecoderConfig, LayerAttn};
use crate::ltx2::text::PaddedPrompt;
use crate::ltx2::text_cache as ltx2;
use crate::rt::tensor::CudaTensor;

fn scratch(tag: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "fv-prompt-cache-eq-{tag}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Deterministic bytes, so a file's head and tail differ.
fn pattern(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761).to_le_bytes()[1] ^ seed)
        .collect()
}

fn reseal(body: &[u8]) -> Vec<u8> {
    let mut b = body.to_vec();
    b.extend_from_slice(&sha256(body));
    b
}

fn names(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

const PROMPTS: &[&str] = &[
    "",
    " ",
    "a red fox",
    "  a red fox\n",
    "a dog <d>hi</d>",
    "\tunicode ✓ 日本語 \u{a0}",
    "ab",
];

// ---------------------------------------------------------------------------
// Keys
// ---------------------------------------------------------------------------

#[test]
fn h3_keys_equal_legacy() {
    let ids: [[u8; 32]; 3] = [[0; 32], [1; 32], sha256(b"enc")];
    for prompt in PROMPTS.iter().copied().chain([&"x".repeat(5000)[..]]) {
        for tok in [[0u8; 32], [9; 32], sha256(b"tokenizer.json")] {
            for tap in [0usize, 1, 24, 50, usize::MAX] {
                for enc in &ids {
                    assert_eq!(
                        h3::cache_key_v1(prompt, &tok, tap, enc),
                        legacy::h3::cache_key(prompt, &tok, tap, enc),
                        "{prompt:?} tap {tap}"
                    );
                }
            }
        }
    }
    // The variant identities, as `h3/text.rs` built them inline.
    let base = sha256(b"identity");
    assert_eq!(
        h3::fp8_rows_identity(&base),
        legacy::h3::sha256(&[base.as_slice(), b"/fp8-rows".as_slice()].concat())
    );
    assert_eq!(
        h3::recovered_identity(24),
        legacy::h3::sha256(format!("recovered-8b-tap{}", 24).as_bytes())
    );
    // Pinned, and recomputed independently (Python hashlib) when written.
    assert_eq!(
        super::hex(&h3::cache_key_v1("a dog <d>hi</d>", &[1; 32], 50, &[2; 32])),
        "86ae3a4c85ef1725102bce88852a0d444061550701b8b40f29a6e3cca11652b5"
    );
}

#[test]
fn ltx2_keys_equal_legacy() {
    let big = pattern(20_000, 3);
    for prompt in PROMPTS {
        for tok in [&b""[..], b"tokenizer", &big] {
            for max_len in [0usize, 128, 1024, 1 << 40] {
                for kind in [
                    &b""[..],
                    b"gemma3-12b",
                    b"gemma3-12b/per-token-rms",
                    b"gemma4-12b-hf",
                    b"gemma4-12b-hf/per-token-rms",
                ] {
                    for (g, c) in [([1u8; 32], [2u8; 32]), ([2; 32], [1; 32])] {
                        assert_eq!(
                            ltx2::cache_key_v1(prompt, tok, max_len, kind, &g, &c),
                            legacy::ltx2::cache_key(prompt, tok, max_len, kind, &g, &c),
                            "{prompt:?}"
                        );
                    }
                }
            }
        }
    }
    assert_eq!(
        ltx2::cache_key_v1(
            "a red fox",
            b"tokenizer",
            1024,
            b"gemma3-12b",
            &[1; 32],
            &[2; 32]
        ),
        "b944238de5c5d7fd7b54990f15928a65e4930ef0da2b4a78241a1035c7d9d27c"
    );
    // The prompt is trimmed (LTX), not elsewhere.
    assert_eq!(
        ltx2::cache_key_v1(
            "  a red fox\n",
            b"tokenizer",
            1024,
            b"gemma3-12b",
            &[1; 32],
            &[2; 32]
        ),
        ltx2::cache_key_v1(
            "a red fox",
            b"tokenizer",
            1024,
            b"gemma3-12b",
            &[1; 32],
            &[2; 32]
        ),
    );
    assert_ne!(
        h3::cache_key_v1("  a red fox\n", &[1; 32], 50, &[2; 32]),
        h3::cache_key_v1("a red fox", &[1; 32], 50, &[2; 32]),
    );
}

/// The Wan key over real files: the tokenizer and `text_encoder/` the
/// legacy method read, the new one reads through `files_head`.
#[test]
fn wan_keys_equal_legacy() {
    let root = scratch("wan-key");
    let te = root.join("text_encoder");
    std::fs::create_dir_all(&te).unwrap();
    for (name, n, seed) in [
        ("model-00001-of-00002.safetensors", 0usize, 1u8),
        ("model-00002-of-00002.safetensors", 10, 2),
        ("a.safetensors", (256 << 10) - 1, 3),
        ("b.safetensors", 256 << 10, 4),
        ("c.safetensors", 300 << 10, 5),
        ("config.json", 50, 6),
    ] {
        std::fs::write(te.join(name), pattern(n, seed)).unwrap();
    }
    // A directory with the extension: listed, sized, opened, unreadable.
    std::fs::create_dir_all(te.join("d.safetensors")).unwrap();
    let empty = root.join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    // One tokenizer path per content: the legacy memo is per path.
    let tok = root.join("tokenizer-a.json");
    std::fs::write(&tok, pattern(70_000, 9)).unwrap();
    let tok_path = Some(tok.to_string_lossy().into_owned());
    let tok_sha = sha256(&std::fs::read(&tok).unwrap());
    for dir in [
        Some(te.as_path()),
        Some(empty.as_path()),
        Some(root.join("missing").as_path()),
        None,
    ] {
        let id = identity::files_head(dir);
        for prompt in PROMPTS {
            for text_len in [0usize, 226, 512] {
                for (path, sha) in [(tok_path.clone(), tok_sha), (None, sha256(b""))] {
                    let new = WAN_KEY_V1.key(prompt, &sha, text_len as u64, &[id.as_slice()]);
                    let old = legacy::wan::text_cache_key(dir, path, text_len, prompt).unwrap();
                    assert_eq!(new, old, "{dir:?} {prompt:?} {text_len}");
                }
            }
        }
    }
    assert_eq!(
        super::hex(&WAN_KEY_V1.key(
            "a red fox",
            &sha256(b"tokenizer"),
            512,
            &[&identity::files_head(None)]
        )),
        "032905eeac0252b0e0917e1567ef0550d7b54cc9691b0ec405701e1ded1bc120"
    );
    // Same inputs, six domains (three families, before and since B4c2):
    // no two recipes ever share an entry name.
    let (t, e) = (sha256(b"t"), [5u8; 32]);
    let keys: Vec<[u8; 32]> = [
        H3_KEY_V1,
        WAN_KEY_V1,
        LTX2_KEY_V1,
        H3_KEY,
        WAN_KEY,
        LTX2_KEY,
    ]
    .iter()
    .map(|r| r.key("p", &t, 50, &[&e]))
    .collect();
    for (i, a) in keys.iter().enumerate() {
        for b in &keys[i + 1..] {
            assert_ne!(a, b);
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// B4c2: no key the new code computes names an entry written before B4c2,
/// whatever the new fields hold — the domain alone separates them — and an
/// entry written by the legacy code under its key is never read by the new
/// code for the same prompt and weights: a miss that computes, leaves the old
/// file alone and stores under its own name.
#[test]
fn b4c2_keys_never_name_a_legacy_entry() {
    let numerics = [
        Numerics {
            gemm: Gemm::Bf16,
            arch: Some((9, 0)),
            bf16_act: true,
        },
        Numerics {
            gemm: Gemm::Host,
            arch: None,
            bf16_act: false,
        },
    ];
    let weights = [
        WeightsRead::Checkpoint,
        WeightsRead::Fp8RowsAtLoad,
        WeightsRead::Fp8RowsTree,
        WeightsRead::Fp8RowsUnknown,
    ];
    let tree = [7u8; 32];
    let (tok, enc) = ([1u8; 32], [2u8; 32]);
    for prompt in PROMPTS {
        let old_h3 = [
            legacy::h3::cache_key(prompt, &tok, 50, &enc),
            legacy::h3::cache_key(prompt, &tok, 50, &h3::fp8_rows_identity(&enc)),
            legacy::h3::cache_key(prompt, &tok, 24, &h3::recovered_identity(24)),
        ];
        let old_ltx =
            legacy::ltx2::cache_key(prompt, b"tokenizer", 1024, b"gemma3-12b", &enc, &tok);
        for n in &numerics {
            for w in weights {
                let t = (w == WeightsRead::Fp8RowsTree).then_some(&tree);
                for tap in [24, 50] {
                    let new = h3::cache_key(prompt, &tok, tap, &enc, w, t, n);
                    assert!(!old_h3.contains(&new), "{prompt:?}");
                }
                for mode in [ltx2::TextMode::Resident, ltx2::TextMode::Streamed] {
                    let s = ltx2::EncodeSettings {
                        weights: w,
                        tree: t,
                        mode,
                        numerics: *n,
                    };
                    let new =
                        ltx2::cache_key(prompt, b"tokenizer", 1024, b"gemma3-12b", &enc, &tok, &s);
                    assert_ne!(new, old_ltx, "{prompt:?}");
                }
            }
            let wan = WAN_KEY.key(prompt, &tok, 512, &[enc.as_slice(), &n.field()]);
            assert_ne!(wan, WAN_KEY_V1.key(prompt, &tok, 512, &[enc.as_slice()]));
        }
        // The domain alone: the new recipes over the *old* fields.
        assert_ne!(
            WAN_KEY.key(prompt, &tok, 512, &[enc.as_slice()]),
            WAN_KEY_V1.key(prompt, &tok, 512, &[enc.as_slice()])
        );
        assert_ne!(H3_KEY.key(prompt, &tok, 50, &[enc.as_slice()]), old_h3[0]);
        assert_ne!(
            super::hex(&LTX2_KEY.key(
                prompt,
                &sha256(b"tokenizer"),
                1024,
                &[b"gemma3-12b".as_slice(), enc.as_slice(), tok.as_slice()]
            )),
            old_ltx
        );
    }

    // The Wan v2 key, pinned (recomputed independently with Python hashlib).
    assert_eq!(
        super::hex(&WAN_KEY.key(
            "a cat",
            &sha256(b""),
            512,
            &[[3u8; 32].as_slice(), &numerics[0].field()]
        )),
        "febc3c3ac5eb10bfb317fb196e46b7e6430eb56396de9ad3f7eed79678185a01"
    );

    // On disk, H3 format (H3 and Wan): a legacy entry for this prompt is not
    // what the new key looks up.
    let dir = scratch("b4c2-h3");
    let ids = [7u32, 8, 9];
    let old_key = legacy::h3::cache_key("p", &tok, 50, &enc);
    let (_, hit) =
        legacy::h3::get_or_compute(&dir, &old_key, &ids, 2, || Ok(vec![1.0; 6])).unwrap();
    assert!(!hit);
    let old_file = std::fs::read(legacy::h3::entry_path(&dir, &old_key)).unwrap();
    let new_key = h3::cache_key(
        "p",
        &tok,
        50,
        &enc,
        WeightsRead::Checkpoint,
        None,
        &numerics[0],
    );
    let mut computed = 0;
    let (entry, hit) = h3::get_or_compute(&dir, &new_key, &ids, 2, || {
        computed += 1;
        Ok(vec![2.0; 6])
    })
    .unwrap();
    assert!(!hit && computed == 1, "the legacy entry was not read");
    assert_eq!(entry.data, vec![2.0; 6]);
    assert_eq!(
        std::fs::read(legacy::h3::entry_path(&dir, &old_key)).unwrap(),
        old_file,
        "and is left as it was"
    );
    assert_eq!(names(&dir).len(), 2);
    let _ = std::fs::remove_dir_all(&dir);

    // On disk, LTX format.
    let dir = scratch("b4c2-ltx");
    let (prompt, video, audio) = &ltx_entries()[0];
    let old_key =
        legacy::ltx2::cache_key("a red fox", b"tokenizer", 1024, b"gemma3-12b", &enc, &tok);
    legacy::ltx2::TextCache::new(&dir)
        .store(&old_key, prompt, video, audio)
        .unwrap();
    let new_key = ltx2::cache_key(
        "a red fox",
        b"tokenizer",
        1024,
        b"gemma3-12b",
        &enc,
        &tok,
        &ltx2::EncodeSettings {
            weights: WeightsRead::Checkpoint,
            tree: None,
            mode: ltx2::TextMode::Streamed,
            numerics: numerics[0],
        },
    );
    let mut computed = 0;
    let (_, outcome) = ltx2::TextCache::new(&dir)
        .get_or_compute(&new_key, prompt, || -> crate::rt::tensor::Result<_> {
            computed += 1;
            Ok(ltx2::CachedContexts {
                video: video.clone(),
                audio: audio.clone(),
            })
        })
        .unwrap();
    assert_eq!(outcome, super::Outcome::Miss);
    assert_eq!(computed, 1, "the legacy entry was not read");
    assert_eq!(names(&dir).len(), 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn file_identities_equal_legacy() {
    let dir = scratch("files-id");
    let probe = identity::PROBE as usize;
    for (name, n, seed) in [
        ("model-00001-of-00004.safetensors", 0usize, 1u8),
        ("model-00002-of-00004.safetensors", probe - 1, 2),
        ("model-00003-of-00004.safetensors", probe, 3),
        ("model-00004-of-00004.safetensors", probe + 1, 4),
        ("diffusion_pytorch_model.safetensors", 2 * probe + 5, 5),
        ("x.safetensors", 3 * probe, 6),
        ("config.json", 10, 7),
    ] {
        std::fs::write(dir.join(name), pattern(n, seed)).unwrap();
    }
    for prefix in [None, Some("model-"), Some("diffusion"), Some("nothing-")] {
        let new = identity::files_head_tail(&dir, prefix).map_err(|e| e.to_string());
        let old = legacy::ltx2::weights_identity(&dir, prefix).map_err(|e| e.to_string());
        assert_eq!(new, old, "{prefix:?}");
    }
    let single = dir.join("x.safetensors");
    assert_eq!(
        identity::files_head_tail(&single, None).unwrap(),
        legacy::ltx2::weights_identity(&single, None).unwrap()
    );
    let missing = dir.join("missing");
    assert_eq!(
        identity::files_head_tail(&missing, None).map_err(|e| e.to_string()),
        legacy::ltx2::weights_identity(&missing, None).map_err(|e| e.to_string())
    );
    let _ = std::fs::remove_dir_all(&dir);
}

fn tiny_decoder() -> DecoderConfig {
    DecoderConfig {
        vocab: 6,
        hidden: 8,
        heads: 4,
        kv_heads: 2,
        head_dim: 4,
        intermediate: 12,
        rms_eps: 1e-6,
        norm_offset: 0.0,
        act: Act::Silu,
        qk_norm: true,
        sandwich_norms: false,
        embed_scale: 1.0,
        attn_scale: 0.5,
        layers: vec![LayerAttn::global(5_000_000.0, 1.0); 4],
        layer_prefix: "model.language_model.layers".into(),
        embed_key: "model.language_model.embed_tokens.weight".into(),
        final_norm_key: "model.language_model.norm.weight".into(),
        attention_k_eq_v: false,
        v_norm: false,
        layer_scalar: false,
        attn_softcap: None,
    }
}

/// H3's tensor identity (unchanged but for its hasher, now the shared
/// `Fields`) over a real lazily opened checkpoint, sandwich norms or not.
#[test]
fn h3_encoder_identity_equals_legacy() {
    for sandwich in [false, true] {
        let dir = scratch("h3-id");
        let mut cfg = tiny_decoder();
        cfg.sandwich_norms = sandwich;
        let mut keys = h3::tap_keys(&cfg, cfg.num_layers());
        keys.sort();
        let specs: Vec<TensorSpec> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| TensorSpec::new(k.clone(), LazyDType::BF16, vec![3, 1 + i % 2000]))
            .collect();
        let mut w = SafetensorsWriter::create(&dir.join("model.safetensors"), &specs, &[]).unwrap();
        for (i, (k, s)) in keys.iter().zip(&specs).enumerate() {
            w.write(k, &pattern(2 * s.shape.iter().product::<usize>(), i as u8))
                .unwrap();
        }
        w.finish().unwrap();
        let store = LazyStore::open(&dir).unwrap();
        for tap in [0usize, 1, 2, 4] {
            assert_eq!(
                h3::encoder_identity(&store, &cfg, tap).unwrap(),
                legacy::h3::encoder_identity(&store, &cfg, tap).unwrap(),
                "tap {tap} sandwich {sandwich}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

// ---------------------------------------------------------------------------
// Bytes on disk
// ---------------------------------------------------------------------------

fn h3_entries() -> Vec<h3::CachedText> {
    vec![
        h3::CachedText {
            ids: vec![7, 8, 9],
            width: 2,
            data: vec![0.5, -1.0, 2.0, 3.5, f32::MIN_POSITIVE, 1e30],
        },
        h3::CachedText {
            ids: vec![u32::MAX],
            width: 5,
            data: vec![-0.0, 0.0, f32::MAX, f32::MIN, 1.0e-45],
        },
        h3::CachedText {
            ids: (0..35).collect(),
            width: 16,
            data: (0..35 * 16).map(|i| (i as f32 * 0.37).sin()).collect(),
        },
    ]
}

fn to_legacy_h3(e: &h3::CachedText) -> legacy::h3::CachedText {
    legacy::h3::CachedText {
        ids: e.ids.clone(),
        width: e.width,
        data: e.data.clone(),
    }
}

#[test]
fn h3_files_are_byte_equal_and_cross_read() {
    let (old_dir, new_dir) = (scratch("h3-old"), scratch("h3-new"));
    for (i, e) in h3_entries().iter().enumerate() {
        let key = sha256(&[i as u8]);
        let old_path = legacy::h3::store(&old_dir, &key, &to_legacy_h3(e)).unwrap();
        let new_path = h3::store(&new_dir, &key, e).unwrap();
        assert_eq!(old_path.file_name(), new_path.file_name());
        assert_eq!(new_path, h3::entry_path(&new_dir, &key));
        assert_eq!(
            std::fs::read(&old_path).unwrap(),
            std::fs::read(&new_path).unwrap()
        );
        // Each reads the other's file.
        assert_eq!(h3::load(&old_dir, &key, &e.ids, e.width).as_ref(), Some(e));
        let back = legacy::h3::load(&new_dir, &key, &e.ids, e.width).unwrap();
        assert_eq!(
            (back.ids, back.width, back.data),
            (e.ids.clone(), e.width, e.data.clone())
        );
    }
    assert_eq!(names(&old_dir), names(&new_dir), "no temporary files left");
    let _ = std::fs::remove_dir_all(&old_dir);
    let _ = std::fs::remove_dir_all(&new_dir);
}

fn ltx_entries() -> Vec<(PaddedPrompt, CudaTensor, CudaTensor)> {
    let t = |n: usize, shape: Vec<usize>, k: f32| {
        CudaTensor::from_vec((0..n).map(|i| (i as f32 * k).sin() * 3.0).collect(), shape).unwrap()
    };
    vec![
        (
            PaddedPrompt::from_ids(&[2, 17, 99], 8).unwrap(),
            t(24, vec![1, 8, 3], 0.5),
            t(16, vec![1, 8, 2], 1.1),
        ),
        (
            PaddedPrompt::from_ids(&[2, 5, 6, 7], 4).unwrap(),
            t(12, vec![4, 3], 0.1),
            t(4, vec![4], 0.7),
        ),
        (
            PaddedPrompt {
                ids: vec![],
                real: 0,
            },
            t(0, vec![1, 0, 3], 0.2),
            t(1, vec![1, 1, 1, 1, 1, 1, 1, 1], 0.3),
        ),
        (
            PaddedPrompt::from_ids(&[2, 1], 6).unwrap(),
            CudaTensor::from_vec(vec![f32::NAN, f32::INFINITY, -0.0, 1.0], vec![1, 2, 2]).unwrap(),
            t(2, vec![1, 2, 1], 0.9),
        ),
    ]
}

fn host(t: &CudaTensor) -> (Vec<usize>, Vec<u32>) {
    (
        t.shape.clone(),
        t.host_cow().unwrap().iter().map(|v| v.to_bits()).collect(),
    )
}

type Contexts = ((Vec<usize>, Vec<u32>), (Vec<usize>, Vec<u32>));

fn ltx_new(c: Option<ltx2::CachedContexts>) -> Option<Contexts> {
    c.map(|c| (host(&c.video), host(&c.audio)))
}

fn ltx_old(c: Option<legacy::ltx2::CachedContexts>) -> Option<Contexts> {
    c.map(|c| (host(&c.video), host(&c.audio)))
}

#[test]
fn ltx2_files_are_byte_equal_and_cross_read() {
    let (old_dir, new_dir) = (scratch("ltx-old"), scratch("ltx-new"));
    let (old, new) = (
        legacy::ltx2::TextCache::new(&old_dir),
        ltx2::TextCache::new(&new_dir),
    );
    for (i, (prompt, video, audio)) in ltx_entries().iter().enumerate() {
        let key = super::hex(&sha256(&[i as u8]));
        old.store(&key, prompt, video, audio).unwrap();
        new.store(&key, prompt, video, audio).unwrap();
        assert_eq!(old.path(&key).file_name(), new.path(&key).file_name());
        assert_eq!(
            std::fs::read(old.path(&key)).unwrap(),
            std::fs::read(new.path(&key)).unwrap()
        );
        let want = Some((host(video), host(audio)));
        // Cross-read: the legacy file by the new code, and the reverse;
        // NaN and infinity come back (LTX does not check finiteness).
        assert_eq!(
            ltx_new(ltx2::TextCache::new(&old_dir).load(&key, prompt)),
            want
        );
        assert_eq!(
            ltx_old(legacy::ltx2::TextCache::new(&new_dir).load(&key, prompt)),
            want
        );
    }
    assert_eq!(names(&old_dir), names(&new_dir), "no .part files left");
    let _ = std::fs::remove_dir_all(&old_dir);
    let _ = std::fs::remove_dir_all(&new_dir);
}

// ---------------------------------------------------------------------------
// Accept / reject: every damaged or foreign file is judged the same
// ---------------------------------------------------------------------------

/// Every truncation, every single-bit flip, trailing bytes, and resealed
/// header lies of `good`: the legacy and new readers agree on each.
fn mutations(good: &[u8], lies: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = (0..good.len()).map(|n| good[..n].to_vec()).collect();
    for byte in 0..good.len() {
        for bit in 0..8 {
            let mut f = good.to_vec();
            f[byte] ^= 1 << bit;
            out.push(f);
        }
    }
    let mut grown = good.to_vec();
    grown.extend_from_slice(&[0; 4]);
    out.push(grown);
    // Trailing bytes inside a valid seal.
    let body = &good[..good.len() - 32];
    out.push(reseal(&[body, &[0, 0, 0, 0][..]].concat()));
    out.push(reseal(&body[..body.len() - 4]));
    out.push(good.to_vec());
    out.extend(lies);
    out
}

fn set_u64(body: &mut [u8], at: usize, v: u64) {
    body[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

fn set_u32(body: &mut [u8], at: usize, v: u32) {
    body[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

#[test]
fn h3_readers_accept_and_reject_the_same_files() {
    let dir = scratch("h3-judge");
    let key = [9u8; 32];
    let e = h3::CachedText {
        ids: vec![1, 2],
        width: 3,
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    };
    let path = h3::store(&dir, &key, &e).unwrap();
    let good = std::fs::read(&path).unwrap();
    let body = good[..good.len() - 32].to_vec();
    let lie = |f: &dyn Fn(&mut Vec<u8>)| {
        let mut b = body.clone();
        f(&mut b);
        reseal(&b)
    };
    let lies = vec![
        lie(&|b| set_u64(b, 8, 0)),
        lie(&|b| set_u64(b, 16, 0)),
        lie(&|b| set_u64(b, 8, u64::MAX)),
        lie(&|b| set_u64(b, 16, u64::MAX)),
        lie(&|b| set_u64(b, 8, 3)),
        lie(&|b| set_u64(b, 16, 2)),
        lie(&|b| set_u32(b, 24, 7)),
        lie(&|b| b[..8].copy_from_slice(b"H3TEXT02")),
        lie(&|b| b[..8].copy_from_slice(b"FVLTX2T1")),
        lie(&|b| set_u32(b, 32, f32::NAN.to_bits())),
        lie(&|b| set_u32(b, 36, f32::INFINITY.to_bits())),
        lie(&|b| set_u32(b, 40, f32::NEG_INFINITY.to_bits())),
        lie(&|b| b.truncate(23)),
        reseal(b""),
        Vec::new(),
        // Zero tokens or zero width with a self-consistent length: refused.
        reseal(&[&b"H3TEXT01"[..], &0u64.to_le_bytes(), &3u64.to_le_bytes()].concat()),
        reseal(
            &[
                &b"H3TEXT01"[..],
                &2u64.to_le_bytes(),
                &0u64.to_le_bytes(),
                &1u32.to_le_bytes(),
                &2u32.to_le_bytes(),
            ]
            .concat(),
        ),
    ];
    let files = mutations(&good, lies);
    let mut accepted = 0;
    for (i, f) in files.iter().enumerate() {
        std::fs::write(&path, f).unwrap();
        for (ids, width) in [
            (&[1u32, 2][..], 3usize),
            (&[1, 2][..], 2),
            (&[1, 3][..], 3),
            (&[7, 2][..], 3),
            (&[][..], 3),
            (&[1, 2][..], 0),
        ] {
            let new = h3::load(&dir, &key, ids, width);
            let old = legacy::h3::load(&dir, &key, ids, width).map(|o| h3::CachedText {
                ids: o.ids,
                width: o.width,
                data: o.data,
            });
            assert_eq!(new, old, "file {i} ids {ids:?} width {width}");
            accepted += usize::from(new.is_some());
        }
    }
    assert!(accepted >= 1, "the intact file is accepted");
    // An LTX entry under the H3 name: a miss for both.
    let ltx = scratch("h3-judge-ltx");
    let (p, v, a) = &ltx_entries()[0];
    ltx2::TextCache::new(&ltx).store("k", p, v, a).unwrap();
    std::fs::write(
        &path,
        std::fs::read(ltx2::TextCache::new(&ltx).path("k")).unwrap(),
    )
    .unwrap();
    assert_eq!(h3::load(&dir, &key, &p.ids, 3), None);
    assert!(legacy::h3::load(&dir, &key, &p.ids, 3).is_none());
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&ltx);
}

/// An LTX entry for `prompt` whose video tensor has `rank` unit dimensions.
fn rank_entry(prompt: &PaddedPrompt, rank: u32) -> Vec<u8> {
    let mut b = b"FVLTX2T1".to_vec();
    b.extend_from_slice(&(prompt.ids.len() as u32).to_le_bytes());
    b.extend_from_slice(&(prompt.real as u32).to_le_bytes());
    prompt
        .ids
        .iter()
        .for_each(|i| b.extend_from_slice(&i.to_le_bytes()));
    b.extend_from_slice(&rank.to_le_bytes());
    (0..rank).for_each(|_| b.extend_from_slice(&1u64.to_le_bytes()));
    b.extend_from_slice(&1.5f32.to_le_bytes());
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    b.extend_from_slice(&2.5f32.to_le_bytes());
    reseal(&b)
}

#[test]
fn ltx2_readers_accept_and_reject_the_same_files() {
    let dir = scratch("ltx-judge");
    let (new, old) = (
        ltx2::TextCache::new(&dir),
        legacy::ltx2::TextCache::new(&dir),
    );
    let (prompt, video, audio) = &ltx_entries()[1];
    new.store("k", prompt, video, audio).unwrap();
    let good = std::fs::read(new.path("k")).unwrap();
    let body = good[..good.len() - 32].to_vec();
    // Layout of entry 1: magic 8 | len 4 | real 4 | ids 16 | rank 4 @32 |
    // shape 2×8 @36 | data 12×4 @52 | rank 4 @100 | shape 8 @104 | data 4×4 @112.
    let lie = |f: &dyn Fn(&mut Vec<u8>)| {
        let mut b = body.clone();
        f(&mut b);
        reseal(&b)
    };
    let lies = vec![
        lie(&|b| set_u32(b, 8, 3)),
        lie(&|b| set_u32(b, 8, u32::MAX)),
        lie(&|b| set_u32(b, 12, 3)),
        lie(&|b| set_u32(b, 16, 9)),
        lie(&|b| set_u32(b, 32, 9)),
        lie(&|b| set_u32(b, 32, 8)),
        lie(&|b| set_u32(b, 32, 0)),
        lie(&|b| set_u32(b, 32, u32::MAX)),
        lie(&|b| set_u64(b, 36, u64::MAX)),
        lie(&|b| set_u64(b, 36, 0)),
        lie(&|b| set_u64(b, 36, 2)),
        lie(&|b| set_u32(b, 100, 9)),
        lie(&|b| set_u64(b, 104, 1 << 62)),
        lie(&|b| set_u32(b, 52, f32::NAN.to_bits())),
        lie(&|b| b[..8].copy_from_slice(b"FVLTX2T2")),
        lie(&|b| b[..8].copy_from_slice(b"H3TEXT01")),
        lie(&|b| b.truncate(8)),
        reseal(b""),
        Vec::new(),
        // Self-consistent tensors of rank 9 (refused) and rank 8 (taken).
        rank_entry(prompt, 9),
        rank_entry(prompt, 8),
    ];
    let other_ids = PaddedPrompt::from_ids(&[2, 5, 6, 8], 4).unwrap();
    let other_real = PaddedPrompt {
        ids: prompt.ids.clone(),
        real: 3,
    };
    let mut accepted = 0;
    for (i, f) in mutations(&good, lies).iter().enumerate() {
        std::fs::write(new.path("k"), f).unwrap();
        for p in [prompt, &other_ids, &other_real] {
            let (n, o) = (ltx_new(new.load("k", p)), ltx_old(old.load("k", p)));
            assert_eq!(n, o, "file {i}");
            accepted += usize::from(n.is_some());
        }
    }
    assert!(
        accepted >= 3,
        "the intact file, the NaN lie and rank 8 are accepted"
    );
    // An H3 entry under the LTX name: a miss for both.
    let h3_dir = scratch("ltx-judge-h3");
    let e = &h3_entries()[0];
    let p = h3::store(&h3_dir, &[1; 32], e).unwrap();
    std::fs::write(new.path("k"), std::fs::read(p).unwrap()).unwrap();
    let pad = PaddedPrompt {
        ids: e.ids.clone(),
        real: e.ids.len(),
    };
    assert!(new.load("k", &pad).is_none() && old.load("k", &pad).is_none());
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&h3_dir);
}

// ---------------------------------------------------------------------------
// Hit and miss
// ---------------------------------------------------------------------------

#[test]
fn h3_hits_and_misses_equal_legacy() {
    let (old_dir, new_dir) = (scratch("h3-proto-old"), scratch("h3-proto-new"));
    let key = h3::cache_key_v1("p", &[0; 32], 50, &[0; 32]);
    let values =
        |ids: &[u32]| -> Vec<f32> { ids.iter().flat_map(|&i| [i as f32, -(i as f32)]).collect() };
    // (ids, width, values the encoder returns)
    let calls: Vec<(Vec<u32>, usize, Option<Vec<f32>>)> = vec![
        (vec![7, 8, 9], 2, None),
        (vec![7, 8, 9], 2, None),
        (vec![7, 8, 10], 2, None),
        (vec![7, 8, 10], 2, None),
        (vec![7, 8, 9], 2, None),
        (vec![7, 8, 9], 3, Some(vec![1.0; 3])),
        (vec![7, 8, 9], 2, Some(vec![f32::NAN; 6])),
        (vec![7, 8, 9], 2, None),
        (vec![], 2, None),
    ];
    let (mut old_n, mut new_n) = (0, 0);
    for (i, (ids, width, out)) in calls.into_iter().enumerate() {
        let data = out.unwrap_or_else(|| values(&ids));
        let old = legacy::h3::get_or_compute(&old_dir, &key, &ids, width, || {
            old_n += 1;
            Ok(data.clone())
        })
        .map(|(e, hit)| {
            (
                e.ids,
                e.width,
                e.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                hit,
            )
        })
        .map_err(|e| e.to_string());
        let new = h3::get_or_compute(&new_dir, &key, &ids, width, || {
            new_n += 1;
            Ok(data.clone())
        })
        .map(|(e, hit)| {
            (
                e.ids,
                e.width,
                e.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                hit,
            )
        })
        .map_err(|e| e.to_string());
        assert_eq!(new, old, "call {i}");
        assert_eq!(new_n, old_n, "call {i}");
        assert_eq!(
            std::fs::read(h3::entry_path(&new_dir, &key)).ok(),
            std::fs::read(legacy::h3::entry_path(&old_dir, &key)).ok(),
            "call {i}"
        );
    }
    assert_eq!(names(&old_dir), names(&new_dir));
    let _ = std::fs::remove_dir_all(&old_dir);
    let _ = std::fs::remove_dir_all(&new_dir);
}

#[test]
fn ltx2_hits_and_misses_equal_legacy() {
    let (old_dir, new_dir) = (scratch("ltx-proto-old"), scratch("ltx-proto-new"));
    let (old, new) = (
        legacy::ltx2::TextCache::new(&old_dir),
        ltx2::TextCache::new(&new_dir),
    );
    let entries = ltx_entries();
    let order = [0usize, 0, 1, 1, 0, 3, 3];
    let (mut old_n, mut new_n) = (0, 0);
    for (i, &e) in order.iter().enumerate() {
        let (prompt, video, audio) = &entries[e];
        let make = |n: &mut usize| -> crate::rt::tensor::Result<legacy::ltx2::CachedContexts> {
            *n += 1;
            let shift = *n as f32;
            let video = CudaTensor::from_vec(
                video.host_cow()?.iter().map(|v| v + shift).collect(),
                video.shape.clone(),
            )?;
            Ok(legacy::ltx2::CachedContexts {
                video,
                audio: audio.clone(),
            })
        };
        let (o, oo) =
            legacy::ltx2::cached_or(Some(&old), Some("k"), prompt, || make(&mut old_n)).unwrap();
        let (n, no) = new
            .get_or_compute("k", prompt, || {
                make(&mut new_n).map(|c| ltx2::CachedContexts {
                    video: c.video,
                    audio: c.audio,
                })
            })
            .unwrap();
        assert_eq!(no.as_str(), format!("{oo:?}").to_lowercase(), "call {i}");
        assert_eq!(
            (host(&n.video), host(&n.audio)),
            (host(&o.video), host(&o.audio)),
            "call {i}"
        );
        assert_eq!(new_n, old_n, "call {i}");
        assert_eq!(
            std::fs::read(new.path("k")).ok(),
            std::fs::read(old.path("k")).ok(),
            "call {i}"
        );
    }
    let _ = std::fs::remove_dir_all(&old_dir);
    let _ = std::fs::remove_dir_all(&new_dir);
}

// ---------------------------------------------------------------------------
// Atomic writes, failures and concurrent writers
// ---------------------------------------------------------------------------

/// A store whose rename cannot succeed (a directory squats the entry name):
/// the same error text and the same files left behind as before — H3 removes
/// its temporary file, LTX leaves its `.part`.
#[test]
fn failed_stores_report_and_leave_what_they_did() {
    let (old_dir, new_dir) = (scratch("fail-h3-old"), scratch("fail-h3-new"));
    let key = [3u8; 32];
    let e = &h3_entries()[0];
    std::fs::create_dir_all(legacy::h3::entry_path(&old_dir, &key).join("x")).unwrap();
    std::fs::create_dir_all(h3::entry_path(&new_dir, &key).join("x")).unwrap();
    let old = legacy::h3::store(&old_dir, &key, &to_legacy_h3(e)).unwrap_err();
    let new = h3::store(&new_dir, &key, e).unwrap_err().to_string();
    assert_eq!(
        new.replace(&*new_dir.to_string_lossy(), "D"),
        old.replace(&*old_dir.to_string_lossy(), "D")
    );
    assert_eq!(names(&old_dir), names(&new_dir));
    // An empty entry is refused before any I/O, with the same text.
    let empty = h3::CachedText {
        ids: vec![],
        width: 4,
        data: vec![],
    };
    assert_eq!(
        h3::store(&new_dir, &key, &empty).unwrap_err().to_string(),
        legacy::h3::store(&old_dir, &key, &to_legacy_h3(&empty)).unwrap_err()
    );

    let (old_dir, new_dir) = (scratch("fail-ltx-old"), scratch("fail-ltx-new"));
    let (old, new) = (
        legacy::ltx2::TextCache::new(&old_dir),
        ltx2::TextCache::new(&new_dir),
    );
    std::fs::create_dir_all(old.path("k").join("x")).unwrap();
    std::fs::create_dir_all(new.path("k").join("x")).unwrap();
    let (p, v, a) = &ltx_entries()[0];
    let o = old.store("k", p, v, a).unwrap_err().to_string();
    let n = new.store("k", p, v, a).unwrap_err().to_string();
    assert_eq!(
        n.replace(&*new_dir.to_string_lossy(), "D"),
        o.replace(&*old_dir.to_string_lossy(), "D")
    );
    assert!(n.starts_with("text cache "), "{n}");
    assert_eq!(names(&old_dir), names(&new_dir));
    assert!(
        names(&new_dir).iter().any(|f| f.ends_with(".part")),
        "{:?}",
        names(&new_dir)
    );
    // This process's temporary name squatted by a directory: the store fails
    // the same way for both (the temporary names are the same).
    let pid = std::process::id();
    let (old_dir, new_dir) = (scratch("squat-old"), scratch("squat-new"));
    std::fs::create_dir_all(
        legacy::h3::entry_path(&old_dir, &key).with_extension(format!("tmp{pid}")),
    )
    .unwrap();
    std::fs::create_dir_all(h3::entry_path(&new_dir, &key).with_extension(format!("tmp{pid}")))
        .unwrap();
    let o = legacy::h3::store(&old_dir, &key, &to_legacy_h3(e)).unwrap_err();
    let n = h3::store(&new_dir, &key, e).unwrap_err().to_string();
    assert_eq!(
        n.replace(&*new_dir.to_string_lossy(), "D"),
        o.replace(&*old_dir.to_string_lossy(), "D")
    );
    let (old, new) = (
        legacy::ltx2::TextCache::new(&old_dir),
        ltx2::TextCache::new(&new_dir),
    );
    std::fs::create_dir_all(old_dir.join(format!("k.{pid}.part"))).unwrap();
    std::fs::create_dir_all(new_dir.join(format!("k.{pid}.part"))).unwrap();
    let o = old.store("k", p, v, a).unwrap_err().to_string();
    let n = new.store("k", p, v, a).unwrap_err().to_string();
    assert_eq!(
        n.replace(&*new_dir.to_string_lossy(), "D"),
        o.replace(&*old_dir.to_string_lossy(), "D")
    );
    assert_eq!(names(&old_dir), names(&new_dir));
    // A cache directory that cannot be created: the same text again.
    let file = scratch("fail-file").join("not-a-dir");
    std::fs::write(&file, b"x").unwrap();
    let o = legacy::ltx2::TextCache::new(file.join("c"))
        .store("k", p, v, a)
        .unwrap_err()
        .to_string();
    let n = ltx2::TextCache::new(file.join("c"))
        .store("k", p, v, a)
        .unwrap_err()
        .to_string();
    assert_eq!(n, o);
    let o = legacy::h3::store(&file.join("c"), &key, &to_legacy_h3(e)).unwrap_err();
    let n = h3::store(&file.join("c"), &key, e).unwrap_err().to_string();
    assert_eq!(n, o);
}

/// A stale temporary file (a writer killed before its rename) is never read
/// and does not stop the next store.
#[test]
fn a_killed_writer_leaves_nothing_a_reader_takes() {
    let dir = scratch("killed");
    let key = [4u8; 32];
    let e = &h3_entries()[2];
    let path = h3::entry_path(&dir, &key);
    let good = {
        let probe = scratch("killed-probe");
        std::fs::read(h3::store(&probe, &key, e).unwrap()).unwrap()
    };
    std::fs::write(path.with_extension("tmp1"), &good[..good.len() / 2]).unwrap();
    assert_eq!(h3::load(&dir, &key, &e.ids, e.width), None);
    // Its own temporary name half-written by a dead process with this pid.
    std::fs::write(
        path.with_extension(format!("tmp{}", std::process::id())),
        b"half",
    )
    .unwrap();
    h3::store(&dir, &key, e).unwrap();
    assert_eq!(h3::load(&dir, &key, &e.ids, e.width).as_ref(), Some(e));
    let ltx = ltx2::TextCache::new(&dir);
    let (p, v, a) = &ltx_entries()[0];
    std::fs::write(dir.join(format!("k.{}.part", std::process::id())), b"half").unwrap();
    assert!(ltx.load("k", p).is_none());
    ltx.store("k", p, v, a).unwrap();
    assert!(ltx.load("k", p).is_some());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Writers as distinct processes (distinct temporary names) and as threads
/// of one process (one shared temporary name), all storing one key with
/// different valid contents while readers load it. A reader sees nothing or
/// exactly one writer's entry, never a mixture; with distinct writers the
/// last rename leaves one whole entry.
#[test]
fn concurrent_writers_never_produce_a_misread() {
    for format in [&super::H3, &LTX2] {
        for shared_temp in [false, true] {
            let dir = scratch("concurrent");
            let cache = PromptCache::new(&dir, format);
            let ids: Vec<u32> = (0..64).collect();
            let header = match format.layout {
                super::Layout::TokensWidth => super::Header::Width(64),
                super::Layout::LenRealShaped { .. } => super::Header::Real(64),
            };
            let contents: Vec<Vec<f32>> = (0..8).map(|w| vec![w as f32; 64 * 64]).collect();
            let check = super::Check { ids: &ids, header };
            let accept = |entry: &super::Entry| {
                let d = &entry.tensors[0].data;
                contents.iter().any(|c| c == d)
                    && entry
                        .tensors
                        .iter()
                        .all(|t| t.data == entry.tensors[0].data)
            };
            std::thread::scope(|s| {
                for (w, data) in contents.iter().enumerate() {
                    let (cache, ids) = (&cache, &ids);
                    s.spawn(move || {
                        let shape = [64usize, 64];
                        let tensors = match format.layout {
                            super::Layout::TokensWidth => vec![super::TensorRef {
                                shape: &shape,
                                data,
                            }],
                            super::Layout::LenRealShaped { .. } => vec![
                                super::TensorRef {
                                    shape: &shape,
                                    data
                                };
                                2
                            ],
                        };
                        let entry = super::EntryRef {
                            ids,
                            header,
                            tensors: &tensors,
                        };
                        let writer = if shared_temp { 1 } else { 100 + w as u32 };
                        for _ in 0..20 {
                            // A shared temporary name may be renamed away
                            // under another writer: that store fails, as it
                            // could before; it is never a torn entry.
                            let _ = cache.store_as("k", &entry, writer);
                        }
                    });
                }
                for _ in 0..4 {
                    s.spawn(|| {
                        for _ in 0..200 {
                            if let Some(e) = cache.load("k", &check) {
                                assert!(accept(&e), "a reader saw a mixed entry");
                            }
                        }
                    });
                }
            });
            match cache.load("k", &check) {
                Some(e) => assert!(accept(&e)),
                None => assert!(shared_temp, "distinct writers leave a whole entry"),
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}

#[test]
fn tokens_width_is_the_h3_format() {
    // Wan stores through the same calls as H3: one format, one directory layout.
    let dir = scratch("tw");
    let e = &h3_entries()[0];
    let p = tokens_width::store(&dir, &[1; 32], e).unwrap();
    assert_eq!(p.extension().and_then(|x| x.to_str()), Some("h3text"));
    assert_eq!(
        PromptCache::new(&dir, &super::H3).path(&super::hex(&[1; 32])),
        p
    );
    let _ = std::fs::remove_dir_all(&dir);
}
