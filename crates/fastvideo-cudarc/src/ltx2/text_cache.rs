//! LTX-2's use of the disk prompt cache ([`crate::rt::prompt_cache`],
//! ADR-0021): its key inputs, its entry type and its default directory. The
//! cache itself — key derivation, file format, atomic write, hit and miss —
//! is the shared one, with the [`prompt_cache::LTX2`] format and the
//! [`prompt_cache::LTX2_KEY`] recipe.
//!
//! A cache of finished text conditioning, because the text encoder is the most
//! expensive thing a repeated prompt pays for and the cheapest to skip.
//!
//! Encoding a prompt reads 47 GB of Gemma shards and uploads 23.5 GB to produce
//! 31 MB: the two `[1, 1024, 3840]` connector outputs. Those depend on nothing
//! but the prompt, the tokenizer, the padded length and the Gemma + connector
//! weights — not on the seed, the resolution, the DiT or the step count — so
//! they are stored under a key made of exactly those things. What is cached is
//! the *connector* output, before the DiT's caption projections: that keeps the
//! entry independent of the DiT checkpoint, and the projections are two small
//! GEMMs.
//!
//! The key must be computable **without opening the weights**, or a hit would
//! still pay for mapping 12 files and parsing their headers. A checkpoint is
//! therefore identified by its files ([`weights_identity`]).
//!
//! An entry carries the padded token ids it was made from and ends in a hash of
//! its own bytes. A hit re-tokenises the prompt and compares; a short, corrupt
//! or foreign file is a miss, never an error — the cache can only make a run
//! faster, not make it fail.
//!
//! A hit returns the stored f32 values as host tensors; a miss returns the
//! connectors' own device tensors. The pipeline puts both into one form
//! (`ltx2::pipeline::canonical_contexts`) before the DiT sees them.

use std::path::PathBuf;

use crate::rt::prompt_cache::{
    self, Cached, Check, Entry, EntryRef, Header, PromptCache, PromptCacheError, TensorRef,
};
use crate::rt::tensor::CudaTensor;

use super::text::PaddedPrompt;

/// Identity of a checkpoint from its files
/// ([`prompt_cache::identity::files_head_tail`]).
pub use crate::rt::prompt_cache::identity::files_head_tail as weights_identity;
pub use crate::rt::prompt_cache::Outcome as CacheOutcome;
pub use crate::rt::prompt_cache::{Gemm, Numerics, WeightsRead};

/// The two contexts of one prompt, as the connectors emit them.
pub struct CachedContexts {
    pub video: CudaTensor,
    pub audio: CudaTensor,
}

impl Cached for CachedContexts {
    fn from_entry(entry: Entry) -> Option<Self> {
        let [video, audio] = <[_; 2]>::try_from(entry.tensors).ok()?;
        Some(Self {
            video: CudaTensor::from_vec(video.data, video.shape).ok()?,
            audio: CudaTensor::from_vec(audio.data, audio.shape).ok()?,
        })
    }

    fn with_tensors<R>(
        &self,
        f: impl FnOnce(&[TensorRef<'_>]) -> R,
    ) -> Result<R, PromptCacheError> {
        with_host(&self.video, &self.audio, f)
    }
}

/// `f` over the two contexts read back as f32 (video, then audio).
fn with_host<R>(
    video: &CudaTensor,
    audio: &CudaTensor,
    f: impl FnOnce(&[TensorRef<'_>]) -> R,
) -> Result<R, PromptCacheError> {
    fn host(t: &CudaTensor) -> Result<std::borrow::Cow<'_, [f32]>, PromptCacheError> {
        t.host_cow()
            .map_err(|e| PromptCacheError::Tensor(e.to_string()))
    }
    let (v, a) = (host(video)?, host(audio)?);
    Ok(f(&[
        TensorRef {
            shape: &video.shape,
            data: &v,
        },
        TensorRef {
            shape: &audio.shape,
            data: &a,
        },
    ]))
}

/// The pre-B4c2 key ([`prompt_cache::LTX2_KEY_V1`]): the stripped prompt,
/// the tokenizer file's digest, the padded length, the encoder kind and the
/// Gemma and connector identities. Hex. Entries on disk written before B4c2
/// have these names; the pipeline no longer computes it.
pub fn cache_key_v1(
    prompt: &str,
    tokenizer_json: &[u8],
    max_len: usize,
    text_encoder_kind: &[u8],
    gemma: &[u8; 32],
    connectors: &[u8; 32],
) -> String {
    prompt_cache::hex(&prompt_cache::LTX2_KEY_V1.key(
        prompt,
        &prompt_cache::sha256(tokenizer_json),
        max_len as u64,
        &[text_encoder_kind, gemma.as_slice(), connectors.as_slice()],
    ))
}

/// Where Gemma's forward ran (ADR-0021 "B4c2", gap 5): the resident decoder
/// or layer by layer. Equal numbers are expected, not proven, so the two are
/// keyed apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextMode {
    Resident,
    Streamed,
}

impl TextMode {
    /// The key field.
    pub fn field(self) -> &'static [u8] {
        match self {
            Self::Resident => b"mode=resident",
            Self::Streamed => b"mode=streamed",
        }
    }
}

/// How an encode runs, beyond which encoder it is (ADR-0021 "B4c2"): every
/// value is what the run resolved, not what was asked for.
#[derive(Debug, Clone, Copy)]
pub struct EncodeSettings<'a> {
    /// Gemma's linears: bf16 checkpoint, FP8 rows quantised at load, or the
    /// pre-quantised `text_encoder_fp8/` tree (gaps 1 and 6).
    pub weights: WeightsRead,
    /// The tree's file identity when `weights` is the tree (gap 1).
    pub tree: Option<&'a [u8; 32]>,
    pub mode: TextMode,
    /// GEMM math, card and activations (gap 4).
    pub numerics: Numerics,
}

/// Everything the conditioning depends on, hashed ([`prompt_cache::LTX2_KEY`]):
/// the fields of [`cache_key_v1`], then the weights read, the tree's
/// identity (an empty field unless read), the mode and the numerics. Hex.
pub fn cache_key(
    prompt: &str,
    tokenizer_json: &[u8],
    max_len: usize,
    text_encoder_kind: &[u8],
    gemma: &[u8; 32],
    connectors: &[u8; 32],
    settings: &EncodeSettings<'_>,
) -> String {
    prompt_cache::hex(&prompt_cache::LTX2_KEY.key(
        prompt,
        &prompt_cache::sha256(tokenizer_json),
        max_len as u64,
        &[
            text_encoder_kind,
            gemma.as_slice(),
            connectors.as_slice(),
            settings.weights.field(),
            settings.tree.map_or(&[][..], |t| t.as_slice()),
            settings.mode.field(),
            &settings.numerics.field(),
        ],
    ))
}

/// `$FASTVIDEO_CACHE`, else `$XDG_CACHE_HOME/fastvideo`, else `~/.cache/fastvideo`,
/// under `ltx2-text`. `None` when the environment names no home at all.
pub fn default_dir() -> Option<PathBuf> {
    let var = |k: &str| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let root = fastvideo_models::settings::env_var_os("FASTVIDEO_CACHE")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| var("XDG_CACHE_HOME").map(|p| p.join("fastvideo")))
        .or_else(|| var("HOME").map(|p| p.join(".cache").join("fastvideo")))?;
    Some(root.join("ltx2-text"))
}

pub struct TextCache {
    cache: PromptCache,
}

impl TextCache {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            cache: PromptCache::new(dir, &prompt_cache::LTX2),
        }
    }

    pub fn path(&self, key: &str) -> PathBuf {
        self.cache.path(key)
    }

    fn check(prompt: &PaddedPrompt) -> Check<'_> {
        Check {
            ids: &prompt.ids,
            header: Header::Real(prompt.real),
        }
    }

    /// Write atomically: a reader never sees half an entry under the final name.
    pub fn store(
        &self,
        key: &str,
        prompt: &PaddedPrompt,
        video: &CudaTensor,
        audio: &CudaTensor,
    ) -> Result<(), PromptCacheError> {
        with_host(video, audio, |tensors| {
            self.cache.store(
                key,
                &EntryRef {
                    ids: &prompt.ids,
                    header: Header::Real(prompt.real),
                    tensors,
                },
            )
        })
        .and_then(|r| r)
        .map(|_| ())
    }

    /// The entry under `key`, if it is whole and was made from exactly these
    /// token ids. Anything else is a miss.
    pub fn load(&self, key: &str, prompt: &PaddedPrompt) -> Option<CachedContexts> {
        self.cache
            .load(key, &Self::check(prompt))
            .and_then(CachedContexts::from_entry)
    }

    /// Load, else `compute` and store (a store that fails is logged).
    pub fn get_or_compute<E>(
        &self,
        key: &str,
        prompt: &PaddedPrompt,
        compute: impl FnOnce() -> Result<CachedContexts, E>,
    ) -> Result<(CachedContexts, CacheOutcome), E> {
        self.cache
            .get_or_compute(key, &Self::check(prompt), compute)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("fv-ltx2-textcache-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn contexts() -> (PaddedPrompt, CudaTensor, CudaTensor) {
        let prompt = PaddedPrompt::from_ids(&[2, 17, 99], 8).unwrap();
        let video = CudaTensor::from_vec(
            (0..24).map(|i| i as f32 * 0.5 - 3.0).collect(),
            vec![1, 8, 3],
        )
        .unwrap();
        let audio =
            CudaTensor::from_vec((0..16).map(|i| (i as f32).sin()).collect(), vec![1, 8, 2])
                .unwrap();
        (prompt, video, audio)
    }

    #[test]
    fn the_v1_key_is_still_the_pre_b4c2_key() {
        let (g, c, kind) = ([1u8; 32], [2u8; 32], b"gemma3-12b");
        let key = cache_key_v1("a red fox", b"tokenizer", 1024, kind, &g, &c);
        assert_eq!(key.len(), 64);
        // Pinned: the names entries written before B4c2 have on disk.
        assert_eq!(
            key,
            "b944238de5c5d7fd7b54990f15928a65e4930ef0da2b4a78241a1035c7d9d27c"
        );
        assert_eq!(
            key,
            cache_key_v1("  a red fox\n", b"tokenizer", 1024, kind, &g, &c),
            "the prompt is stripped, as the pipeline strips it"
        );
        for other in [
            cache_key_v1("a red fox.", b"tokenizer", 1024, kind, &g, &c),
            cache_key_v1("a red fox", b"tokenizer2", 1024, kind, &g, &c),
            cache_key_v1("a red fox", b"tokenizer", 512, kind, &g, &c),
            cache_key_v1("a red fox", b"tokenizer", 1024, b"gemma4-12b", &g, &c),
            cache_key_v1("a red fox", b"tokenizer", 1024, kind, &c, &c),
            cache_key_v1("a red fox", b"tokenizer", 1024, kind, &g, &g),
        ] {
            assert_ne!(key, other);
        }
    }

    /// The settings of the pinned keys: bf16 Gemma streamed on an H100-class
    /// card with the defaults.
    const BF16_STREAMED: EncodeSettings<'static> = EncodeSettings {
        weights: WeightsRead::Checkpoint,
        tree: None,
        mode: TextMode::Streamed,
        numerics: Numerics {
            gemm: Gemm::Bf16,
            arch: Some((9, 0)),
            bf16_act: true,
        },
    };

    #[test]
    fn the_key_is_stable_and_moves_with_every_input() {
        let (g, c, kind) = ([1u8; 32], [2u8; 32], b"gemma3-12b");
        let s = BF16_STREAMED;
        let key = cache_key("a red fox", b"tokenizer", 1024, kind, &g, &c, &s);
        assert_eq!(key.len(), 64);
        // Pinned (recomputed independently with Python hashlib, ADR-0021
        // "B4c2"): a changed key format silently orphans every cache on disk.
        assert_eq!(
            key,
            "f961d9c7c4c55c628c74bf5c8c0e8d186afd55a2da24fe43907f86da65f912e9"
        );
        let tree = [5u8; 32];
        let fp8_tree_resident = EncodeSettings {
            weights: WeightsRead::Fp8RowsTree,
            tree: Some(&tree),
            mode: TextMode::Resident,
            ..s
        };
        assert_eq!(
            cache_key(
                "a red fox",
                b"tokenizer",
                1024,
                kind,
                &g,
                &c,
                &fp8_tree_resident
            ),
            "d3ec427e9e0dbe0c1a4834fab62ecc8df6c2af0185175519d419748ffe446979"
        );
        assert_eq!(
            key,
            cache_key("  a red fox\n", b"tokenizer", 1024, kind, &g, &c, &s),
            "the prompt is stripped, as the pipeline strips it"
        );
        let other_tree = [6u8; 32];
        let with = |settings: EncodeSettings<'_>| {
            cache_key("a red fox", b"tokenizer", 1024, kind, &g, &c, &settings)
        };
        let others = [
            cache_key("a red fox.", b"tokenizer", 1024, kind, &g, &c, &s),
            cache_key("a red fox", b"tokenizer2", 1024, kind, &g, &c, &s),
            cache_key("a red fox", b"tokenizer", 512, kind, &g, &c, &s),
            cache_key("a red fox", b"tokenizer", 1024, b"gemma4-12b", &g, &c, &s),
            cache_key("a red fox", b"tokenizer", 1024, kind, &c, &c, &s),
            cache_key("a red fox", b"tokenizer", 1024, kind, &g, &g, &s),
            // Gap 1 and 6: FP8 rows at load, from the tree, from another tree.
            with(EncodeSettings {
                weights: WeightsRead::Fp8RowsAtLoad,
                ..s
            }),
            with(EncodeSettings {
                weights: WeightsRead::Fp8RowsTree,
                tree: Some(&tree),
                ..s
            }),
            with(EncodeSettings {
                weights: WeightsRead::Fp8RowsTree,
                tree: Some(&other_tree),
                ..s
            }),
            // Gap 5: resident.
            with(EncodeSettings {
                mode: TextMode::Resident,
                ..s
            }),
            // Gap 4: GEMM math, card, activations.
            with(EncodeSettings {
                numerics: Numerics {
                    gemm: Gemm::Tf32,
                    ..s.numerics
                },
                ..s
            }),
            with(EncodeSettings {
                numerics: Numerics {
                    arch: Some((12, 0)),
                    ..s.numerics
                },
                ..s
            }),
            with(EncodeSettings {
                numerics: Numerics {
                    bf16_act: false,
                    ..s.numerics
                },
                ..s
            }),
        ];
        for (i, a) in others.iter().enumerate() {
            assert_ne!(*a, key, "{i}");
            for (j, b) in others.iter().enumerate().skip(i + 1) {
                assert_ne!(a, b, "{i} vs {j}");
            }
        }
    }

    #[test]
    fn the_new_domain_alone_keeps_old_entries_out() {
        // The v2 key with the v1 fields only is not the v1 key: the domain
        // moves every name, so no entry written before B4c2 is looked up.
        let (g, c, kind) = ([1u8; 32], [2u8; 32], b"gemma3-12b");
        let v1 = cache_key_v1("a red fox", b"tokenizer", 1024, kind, &g, &c);
        let new_domain_old_fields = prompt_cache::hex(&prompt_cache::LTX2_KEY.key(
            "a red fox",
            &prompt_cache::sha256(b"tokenizer"),
            1024,
            &[kind, g.as_slice(), c.as_slice()],
        ));
        assert_ne!(v1, new_domain_old_fields);
        assert_ne!(
            v1,
            cache_key(
                "a red fox",
                b"tokenizer",
                1024,
                kind,
                &g,
                &c,
                &BF16_STREAMED
            )
        );
        assert_ne!(
            prompt_cache::LTX2_KEY.domain,
            prompt_cache::LTX2_KEY_V1.domain
        );
    }

    #[test]
    fn a_stored_entry_comes_back_bit_for_bit() {
        let cache = TextCache::new(scratch("hit"));
        let (prompt, video, audio) = contexts();
        assert!(cache.load("k", &prompt).is_none(), "an empty cache misses");
        cache.store("k", &prompt, &video, &audio).unwrap();
        let hit = cache.load("k", &prompt).expect("hit");
        assert_eq!(hit.video.shape, vec![1, 8, 3]);
        assert_eq!(&*hit.video.host_cow().unwrap(), &*video.host_cow().unwrap());
        assert_eq!(&*hit.audio.host_cow().unwrap(), &*audio.host_cow().unwrap());
        assert!(cache.load("another-key", &prompt).is_none());
        // No .part files are left behind.
        let names: Vec<_> = std::fs::read_dir(cache.path("k").parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names.len(), 1, "{names:?}");
    }

    #[test]
    fn an_entry_made_from_other_token_ids_is_a_miss() {
        let cache = TextCache::new(scratch("ids"));
        let (prompt, video, audio) = contexts();
        cache.store("k", &prompt, &video, &audio).unwrap();
        let other = PaddedPrompt::from_ids(&[2, 17, 100], 8).unwrap();
        assert!(cache.load("k", &other).is_none());
        let longer = PaddedPrompt::from_ids(&[2, 17, 99], 16).unwrap();
        assert!(cache.load("k", &longer).is_none());
    }

    #[test]
    fn a_truncated_or_corrupted_entry_is_a_miss_not_an_error() {
        let cache = TextCache::new(scratch("trunc"));
        let (prompt, video, audio) = contexts();
        cache.store("k", &prompt, &video, &audio).unwrap();
        let whole = std::fs::read(cache.path("k")).unwrap();
        for cut in [0, 7, 40, whole.len() / 2, whole.len() - 1] {
            std::fs::write(cache.path("k"), &whole[..cut]).unwrap();
            assert!(
                cache.load("k", &prompt).is_none(),
                "{cut} of {} bytes",
                whole.len()
            );
        }
        let mut flipped = whole.clone();
        flipped[whole.len() / 2] ^= 1;
        std::fs::write(cache.path("k"), &flipped).unwrap();
        assert!(cache.load("k", &prompt).is_none(), "one flipped bit");
        std::fs::write(cache.path("k"), &whole).unwrap();
        assert!(
            cache.load("k", &prompt).is_some(),
            "the intact entry still hits"
        );
    }

    #[test]
    fn a_checkpoint_is_identified_by_its_files_without_being_parsed() {
        let dir = scratch("ident");
        std::fs::write(
            dir.join("model-00001-of-00002.safetensors"),
            vec![7u8; 3000],
        )
        .unwrap();
        std::fs::write(dir.join("model-00002-of-00002.safetensors"), vec![9u8; 100]).unwrap();
        std::fs::write(
            dir.join("diffusion_pytorch_model-00001-of-00012.safetensors"),
            vec![1u8; 50],
        )
        .unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        let id = weights_identity(&dir, Some("model-")).unwrap();
        assert_eq!(id, weights_identity(&dir, Some("model-")).unwrap());
        // The stale duplicate set is outside the prefix; touching it changes nothing.
        std::fs::write(
            dir.join("diffusion_pytorch_model-00001-of-00012.safetensors"),
            vec![2u8; 60],
        )
        .unwrap();
        assert_eq!(id, weights_identity(&dir, Some("model-")).unwrap());
        assert_ne!(id, weights_identity(&dir, None).unwrap());
        // Same size, different bytes: a fine-tune with the same layout.
        std::fs::write(dir.join("model-00002-of-00002.safetensors"), vec![8u8; 100]).unwrap();
        assert_ne!(id, weights_identity(&dir, Some("model-")).unwrap());
        assert!(weights_identity(&dir, Some("nothing-")).is_err());
        // A single file identifies itself.
        assert!(weights_identity(&dir.join("model-00001-of-00002.safetensors"), None).is_ok());
    }
}
