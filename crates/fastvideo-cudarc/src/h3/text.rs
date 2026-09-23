//! H3 text conditioning: the prompt through Qwen3-VL-32B's language model, read
//! at HF `hidden_states[50]`.
//!
//! Three things are easy to get wrong and all still produce a plausible
//! `[1, S, 5120]` tensor (docs/ports/h3.md, section b):
//!
//! * the prompt is tokenized verbatim — no chat template, no BOS — but with the
//!   seven `<d>`-style markers transformers appends to the vocabulary
//!   ([`fastvideo_models::h3::tokenizer`]);
//! * index 50 of `output_hidden_states` is the residual stream after decoder
//!   layer **49**, and it is *not* normed — [`crate::llm::hidden_states`] uses
//!   the same numbering, so this is tap 50 and layers 50..63 are never read;
//! * attention is causal even though the result is used as an encoder output.

use std::path::Path;

use fastvideo_models::h3::config::H3TextEncoderConfig;
use fastvideo_models::h3::tokenizer::H3Tokenizer;

use crate::llm::{self, DecoderConfig};
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// Token ids of `prompt` under the checkpoint at `root` (`root/tokenizer/tokenizer.json`).
pub fn tokenize(root: &Path, prompt: &str) -> Result<Vec<u32>> {
    let path = root.join("tokenizer").join("tokenizer.json");
    let tokenizer = H3Tokenizer::from_file(&path).map_err(msg)?;
    if !tokenizer.added_ids_match_reference() {
        return Err(msg(format!(
            "{}: the H3 marker tokens landed on {:?}, not 151669..=151675; this is not the checkpoint's tokenizer",
            path.display(),
            tokenizer.added_special_token_ids()
        )));
    }
    tokenizer.encode(prompt).map_err(msg)
}

/// `hidden_states[tap]` for already-tokenized text: `[1, S, hidden]`, un-normed
/// for any tap short of the last layer. One unpadded sequence, so positions
/// are `0..S` and every token may be attended.
pub fn encode_ids(
    map: &WeightMap,
    cfg: &DecoderConfig,
    ids: &[u32],
    tap: usize,
) -> Result<CudaTensor> {
    if tap >= cfg.num_layers() {
        // The last tap is the one entry HF norms; H3 was trained on an
        // un-normed mid-stack state and a normed one is a different input.
        return Err(msg(format!(
            "h3 text: tap {tap} of a {}-layer decoder would be post-norm",
            cfg.num_layers()
        )));
    }
    let positions: Vec<u32> = (0..ids.len() as u32).collect();
    let attend = vec![true; ids.len()];
    let mut taps = llm::hidden_states(map, cfg, ids, &positions, &attend, &[tap])?;
    taps.pop()
        .ok_or_else(|| msg("h3 text: the decoder returned no hidden state"))
}

/// Something that turns token ids into `hidden_states[tap]`. Two kinds exist:
/// the streaming encoder below, which holds nothing between prompts and pays
/// ~10 s of weight transfer per prompt, and a resident one ([`crate::llm`]'s
/// resident mode, FP8 to fit beside the DiT), which pays once. The pipeline
/// only ever sees this trait, so which one runs is a request-level choice.
pub trait HiddenStateEncoder {
    fn hidden_state(&self, ids: &[u32], tap: usize) -> Result<CudaTensor>;
    /// For reports: `"streamed"`, `"resident-bf16"`, `"resident-fp8"`.
    fn kind(&self) -> &'static str;
    /// Device bytes this encoder keeps between prompts.
    fn resident_bytes(&self) -> u64 {
        0
    }
}

/// One layer on the device at a time, nothing kept: the ~50 GB of decoder
/// weights stream through and are gone when a call returns.
pub struct StreamedEncoder<'a> {
    pub map: &'a WeightMap,
    pub cfg: &'a DecoderConfig,
}

impl HiddenStateEncoder for StreamedEncoder<'_> {
    fn hidden_state(&self, ids: &[u32], tap: usize) -> Result<CudaTensor> {
        encode_ids(self.map, self.cfg, ids, tap)
    }

    fn kind(&self) -> &'static str {
        "streamed"
    }
}

/// [`crate::llm::ResidentDecoder`]: layers 0..=49 stay on the device, a new
/// prompt costs a forward instead of 50 GB of transfers. At the checkpoint's
/// bf16 that is 50 GB, which does not fit beside the 41 GB DiT on a 96 GB card;
/// with weight-only FP8 rows (E4M3 codes + a scale per output row, dequantized
/// to bf16 per GEMM, activations untouched) it is 24.4 GB and does.
impl HiddenStateEncoder for crate::llm::ResidentDecoder {
    fn hidden_state(&self, ids: &[u32], tap: usize) -> Result<CudaTensor> {
        let positions: Vec<u32> = (0..ids.len() as u32).collect();
        let mut taps = self.hidden_states(ids, &positions, &vec![true; ids.len()], &[tap])?;
        taps.pop()
            .ok_or_else(|| msg("h3 text: the resident decoder returned no hidden state"))
    }

    fn kind(&self) -> &'static str {
        match self.precision() {
            crate::llm::WeightPrecision::Native => "resident-bf16",
            crate::llm::WeightPrecision::Fp8Rows => "resident-fp8",
        }
    }

    fn resident_bytes(&self) -> u64 {
        self.device_bytes()
    }
}

/// Load the resident encoder for `root` (`root/text_encoder`, either layout):
/// exactly the layers tap 50 needs, no final norm. The precision is
/// per-instance; the process-wide `FASTVIDEO_FP8` flag is not involved (it
/// would quantize the DiT as well).
pub fn load_resident_encoder(
    root: &Path,
    precision: crate::llm::WeightPrecision,
) -> Result<crate::llm::ResidentDecoder> {
    let map = WeightMap::open(&root.join("text_encoder"))?;
    let cfg = DecoderConfig::qwen3_vl_32b_text().for_bf16_reference();
    crate::llm::ResidentDecoder::load_with(
        &map,
        &cfg,
        H3TextEncoderConfig::fasth3_8step().output_hidden_state_index,
        precision,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStatus {
    /// No cache directory was given.
    Disabled,
    Hit,
    Miss,
}

impl CacheStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            CacheStatus::Disabled => "disabled",
            CacheStatus::Hit => "hit",
            CacheStatus::Miss => "miss",
        }
    }
}

pub struct TextConditioning {
    pub ids: Vec<u32>,
    /// `[1, S, 5120]`, un-normed `hidden_states[50]`.
    pub hidden: CudaTensor,
    pub cache: CacheStatus,
    /// Which encoder ran; `"cache"` on a hit, when none did.
    pub encoder: &'static str,
    /// Per-token DiT AdaLN tags for the text span (`TAG_TEXT` / `TAG_VIDEO`).
    /// Empty for text-only prompts (packing defaults every text row to TAG_TEXT).
    pub token_tags: Vec<u8>,
}

/// The conditioning the DiT consumes for `prompt`.
///
/// `root` holds `tokenizer/tokenizer.json` and `text_encoder/*.safetensors`:
/// either the published snapshot or the slim re-pack of [`super::slim`] — the
/// loader resolves tensors by name, so the layouts are interchangeable (and
/// share cache entries, whose key is built from tensors, not files).
///
/// With `cache_dir`, a prompt seen before costs a file read: the shards are
/// mapped for their headers (the key needs the encoder's identity) but no
/// weight is streamed and no forward runs. `resident` replaces the streaming
/// encoder on a miss.
pub fn encode_prompt_with(
    root: &Path,
    prompt: &str,
    cache_dir: Option<&Path>,
    resident: Option<&dyn HiddenStateEncoder>,
) -> Result<TextConditioning> {
    use super::text_cache as cache;
    let ids = tokenize(root, prompt)?;
    let map = WeightMap::open(&root.join("text_encoder"))?;
    // The reference encoder runs in bf16; follow its constant casts.
    let cfg = DecoderConfig::qwen3_vl_32b_text().for_bf16_reference();
    let tap = H3TextEncoderConfig::fasth3_8step().output_hidden_state_index;
    let streamed = StreamedEncoder {
        map: &map,
        cfg: &cfg,
    };
    let encoder: &dyn HiddenStateEncoder = resident.unwrap_or(&streamed);
    let Some(dir) = cache_dir else {
        let hidden = encoder.hidden_state(&ids, tap)?;
        return Ok(TextConditioning {
            ids,
            hidden,
            cache: CacheStatus::Disabled,
            encoder: encoder.kind(),
            token_tags: Vec::new(),
        });
    };

    let tokenizer_path = root.join("tokenizer").join("tokenizer.json");
    let tokenizer_bytes = std::fs::read(&tokenizer_path)
        .map_err(|e| msg(format!("{}: {e}", tokenizer_path.display())))?;
    let store = map
        .lazy()
        .ok_or_else(|| msg("h3 text: the encoder checkpoint was not opened lazily"))?;
    let key = cache::cache_key(
        prompt,
        &cache::sha256(&tokenizer_bytes),
        tap,
        &cache::encoder_identity(store, &cfg, tap)?,
    );
    let (entry, hit) = cache::get_or_compute(dir, &key, &ids, cfg.hidden, || {
        Ok(encoder.hidden_state(&ids, tap)?.host_cow()?.into_owned())
    })?;
    let hidden = CudaTensor::from_vec(entry.data, vec![1, ids.len(), cfg.hidden])?.to_device()?;
    Ok(TextConditioning {
        ids,
        hidden,
        cache: if hit {
            CacheStatus::Hit
        } else {
            CacheStatus::Miss
        },
        encoder: if hit { "cache" } else { encoder.kind() },
        token_tags: Vec::new(),
    })
}

/// [`encode_prompt_with`] without a cache or a resident encoder.
pub fn encode_prompt(root: &Path, prompt: &str) -> Result<(Vec<u32>, CudaTensor)> {
    let text = encode_prompt_with(root, prompt, None, None)?;
    Ok((text.ids, text.hidden))
}

/// SearchingMan recovered-8B path: tokenize with the DiT snapshot's H3
/// tokenizer, encode with a resident [`super::recovered_8b::Recovered8bEncoder`]
/// (tap 24 + adapter → `[1, S, 5120]`). Cache entries are keyed as width 5120.
pub fn encode_prompt_recovered(
    tokenizer_root: &Path,
    _encoder_root: &Path,
    prompt: &str,
    cache_dir: Option<&Path>,
    resident: Option<&dyn HiddenStateEncoder>,
) -> Result<TextConditioning> {
    use super::recovered_8b::RECOVERED_8B_TAP;
    use super::text_cache as cache;
    let encoder = resident.ok_or_else(|| msg("recovered-8b: resident encoder required"))?;
    let ids = tokenize(tokenizer_root, prompt)?;
    let tap = RECOVERED_8B_TAP;
    let out_hidden = 5120usize;
    let Some(dir) = cache_dir else {
        let hidden = encoder.hidden_state(&ids, tap)?;
        return Ok(TextConditioning {
            ids,
            hidden,
            cache: CacheStatus::Disabled,
            encoder: encoder.kind(),
            token_tags: Vec::new(),
        });
    };
    let tokenizer_path = tokenizer_root.join("tokenizer").join("tokenizer.json");
    let tokenizer_bytes = std::fs::read(&tokenizer_path)
        .map_err(|e| msg(format!("{}: {e}", tokenizer_path.display())))?;
    // Identity is the encoder kind + tap; recovered weights are not a LazyStore
    // under text_encoder/, so we fingerprint the kind string instead.
    let identity = cache::sha256(format!("recovered-8b-tap{tap}").as_bytes());
    let key = cache::cache_key(prompt, &cache::sha256(&tokenizer_bytes), tap, &identity);
    let (entry, hit) = cache::get_or_compute(dir, &key, &ids, out_hidden, || {
        Ok(encoder.hidden_state(&ids, tap)?.host_cow()?.into_owned())
    })?;
    let hidden = CudaTensor::from_vec(entry.data, vec![1, ids.len(), out_hidden])?.to_device()?;
    Ok(TextConditioning {
        ids,
        hidden,
        cache: if hit {
            CacheStatus::Hit
        } else {
            CacheStatus::Miss
        },
        encoder: if hit { "cache" } else { encoder.kind() },
        token_tags: Vec::new(),
    })
}

/// One RGB image for multimodal encode (HWC u8).
pub struct VisionImage<'a> {
    pub rgb: &'a [u8],
    pub height: usize,
    pub width: usize,
}

/// One video for multimodal encode: packed HWC frames at 24 fps canvas size.
pub struct VisionVideo<'a> {
    pub frames: &'a [u8],
    pub num_frames: usize,
    pub height: usize,
    pub width: usize,
}

/// FL2VA / Ref2VA multimodal text: presentation → vision tower → mRoPE LM.
pub fn encode_multimodal(
    root: &Path,
    prompt: &str,
    images: &[VisionImage<'_>],
    videos: &[VisionVideo<'_>],
    refs: Option<&[fastvideo_models::h3::presentation::PresentationRef]>,
) -> Result<TextConditioning> {
    use fastvideo_models::h3::config::{H3TextEncoderConfig, H3VisionConfig};
    use fastvideo_models::h3::mrope::build_mrope_positions;
    use fastvideo_models::h3::presentation::{
        build_fl2va_presentation, build_ref2va_presentation, PresentationRef,
    };
    use fastvideo_models::h3::vision_preprocess::{
        prepare_vision_image, prepare_vision_video, sample_qwen_video_frames,
    };

    let text_cfg = H3TextEncoderConfig::fasth3_8step();
    let vision_cfg = H3VisionConfig::fasth3_8step();
    let tok_path = root.join("tokenizer").join("tokenizer.json");
    let tokenizer = H3Tokenizer::from_file(&tok_path).map_err(msg)?;

    // Prepare vision pixels + grids; collect pad counts for the presentation.
    let map = WeightMap::open(&root.join("text_encoder"))?;
    let vision = super::vision::H3VisionTower::load(vision_cfg.clone(), &map)?;

    let mut image_prepared = Vec::with_capacity(images.len());
    let mut image_grids = Vec::with_capacity(images.len());
    let mut image_token_counts = Vec::with_capacity(images.len());
    for img in images {
        let prep =
            prepare_vision_image(img.rgb, img.height, img.width, &vision_cfg).map_err(msg)?;
        image_token_counts.push(
            prep.grid
                .num_tokens(vision_cfg.spatial_merge_size)
                .map_err(msg)?,
        );
        image_grids.push([prep.grid.temporal, prep.grid.height, prep.grid.width]);
        image_prepared.push(prep);
    }

    let mut video_prepared = Vec::with_capacity(videos.len());
    let mut video_grids = Vec::with_capacity(videos.len());
    let mut video_token_counts = Vec::with_capacity(videos.len());
    let mut video_timestamps = Vec::with_capacity(videos.len());
    for vid in videos {
        let (indices, timestamps) = sample_qwen_video_frames(
            vid.num_frames,
            vision_cfg.video_sample_fps(),
            vision_cfg.temporal_patch_size,
        )
        .map_err(msg)?;
        let frame_bytes = vid.height * vid.width * 3;
        let mut sampled = Vec::with_capacity(indices.len() * frame_bytes);
        for &idx in &indices {
            let start = idx.min(vid.num_frames - 1) * frame_bytes;
            sampled.extend_from_slice(&vid.frames[start..start + frame_bytes]);
        }
        let prep =
            prepare_vision_video(&sampled, indices.len(), vid.height, vid.width, &vision_cfg)
                .map_err(msg)?;
        if prep.grid.temporal != timestamps.len() {
            return Err(msg(format!(
                "vision video: grid T={} vs {} timestamps",
                prep.grid.temporal,
                timestamps.len()
            )));
        }
        video_token_counts.push(
            prep.grid
                .num_tokens(vision_cfg.spatial_merge_size)
                .map_err(msg)?,
        );
        video_grids.push([prep.grid.temporal, prep.grid.height, prep.grid.width]);
        video_timestamps.push(timestamps);
        video_prepared.push(prep);
    }

    let presentation = if let Some(ordered) = refs {
        // Rebuild ordered refs with computed token counts / timestamps.
        let mut rebuilt = Vec::with_capacity(ordered.len());
        let mut ii = 0usize;
        let mut vi = 0usize;
        for r in ordered {
            match r {
                PresentationRef::Image { .. } => {
                    rebuilt.push(PresentationRef::Image {
                        token_count: *image_token_counts
                            .get(ii)
                            .ok_or_else(|| msg("ref presentation: fewer images than Image refs"))?,
                    });
                    ii += 1;
                }
                PresentationRef::Video { .. } => {
                    rebuilt.push(PresentationRef::Video {
                        token_count: *video_token_counts
                            .get(vi)
                            .ok_or_else(|| msg("ref presentation: fewer videos than Video refs"))?,
                        block_timestamps: video_timestamps[vi].clone(),
                    });
                    vi += 1;
                }
                PresentationRef::Audio => rebuilt.push(PresentationRef::Audio),
            }
        }
        if ii != image_token_counts.len() || vi != video_token_counts.len() {
            return Err(msg(
                "ref presentation: image/video counts do not match refs",
            ));
        }
        build_ref2va_presentation(&tokenizer, &text_cfg, prompt, &rebuilt).map_err(msg)?
    } else {
        if !videos.is_empty() {
            return Err(msg("FL2VA multimodal encode does not take video refs"));
        }
        build_fl2va_presentation(&tokenizer, &text_cfg, prompt, &image_token_counts).map_err(msg)?
    };

    let ids = presentation.token_ids.clone();
    let token_tags = presentation.token_tags.clone();

    // Vision forward (batch all images / all videos).
    let (image_features, image_deepstack) = if image_prepared.is_empty() {
        (None, None)
    } else {
        let mut pixels = Vec::new();
        let mut grids = Vec::new();
        let patch_dim = image_prepared[0].patch_dim;
        for p in &image_prepared {
            if p.patch_dim != patch_dim {
                return Err(msg("image patch_dim mismatch across batch"));
            }
            pixels.extend_from_slice(&p.pixels);
            grids.push(p.grid.clone());
        }
        let (feat, deep) = vision.forward_grids(&pixels, patch_dim, &grids)?;
        (Some(feat), Some(deep))
    };

    let (video_features, video_deepstack) = if video_prepared.is_empty() {
        (None, None)
    } else {
        let mut pixels = Vec::new();
        let mut grids = Vec::new();
        let patch_dim = video_prepared[0].patch_dim;
        for p in &video_prepared {
            if p.patch_dim != patch_dim {
                return Err(msg("video patch_dim mismatch across batch"));
            }
            pixels.extend_from_slice(&p.pixels);
            grids.push(p.grid.clone());
        }
        let (feat, deep) = vision.forward_grids(&pixels, patch_dim, &grids)?;
        (Some(feat), Some(deep))
    };

    let cfg = DecoderConfig::qwen3_vl_32b_text().for_bf16_reference();
    let tap = text_cfg.output_hidden_state_index;
    let (embedded, image_mask, video_mask) = llm::embed_with_vision(
        &map,
        &cfg,
        &ids,
        text_cfg.image_token_id,
        image_features.as_ref(),
        text_cfg.video_token_id,
        video_features.as_ref(),
    )?;

    let visual_mask: Vec<bool> = image_mask
        .iter()
        .zip(video_mask.iter())
        .map(|(&i, &v)| i || v)
        .collect();
    let deepstack = merge_deepstack(
        image_deepstack,
        video_deepstack,
        &image_mask,
        &video_mask,
        &visual_mask,
        cfg.hidden,
    )?;

    let mrope_positions = build_mrope_positions(
        &ids,
        &text_cfg,
        vision_cfg.spatial_merge_size,
        &image_grids,
        &video_grids,
    )
    .map_err(msg)?;
    let mm = llm::MultimodalCtx {
        mrope_positions,
        mrope_section: text_cfg.mrope_section,
        visual_mask,
        deepstack,
    };
    let attend = vec![true; ids.len()];
    let mut taps = llm::hidden_states_multimodal(&map, &cfg, embedded, &attend, &[tap], &mm)?;
    let hidden = taps
        .pop()
        .ok_or_else(|| msg("h3 multimodal: decoder returned no hidden state"))?;

    Ok(TextConditioning {
        ids,
        hidden,
        cache: CacheStatus::Disabled,
        encoder: "streamed-multimodal",
        token_tags,
    })
}

fn merge_deepstack(
    image: Option<Vec<CudaTensor>>,
    video: Option<Vec<CudaTensor>>,
    image_mask: &[bool],
    video_mask: &[bool],
    visual_mask: &[bool],
    hidden: usize,
) -> Result<Vec<CudaTensor>> {
    match (image, video) {
        (None, None) => Ok(Vec::new()),
        (Some(img), None) => Ok(img),
        (None, Some(vid)) => Ok(vid),
        (Some(img), Some(vid)) => {
            if img.len() != vid.len() {
                return Err(msg("deepstack depth mismatch between image and video"));
            }
            let n_vis = visual_mask.iter().filter(|&&m| m).count();
            let mut out = Vec::with_capacity(img.len());
            for (im, vd) in img.into_iter().zip(vid.into_iter()) {
                let ih = im.host_cow()?;
                let vh = vd.host_cow()?;
                let mut combined = vec![0f32; n_vis * hidden];
                let mut ii = 0usize;
                let mut vi = 0usize;
                let mut oi = 0usize;
                for s in 0..visual_mask.len() {
                    if !visual_mask[s] {
                        continue;
                    }
                    if image_mask[s] {
                        combined[oi * hidden..(oi + 1) * hidden]
                            .copy_from_slice(&ih[ii * hidden..(ii + 1) * hidden]);
                        ii += 1;
                    } else if video_mask[s] {
                        combined[oi * hidden..(oi + 1) * hidden]
                            .copy_from_slice(&vh[vi * hidden..(vi + 1) * hidden]);
                        vi += 1;
                    }
                    oi += 1;
                }
                out.push(CudaTensor::from_vec(combined, vec![n_vis, hidden])?.to_device()?);
            }
            Ok(out)
        }
    }
}

/// Convenience: FL2VA keyframe images only.
pub fn encode_fl2va_multimodal(
    root: &Path,
    prompt: &str,
    images: &[VisionImage<'_>],
) -> Result<TextConditioning> {
    encode_multimodal(root, prompt, images, &[], None)
}

/// Convenience: Ref2VA ordered presentation refs (media buffers parallel to
/// Image/Video entries in `refs`; Audio entries carry no vision buffer).
pub fn encode_ref2va_multimodal(
    root: &Path,
    prompt: &str,
    refs: &[fastvideo_models::h3::presentation::PresentationRef],
    images: &[VisionImage<'_>],
    videos: &[VisionVideo<'_>],
) -> Result<TextConditioning> {
    encode_multimodal(root, prompt, images, videos, Some(refs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{Act, LayerAttn};

    fn tiny() -> DecoderConfig {
        DecoderConfig {
            vocab: 16,
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
            layers: vec![LayerAttn::global(5_000_000.0, 1.0); 3],
            layer_prefix: "m.layers".into(),
            embed_key: "m.embed.weight".into(),
            final_norm_key: "m.norm.weight".into(),
            attention_k_eq_v: false,
        }
    }

    fn weights() -> WeightMap {
        WeightMap::generated(|key, shape| {
            let seed = key
                .bytes()
                .fold(3u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
            (0..shape.iter().product::<usize>())
                .map(|i| {
                    let v = (seed.wrapping_add(i as u32).wrapping_mul(2_654_435_761) >> 8) as f32
                        / (1u32 << 24) as f32;
                    if key.contains("norm") {
                        0.5 + v
                    } else {
                        v - 0.5
                    }
                })
                .collect()
        })
    }

    #[test]
    fn the_tap_is_the_stream_after_that_many_layers_and_is_not_normed() {
        let (cfg, ids) = (tiny(), [1u32, 3, 0, 2]);
        let got = encode_ids(&weights(), &cfg, &ids, 2).unwrap();
        assert_eq!(got.shape, vec![1, 4, 8]);
        // A two-layer model's last tap is the same stream, but normed: the
        // un-normed tap must differ from it by exactly that norm.
        let mut two = cfg.clone();
        two.layers.truncate(2);
        let pos: Vec<u32> = (0..4).collect();
        let normed = llm::hidden_states(&weights(), &two, &ids, &pos, &[true; 4], &[2])
            .unwrap()
            .remove(0);
        let w = crate::wan::weights::cuda_tensor_shaped(&weights(), "m.norm.weight", &[8]).unwrap();
        let want = got.rms_norm(&w, cfg.rms_eps).unwrap();
        let (a, b) = (want.host_cow().unwrap(), normed.host_cow().unwrap());
        assert!(a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-6));
        assert!(got
            .host_cow()
            .unwrap()
            .iter()
            .zip(b.iter())
            .any(|(x, y)| (x - y).abs() > 1e-3));
    }

    #[test]
    fn the_resident_encoder_is_the_streamed_one_with_the_weights_left_in_place() {
        let (cfg, ids, map) = (tiny(), [1u32, 3, 0, 2], weights());
        let streamed = StreamedEncoder {
            map: &map,
            cfg: &cfg,
        }
        .hidden_state(&ids, 2)
        .unwrap();
        let resident = crate::llm::ResidentDecoder::load(&map, &cfg, 2).unwrap();
        let again = HiddenStateEncoder::hidden_state(&resident, &ids, 2).unwrap();
        assert_eq!(&*streamed.host_cow().unwrap(), &*again.host_cow().unwrap());
        assert_eq!(
            (
                StreamedEncoder {
                    map: &map,
                    cfg: &cfg
                }
                .kind(),
                resident.kind()
            ),
            ("streamed", "resident-bf16")
        );
        // Weight-only FP8 is a different, nearby function: close, not equal, and it says what it is.
        let fp8 = crate::llm::ResidentDecoder::load_with(
            &map,
            &cfg,
            2,
            crate::llm::WeightPrecision::Fp8Rows,
        )
        .unwrap();
        let quantized = HiddenStateEncoder::hidden_state(&fp8, &ids, 2).unwrap();
        let (a, b) = (streamed.host_cow().unwrap(), quantized.host_cow().unwrap());
        let (err, norm) = a.iter().zip(b.iter()).fold((0f64, 0f64), |(e, n), (x, y)| {
            (e + f64::from(x - y).powi(2), n + f64::from(*x).powi(2))
        });
        let rel = (err / norm).sqrt();
        assert!(rel > 0.0 && rel < 0.1, "fp8 rows vs native: rel {rel}");
        assert_eq!(fp8.kind(), "resident-fp8");
    }

    #[test]
    fn a_tap_that_would_be_normed_is_refused() {
        let err = encode_ids(&weights(), &tiny(), &[1, 2], 3).unwrap_err();
        assert!(err.to_string().contains("post-norm"), "{err}");
    }
}
