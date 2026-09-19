//! `LTX2VideoTransformer3DModel`: the 19B dual-stream DiT.
//!
//! Two token streams of different widths — video (4096) and audio (2048) — run
//! through the *same* 48 blocks. Each block, in order:
//!
//! 1. self-attention per stream (AdaLN in, gate out, the stream's own rotary);
//! 2. text cross-attention per stream — no modulation, no gate, no rotary;
//! 3. audio↔video cross-attention in both directions, both computed from the
//!    states as they were *before* either update, each side rotated by its own
//!    time-only table so tokens at the same instant line up;
//! 4. feed-forward per stream (AdaLN in, gate out).
//!
//! Everything that depends on the timestep is a per-step constant *vector*
//! (T2AV has one sigma per stream and the two are equal), so modulation is
//! broadcast over tokens. Every block norm is a weightless RMSNorm; the
//! `rms(x)·(1 + scale)` of AdaLN is therefore the RMSNorm kernel with
//! `1 + scale` as its weight, followed by one broadcast add.
//!
//! LTX-2.0 only: no gated attention, no prompt AdaLN, no STG/perturbed
//! attention, no masks (the connectors leave no padding), batch 1.
//! See docs/ports/ltx2.md §e.

use fastvideo_models::ltx2::config::Ltx2TransformerConfig;
use fastvideo_models::ltx2::Ltx2RopeTables;

use crate::wan::nn::{sinusoidal_timesteps, Linear};
use crate::wan::tensor::{CudaTensor, Result};
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

use super::attention::{Attention, AttentionDims, DeviceRope, FeedForward};
use super::keys::Keys;
use super::{msg, ones};

/// `LTX2AdaLayerNormSingle`: sinusoid → MLP → (`embedded`, `rows` modulation
/// vectors).
struct AdaLnSingle {
    linear_1: Linear,
    linear_2: Linear,
    linear: Linear,
    dim: usize,
    rows: usize,
    sinusoid: usize,
}

impl AdaLnSingle {
    fn load(map: &WeightMap, keys: &Keys, name: &str, dim: usize, rows: usize, sinusoid: usize) -> Result<Self> {
        let lin = |suffix: &str, i: usize, o: usize| Linear::load(map, &keys.key(&format!("{name}.{suffix}")), i, o, true);
        Ok(Self {
            linear_1: lin("emb.timestep_embedder.linear_1", sinusoid, dim)?,
            linear_2: lin("emb.timestep_embedder.linear_2", dim, dim)?,
            linear: lin("linear", dim, rows * dim)?,
            dim,
            rows,
            sinusoid,
        })
    }

    /// `(modulation [rows, dim], embedded [1, dim])` for one timestep
    /// (`1000·sigma`).
    fn forward(&self, timestep: f32) -> Result<(CudaTensor, CudaTensor)> {
        let s = sinusoidal_timesteps(&CudaTensor::from_vec(vec![timestep], vec![1])?, self.sinusoid)?;
        let e = self.linear_2.forward(&self.linear_1.forward(&s)?.silu())?;
        let m = self.linear.forward(&e.silu())?.reshape(vec![self.rows, self.dim])?;
        Ok((m, e))
    }
}

/// `PixArtAlphaTextProjection`: `Linear → tanh-GELU → Linear`.
struct CaptionProjection {
    linear_1: Linear,
    linear_2: Linear,
}

impl CaptionProjection {
    fn load(map: &WeightMap, keys: &Keys, name: &str, caption: usize, dim: usize) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::load(map, &keys.key(&format!("{name}.linear_1")), caption, dim, true)?,
            linear_2: Linear::load(map, &keys.key(&format!("{name}.linear_2")), dim, dim, true)?,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        self.linear_2.forward(&self.linear_1.forward_gelu(x)?)
    }
}

/// Row `r` of a `[rows, dim]` table as `[1, dim]`.
fn row(t: &CudaTensor, r: usize) -> Result<CudaTensor> {
    t.narrow(0, r, 1)
}

/// `rms(x) · (1 + scale) + shift` with a weightless RMSNorm.
fn rms_adaln(x: &CudaTensor, scale: &CudaTensor, shift: &CudaTensor, eps: f32) -> Result<CudaTensor> {
    x.rms_norm(&scale.try_add_scalar(1.0)?, eps)?.add(shift)
}

/// One stream's half of a block.
struct StreamBlock {
    attn1: Attention,
    attn2: Attention,
    ff: FeedForward,
    /// `[6, dim]`: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
    scale_shift_table: CudaTensor,
    /// `[5, dim]`: a2v_scale, a2v_shift, v2a_scale, v2a_shift, gate.
    cross_table: CudaTensor,
}

struct Block {
    video: StreamBlock,
    audio: StreamBlock,
    audio_to_video: Attention,
    video_to_audio: Attention,
}

/// The per-step modulation every block adds its own tables to.
struct StepModulation {
    /// `[6, dim]` from `time_embed` / `audio_time_embed`.
    main: CudaTensor,
    /// `[4, dim]` from `av_cross_attn_*_scale_shift`.
    cross: CudaTensor,
    /// `[1, dim]` from the stream's a↔v gate embedder.
    gate: CudaTensor,
}

/// The four rotary tables of one request geometry, on the device.
pub struct Ropes {
    pub video: DeviceRope,
    pub audio: DeviceRope,
    pub cross_video: DeviceRope,
    pub cross_audio: DeviceRope,
}

impl Ropes {
    pub fn new(cfg: &Ltx2TransformerConfig, grid: [usize; 3], audio_tokens: usize, fps: f32) -> Result<Self> {
        Self::upload(&Ltx2RopeTables::new(cfg, grid, audio_tokens, fps))
    }

    pub fn upload(t: &Ltx2RopeTables) -> Result<Self> {
        Ok(Self {
            video: DeviceRope::upload(&t.video)?,
            audio: DeviceRope::upload(&t.audio)?,
            cross_video: DeviceRope::upload(&t.cross_video)?,
            cross_audio: DeviceRope::upload(&t.cross_audio)?,
        })
    }
}

/// The connector outputs after the DiT's caption projections: `[1, T, 4096]`
/// and `[1, T, 2048]`. Timestep-independent, so computed once per prompt.
pub struct TextConditioning {
    pub video: CudaTensor,
    pub audio: CudaTensor,
}

/// Called after each block with `(index, video, audio)`.
pub type BlockObserver<'a> = &'a mut dyn FnMut(usize, &CudaTensor, &CudaTensor) -> Result<()>;

pub struct Ltx2Transformer {
    cfg: Ltx2TransformerConfig,
    proj_in: Linear,
    audio_proj_in: Linear,
    caption_projection: CaptionProjection,
    audio_caption_projection: CaptionProjection,
    time_embed: AdaLnSingle,
    audio_time_embed: AdaLnSingle,
    cross_video_scale_shift: AdaLnSingle,
    cross_audio_scale_shift: AdaLnSingle,
    cross_video_gate: AdaLnSingle,
    cross_audio_gate: AdaLnSingle,
    /// `[2, dim]`: shift, scale of the output head.
    scale_shift_table: CudaTensor,
    audio_scale_shift_table: CudaTensor,
    proj_out: Linear,
    audio_proj_out: Linear,
    blocks: Vec<Block>,
    ones_video: CudaTensor,
    ones_audio: CudaTensor,
}

fn table(map: &WeightMap, key: &str, rows: usize, dim: usize) -> Result<CudaTensor> {
    let mut t = cuda_tensor_shaped(map, key, &[rows, dim])?;
    t.pin_device()?;
    Ok(t)
}

impl Ltx2Transformer {
    /// Load every block onto the device (37.8 GB as bf16). `keys` names the
    /// checkpoint's layout: the diffusers `transformer/` folder or the single
    /// `ltx-2-19b-*.safetensors`.
    pub fn load(map: &WeightMap, keys: &Keys, cfg: &Ltx2TransformerConfig) -> Result<Self> {
        if cfg.gated_attn || cfg.audio_gated_attn || cfg.cross_attn_mod || cfg.audio_cross_attn_mod || cfg.perturbed_attn || !cfg.use_prompt_embeddings {
            return Err(msg("ltx2 dit: gated attention, prompt modulation and perturbed attention are LTX-2.3+, not supported"));
        }
        if cfg.norm_elementwise_affine || cfg.patch_size != 1 || cfg.patch_size_t != 1 || !cfg.attention_bias || !cfg.attention_out_bias {
            return Err(msg("ltx2 dit: expected weightless block norms, 1x1x1 patches and biased attention"));
        }
        if cfg.cross_attention_dim != cfg.inner_dim() || cfg.audio_cross_attention_dim != cfg.audio_inner_dim() {
            return Err(msg("ltx2 dit: the caption projections must lift the text to each stream's own width"));
        }
        let (dv, da, eps) = (cfg.inner_dim(), cfg.audio_inner_dim(), cfg.norm_eps as f32);
        let (hv, ha) = (cfg.num_attention_heads, cfg.audio_num_attention_heads);
        let video_dims = AttentionDims { query_dim: dv, context_dim: dv, heads: hv, head_dim: cfg.attention_head_dim };
        let audio_dims = AttentionDims { query_dim: da, context_dim: da, heads: ha, head_dim: cfg.audio_attention_head_dim };
        // Both directions attend in the audio head layout.
        let a2v_dims = AttentionDims { query_dim: dv, context_dim: da, ..audio_dims };
        let v2a_dims = AttentionDims { query_dim: da, context_dim: dv, ..audio_dims };

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let p = format!("transformer_blocks.{i}");
            let attn = |name: &str, dims: AttentionDims| Attention::load(map, keys, &format!("{p}.{name}"), dims, eps);
            blocks.push(Block {
                video: StreamBlock {
                    attn1: attn("attn1", video_dims)?,
                    attn2: attn("attn2", video_dims)?,
                    ff: FeedForward::load(map, keys, &format!("{p}.ff"), dv, cfg.ff_inner_dim())?,
                    scale_shift_table: table(map, &keys.key(&format!("{p}.scale_shift_table")), 6, dv)?,
                    cross_table: table(map, &keys.key(&format!("{p}.video_a2v_cross_attn_scale_shift_table")), 5, dv)?,
                },
                audio: StreamBlock {
                    attn1: attn("audio_attn1", audio_dims)?,
                    attn2: attn("audio_attn2", audio_dims)?,
                    ff: FeedForward::load(map, keys, &format!("{p}.audio_ff"), da, cfg.audio_ff_inner_dim())?,
                    scale_shift_table: table(map, &keys.key(&format!("{p}.audio_scale_shift_table")), 6, da)?,
                    cross_table: table(map, &keys.key(&format!("{p}.audio_a2v_cross_attn_scale_shift_table")), 5, da)?,
                },
                audio_to_video: attn("audio_to_video_attn", a2v_dims)?,
                video_to_audio: attn("video_to_audio_attn", v2a_dims)?,
            });
            if (i + 1) % 8 == 0 || i + 1 == cfg.num_layers {
                crate::wan::log::info(format_args!("ltx2 dit: loaded block {}/{}", i + 1, cfg.num_layers));
            }
        }
        let ada = |name: &str, dim: usize, rows: usize| AdaLnSingle::load(map, keys, name, dim, rows, cfg.timestep_proj_dim);
        Ok(Self {
            proj_in: Linear::load(map, &keys.key("proj_in"), cfg.in_channels, dv, true)?,
            audio_proj_in: Linear::load(map, &keys.key("audio_proj_in"), cfg.audio_in_channels, da, true)?,
            caption_projection: CaptionProjection::load(map, keys, "caption_projection", cfg.caption_channels, dv)?,
            audio_caption_projection: CaptionProjection::load(map, keys, "audio_caption_projection", cfg.caption_channels, da)?,
            time_embed: ada("time_embed", dv, 6)?,
            audio_time_embed: ada("audio_time_embed", da, 6)?,
            cross_video_scale_shift: ada("av_cross_attn_video_scale_shift", dv, 4)?,
            cross_audio_scale_shift: ada("av_cross_attn_audio_scale_shift", da, 4)?,
            cross_video_gate: ada("av_cross_attn_video_a2v_gate", dv, 1)?,
            cross_audio_gate: ada("av_cross_attn_audio_v2a_gate", da, 1)?,
            scale_shift_table: table(map, &keys.key("scale_shift_table"), 2, dv)?,
            audio_scale_shift_table: table(map, &keys.key("audio_scale_shift_table"), 2, da)?,
            proj_out: Linear::load(map, &keys.key("proj_out"), dv, cfg.out_channels, true)?,
            audio_proj_out: Linear::load(map, &keys.key("audio_proj_out"), da, cfg.audio_out_channels, true)?,
            blocks,
            ones_video: ones(dv)?,
            ones_audio: ones(da)?,
            cfg: cfg.clone(),
        })
    }

    pub fn config(&self) -> &Ltx2TransformerConfig {
        &self.cfg
    }

    /// The connectors' `[1, T, 3840]` contexts, lifted to each stream's width.
    pub fn project_text(&self, video: &CudaTensor, audio: &CudaTensor) -> Result<TextConditioning> {
        Ok(TextConditioning { video: self.caption_projection.forward(video)?, audio: self.audio_caption_projection.forward(audio)? })
    }

    /// One joint forward. `video`: packed latents `[1, S, 128]`; `audio`:
    /// `[1, L, 128]`; `timestep` = `1000·sigma`, shared by both streams.
    /// Returns the two velocities, same shapes.
    pub fn forward(
        &self,
        video: &CudaTensor,
        audio: &CudaTensor,
        text: &TextConditioning,
        timestep: f32,
        ropes: &Ropes,
        mut observer: Option<BlockObserver<'_>>,
    ) -> Result<(CudaTensor, CudaTensor)> {
        if video.rank() != 3 || audio.rank() != 3 || video.shape[0] != 1 || audio.shape[0] != 1 {
            return Err(msg(format!("ltx2 dit expects [1, S, C] and [1, L, C], got {:?} and {:?}", video.shape, audio.shape)));
        }
        let eps = self.cfg.norm_eps as f32;
        // The a↔v gate embedders see the timestep rescaled by
        // cross_attn_timestep_scale_multiplier / timestep_scale_multiplier (= 1).
        let gate_t = timestep * (self.cfg.cross_attn_timestep_scale_multiplier / self.cfg.timestep_scale_multiplier) as f32;
        let (v_main, v_embedded) = self.time_embed.forward(timestep)?;
        let (a_main, a_embedded) = self.audio_time_embed.forward(timestep)?;
        let v_mod = StepModulation { main: v_main, cross: self.cross_video_scale_shift.forward(timestep)?.0, gate: self.cross_video_gate.forward(gate_t)?.0 };
        let a_mod = StepModulation { main: a_main, cross: self.cross_audio_scale_shift.forward(timestep)?.0, gate: self.cross_audio_gate.forward(gate_t)?.0 };

        let mut xv = self.proj_in.forward(video)?;
        let mut xa = self.audio_proj_in.forward(audio)?;
        for (i, block) in self.blocks.iter().enumerate() {
            (xv, xa) = self.block(block, xv, xa, text, &v_mod, &a_mod, ropes, eps)?;
            if let Some(obs) = observer.as_mut() {
                obs(i, &xv, &xa)?;
            }
        }
        let head = |x: &CudaTensor, tab: &CudaTensor, e: &CudaTensor, out: &Linear| -> Result<CudaTensor> {
            // (shift, scale) = table[2, D] + embedded; LayerNorm here, not RMSNorm.
            let dim = e.shape[1];
            let mods = tab.add(e)?.reshape(vec![1, 2, dim])?;
            out.forward(&x.ln_adaln_e(&mods, 1, 0, eps)?)
        };
        Ok((
            head(&xv, &self.scale_shift_table, &v_embedded, &self.proj_out)?,
            head(&xa, &self.audio_scale_shift_table, &a_embedded, &self.audio_proj_out)?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        b: &Block,
        xv: CudaTensor,
        xa: CudaTensor,
        text: &TextConditioning,
        v_mod: &StepModulation,
        a_mod: &StepModulation,
        ropes: &Ropes,
        eps: f32,
    ) -> Result<(CudaTensor, CudaTensor)> {
        let (dv, da) = (xv.shape[2], xa.shape[2]);
        // table + per-step modulation, kept as [1, rows, D] for the gated adds.
        let v_tab = b.video.scale_shift_table.add(&v_mod.main)?;
        let a_tab = b.audio.scale_shift_table.add(&a_mod.main)?;
        let (v_gates, a_gates) = (v_tab.reshape(vec![1, 6, dv])?, a_tab.reshape(vec![1, 6, da])?);

        // 1. self-attention.
        let h = rms_adaln(&xv, &row(&v_tab, 1)?, &row(&v_tab, 0)?, eps)?;
        let xv = xv.residual_gate_add_e(&b.video.attn1.forward(&h, None, Some(&ropes.video), None)?, &v_gates, 2)?;
        let h = rms_adaln(&xa, &row(&a_tab, 1)?, &row(&a_tab, 0)?, eps)?;
        let xa = xa.residual_gate_add_e(&b.audio.attn1.forward(&h, None, Some(&ropes.audio), None)?, &a_gates, 2)?;

        // 2. text cross-attention: plain residual.
        let xv = xv.add(&b.video.attn2.forward(&xv.rms_norm(&self.ones_video, eps)?, Some(&text.video), None, None)?)?;
        let xa = xa.add(&b.audio.attn2.forward(&xa.rms_norm(&self.ones_audio, eps)?, Some(&text.audio), None, None)?)?;

        // 3. audio↔video, both directions from the same pre-update states.
        // Rows 0..3 of each side's table: a2v_scale, a2v_shift, v2a_scale, v2a_shift
        // (scale first here); row 4 is the gate, modulated by its own embedder.
        let v_cross = b.video.cross_table.narrow(0, 0, 4)?.add(&v_mod.cross)?;
        let a_cross = b.audio.cross_table.narrow(0, 0, 4)?.add(&a_mod.cross)?;
        let a2v_gate = row(&b.video.cross_table, 4)?.add(&v_mod.gate)?.reshape(vec![1, 1, dv])?;
        let v2a_gate = row(&b.audio.cross_table, 4)?.add(&a_mod.gate)?.reshape(vec![1, 1, da])?;
        let side = |x: &CudaTensor, cross: &CudaTensor, first: usize| rms_adaln(x, &row(cross, first)?, &row(cross, first + 1)?, eps);
        let a2v = b.audio_to_video.forward(&side(&xv, &v_cross, 0)?, Some(&side(&xa, &a_cross, 0)?), Some(&ropes.cross_video), Some(&ropes.cross_audio))?;
        let v2a = b.video_to_audio.forward(&side(&xa, &a_cross, 2)?, Some(&side(&xv, &v_cross, 2)?), Some(&ropes.cross_audio), Some(&ropes.cross_video))?;
        let xv = xv.residual_gate_add_e(&a2v, &a2v_gate, 0)?;
        let xa = xa.residual_gate_add_e(&v2a, &v2a_gate, 0)?;

        // 4. feed-forward.
        let h = rms_adaln(&xv, &row(&v_tab, 4)?, &row(&v_tab, 3)?, eps)?;
        let xv = xv.residual_gate_add_e(&b.video.ff.forward(&h)?, &v_gates, 5)?;
        let h = rms_adaln(&xa, &row(&a_tab, 4)?, &row(&a_tab, 3)?, eps)?;
        let xa = xa.residual_gate_add_e(&b.audio.ff.forward(&h)?, &a_gates, 5)?;
        Ok((xv, xa))
    }
}

/// `[1, C, F, H, W]` → `[1, F·H·W, C]`, tokens frame-major, then row, then column.
pub fn pack_video(latents: &CudaTensor) -> Result<CudaTensor> {
    let [b, c, f, h, w] = latents.shape[..] else {
        return Err(msg(format!("pack_video expects [1, C, F, H, W], got {:?}", latents.shape)));
    };
    latents.reshape(vec![b, c, f * h * w])?.permute(&[0, 2, 1])
}

/// The inverse of [`pack_video`] for a `[frames, height, width]` grid.
pub fn unpack_video(tokens: &CudaTensor, grid: [usize; 3]) -> Result<CudaTensor> {
    let [b, s, c] = tokens.shape[..] else {
        return Err(msg(format!("unpack_video expects [1, S, C], got {:?}", tokens.shape)));
    };
    let [f, h, w] = grid;
    if s != f * h * w {
        return Err(msg(format!("unpack_video: {s} tokens for a {f}x{h}x{w} grid")));
    }
    tokens.permute(&[0, 2, 1])?.reshape(vec![b, c, f, h, w])
}

#[cfg(test)]
mod tests {
    use super::super::attention::tests::{assert_close, attention_reference, get, linear, rms, rows, tensor, tokens, weights};
    use super::super::keys::Layout;
    use super::*;

    fn tiny() -> Ltx2TransformerConfig {
        Ltx2TransformerConfig {
            in_channels: 6,
            out_channels: 6,
            num_attention_heads: 2,
            attention_head_dim: 8,
            cross_attention_dim: 16,
            audio_in_channels: 5,
            audio_out_channels: 5,
            audio_num_attention_heads: 2,
            audio_attention_head_dim: 4,
            audio_cross_attention_dim: 8,
            num_layers: 2,
            caption_channels: 12,
            timestep_proj_dim: 8,
            ..Ltx2TransformerConfig::ltx2_19b()
        }
    }

    fn add(a: &mut [f32], b: &[f32]) {
        a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
    }

    fn gelu(v: f32) -> f32 {
        0.5 * v * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (v + 0.044_715 * v * v * v)).tanh())
    }

    fn silu(v: f32) -> f32 {
        v / (1.0 + (-v).exp())
    }

    fn lin(map: &WeightMap, prefix: &str, x: &[f32], o: usize) -> Vec<f32> {
        linear(x, &get(map, &format!("{prefix}.weight"), &[o, x.len()]), &get(map, &format!("{prefix}.bias"), &[o]))
    }

    /// `(modulation rows, embedded)` of an AdaLN-single, from its definition.
    fn adaln(map: &WeightMap, name: &str, t: f32, dim: usize, rows_n: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
        let half = 4;
        let s: Vec<f32> = (0..8)
            .map(|i| {
                let arg = t * (-(10000f32.ln()) * (i % half) as f32 / half as f32).exp();
                if i < half { arg.cos() } else { arg.sin() }
            })
            .collect();
        let h: Vec<f32> = lin(map, &format!("{name}.emb.timestep_embedder.linear_1"), &s, dim).into_iter().map(silu).collect();
        let e = lin(map, &format!("{name}.emb.timestep_embedder.linear_2"), &h, dim);
        let m = lin(map, &format!("{name}.linear"), &e.iter().map(|v| silu(*v)).collect::<Vec<_>>(), rows_n * dim);
        (m.chunks_exact(dim).map(<[f32]>::to_vec).collect(), e)
    }

    fn table_plus(map: &WeightMap, key: &str, rows_n: usize, dim: usize, m: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let t = get(map, key, &[rows_n, dim]);
        (0..m.len()).map(|r| t[r * dim..(r + 1) * dim].iter().zip(&m[r]).map(|(a, b)| a + b).collect()).collect()
    }

    fn adaln_norm(x: &[Vec<f32>], scale: &[f32], shift: &[f32]) -> Vec<Vec<f32>> {
        x.iter().map(|v| rms(v, None, 1e-6).iter().enumerate().map(|(i, n)| n * (1.0 + scale[i]) + shift[i]).collect()).collect()
    }

    fn gated_add(x: &mut [Vec<f32>], update: &[Vec<f32>], gate: &[f32]) {
        for (v, u) in x.iter_mut().zip(update) {
            v.iter_mut().enumerate().for_each(|(i, a)| *a += u[i] * gate[i]);
        }
    }

    fn ff(map: &WeightMap, prefix: &str, x: &[Vec<f32>], dim: usize) -> Vec<Vec<f32>> {
        x.iter()
            .map(|v| {
                let up: Vec<f32> = lin(map, &format!("{prefix}.net.0.proj"), v, dim * 4).into_iter().map(gelu).collect();
                lin(map, &format!("{prefix}.net.2"), &up, dim)
            })
            .collect()
    }

    /// The whole model against the block order of docs/ports/ltx2.md §e written
    /// as loops: every modulation row, both a↔v directions from pre-update
    /// states, the four rotary tables, the LayerNorm heads.
    #[test]
    fn forward_matches_a_loop_reference() {
        let cfg = tiny();
        let map = weights();
        let model = Ltx2Transformer::load(&map, &Keys::transformer(Layout::Diffusers), &cfg).unwrap();
        let (grid, l, t_len, timestep) = ([2usize, 1, 3], 4usize, 5usize, 725.0f32);
        let s = 6;
        let tables = Ltx2RopeTables::new(&cfg, grid, l, 24.0);
        let ropes = Ropes::upload(&tables).unwrap();
        let (video, audio) = (tokens(s, 6, 0.41), tokens(l, 5, 0.83));
        let (ctx_v, ctx_a) = (tokens(t_len, 12, 0.29), tokens(t_len, 12, 0.57));
        let text = model.project_text(&tensor(&ctx_v), &tensor(&ctx_a)).unwrap();
        let mut seen = Vec::new();
        let mut obs = |i: usize, v: &CudaTensor, a: &CudaTensor| -> Result<()> {
            seen.push((i, rows(v, 16), rows(a, 8)));
            Ok(())
        };
        let (got_v, got_a) = model.forward(&tensor(&video), &tensor(&audio), &text, timestep, &ropes, Some(&mut obs)).unwrap();
        assert_eq!((got_v.shape.clone(), got_a.shape.clone()), (vec![1, s, 6], vec![1, l, 5]));
        assert_eq!(seen.len(), 2, "one observation per block");

        let (dv, da) = (16usize, 8usize);
        let caption = |name: &str, x: &[Vec<f32>], d: usize| -> Vec<Vec<f32>> {
            x.iter().map(|v| lin(&map, &format!("{name}.linear_2"), &lin(&map, &format!("{name}.linear_1"), v, d).into_iter().map(gelu).collect::<Vec<_>>(), d)).collect()
        };
        let (tv, ta) = (caption("caption_projection", &ctx_v, dv), caption("audio_caption_projection", &ctx_a, da));
        let (v_main, v_emb) = adaln(&map, "time_embed", timestep, dv, 6);
        let (a_main, a_emb) = adaln(&map, "audio_time_embed", timestep, da, 6);
        let v_cross = adaln(&map, "av_cross_attn_video_scale_shift", timestep, dv, 4).0;
        let a_cross = adaln(&map, "av_cross_attn_audio_scale_shift", timestep, da, 4).0;
        let v_gate = adaln(&map, "av_cross_attn_video_a2v_gate", timestep, dv, 1).0;
        let a_gate = adaln(&map, "av_cross_attn_audio_v2a_gate", timestep, da, 1).0;

        let mut xv: Vec<Vec<f32>> = video.iter().map(|v| lin(&map, "proj_in", v, dv)).collect();
        let mut xa: Vec<Vec<f32>> = audio.iter().map(|v| lin(&map, "audio_proj_in", v, da)).collect();
        let vd = AttentionDims { query_dim: dv, context_dim: dv, heads: 2, head_dim: 8 };
        let ad = AttentionDims { query_dim: da, context_dim: da, heads: 2, head_dim: 4 };
        for (i, tap) in seen.iter().enumerate() {
            let p = format!("transformer_blocks.{i}");
            let vt = table_plus(&map, &format!("{p}.scale_shift_table"), 6, dv, &v_main);
            let at = table_plus(&map, &format!("{p}.audio_scale_shift_table"), 6, da, &a_main);
            // 1. self-attention: rows shift, scale, gate.
            let h = adaln_norm(&xv, &vt[1], &vt[0]);
            let u = attention_reference(&map, &format!("{p}.attn1"), vd, &h, &h, Some(&tables.video), None);
            gated_add(&mut xv, &u, &vt[2]);
            let h = adaln_norm(&xa, &at[1], &at[0]);
            let u = attention_reference(&map, &format!("{p}.audio_attn1"), ad, &h, &h, Some(&tables.audio), None);
            gated_add(&mut xa, &u, &at[2]);
            // 2. text cross-attention.
            let h: Vec<Vec<f32>> = xv.iter().map(|v| rms(v, None, 1e-6)).collect();
            let u = attention_reference(&map, &format!("{p}.attn2"), vd, &h, &tv, None, None);
            xv.iter_mut().zip(&u).for_each(|(a, b)| add(a, b));
            let h: Vec<Vec<f32>> = xa.iter().map(|v| rms(v, None, 1e-6)).collect();
            let u = attention_reference(&map, &format!("{p}.audio_attn2"), ad, &h, &ta, None, None);
            xa.iter_mut().zip(&u).for_each(|(a, b)| add(a, b));
            // 3. a↔v from the same pre-update states; rows scale, shift per direction.
            let vc = table_plus(&map, &format!("{p}.video_a2v_cross_attn_scale_shift_table"), 5, dv, &v_cross);
            let ac = table_plus(&map, &format!("{p}.audio_a2v_cross_attn_scale_shift_table"), 5, da, &a_cross);
            let g_v: Vec<f32> = get(&map, &format!("{p}.video_a2v_cross_attn_scale_shift_table"), &[5, dv])[4 * dv..].iter().zip(&v_gate[0]).map(|(a, b)| a + b).collect();
            let g_a: Vec<f32> = get(&map, &format!("{p}.audio_a2v_cross_attn_scale_shift_table"), &[5, da])[4 * da..].iter().zip(&a_gate[0]).map(|(a, b)| a + b).collect();
            let a2v = attention_reference(
                &map,
                &format!("{p}.audio_to_video_attn"),
                AttentionDims { query_dim: dv, context_dim: da, ..ad },
                &adaln_norm(&xv, &vc[0], &vc[1]),
                &adaln_norm(&xa, &ac[0], &ac[1]),
                Some(&tables.cross_video),
                Some(&tables.cross_audio),
            );
            let v2a = attention_reference(
                &map,
                &format!("{p}.video_to_audio_attn"),
                AttentionDims { query_dim: da, context_dim: dv, ..ad },
                &adaln_norm(&xa, &ac[2], &ac[3]),
                &adaln_norm(&xv, &vc[2], &vc[3]),
                Some(&tables.cross_audio),
                Some(&tables.cross_video),
            );
            gated_add(&mut xv, &a2v, &g_v);
            gated_add(&mut xa, &v2a, &g_a);
            // 4. feed-forward.
            let u = ff(&map, &format!("{p}.ff"), &adaln_norm(&xv, &vt[4], &vt[3]), dv);
            gated_add(&mut xv, &u, &vt[5]);
            let u = ff(&map, &format!("{p}.audio_ff"), &adaln_norm(&xa, &at[4], &at[3]), da);
            gated_add(&mut xa, &u, &at[5]);
            assert_eq!(tap.0, i);
            assert_close(&tap.1, &xv, 2e-4, &format!("block {i} video"));
            assert_close(&tap.2, &xa, 2e-4, &format!("block {i} audio"));
        }
        let head = |x: &[Vec<f32>], tab: &str, e: &[f32], out: &str, d: usize, c: usize| -> Vec<Vec<f32>> {
            let t = get(&map, tab, &[2, d]);
            x.iter()
                .map(|v| {
                    let mean = v.iter().sum::<f32>() / d as f32;
                    let var = v.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / d as f32;
                    let n: Vec<f32> = v.iter().enumerate().map(|(i, a)| (a - mean) / (var + 1e-6).sqrt() * (1.0 + t[d + i] + e[i]) + t[i] + e[i]).collect();
                    lin(&map, out, &n, c)
                })
                .collect()
        };
        assert_close(&rows(&got_v, 6), &head(&xv, "scale_shift_table", &v_emb, "proj_out", dv, 6), 2e-4, "video velocity");
        assert_close(&rows(&got_a, 5), &head(&xa, "audio_scale_shift_table", &a_emb, "audio_proj_out", da, 5), 2e-4, "audio velocity");
    }

    #[test]
    fn packing_is_frame_major_then_row_then_column() {
        let z = CudaTensor::from_vec((0..12).map(|i| i as f32).collect(), vec![1, 2, 2, 1, 3]).unwrap();
        let p = pack_video(&z).unwrap();
        assert_eq!(p.shape, vec![1, 6, 2]);
        // Token (f=1, h=0, w=2) = index 5: channel 0 → 5, channel 1 → 11.
        assert_eq!(&p.host_cow().unwrap()[10..12], &[5.0, 11.0]);
        let back = unpack_video(&p, [2, 1, 3]).unwrap();
        assert_eq!(&*back.host_cow().unwrap(), &*z.host_cow().unwrap());
        assert!(unpack_video(&p, [2, 2, 3]).is_err());
    }

    #[test]
    fn features_of_later_checkpoints_are_refused_at_load() {
        let cfg = Ltx2TransformerConfig { gated_attn: true, ..tiny() };
        let err = Ltx2Transformer::load(&weights(), &Keys::transformer(Layout::Diffusers), &cfg).err().map(|e| e.to_string());
        assert!(err.is_some_and(|e| e.contains("LTX-2.3")));
    }
}
