//! CLIP ViT-H/14 vision encoder for Wan I2V (`image_encoder/`).
//!
//! Port of Candle `fastvideo_models::wan::clip` onto cudarc `CudaTensor`.

use super::nn::{self, Linear};
use super::tensor::{CudaTensor, Result, TensorError};
use super::weights::{self, WeightMap};

const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

#[derive(Debug, Clone)]
pub struct ClipVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub layer_norm_eps: f32,
}

impl ClipVisionConfig {
    pub fn vit_h_14() -> Self {
        Self {
            hidden_size: 1280,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            image_size: 224,
            patch_size: 14,
            layer_norm_eps: 1e-5,
        }
    }

    pub fn tiny() -> Self {
        Self {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            image_size: 16,
            patch_size: 8,
            layer_norm_eps: 1e-5,
        }
    }

    pub fn num_patches(&self) -> usize {
        let n = self.image_size / self.patch_size;
        n * n
    }

    pub fn num_positions(&self) -> usize {
        self.num_patches() + 1
    }
}

#[derive(Debug, Clone)]
struct ClipAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    dim_head: usize,
}

impl ClipAttention {
    fn load(map: &WeightMap, prefix: &str, dim: usize, heads: usize) -> Result<Self> {
        Ok(Self {
            q: Linear::load(map, &weights::join_key(prefix, "q_proj"), dim, dim, true)?,
            k: Linear::load(map, &weights::join_key(prefix, "k_proj"), dim, dim, true)?,
            v: Linear::load(map, &weights::join_key(prefix, "v_proj"), dim, dim, true)?,
            out: Linear::load(map, &weights::join_key(prefix, "out_proj"), dim, dim, true)?,
            heads,
            dim_head: dim / heads,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        let (b, s, _) = (xs.shape[0], xs.shape[1], xs.shape[2]);
        let q = self.q.forward(xs)?;
        let k = self.k.forward(xs)?;
        let v = self.v.forward(xs)?;
        let q = q
            .reshape(vec![b, s, self.heads, self.dim_head])?
            .transpose(1, 2)?;
        let k = k
            .reshape(vec![b, s, self.heads, self.dim_head])?
            .transpose(1, 2)?;
        let v = v
            .reshape(vec![b, s, self.heads, self.dim_head])?
            .transpose(1, 2)?;
        let attn = nn::scaled_dot_product_attention(&q, &k, &v, None)?;
        let attn = attn
            .transpose(1, 2)?
            .reshape(vec![b, s, self.heads * self.dim_head])?;
        self.out.forward(&attn)
    }
}

#[derive(Debug, Clone)]
struct ClipMlp {
    fc1: Linear,
    fc2: Linear,
}

impl ClipMlp {
    fn load(map: &WeightMap, prefix: &str, dim: usize, inner: usize) -> Result<Self> {
        Ok(Self {
            fc1: Linear::load(map, &weights::join_key(prefix, "fc1"), dim, inner, true)?,
            fc2: Linear::load(map, &weights::join_key(prefix, "fc2"), inner, dim, true)?,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        self.fc2.forward(&nn::gelu(&self.fc1.forward(xs)?))
    }
}

#[derive(Debug, Clone)]
struct ClipEncoderLayer {
    attn: ClipAttention,
    ln1_w: CudaTensor,
    ln1_b: CudaTensor,
    mlp: ClipMlp,
    ln2_w: CudaTensor,
    ln2_b: CudaTensor,
    eps: f32,
}

impl ClipEncoderLayer {
    fn load(map: &WeightMap, prefix: &str, cfg: &ClipVisionConfig) -> Result<Self> {
        let dim = cfg.hidden_size;
        Ok(Self {
            attn: ClipAttention::load(
                map,
                &weights::join_key(prefix, "self_attn"),
                dim,
                cfg.num_attention_heads,
            )?,
            ln1_w: weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "layer_norm1.weight"),
                &[dim],
            )?,
            ln1_b: weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "layer_norm1.bias"),
                &[dim],
            )?,
            mlp: ClipMlp::load(
                map,
                &weights::join_key(prefix, "mlp"),
                dim,
                cfg.intermediate_size,
            )?,
            ln2_w: weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "layer_norm2.weight"),
                &[dim],
            )?,
            ln2_b: weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "layer_norm2.bias"),
                &[dim],
            )?,
            eps: cfg.layer_norm_eps,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        let n1 = nn::layer_norm(xs, self.eps, Some(&self.ln1_w), Some(&self.ln1_b))?;
        let xs = xs.add(&self.attn.forward(&n1)?)?;
        let n2 = nn::layer_norm(&xs, self.eps, Some(&self.ln2_w), Some(&self.ln2_b))?;
        xs.add(&self.mlp.forward(&n2)?)
    }
}

#[derive(Debug, Clone)]
pub struct ClipVision {
    pub cfg: ClipVisionConfig,
    class_embedding: CudaTensor,
    patch_weight: CudaTensor,
    position_embedding: CudaTensor,
    pre_ln_w: CudaTensor,
    pre_ln_b: CudaTensor,
    layers: Vec<ClipEncoderLayer>,
}

impl ClipVision {
    pub fn load(cfg: ClipVisionConfig, map: &WeightMap) -> Result<Self> {
        match Self::load_inner(&cfg, map, "vision_model") {
            Ok(m) => Ok(m),
            Err(_) => Self::load_inner(&cfg, map, ""),
        }
    }

    fn load_inner(cfg: &ClipVisionConfig, map: &WeightMap, root: &str) -> Result<Self> {
        let dim = cfg.hidden_size;
        let p = cfg.patch_size;
        let emb = if root.is_empty() {
            "embeddings".to_string()
        } else {
            format!("{root}.embeddings")
        };
        let pre = if root.is_empty() {
            if map.contains("pre_layrnorm.weight") {
                "pre_layrnorm".to_string()
            } else {
                "pre_layernorm".to_string()
            }
        } else if map.contains(&format!("{root}.pre_layrnorm.weight")) {
            format!("{root}.pre_layrnorm")
        } else {
            format!("{root}.pre_layernorm")
        };
        let enc = if root.is_empty() {
            "encoder.layers".to_string()
        } else {
            format!("{root}.encoder.layers")
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(ClipEncoderLayer::load(
                map,
                &format!("{enc}.{i}"),
                cfg,
            )?);
        }
        Ok(Self {
            class_embedding: weights::cuda_tensor_shaped(
                map,
                &format!("{emb}.class_embedding"),
                &[dim],
            )?,
            patch_weight: weights::cuda_tensor_shaped(
                map,
                &format!("{emb}.patch_embedding.weight"),
                &[dim, 3, p, p],
            )?,
            position_embedding: weights::cuda_tensor_shaped(
                map,
                &format!("{emb}.position_embedding.weight"),
                &[cfg.num_positions(), dim],
            )?,
            pre_ln_w: weights::cuda_tensor_shaped(map, &format!("{pre}.weight"), &[dim])?,
            pre_ln_b: weights::cuda_tensor_shaped(map, &format!("{pre}.bias"), &[dim])?,
            layers,
            cfg: cfg.clone(),
        })
    }

    pub fn forward_penultimate(&self, pixels: &CudaTensor) -> Result<CudaTensor> {
        let mut hidden = self.embeddings(pixels)?;
        let last = self.layers.len().saturating_sub(1);
        for (i, layer) in self.layers.iter().enumerate() {
            if i == last {
                return Ok(hidden);
            }
            hidden = layer.forward(&hidden)?;
        }
        Ok(hidden)
    }

    fn embeddings(&self, pixels: &CudaTensor) -> Result<CudaTensor> {
        let b = pixels.shape[0];
        let p = self.cfg.patch_size;
        let patches = nn::conv2d(pixels, &self.patch_weight, 0, p)?;
        let patches = patches.flatten_from(2)?.transpose(1, 2)?;
        let class = self
            .class_embedding
            .reshape(vec![1, 1, self.cfg.hidden_size])?;
        // broadcast class to batch
        let class = if b == 1 {
            class
        } else {
            let mut reps = Vec::new();
            for _ in 0..b {
                reps.push(class.clone());
            }
            let refs: Vec<&CudaTensor> = reps.iter().collect();
            CudaTensor::cat(&refs, 0)?
        };
        let tokens = CudaTensor::cat(&[&class, &patches], 1)?;
        let tokens = tokens.add(&self.position_embedding)?;
        nn::layer_norm(
            &tokens,
            self.cfg.layer_norm_eps,
            Some(&self.pre_ln_w),
            Some(&self.pre_ln_b),
        )
    }

    pub fn encode_image_file(&self, path: &str) -> Result<CudaTensor> {
        let pixels = clip_preprocess(path, self.cfg.image_size)?;
        self.forward_penultimate(&pixels)
    }
}

pub fn clip_preprocess(path: &str, size: usize) -> Result<CudaTensor> {
    let img = image::open(path)
        .map_err(|e| TensorError::Message(format!("CLIP image load failed: {e}")))?
        .to_rgb8();
    let (w, h) = img.dimensions();
    let (nw, nh) = if w < h {
        let nw = size as u32;
        let nh = ((h as u64 * size as u64) / w as u64) as u32;
        (nw.max(1), nh.max(1))
    } else {
        let nh = size as u32;
        let nw = ((w as u64 * size as u64) / h as u64) as u32;
        (nw.max(1), nh.max(1))
    };
    let resized = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::CatmullRom);
    let x0 = nw.saturating_sub(size as u32) / 2;
    let y0 = nh.saturating_sub(size as u32) / 2;
    let cropped = image::imageops::crop_imm(&resized, x0, y0, size as u32, size as u32).to_image();
    let mut data = Vec::with_capacity(3 * size * size);
    for c in 0..3 {
        for y in 0..size {
            for x in 0..size {
                let p = cropped.get_pixel(x as u32, y as u32)[c] as f32 / 255.0;
                data.push((p - CLIP_MEAN[c]) / CLIP_STD[c]);
            }
        }
    }
    CudaTensor::from_vec(data, vec![1, 3, size, size])
}
