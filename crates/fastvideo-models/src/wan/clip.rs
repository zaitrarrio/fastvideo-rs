//! CLIP ViT-H/14 vision encoder (Diffusers `image_encoder/` for Wan I2V).
//!
//! Matches Hugging Face `CLIPVisionModel` / `CLIPVisionModelWithProjection`
//! (`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`). Wan I2V uses `hidden_states[-2]`.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::nn::{self, Linear};

/// OpenAI / OpenCLIP pixel mean and std (CLIPImageProcessor defaults).
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
    pub layer_norm_eps: f64,
}

impl ClipVisionConfig {
    /// Wan 2.1 I2V 14B `image_encoder/config.json`.
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

/// Keys that must exist in Wan I2V Diffusers `image_encoder` (with `vision_model.` prefix).
pub const CLIP_VIT_H_REQUIRED_KEYS: &[&str] = &[
    "vision_model.embeddings.class_embedding",
    "vision_model.embeddings.patch_embedding.weight",
    "vision_model.embeddings.position_embedding.weight",
    "vision_model.pre_layrnorm.weight",
    "vision_model.encoder.layers.0.self_attn.q_proj.weight",
    "vision_model.encoder.layers.0.mlp.fc1.weight",
    "vision_model.encoder.layers.31.self_attn.q_proj.weight",
    "vision_model.encoder.layers.31.mlp.fc2.weight",
];

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
    fn load(dim: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q: Linear::load(dim, dim, vb.pp("q_proj"))?,
            k: Linear::load(dim, dim, vb.pp("k_proj"))?,
            v: Linear::load(dim, dim, vb.pp("v_proj"))?,
            out: Linear::load(dim, dim, vb.pp("out_proj"))?,
            heads,
            dim_head: dim / heads,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, s, _) = xs.dims3()?;
        let q = self.q.forward(xs)?;
        let k = self.k.forward(xs)?;
        let v = self.v.forward(xs)?;
        let q = q
            .reshape((b, s, self.heads, self.dim_head))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, s, self.heads, self.dim_head))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, s, self.heads, self.dim_head))?
            .transpose(1, 2)?
            .contiguous()?;
        let attn = nn::scaled_dot_product_attention(&q, &k, &v, None)?;
        let attn = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, self.heads * self.dim_head))?;
        self.out.forward(&attn)
    }
}

#[derive(Debug, Clone)]
struct ClipMlp {
    fc1: Linear,
    fc2: Linear,
}

impl ClipMlp {
    fn load(dim: usize, inner: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: Linear::load(dim, inner, vb.pp("fc1"))?,
            fc2: Linear::load(inner, dim, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&nn::gelu(&self.fc1.forward(xs)?)?)
    }
}

#[derive(Debug, Clone)]
struct ClipEncoderLayer {
    attn: ClipAttention,
    ln1_w: Tensor,
    ln1_b: Tensor,
    mlp: ClipMlp,
    ln2_w: Tensor,
    ln2_b: Tensor,
    eps: f64,
}

impl ClipEncoderLayer {
    fn load(cfg: &ClipVisionConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size;
        Ok(Self {
            attn: ClipAttention::load(dim, cfg.num_attention_heads, vb.pp("self_attn"))?,
            ln1_w: vb.pp("layer_norm1").get(dim, "weight")?,
            ln1_b: vb.pp("layer_norm1").get(dim, "bias")?,
            mlp: ClipMlp::load(dim, cfg.intermediate_size, vb.pp("mlp"))?,
            ln2_w: vb.pp("layer_norm2").get(dim, "weight")?,
            ln2_b: vb.pp("layer_norm2").get(dim, "bias")?,
            eps: cfg.layer_norm_eps,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let n1 = nn::layer_norm(
            &xs.to_dtype(DType::F32)?,
            self.eps,
            Some(&self.ln1_w),
            Some(&self.ln1_b),
        )?
        .to_dtype(xs.dtype())?;
        let xs = (xs + self.attn.forward(&n1)?)?;
        let n2 = nn::layer_norm(
            &xs.to_dtype(DType::F32)?,
            self.eps,
            Some(&self.ln2_w),
            Some(&self.ln2_b),
        )?
        .to_dtype(xs.dtype())?;
        xs + self.mlp.forward(&n2)?
    }
}

#[derive(Debug, Clone)]
pub struct ClipVision {
    pub cfg: ClipVisionConfig,
    class_embedding: Tensor,
    patch_weight: Tensor,
    position_embedding: Tensor,
    pre_ln_w: Tensor,
    pre_ln_b: Tensor,
    layers: Vec<ClipEncoderLayer>,
}

impl ClipVision {
    pub fn load(cfg: ClipVisionConfig, vb: VarBuilder) -> Result<Self> {
        match Self::load_inner(&cfg, vb.pp("vision_model")) {
            Ok(model) => Ok(model),
            Err(_) => Self::load_inner(&cfg, vb),
        }
    }

    fn load_inner(cfg: &ClipVisionConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size;
        let p = cfg.patch_size;
        let emb = vb.pp("embeddings");
        let pre = match vb.pp("pre_layrnorm").get(dim, "weight") {
            Ok(_) => vb.pp("pre_layrnorm"),
            Err(_) => vb.pp("pre_layernorm"),
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(ClipEncoderLayer::load(
                cfg,
                vb.pp("encoder").pp("layers").pp(&i.to_string()),
            )?);
        }
        Ok(Self {
            class_embedding: emb.get(dim, "class_embedding")?,
            patch_weight: emb
                .pp("patch_embedding")
                .get((dim, 3, p, p), "weight")?,
            position_embedding: emb
                .pp("position_embedding")
                .get((cfg.num_positions(), dim), "weight")?,
            pre_ln_w: pre.get(dim, "weight")?,
            pre_ln_b: pre.get(dim, "bias")?,
            layers,
            cfg: cfg.clone(),
        })
    }

    /// Pixel values `[B, 3, H, W]` already CLIP-normalized. Returns `hidden_states[-2]`.
    pub fn forward_penultimate(&self, pixels: &Tensor) -> Result<Tensor> {
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

    fn embeddings(&self, pixels: &Tensor) -> Result<Tensor> {
        let (b, _, _, _) = pixels.dims4()?;
        let p = self.cfg.patch_size;
        let patches = nn::conv2d(pixels, &self.patch_weight.to_dtype(pixels.dtype())?, 0, p)?;
        let patches = patches.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        let class = self
            .class_embedding
            .to_dtype(pixels.dtype())?
            .reshape((1, 1, self.cfg.hidden_size))?
            .broadcast_as((b, 1, self.cfg.hidden_size))?;
        let tokens = Tensor::cat(&[&class, &patches], 1)?;
        let tokens = tokens.broadcast_add(&self.position_embedding.to_dtype(pixels.dtype())?)?;
        nn::layer_norm(
            &tokens.to_dtype(DType::F32)?,
            self.cfg.layer_norm_eps,
            Some(&self.pre_ln_w),
            Some(&self.pre_ln_b),
        )?
        .to_dtype(pixels.dtype())
    }

    /// Load a PNG/JPEG, CLIP-preprocess to 224², return `[1, 257, 1280]` (or tiny shape).
    pub fn encode_image_file(&self, path: &str, device: &Device, dtype: DType) -> Result<Tensor> {
        let pixels = clip_preprocess(path, self.cfg.image_size, device)?.to_dtype(dtype)?;
        self.forward_penultimate(&pixels)
    }
}

/// CLIPImageProcessor: shortest-edge resize, center crop, /255, OpenAI mean/std.
pub fn clip_preprocess(path: &str, size: usize, device: &Device) -> Result<Tensor> {
    let img = image::open(path)
        .map_err(|e| candle_core::Error::Msg(format!("CLIP image load failed: {e}")))?
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
    Tensor::from_vec(data, (1, 3, size, size), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vit_h_config_matches_wan_i2v() {
        let c = ClipVisionConfig::vit_h_14();
        assert_eq!(c.hidden_size, 1280);
        assert_eq!(c.num_hidden_layers, 32);
        assert_eq!(c.num_positions(), 257);
    }

    #[test]
    fn tiny_clip_penultimate_shape() {
        let device = Device::Cpu;
        let cfg = ClipVisionConfig::tiny();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ClipVision::load(cfg.clone(), vb).unwrap();
        let pixels = Tensor::zeros(
            (1, 3, cfg.image_size, cfg.image_size),
            DType::F32,
            &device,
        )
        .unwrap();
        let out = model.forward_penultimate(&pixels).unwrap();
        assert_eq!(out.dims(), &[1, cfg.num_positions(), cfg.hidden_size]);
    }

    #[test]
    fn clip_preprocess_chw() {
        let dir = std::env::temp_dir().join("fastvideo-clip-pp");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.png");
        let img = image::RgbImage::from_pixel(32, 48, image::Rgb([128, 64, 32]));
        img.save(&path).unwrap();
        let t = clip_preprocess(path.to_str().unwrap(), 16, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[1, 3, 16, 16]);
        let n = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(n.iter().all(|v| v.is_finite()));
    }
}
