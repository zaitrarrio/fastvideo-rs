//! Spark stage-2 bridge: `LatentResizer3D` (`minimax_h3_latent_upscaler_3d.py`
//! at `d7c01b9011f2e8439493f6c02c29995a27df276f`, attention forced off) and
//! `H3ToLTXConvAdapter` (`stage2_ops/h3_ltx_adapter/model.py`).
//!
//! The joint 3-step LTX refiner is `Ltx2Pipeline::refine_joint`. It runs when
//! `FASTVIDEO_LTX2_WEIGHTS` is set, on this cropped video latent plus the H3 PCM.
//! The pipeline passes the upstream fixed prompt and the Gemma text cache;
//! the sampler stays on the LTX refine path.

use std::path::{Path, PathBuf};

use fastvideo_models::h3::spark::{
    align_h3_to_ltx, author_node_input, author_node_output, trilinear_ncdhw, ADAPTER_CONFIG,
    ADAPTER_OUTPUT, ADAPTER_WEIGHTS, H3_INPUT, H3_UPSCALED, PIXEL_FRAMES, PIXEL_HEIGHT,
    PIXEL_WIDTH, REFINER_INPUT, UPSCALER_FILE,
};

use crate::wan::nn::Linear;
use crate::wan::ops::PadMode;
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{cuda_tensor, WeightMap};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

struct Norm {
    weight: CudaTensor,
    bias: CudaTensor,
    groups: usize,
    eps: f32,
}

impl Norm {
    fn load(map: &WeightMap, prefix: &str, groups: usize, eps: f32) -> Result<Self> {
        let mut weight = cuda_tensor(map, &format!("{prefix}.weight"))?;
        let mut bias = cuda_tensor(map, &format!("{prefix}.bias"))?;
        weight.pin_device()?;
        bias.pin_device()?;
        Ok(Self {
            weight,
            bias,
            groups,
            eps,
        })
    }

    fn apply(&self, x: &CudaTensor, silu: bool) -> Result<CudaTensor> {
        x.group_norm(self.groups, &self.weight, &self.bias, self.eps, silu)
    }
}

struct Conv3d {
    weight: CudaTensor,
    bias: CudaTensor,
    pad: [usize; 3],
    groups: usize,
}

impl Conv3d {
    fn load(map: &WeightMap, prefix: &str, pad: [usize; 3], groups: usize) -> Result<Self> {
        let mut weight = cuda_tensor(map, &format!("{prefix}.weight"))?;
        let mut bias = cuda_tensor(map, &format!("{prefix}.bias"))?;
        if weight.rank() != 5 {
            return Err(msg(format!(
                "{prefix}.weight rank {} , a conv3d weight is 5-D",
                weight.rank()
            )));
        }
        weight.pin_device()?;
        bias.pin_device()?;
        Ok(Self {
            weight,
            bias,
            pad,
            groups,
        })
    }

    fn apply(&self, x: &CudaTensor) -> Result<CudaTensor> {
        x.conv3d_groups(
            &self.weight,
            Some(&self.bias),
            self.pad,
            [1, 1, 1],
            self.groups,
        )
    }
}

struct ResBlock {
    in_norm: Norm,
    in_conv: Conv3d,
    emb: Linear,
    out_norm: Norm,
    out_conv: Conv3d,
    skip: Option<Conv3d>,
}

impl ResBlock {
    fn load(map: &WeightMap, prefix: &str, emb_dim: usize) -> Result<Self> {
        let in_conv = Conv3d::load(map, &format!("{prefix}.in_layers.2"), [1, 1, 1], 1)?;
        let channels = in_conv.weight.shape[1];
        let out_channels = in_conv.weight.shape[0];
        let skip = if map.has_tensor(&format!("{prefix}.skip.weight")) {
            Some(Conv3d::load(map, &format!("{prefix}.skip"), [0, 0, 0], 1)?)
        } else if out_channels != channels {
            return Err(msg(format!(
                "{prefix}: channels {channels} → {out_channels} without a skip conv"
            )));
        } else {
            None
        };
        Ok(Self {
            in_norm: Norm::load(map, &format!("{prefix}.in_layers.0"), 32, 1e-5)?,
            in_conv,
            emb: Linear::load(
                map,
                &format!("{prefix}.emb_layers.1"),
                emb_dim,
                2 * out_channels,
                true,
            )?,
            out_norm: Norm::load(map, &format!("{prefix}.out_norm"), 32, 1e-5)?,
            out_conv: Conv3d::load(map, &format!("{prefix}.out_layers.2"), [1, 1, 1], 1)?,
            skip,
        })
    }

    fn apply(&self, x: &CudaTensor, emb: &CudaTensor) -> Result<CudaTensor> {
        let h = self.in_conv.apply(&self.in_norm.apply(x, true)?)?;
        let emb = self.emb.forward(&emb.silu())?;
        let channels = emb.shape[emb.rank() - 1] / 2;
        let emb = emb.reshape(vec![x.shape[0], channels * 2, 1, 1, 1])?;
        let scale = emb.narrow(1, 0, channels)?;
        let shift = emb.narrow(1, channels, channels)?;
        let one = CudaTensor::from_vec(vec![1.0], vec![1, 1, 1, 1, 1])?;
        let h = self
            .out_norm
            .apply(&h, false)?
            .mul(&scale.add(&one)?)?
            .add(&shift)?;
        let h = self.out_conv.apply(&h.silu())?;
        match &self.skip {
            Some(skip) => skip.apply(x)?.add(&h),
            None => x.add(&h),
        }
    }
}

struct Temporal {
    norm: Norm,
    dw: Conv3d,
    pw: Conv3d,
}

impl Temporal {
    fn load(map: &WeightMap, prefix: &str) -> Result<Self> {
        let dw = Conv3d::load(
            map,
            &format!("{prefix}.dwconv"),
            [dw_pad(&map, prefix)?, 0, 0],
            1,
        )?;
        let channels = dw.weight.shape[0];
        if dw.weight.shape[1] != 1 {
            return Err(msg(format!(
                "{prefix}.dwconv is not depthwise: {:?}",
                dw.weight.shape
            )));
        }
        let mut dw = dw;
        dw.groups = channels;
        Ok(Self {
            norm: Norm::load(map, &format!("{prefix}.norm"), 32, 1e-5)?,
            dw,
            pw: Conv3d::load(map, &format!("{prefix}.pwconv"), [0, 0, 0], 1)?,
        })
    }

    fn apply(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let h = self.pw.apply(&self.dw.apply(&self.norm.apply(x, true)?)?)?;
        x.add(&h)
    }
}

fn dw_pad(map: &WeightMap, prefix: &str) -> Result<usize> {
    let weight = cuda_tensor(map, &format!("{prefix}.dwconv.weight"))?;
    let k = *weight
        .shape
        .get(2)
        .ok_or_else(|| msg("dwconv weight rank"))?;
    Ok(k / 2)
}

enum Block {
    Res(ResBlock),
    Temporal(Temporal),
}

impl Block {
    fn apply(&self, x: &CudaTensor, emb: &CudaTensor) -> Result<CudaTensor> {
        match self {
            Self::Res(block) => block.apply(x, emb),
            Self::Temporal(block) => block.apply(x),
        }
    }
}

struct Upscaler {
    conv_in: Conv3d,
    embed_in: Linear,
    embed_up: Linear,
    in_blocks: Vec<Block>,
    out_blocks: Vec<Block>,
    norm_out: Norm,
    conv_out: Conv3d,
    temporal_kernel: usize,
}

impl Upscaler {
    fn load(path: &Path) -> Result<Self> {
        let map = WeightMap::open_files(&[path.to_path_buf()])?;
        let root = if map.has_tensor("conv_in.weight") {
            ""
        } else if map.has_tensor("upscaler.conv_in.weight") {
            "upscaler."
        } else {
            return Err(msg(format!(
                "h3 upscaler: no conv_in.weight in {}",
                path.display()
            )));
        };
        if map.lazy().is_some_and(|lazy| {
            lazy.keys()
                .any(|key| key.contains("attn") || key.ends_with(".q.weight"))
        }) {
            return Err(msg(
                "h3 upscaler: checkpoint has attention keys; the published loader forces attn off and would reject them",
            ));
        }
        let key = |name: &str| format!("{root}{name}");
        let conv_in = Conv3d::load(&map, &key("conv_in"), [1, 1, 1], 1)?;
        let mut in_blocks = Vec::new();
        let mut out_blocks = Vec::new();
        let mut temporal_kernel = 0usize;
        for (which, blocks) in [
            ("in_blocks", &mut in_blocks),
            ("out_blocks", &mut out_blocks),
        ] {
            for i in 0..256 {
                let prefix = key(&format!("{which}.{i}"));
                if map.has_tensor(&format!("{prefix}.dwconv.weight")) {
                    let block = Temporal::load(&map, &prefix)?;
                    temporal_kernel = block.dw.weight.shape[2];
                    blocks.push(Block::Temporal(block));
                } else if map.has_tensor(&format!("{prefix}.in_layers.2.weight")) {
                    blocks.push(Block::Res(ResBlock::load(&map, &prefix, 64)?));
                } else {
                    break;
                }
            }
        }
        if in_blocks.is_empty() || out_blocks.is_empty() {
            return Err(msg(format!(
                "h3 upscaler: no residual blocks in {}",
                path.display()
            )));
        }
        Ok(Self {
            conv_in,
            embed_in: Linear::load(&map, &key("embed.0"), 1, 64, true)?,
            embed_up: Linear::load(&map, &key("embed.2"), 64, 64, true)?,
            in_blocks,
            out_blocks,
            norm_out: Norm::load(&map, &key("norm_out"), 32, 1e-5)?,
            conv_out: Conv3d::load(&map, &key("conv_out"), [1, 1, 1], 1)?,
            temporal_kernel,
        })
    }

    fn forward(&self, x: &CudaTensor, scale: f32, target: [usize; 3]) -> Result<CudaTensor> {
        if x.rank() != 5 {
            return Err(msg(format!(
                "h3 upscaler: expected NCDHW, got {:?}",
                x.shape
            )));
        }
        let (b, c, t) = (x.shape[0], x.shape[1], x.shape[2]);
        if [t, x.shape[3], x.shape[4]] == target {
            return Ok(x.clone());
        }
        let overlap = self.temporal_kernel;
        let chunk = 32;
        if t <= chunk {
            return self.forward_seg(x, scale, target);
        }
        let padded = x.pad(2, overlap, overlap, PadMode::Replicate)?;
        let (th, tw) = (target[1], target[2]);
        let mut acc = vec![0.0f32; b * c * t * th * tw];
        let mut weight_acc = vec![0.0f32; t];
        let mut start = 0;
        while start < t {
            let seg_start = start;
            let seg_end = (start + chunk).min(t);
            let out_start = seg_start.saturating_sub(overlap);
            let out_end = (seg_end + overlap).min(t);
            let lo = out_start.saturating_sub(overlap);
            let hi = (out_end + overlap).min(t + 2 * overlap);
            let seg = padded.narrow(2, lo, hi - lo)?;
            let seg_out = self.forward_seg(&seg, scale, [hi - lo, th, tw])?;
            let s0 = (out_start + overlap) - lo;
            let n_valid = out_end - out_start;
            let valid = seg_out.narrow(2, s0, n_valid)?.host_cow()?.into_owned();
            let mut weight = vec![1.0f32; n_valid];
            if seg_start > out_start {
                let blend_len = seg_start - out_start;
                for i in 0..blend_len {
                    weight[i] = (i + 1) as f32 / (blend_len + 1) as f32;
                }
            }
            if out_end > seg_end {
                let blend_len = out_end - seg_end;
                for i in 0..blend_len {
                    weight[n_valid - blend_len + i] =
                        (blend_len - i) as f32 / (blend_len + 1) as f32;
                }
            }
            let spatial = th * tw;
            for batch in 0..b {
                for ch in 0..c {
                    for i in 0..n_valid {
                        let src = ((batch * c + ch) * n_valid + i) * spatial;
                        let dst = ((batch * c + ch) * t + out_start + i) * spatial;
                        let w = weight[i];
                        for k in 0..spatial {
                            acc[dst + k] += valid[src + k] * w;
                        }
                    }
                }
            }
            for i in 0..n_valid {
                weight_acc[out_start + i] += weight[i];
            }
            start += chunk;
        }
        for i in 0..t {
            let w = weight_acc[i].max(1e-8);
            for batch in 0..b {
                for ch in 0..c {
                    let dst = ((batch * c + ch) * t + i) * th * tw;
                    for k in 0..th * tw {
                        acc[dst + k] /= w;
                    }
                }
            }
        }
        CudaTensor::from_vec(acc, vec![b, c, t, th, tw])
    }

    fn forward_seg(&self, x: &CudaTensor, scale: f32, size: [usize; 3]) -> Result<CudaTensor> {
        if x.shape[0] != 1 {
            return Err(msg("h3 upscaler: batch must be 1"));
        }
        let scale_in = CudaTensor::from_vec(vec![scale - 1.0], vec![1, 1])?;
        let emb = self
            .embed_up
            .forward(&self.embed_in.forward(&scale_in)?.silu())?;
        let mut y = self.conv_in.apply(x)?;
        for block in &self.in_blocks {
            y = block.apply(&y, &emb)?;
        }
        y = trilinear(&y, size)?;
        for block in &self.out_blocks {
            y = block.apply(&y, &emb)?;
        }
        y = self.norm_out.apply(&y, true)?;
        self.conv_out.apply(&y)
    }
}

fn trilinear(x: &CudaTensor, size: [usize; 3]) -> Result<CudaTensor> {
    if x.rank() != 5 {
        return Err(msg(format!("trilinear: {:?}", x.shape)));
    }
    let shape = [x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]];
    let host = x.host_cow()?.into_owned();
    let out = trilinear_ncdhw(&host, shape, size).map_err(msg)?;
    CudaTensor::from_vec(out, vec![shape[0], shape[1], size[0], size[1], size[2]])
}

struct FactorBlock {
    norm: Norm,
    spatial: Conv3d,
    temporal: Conv3d,
    in_proj: Conv3d,
    out_proj: Conv3d,
}

impl FactorBlock {
    fn load(map: &WeightMap, prefix: &str, groups: usize) -> Result<Self> {
        let temporal = Conv3d::load(map, &format!("{prefix}.temporal"), [1, 0, 0], 1)?;
        let channels = temporal.weight.shape[0];
        let mut temporal = temporal;
        temporal.groups = channels;
        Ok(Self {
            norm: Norm::load(map, &format!("{prefix}.norm"), groups, 1e-6)?,
            spatial: Conv3d::load(map, &format!("{prefix}.spatial"), [0, 1, 1], 1)?,
            temporal,
            in_proj: Conv3d::load(map, &format!("{prefix}.in_proj"), [0, 0, 0], 1)?,
            out_proj: Conv3d::load(map, &format!("{prefix}.out_proj"), [0, 0, 0], 1)?,
        })
    }

    fn apply(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let h = self
            .temporal
            .apply(&self.spatial.apply(&self.norm.apply(x, true)?)?)?;
        let projected = self.in_proj.apply(&h.silu())?;
        let hidden = projected.shape[1] / 2;
        let value = projected.narrow(1, 0, hidden)?;
        let gate = projected.narrow(1, hidden, hidden)?;
        let h = self.out_proj.apply(&value.mul(&gate.silu())?)?;
        x.add(&h)
    }
}

struct Adapter {
    spec_in: usize,
    skip: Conv3d,
    stem: Conv3d,
    blocks: Vec<FactorBlock>,
    final_norm: Norm,
    head: Conv3d,
}

impl Adapter {
    fn load(weights: &Path, config: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(config).map_err(|e| msg(e.to_string()))?;
        let value: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| msg(e.to_string()))?;
        if value.get("model_type").and_then(|v| v.as_str()) != Some("tiny") {
            return Err(msg(format!(
                "h3 adapter: only model_type \"tiny\" is published, got {}",
                value.get("model_type").unwrap_or(&serde_json::Value::Null)
            )));
        }
        for key in [
            "dense_temporal_blocks",
            "coordinate_channels",
            "output_refiner_blocks",
            "source_detail_blocks",
        ] {
            let n = value.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
            if n != 0 {
                return Err(msg(format!(
                    "h3 adapter: checkpoint enables unsupported {key}={n}"
                )));
            }
        }
        let need = |key: &str| -> Result<usize> {
            value
                .get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| msg(format!("h3 adapter config missing {key}")))
        };
        let groups = need("groups")?;
        let num_blocks = need("num_blocks")?;
        let map = WeightMap::open_files(&[weights.to_path_buf()])?;
        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            blocks.push(FactorBlock::load(&map, &format!("blocks.{i}"), groups)?);
        }
        Ok(Self {
            spec_in: need("in_channels")?,
            skip: Conv3d::load(&map, "skip", [0, 0, 0], 1)?,
            stem: Conv3d::load(&map, "stem", [1, 1, 1], 1)?,
            blocks,
            final_norm: Norm::load(&map, "final_norm", groups, 1e-6)?,
            head: Conv3d::load(&map, "head", [0, 0, 0], 1)?,
        })
    }

    fn forward(&self, aligned: &CudaTensor) -> Result<CudaTensor> {
        if aligned.shape[1] != self.spec_in {
            return Err(msg(format!(
                "h3 adapter: aligned channels {} != config in_channels {}",
                aligned.shape[1], self.spec_in
            )));
        }
        let mut y = self.stem.apply(aligned)?;
        for block in &self.blocks {
            y = block.apply(&y)?;
        }
        let y = self.head.apply(&self.final_norm.apply(&y, true)?)?;
        self.skip.apply(aligned)?.add(&y)
    }
}

pub struct SparkBridge {
    upscaler: Upscaler,
    adapter: Adapter,
}

impl SparkBridge {
    pub fn resolve(root: &Path) -> Result<Option<Self>> {
        let up = match std::env::var("FASTVIDEO_H3_UPSCALER") {
            Ok(raw) => {
                let path = PathBuf::from(raw);
                if !path.is_file() {
                    return Err(msg(format!(
                        "FASTVIDEO_H3_UPSCALER {} is not a file",
                        path.display()
                    )));
                }
                Some(path)
            }
            Err(_) => find_file(root, UPSCALER_FILE),
        };
        let adapter = match std::env::var("FASTVIDEO_H3_LTX_ADAPTER") {
            Ok(raw) => Some(adapter_paths(Path::new(&raw))?),
            Err(_) => find_adapter(root),
        };
        match (up, adapter) {
            (None, None) => Ok(None),
            (Some(up), Some((weights, config))) => Ok(Some(Self {
                upscaler: Upscaler::load(&up)?,
                adapter: Adapter::load(&weights, &config)?,
            })),
            (Some(_), None) => Err(msg(
                "h3 spark: upscaler found, H3-to-LTX adapter (config.json + model.safetensors) not found",
            )),
            (None, Some(_)) => Err(msg(
                "h3 spark: H3-to-LTX adapter found, latent upscaler not found",
            )),
        }
    }

    /// Official canvas only: N, upscale ×2, D, align, adapter, crop time to 16.
    pub fn forward(&self, h3: &CudaTensor) -> Result<CudaTensor> {
        let expect = [1, H3_INPUT[0], H3_INPUT[1], H3_INPUT[2], H3_INPUT[3]];
        if h3.shape.as_slice() != expect {
            return Err(msg(format!(
                "h3 spark bridge: latent {:?} , official canvas is {expect:?}",
                h3.shape
            )));
        }
        let host = h3.host_cow()?.into_owned();
        let normed = author_node_input(&host, expect).map_err(msg)?;
        let mut x = CudaTensor::from_vec(normed, expect.to_vec())?;
        x.pin_device()?;
        let target = [H3_UPSCALED[1], H3_UPSCALED[2], H3_UPSCALED[3]];
        let up = self.upscaler.forward(&x, 2.0, target)?;
        let up_host = up.host_cow()?.into_owned();
        let up_shape = [
            1,
            H3_UPSCALED[0],
            H3_UPSCALED[1],
            H3_UPSCALED[2],
            H3_UPSCALED[3],
        ];
        if up.shape.as_slice() != up_shape {
            return Err(msg(format!(
                "h3 upscaler: {:?} , expected {up_shape:?}",
                up.shape
            )));
        }
        let denorm = author_node_output(&up_host, up_shape).map_err(msg)?;
        let (aligned, aligned_shape) = align_h3_to_ltx(
            &denorm,
            up_shape,
            PIXEL_FRAMES,
            PIXEL_HEIGHT / 32,
            PIXEL_WIDTH / 32,
            3,
        )
        .map_err(msg)?;
        // 24 * (1 linear + 3 packed slots) * 4 unshuffle = 384.
        if aligned_shape
            != [
                1,
                384,
                ADAPTER_OUTPUT[1],
                ADAPTER_OUTPUT[2],
                ADAPTER_OUTPUT[3],
            ]
        {
            return Err(msg(format!(
                "h3 align: {aligned_shape:?} , expected [1, 384, {}, {}, {}]",
                ADAPTER_OUTPUT[1], ADAPTER_OUTPUT[2], ADAPTER_OUTPUT[3]
            )));
        }
        let mut aligned = CudaTensor::from_vec(aligned, aligned_shape.to_vec())?;
        aligned.pin_device()?;
        let y = self.adapter.forward(&aligned)?;
        let out_shape = [
            1,
            ADAPTER_OUTPUT[0],
            ADAPTER_OUTPUT[1],
            ADAPTER_OUTPUT[2],
            ADAPTER_OUTPUT[3],
        ];
        if y.shape.as_slice() != out_shape {
            return Err(msg(format!(
                "h3 adapter: {:?} , expected {out_shape:?}",
                y.shape
            )));
        }
        let cropped = y.narrow(2, 0, REFINER_INPUT[1])?;
        let refiner = [
            1,
            REFINER_INPUT[0],
            REFINER_INPUT[1],
            REFINER_INPUT[2],
            REFINER_INPUT[3],
        ];
        if cropped.shape.as_slice() != refiner {
            return Err(msg(format!(
                "h3 spark bridge: cropped {:?} , expected {refiner:?}",
                cropped.shape
            )));
        }
        Ok(cropped)
    }
}

fn find_file(root: &Path, name: &str) -> Option<PathBuf> {
    let mut roots = vec![root.to_path_buf()];
    if let Some(parent) = root.parent() {
        roots.push(parent.to_path_buf());
    }
    let nested = [
        "upscaler",
        "h3-spark-upscaler",
        "h3-spark",
        "h3_ltx_adapter",
    ];
    for dir in roots {
        let direct = dir.join(name);
        if direct.is_file() {
            return Some(direct);
        }
        for sub in nested {
            let candidate = dir.join(sub).join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn find_adapter(root: &Path) -> Option<(PathBuf, PathBuf)> {
    // The H3 snapshot itself has a config and weights. Only a dedicated
    // directory is the adapter.
    let mut dirs = vec![
        root.join("h3_ltx_adapter"),
        root.join("h3-to-ltx"),
        root.join("H3-to-LTX-Latent-Adapter"),
    ];
    if let Some(parent) = root.parent() {
        dirs.push(parent.join("h3_ltx_adapter"));
        dirs.push(parent.join("h3-to-ltx"));
        dirs.push(parent.join("H3-to-LTX-Latent-Adapter"));
    }
    for dir in dirs {
        let weights = dir.join(ADAPTER_WEIGHTS);
        let config = dir.join(ADAPTER_CONFIG);
        if weights.is_file() && config.is_file() {
            return Some((weights, config));
        }
    }
    None
}

fn adapter_paths(path: &Path) -> Result<(PathBuf, PathBuf)> {
    if path.is_dir() {
        let weights = path.join(ADAPTER_WEIGHTS);
        let config = path.join(ADAPTER_CONFIG);
        if weights.is_file() && config.is_file() {
            return Ok((weights, config));
        }
        return Err(msg(format!(
            "FASTVIDEO_H3_LTX_ADAPTER {} needs {ADAPTER_CONFIG} and {ADAPTER_WEIGHTS}",
            path.display()
        )));
    }
    if path.is_file() {
        let config = path
            .parent()
            .map(|dir| dir.join(ADAPTER_CONFIG))
            .filter(|p| p.is_file())
            .ok_or_else(|| {
                msg(format!(
                    "FASTVIDEO_H3_LTX_ADAPTER {} has no sibling {ADAPTER_CONFIG}",
                    path.display()
                ))
            })?;
        return Ok((path.to_path_buf(), config));
    }
    Err(msg(format!(
        "FASTVIDEO_H3_LTX_ADAPTER {} is not a file or directory",
        path.display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fv-spark-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn find_adapter_ignores_the_h3_snapshot_and_uses_a_sibling_dir() {
        let root = tmp("weights");
        let h3 = root.join("h3-base");
        std::fs::create_dir_all(&h3).unwrap();
        std::fs::write(h3.join("config.json"), b"{}").unwrap();
        std::fs::write(h3.join("model.safetensors"), b"not-the-adapter").unwrap();
        assert!(find_adapter(&h3).is_none());

        let adapter = root.join("h3-to-ltx");
        std::fs::create_dir_all(&adapter).unwrap();
        std::fs::write(adapter.join(ADAPTER_CONFIG), b"{}").unwrap();
        std::fs::write(adapter.join(ADAPTER_WEIGHTS), b"weights").unwrap();
        let found = find_adapter(&h3).expect("sibling adapter");
        assert_eq!(found.0, adapter.join(ADAPTER_WEIGHTS));
        assert_eq!(found.1, adapter.join(ADAPTER_CONFIG));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn find_file_looks_beside_h3_and_under_upscaler() {
        let root = tmp("up");
        let h3 = root.join("h3-base");
        std::fs::create_dir_all(root.join("upscaler")).unwrap();
        std::fs::create_dir_all(&h3).unwrap();
        let file = root.join("upscaler").join(UPSCALER_FILE);
        std::fs::write(&file, b"up").unwrap();
        assert_eq!(
            find_file(&h3, UPSCALER_FILE).as_deref(),
            Some(file.as_path())
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
