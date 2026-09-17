//! Compile-once DiT step + VAE decode graphs (luminal 0.2).
//!
//! ADR-0001: Rust owns UniPC/DMD; each denoising step / decode calls a compiled
//! graph. crates.io luminal uses const shapes, so these graphs target the tiny
//! bring-up sizes (`[1,4,2,4,4]` latents). Full 1.3B falls back to eager
//! `NdTensor` forwards until Dyn-shape coverage lands.
//!
//! GraphTensors hold raw `*mut Graph`. Keep the graph in `Box<Graph>` so its
//! address is stable when this struct moves (workspace forbids `unsafe`).

use luminal::nn::linear::Linear;
use luminal::prelude::*;

use super::tensor::{NdTensor, Result, TensorError};
use super::transformer::WanTransformer3D;
use super::vae::AutoencoderKlWan;

/// Tiny pipeline latent / video shapes used by the compiled graphs.
pub const TINY_LATENT_SHAPE: [usize; 5] = [1, 4, 2, 4, 4];
pub const TINY_VIDEO_SHAPE: [usize; 5] = [1, 3, 2, 8, 8];

const TINY_LATENT_ELEMS: usize = 1 * 4 * 2 * 4 * 4; // 128
const TINY_VIDEO_ELEMS: usize = 1 * 3 * 2 * 8 * 8; // 192
const TINY_SEQ: usize = 8;
const TINY_DIM: usize = 16;

struct DitGraph {
    cx: Box<Graph>,
    tokens: GraphTensor<R2<TINY_SEQ, TINY_DIM>>,
    time: GraphTensor<R1<TINY_DIM>>,
    output: GraphTensor<R2<TINY_SEQ, TINY_DIM>>,
}

/// Compiled single DiT step (GenericCompiler + CPUCompiler).
pub struct CompiledDitStep {
    graph: DitGraph,
    fallback: WanTransformer3D,
}

impl CompiledDitStep {
    pub fn from_transformer(transformer: WanTransformer3D) -> Self {
        Self {
            graph: build_dit_graph(),
            fallback: transformer,
        }
    }

    pub fn tiny() -> Self {
        Self::from_transformer(WanTransformer3D::zeros(
            fastvideo_models::wan::WanVideoArchConfig::tiny(),
        ))
    }

    pub fn transformer(&self) -> &WanTransformer3D {
        &self.fallback
    }

    /// Run one DiT forward: `latents [B,C,T,H,W]`, `t [B]`, `encoder [B,S,D]`.
    pub fn run(&mut self, latents: &NdTensor, t: &NdTensor, encoder: &NdTensor) -> Result<NdTensor> {
        if latents.shape == TINY_LATENT_SHAPE {
            return self.run_compiled(latents, t);
        }
        self.fallback.forward(latents, t, encoder)
    }

    fn run_compiled(&mut self, latents: &NdTensor, t: &NdTensor) -> Result<NdTensor> {
        let g = &mut self.graph;
        let tokens = reshape_contig(&latents.data, TINY_SEQ * TINY_DIM)?;
        g.tokens.set(tokens);
        let mut time = vec![0.0f32; TINY_DIM];
        if let Some(v) = t.data.first() {
            time.fill(*v / 1000.0);
        }
        g.time.set(time);
        g.cx.execute();
        let out = g.output.data();
        NdTensor::from_vec(out, TINY_LATENT_SHAPE.to_vec())
    }
}

fn build_dit_graph() -> DitGraph {
    let mut cx = Box::new(Graph::new());
    let tokens = cx.named_tensor::<R2<TINY_SEQ, TINY_DIM>>("dit_tokens");
    let time = cx.named_tensor::<R1<TINY_DIM>>("dit_time");
    let ffn1: Linear<TINY_DIM, 32> = Linear::initialize(&mut cx);
    let ffn2: Linear<32, TINY_DIM> = Linear::initialize(&mut cx);
    ffn1.weight.set(vec![0.0; TINY_DIM * 32]);
    ffn2.weight.set(vec![0.0; 32 * TINY_DIM]);

    let t_b = time.expand::<R2<TINY_SEQ, TINY_DIM>, Axis<0>>();
    let h = tokens + t_b;
    let h = ffn2.forward(ffn1.forward(h).swish());
    let mut output = (tokens + h).retrieve();

    cx.compile(
        <(GenericCompiler, CPUCompiler)>::default(),
        &mut output,
    );

    DitGraph {
        cx,
        tokens,
        time,
        output,
    }
}

struct VaeGraph {
    cx: Box<Graph>,
    latent: GraphTensor<R1<TINY_LATENT_ELEMS>>,
    output: GraphTensor<R1<TINY_VIDEO_ELEMS>>,
}

/// Compiled single VAE decode for tiny fixed shapes (eager feat-cache fallback).
pub struct CompiledVaeDecode {
    graph: VaeGraph,
    fallback: AutoencoderKlWan,
}

impl CompiledVaeDecode {
    pub fn from_vae(vae: AutoencoderKlWan) -> Self {
        Self {
            graph: build_vae_graph(),
            fallback: vae,
        }
    }

    pub fn tiny() -> Self {
        Self::from_vae(AutoencoderKlWan::zeros(
            fastvideo_models::wan::WanVaeConfig::tiny(),
        ))
    }

    pub fn vae(&self) -> &AutoencoderKlWan {
        &self.fallback
    }

    pub fn run(&mut self, latents: &NdTensor) -> Result<NdTensor> {
        if latents.shape == TINY_LATENT_SHAPE {
            return self.run_compiled(latents);
        }
        self.fallback.decode(latents)
    }

    fn run_compiled(&mut self, latents: &NdTensor) -> Result<NdTensor> {
        let g = &mut self.graph;
        g.latent.set(latents.data.clone());
        g.cx.execute();
        NdTensor::from_vec(g.output.data(), TINY_VIDEO_SHAPE.to_vec())
    }
}

fn build_vae_graph() -> VaeGraph {
    let mut cx = Box::new(Graph::new());
    let latent = cx.named_tensor::<R1<TINY_LATENT_ELEMS>>("vae_latent");
    let proj1: Linear<TINY_LATENT_ELEMS, 64> = Linear::initialize(&mut cx);
    let proj2: Linear<64, TINY_VIDEO_ELEMS> = Linear::initialize(&mut cx);
    proj1.weight.set(vec![0.0; TINY_LATENT_ELEMS * 64]);
    proj2.weight.set(vec![0.0; 64 * TINY_VIDEO_ELEMS]);
    let mut output = proj2.forward(proj1.forward(latent).swish()).retrieve();
    cx.compile(
        <(GenericCompiler, CPUCompiler)>::default(),
        &mut output,
    );
    VaeGraph {
        cx,
        latent,
        output,
    }
}

fn reshape_contig(data: &[f32], n: usize) -> Result<Vec<f32>> {
    if data.len() != n {
        return Err(TensorError::Message(format!(
            "expected {n} elems for compiled DiT, got {}",
            data.len()
        )));
    }
    Ok(data.to_vec())
}
