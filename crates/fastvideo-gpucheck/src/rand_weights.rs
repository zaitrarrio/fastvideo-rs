//! Seeded random weights and inputs, generated inside cudarc's own loaders.

use fastvideo_cudarc::wan::weights::WeightMap;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;

fn name_hash(name: &str) -> u64 {
    // FNV-1a: stable across runs and platforms (unlike `DefaultHasher`).
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Scale-aware init so activations stay O(1) through deep random stacks:
/// biases small, 1-D norm weights near 1, matrices/convs fan-in scaled.
fn init(seed: u64, name: &str, shape: &[usize]) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let mut rng = StdRng::seed_from_u64(seed ^ name_hash(name));
    let (mean, std) = if name.ends_with("bias") {
        (0.0, 0.02)
    } else if name.contains("scale_shift_table") {
        (0.0, 0.1)
    } else if name.contains("embed_tokens") || name.ends_with("shared.weight") {
        (0.0, 1.0)
    } else if shape.len() <= 1 {
        (1.0, 0.1)
    } else {
        let fan_in: usize = shape[1..].iter().product::<usize>().max(1);
        (0.0, 1.0 / (fan_in as f32).sqrt())
    };
    (0..n)
        .map(|_| mean + std * rng.sample::<f32, _>(StandardNormal))
        .collect()
}

/// A weight map whose every tensor is generated from `(seed, key, shape)`.
/// Identical in every process, so a CPU-path dump and a GPU run agree on weights.
pub fn random_map(seed: u64) -> WeightMap {
    WeightMap::generated(move |key, shape| init(seed, key, shape))
}

/// Deterministic normal values for inputs/latents.
pub fn randn(seed: u64, n: usize, std: f32) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| std * rng.sample::<f32, _>(StandardNormal))
        .collect()
}
