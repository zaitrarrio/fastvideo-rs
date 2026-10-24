//! Sequence-parallel helpers for multi-GPU Wan DiT (single-node).
//!
//! When `FASTVIDEO_SP_WORLD=N` (`N>1`), the attention sequence dim is
//! sharded across `N` ranks; each rank's shard is dispatched to its own
//! physical CUDA device (`device_for_rank` below → `device::device_for_index`
//! in `nn::dispatch_sharded_multi_gpu`) on its own OS thread, running in
//! genuine parallel, then gathered back after SDPA. The gather is
//! host-mediated (each rank downloads its shard before the join, since a
//! `CudaSlice` belongs to the CUDA context that allocated it) rather than a
//! device-to-device NCCL all-gather — NCCL P2P would skip that round trip
//! and remains a follow-up.

use super::tensor::{CudaTensor, Result, TensorError};

/// Split sequence length `seq` across `world` ranks (contiguous shards).
pub fn shard_ranges(seq: usize, world: usize) -> Vec<(usize, usize)> {
    let world = world.max(1);
    let base = seq / world;
    let rem = seq % world;
    let mut out = Vec::with_capacity(world);
    let mut start = 0;
    for r in 0..world {
        let len = base + if r < rem { 1 } else { 0 };
        out.push((start, len));
        start += len;
    }
    out
}

/// All-gather sequence shards along axis `dim` (typically 2 for BHSD).
pub fn all_gather_seq(shards: &[CudaTensor], dim: usize) -> Result<CudaTensor> {
    if shards.is_empty() {
        return Err(TensorError::Message("all_gather empty".into()));
    }
    let refs: Vec<&CudaTensor> = shards.iter().collect();
    CudaTensor::cat(&refs, dim)
}

/// Narrow a BHSD / BSHD tensor to this rank's sequence shard.
pub fn shard_tensor(xs: &CudaTensor, dim: usize, rank: usize, world: usize) -> Result<CudaTensor> {
    let seq = xs.dim(dim)?;
    let ranges = shard_ranges(seq, world);
    let (start, len) = ranges
        .get(rank)
        .copied()
        .ok_or_else(|| TensorError::Message(format!("SP rank {rank} out of world {world}")))?;
    xs.narrow(dim, start, len)
}

/// Bind logical rank → CUDA device index when multiple GPUs are present.
pub fn device_for_rank(rank: usize, world: usize) -> usize {
    let _ = world;
    let n = std::env::var("CUDA_VISIBLE_DEVICES")
        .ok()
        .map(|s| s.split(',').filter(|x| !x.is_empty()).count())
        .unwrap_or(1)
        .max(1);
    rank % n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_ranges_cover_seq() {
        let r = shard_ranges(10, 2);
        assert_eq!(r, vec![(0, 5), (5, 5)]);
        let r = shard_ranges(11, 2);
        assert_eq!(r, vec![(0, 6), (6, 5)]);
        let total: usize = r.iter().map(|(_, l)| l).sum();
        assert_eq!(total, 11);
    }

    #[test]
    fn all_gather_roundtrip() {
        let a =
            CudaTensor::from_vec((0..12).map(|x| x as f32).collect(), vec![1, 1, 4, 3]).unwrap();
        let s0 = shard_tensor(&a, 2, 0, 2).unwrap();
        let s1 = shard_tensor(&a, 2, 1, 2).unwrap();
        let g = all_gather_seq(&[s0, s1], 2).unwrap();
        assert_eq!(g.shape, a.shape);
        assert_eq!(g.data, a.data);
    }
}
