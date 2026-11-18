//! The H3 port against diffusers itself, on the CPU, at toy sizes.
//!
//! The loop references in each module's tests are written from the *spec*; if
//! the spec misreads the reference, they inherit the mistake. These tests do
//! not: `scripts/gpu/h3_tiny_reference.py` instantiates diffusers' own
//! MiniMax-H3 classes with tiny configs and seeded random weights, runs them
//! once in float32, and stores weights, inputs, outputs and taps in
//! `fixtures/*.st`. Here the same weights go through the **production loaders**
//! and the outputs must agree.
//!
//! Everything structural that was first confirmed on rented hardware is pinned
//! here on a laptop: the packed layout and its float positions, channel-major
//! patch features, the AdaLN row layout behind the precomputed table, MM-RoPE's
//! channel map with pass-through channels, value-first SwiGLU, the data-ward
//! scheduler step over two shifted ladders, the ViT decoder with its register
//! tokens, raw-neighbour tile stitching and temporal cross-fades, the
//! weight-norm axis of transposed convs and the alias-free SnakeBeta.
//! The configs mirror the ones in that script; change both together.

use std::path::PathBuf;

use fastvideo_models::h3::config::{
    H3AudioVaeConfig, H3TransformerConfig, H3VideoVaeConfig, TAG_AUDIO, TAG_TEXT, TAG_VIDEO,
};
use fastvideo_models::h3::packing::{patchify, H3PackedLayout};
use fastvideo_models::h3::schedule::H3JointSchedule;

use crate::rt::tensor::CudaTensor;
use crate::rt::weights::WeightMap;

use super::audio_vae::H3AudioDecoder;
use super::pipeline::denoise;
use super::transformer::{AttnMode, DeviceLayout, H3TextRefiner, H3Transformer};
use super::vae::H3VideoDecoder;

fn fixture(name: &str) -> WeightMap {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/h3/fixtures")
        .join(name);
    WeightMap::open_files(std::slice::from_ref(&path))
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn values(map: &WeightMap, key: &str) -> (Vec<usize>, Vec<f32>) {
    map.get_f32(key).unwrap_or_else(|e| panic!("{key}: {e}"))
}

fn tensor(map: &WeightMap, key: &str) -> CudaTensor {
    let (shape, data) = values(map, key);
    CudaTensor::from_vec(data, shape).expect("fixture tensor")
}

/// Relative L2 against a reference slice, naming the worst element.
fn assert_close(what: &str, got: &[f32], want: &[f32], max_rel: f64) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: {} values vs the reference's {}",
        got.len(),
        want.len()
    );
    let (mut err, mut norm, mut worst) = (0f64, 0f64, (0usize, 0f32, 0f32));
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        let d = f64::from(a - b);
        err += d * d;
        norm += f64::from(*b) * f64::from(*b);
        if (a - b).abs() > (worst.1 - worst.2).abs() {
            worst = (i, *a, *b);
        }
    }
    assert!(norm > 0.0, "{what}: an all-zero reference proves nothing");
    let rel = (err / norm).sqrt();
    assert!(
        rel <= max_rel,
        "{what}: rel L2 {rel:.3e} > {max_rel:.0e}; worst at {}: ours {} vs reference {}",
        worst.0,
        worst.1,
        worst.2
    );
}

fn assert_matches(what: &str, got: &CudaTensor, map: &WeightMap, key: &str, max_rel: f64) {
    assert_close(
        what,
        &got.host_cow().expect("host"),
        &values(map, key).1,
        max_rel,
    );
}

fn dit_config() -> H3TransformerConfig {
    H3TransformerConfig {
        num_attention_heads: 3,
        attention_head_dim: 16,
        hidden_size: 20,
        num_layers: 3,
        num_refiner_layers: 2,
        ffn_dim: 14,
        in_channels: 2,
        audio_in_channels: 3,
        patch_size: [1, 2, 2],
        text_dim: 7,
        freq_dim: 8,
        time_embed_hidden_dim: 9,
        time_embed_dim: 5,
        rope_freq_dim: 2, // 12 of 16 channels rotate: 3 axes x 2 frequencies x 2, 4 pass through
        rope_theta: 10000.0,
        norm_eps: 1e-5,
        qk_norm_eps: 1e-5,
        final_norm_eps: 1e-5,
        adaln_rank: None,
        source_num_layers: 2,
        kept_block_indices: vec![0, 1],
    }
}

#[test]
fn dit_layout_forward_and_ladder_match_diffusers() {
    let (cfg, map) = (dit_config(), fixture("dit_tiny.st"));
    let h = cfg.hidden_size;
    // 5 text tokens, 3 audio latents per channel, a 3 x 4 x 6 latent (h3_tiny_reference.py).
    let layout = H3PackedLayout::new(5, (3, 4, 6), 3, cfg.patch_size).expect("layout");
    let positions: Vec<f32> = layout
        .position_ids
        .iter()
        .flatten()
        .map(|&p| p as f32)
        .collect();
    assert_eq!(
        positions,
        values(&map, "ref.position_ids").1,
        "packed positions, exact as float32"
    );
    let tags: Vec<f32> = layout.token_tags.iter().map(|&t| f32::from(t)).collect();
    assert_eq!(tags, values(&map, "ref.token_tags").1);

    let (noise_shape, noise) = values(&map, "in.video_noise");
    let rows = patchify(
        &noise,
        [
            noise_shape[1],
            noise_shape[2],
            noise_shape[3],
            noise_shape[4],
        ],
        cfg.patch_size,
    )
    .expect("patchify");
    assert_eq!(
        rows,
        values(&map, "ref.video_rows").1,
        "channel-major patch features"
    );

    let refined = H3TextRefiner::load(&cfg, &map)
        .expect("refiner")
        .forward(&tensor(&map, "in.text"))
        .expect("refine");
    assert_matches("text_refined", &refined, &map, "ref.text_refined", 1e-5);

    let schedule = H3JointSchedule::fasth3_8step();
    let model = H3Transformer::load(cfg.clone(), &map, &schedule, false).expect("load");
    let step = values(&map, "in.step").1[0] as usize;
    let table = model.adaln_table();
    // temb rows are the sorted-unique timesteps: video (smaller t) then audio.
    let temb: Vec<f32> = table.temb[2 * step]
        .iter()
        .chain(&table.temb[2 * step + 1])
        .copied()
        .collect();
    assert_close("temb", &temb, &values(&map, "ref.temb").1, 1e-6);
    // ref.adaln_<b> is [6 params, n_t * 3 rows, hidden]; T2AV reads rows 0, 1 and 5.
    for block in 0..cfg.num_layers {
        let (shape, want) = values(&map, &format!("ref.adaln_{block}"));
        assert_eq!(shape, vec![6, 6, h]);
        for (tag, row) in [(TAG_VIDEO, 0usize), (TAG_TEXT, 1), (TAG_AUDIO, 5)] {
            let slot = table.block_slot(step, block, tag);
            for p in 0..6 {
                let plus = if p == 1 || p == 4 { 1.0 } else { 0.0 }; // the table stores 1 + scale
                let ours: Vec<f32> = slot[p * h..(p + 1) * h].iter().map(|v| v - plus).collect();
                assert_close(
                    &format!("adaln block {block} tag {tag} param {p}"),
                    &ours,
                    &want[(p * 6 + row) * h..(p * 6 + row + 1) * h],
                    1e-5,
                );
            }
        }
    }

    let device_layout = DeviceLayout::new(&cfg, layout.clone()).expect("rope");
    let video_rows =
        CudaTensor::from_vec(rows, vec![layout.video.len, cfg.video_patch_dim()]).expect("rows");
    let audio_rows = tensor(&map, "in.audio_rows");
    let mut blocks: Vec<(String, Vec<f32>)> = Vec::new();
    let (video, audio) = model
        .forward(
            step,
            &video_rows,
            &audio_rows,
            &refined,
            &device_layout,
            AttnMode::Dense,
            Some(&mut |name, x| {
                blocks.push((name.to_string(), x.host_cow()?.into_owned()));
                Ok(())
            }),
            None,
            None,
        )
        .expect("forward");
    assert_eq!(blocks.len(), cfg.num_layers);
    for (name, got) in &blocks {
        assert_close(name, got, &values(&map, &format!("ref.{name}")).1, 1e-5);
    }
    assert_matches("video velocity", &video, &map, "ref.video", 1e-5);
    assert_matches("audio velocity", &audio, &map, "ref.audio", 1e-5);

    // The whole ladder: both shifts, the plus sign, sigma-from-timestep and the ratio.
    let (nv, na) = (video_rows.numel(), audio_rows.numel());
    let (want_video, want_audio) = (
        values(&map, "ref.loop_video").1,
        values(&map, "ref.loop_audio").1,
    );
    denoise(
        &model,
        &device_layout,
        &refined,
        video_rows,
        audio_rows,
        &schedule,
        0,
        AttnMode::Dense,
        None,
        None,
        &mut |i, v, a| {
            assert_close(
                &format!("loop video step {i}"),
                &v.host_cow()?,
                &want_video[i * nv..(i + 1) * nv],
                2e-5,
            );
            assert_close(
                &format!("loop audio step {i}"),
                &a.host_cow()?,
                &want_audio[i * na..(i + 1) * na],
                2e-5,
            );
            Ok(())
        },
    )
    .expect("ladder");
}

fn vae_config(map: &WeightMap) -> H3VideoVaeConfig {
    let mut cfg = H3VideoVaeConfig::fasth3_8step();
    cfg.latent_channels = 5;
    cfg.spatial_downsample_factors = [2, 2, 1, 1, 1, 1]; // 4 px per latent
    cfg.decoder_num_layers = 2;
    cfg.decoder_num_attention_heads = 2;
    cfg.decoder_attention_head_dim = 16; // 12 rotary channels: 3 axes x 2 frequencies x 2
    cfg.decoder_ffn_mult = 2;
    cfg.tile_sample_min_size = 8;
    cfg.tile_sample_min_overlap = 4;
    for (name, dst) in [
        ("in.latents_mean", &mut cfg.latents_mean),
        ("in.latents_std", &mut cfg.latents_std),
    ] {
        for (d, v) in dst.iter_mut().zip(values(map, name).1) {
            *d = f64::from(v);
        }
    }
    cfg
}

#[test]
fn video_decoder_matches_diffusers_tile_by_tile_and_stitched() {
    let map = fixture("vae_tiny.st");
    let decoder = H3VideoDecoder::load(vae_config(&map), &map).expect("load");

    // One tile through the ViT alone: [1, C, 7, 2, 2] denormalized, as the reference feeds it.
    let tile = tensor(&map, "in.tile");
    let tokens = tile
        .reshape(tile.shape[1..].to_vec())
        .expect("drop batch")
        .permute(&[1, 2, 3, 0])
        .expect("channel last")
        .reshape(vec![1, 28, 5])
        .expect("tokens");
    let mut taps: Vec<(usize, Vec<f32>)> = Vec::new();
    let out = decoder
        .decode_tiles_observed(&tokens, [7, 2, 2], &mut |block, x| {
            taps.push((block, x.host_cow()?.into_owned()));
            Ok(())
        })
        .expect("tile")
        .remove(0);
    for (block, got) in &taps {
        assert_close(
            &format!("tile block {block}"),
            got,
            &values(&map, &format!("ref.tile_block_{block}")).1,
            1e-5,
        );
    }
    assert_matches("tile", &out, &map, "ref.tile", 1e-5);

    // The full decode: mean/std, 2 x 4 tiles with raw-neighbour blending, 2 temporal chunks.
    let (shape, want) = values(&map, "ref.video");
    let (frames, plane) = (shape[2], shape[3] * shape[4]);
    let mut video = vec![f32::NAN; want.len()];
    let emitted = decoder
        .decode_raw_streaming(&tensor(&map, "in.latent"), &mut |offset, chunk| {
            let host = chunk.host_cow()?;
            let f = chunk.shape[1];
            for ch in 0..3 {
                video[(ch * frames + offset) * plane..][..f * plane]
                    .copy_from_slice(&host[ch * f * plane..(ch + 1) * f * plane]);
            }
            Ok(())
        })
        .expect("decode");
    assert_eq!((emitted, shape), (39, vec![1, 3, 39, 12, 20]));
    assert_close("video", &video, &want, 1e-5);
}

#[test]
fn audio_decoder_matches_diffusers_stage_by_stage() {
    let map = fixture("audio_vae_tiny.st");
    let mut cfg = H3AudioVaeConfig::fasth3_8step();
    cfg.latent_dim = 8;
    cfg.latent_channels = 4;
    cfg.decoder_dim = 128;
    cfg.decoder_rates = [5, 2, 2, 2, 2, 2, 2];
    cfg.decoder_kernel_sizes = [9, 4, 4, 4, 4, 4, 4];
    for (name, dst) in [
        ("in.latents_mean", &mut cfg.latents_mean),
        ("in.latents_std", &mut cfg.latents_std),
    ] {
        for (d, v) in dst.iter_mut().zip(values(&map, name).1) {
            *d = f64::from(v);
        }
    }
    let decoder = H3AudioDecoder::load(cfg, &map).expect("load");
    let mut seen = 0usize;
    let wave = decoder
        .decode_observed(&tensor(&map, "in.latent"), &mut |name, x| {
            assert_close(
                name,
                &x.host_cow()?,
                &values(&map, &format!("ref.{name}")).1,
                1e-5,
            );
            seen += 1;
            Ok(())
        })
        .expect("decode");
    assert_eq!(
        seen,
        1 + 2 * 7,
        "conv_pre, then up_i and stage_i for seven stages"
    );
    assert_matches("wave", &wave, &map, "ref.wave", 1e-5);
    // The DiT's row layout (left channel's latents, then the right's) decodes to the same stereo pair.
    let from_rows = decoder
        .decode_rows(&tensor(&map, "in.rows"), 2)
        .expect("rows");
    assert_matches("wave from rows", &from_rows, &map, "ref.wave", 1e-5);
}

/// A rank-2 PEFT adapter on every attention and FFN linear of the tiny DiT
/// and its refiner, merged (`LoraMode::Merge`) and as an unmerged branch
/// (`LoraMode::Branch`, the LongLive-Plug default), through the production
/// loaders: the two forwards agree up to the branch's bfloat16 factors, and
/// both move the velocity away from the unadapted model.
#[test]
fn an_unmerged_lora_branch_matches_the_merge_on_the_tiny_dit() {
    use super::lora::{H3LoraFuse, LoraMode};
    use fastvideo_models::h3::lora::plan_from_keys;
    use std::collections::HashMap;

    let (cfg, map) = (dit_config(), fixture("dit_tiny.st"));
    let rank = 2usize;
    let mut modules = Vec::new();
    for i in 0..cfg.num_layers {
        modules.push(format!("transformer_blocks.{i}"));
    }
    for i in 0..cfg.num_refiner_layers {
        modules.push(format!("token_refiner.refiner_blocks.{i}"));
    }
    let ramp = |n: usize, seed: f32| -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + seed) * 0.37).sin() * 0.25)
            .collect()
    };
    let mut tensors = Vec::new();
    let mut seed = 1.0f32;
    for block in &modules {
        for target in [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
        ] {
            let module = format!("{block}.{target}");
            let shape = map
                .shape(&format!("{module}.weight"))
                .unwrap_or_else(|| panic!("{module}"));
            let (out, inn) = (shape[0], shape[1]);
            tensors.push((
                format!("{module}.lora_A.weight"),
                vec![rank, inn],
                ramp(rank * inn, seed),
            ));
            tensors.push((
                format!("{module}.lora_B.weight"),
                vec![out, rank],
                ramp(out * rank, seed + 0.5),
            ));
            seed += 1.0;
        }
    }
    let keys: Vec<String> = tensors.iter().map(|(k, _, _)| k.clone()).collect();
    let adapter = WeightMap::from_f32_tensors(tensors);
    let open = |mode: Option<LoraMode>| -> Option<H3LoraFuse> {
        let mode = mode?;
        let plan = plan_from_keys(&keys, &HashMap::new(), "tiny-adapter").unwrap();
        let mut fuse = H3LoraFuse::from_plan(
            &map,
            &adapter,
            std::path::Path::new("tiny-adapter"),
            plan,
            rank as u32,
            1.0,
        )
        .unwrap();
        assert_eq!(fuse.pairs_total, 6 * modules.len());
        fuse.set_mode(mode);
        Some(fuse)
    };

    let layout = H3PackedLayout::new(5, (3, 4, 6), 3, cfg.patch_size).expect("layout");
    let (noise_shape, noise) = values(&map, "in.video_noise");
    let rows = patchify(
        &noise,
        [
            noise_shape[1],
            noise_shape[2],
            noise_shape[3],
            noise_shape[4],
        ],
        cfg.patch_size,
    )
    .expect("patchify");
    let schedule = H3JointSchedule::fasth3_8step();
    let step = values(&map, "in.step").1[0] as usize;
    let device_layout = DeviceLayout::new(&cfg, layout.clone()).expect("rope");
    let video_rows =
        CudaTensor::from_vec(rows, vec![layout.video.len, cfg.video_patch_dim()]).expect("rows");
    let audio_rows = tensor(&map, "in.audio_rows");

    let run = |mode: Option<LoraMode>| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut lora = open(mode);
        let refiner = H3TextRefiner::load_with(&cfg, &map, &mut lora).expect("refiner");
        let refined = refiner.forward(&tensor(&map, "in.text")).expect("refine");
        let model = H3Transformer::load_with(cfg.clone(), &map, &schedule, false, None, &mut lora)
            .expect("load");
        if let Some(f) = &lora {
            f.finish().expect("every pair consumed");
        }
        let (video, audio) = model
            .forward(
                step,
                &video_rows,
                &audio_rows,
                &refined,
                &device_layout,
                AttnMode::Dense,
                None,
                None,
                None,
            )
            .expect("forward");
        (
            refined.host_cow().unwrap().into_owned(),
            video.host_cow().unwrap().into_owned(),
            audio.host_cow().unwrap().into_owned(),
        )
    };
    let base = run(None);
    let merged = run(Some(LoraMode::Merge));
    let branched = run(Some(LoraMode::Branch));
    for (name, b, m, br) in [
        ("text_refined", &base.0, &merged.0, &branched.0),
        ("video velocity", &base.1, &merged.1, &branched.1),
        ("audio velocity", &base.2, &merged.2, &branched.2),
    ] {
        let moved = b
            .iter()
            .zip(m)
            .fold(0.0f32, |acc, (p, q)| acc.max((p - q).abs()));
        assert!(
            moved > 1e-2,
            "{name}: the adapter moves the output ({moved})"
        );
        let gap = m
            .iter()
            .zip(br)
            .fold(0.0f32, |acc, (p, q)| acc.max((p - q).abs()));
        // The branch's bf16 factors and bf16 `A x` against the f32 merge.
        assert!(
            gap <= 3e-2 * moved + 1e-5,
            "{name}: branch against merge {gap} (the adapter moved {moved})"
        );
    }
}
