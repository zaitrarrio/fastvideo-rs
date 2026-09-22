//! Qwen3-VL 3-axis mRoPE position ids (FastVideo `_get_rope_index`).

use crate::h3::config::H3TextEncoderConfig;

/// One 3-axis position per sequence token: `[temporal, height, width]`.
pub type MropePos = [f64; 3];

/// Build interleaved mRoPE positions for a presentation that may contain
/// image/video pad spans. Grids are in **patch** cells `(t, h, w)` before the
/// spatial merge (same as vision processor `grid_thw`). Each vision_start in
/// `input_ids` consumes one grid; pad count must equal `t*(h/merge)*(w/merge)`.
pub fn build_mrope_positions(
    input_ids: &[u32],
    cfg: &H3TextEncoderConfig,
    spatial_merge: usize,
    image_grids: &[[usize; 3]],
    video_grids: &[[usize; 3]],
) -> Result<Vec<MropePos>, String> {
    let spatial_merge = spatial_merge.max(1);
    let mut image_index = 0usize;
    let mut video_index = 0usize;
    let mut positions: Vec<MropePos> = Vec::with_capacity(input_ids.len());

    let mut start = 0usize;
    while start < input_ids.len() {
        let mut next_pad = None;
        let mut is_image = false;
        for i in start..input_ids.len() {
            if input_ids[i] == cfg.image_token_id {
                next_pad = Some(i);
                is_image = true;
                break;
            }
            if input_ids[i] == cfg.video_token_id {
                next_pad = Some(i);
                is_image = false;
                break;
            }
        }
        let Some(end) = next_pad else {
            let remain = input_ids.len() - start;
            let offset = positions
                .iter()
                .map(|p| p[0].max(p[1]).max(p[2]))
                .fold(f64::NEG_INFINITY, f64::max);
            let offset = if offset.is_finite() { offset + 1.0 } else { 0.0 };
            append_text_span(&mut positions, remain, offset);
            break;
        };

        let text_len = end - start;
        let offset = positions
            .iter()
            .map(|p| p[0].max(p[1]).max(p[2]))
            .fold(f64::NEG_INFINITY, f64::max);
        let offset = if offset.is_finite() { offset + 1.0 } else { 0.0 };
        append_text_span(&mut positions, text_len, offset);

        let grid = if is_image {
            let g = image_grids
                .get(image_index)
                .copied()
                .ok_or_else(|| format!("missing image grid at index {image_index}"))?;
            image_index += 1;
            g
        } else {
            let g = video_grids
                .get(video_index)
                .copied()
                .ok_or_else(|| format!("missing video grid at index {video_index}"))?;
            video_index += 1;
            g
        };

        let (frames, height, width) = (grid[0].max(1), grid[1].max(1), grid[2].max(1));
        if height % spatial_merge != 0 || width % spatial_merge != 0 {
            return Err(format!(
                "vision grid HxW ({height}x{width}) not divisible by spatial_merge={spatial_merge}"
            ));
        }
        let gh = height / spatial_merge;
        let gw = width / spatial_merge;
        let n_vision = frames * gh * gw;

        let pad_end = end + n_vision;
        if pad_end > input_ids.len() {
            return Err(format!(
                "vision pad span overflows sequence: need {n_vision} pads from {end}"
            ));
        }
        let expected_id = if is_image {
            cfg.image_token_id
        } else {
            cfg.video_token_id
        };
        for &id in &input_ids[end..pad_end] {
            if id != expected_id {
                return Err("vision pad span interrupted by non-pad token".into());
            }
        }

        let llm_pos = vision_llm_positions(frames, gh, gw, text_len as f64 + offset);
        if llm_pos.len() != n_vision {
            return Err("vision position count mismatch".into());
        }
        positions.extend(llm_pos);
        start = pad_end;
    }

    if positions.len() != input_ids.len() {
        return Err(format!(
            "mRoPE length {} != sequence {}",
            positions.len(),
            input_ids.len()
        ));
    }
    Ok(positions)
}

fn append_text_span(out: &mut Vec<MropePos>, len: usize, offset: f64) {
    for i in 0..len {
        let v = offset + i as f64;
        out.push([v, v, v]);
    }
}

fn vision_llm_positions(frames: usize, gh: usize, gw: usize, offset: f64) -> Vec<MropePos> {
    let n = frames * gh * gw;
    let mut out = Vec::with_capacity(n);
    for t in 0..frames {
        for h in 0..gh {
            for w in 0..gw {
                out.push([
                    offset + t as f64,
                    offset + h as f64,
                    offset + w as f64,
                ]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h3::config::H3TextEncoderConfig;

    fn cfg() -> H3TextEncoderConfig {
        let mut c = H3TextEncoderConfig::fasth3_8step();
        c.vision_start_token_id = 10;
        c.vision_end_token_id = 11;
        c.image_token_id = 12;
        c.video_token_id = 13;
        c
    }

    #[test]
    fn text_only_is_identity() {
        let ids = vec![1u32, 2, 3, 4];
        let pos = build_mrope_positions(&ids, &cfg(), 2, &[], &[]).unwrap();
        assert_eq!(pos.len(), 4);
        assert_eq!(pos[2], [2.0, 2.0, 2.0]);
    }

    #[test]
    fn image_span_gets_spatial_axes() {
        let c = cfg();
        // text, vision_start, 4 pads (1x4x4 / merge2² = 4), vision_end, text
        let mut ids = vec![1u32, c.vision_start_token_id];
        ids.extend(std::iter::repeat_n(c.image_token_id, 4));
        ids.push(c.vision_end_token_id);
        ids.push(2);
        let grids = [[1usize, 4, 4]];
        let pos = build_mrope_positions(&ids, &c, 2, &grids, &[]).unwrap();
        assert_eq!(pos.len(), ids.len());
        assert_eq!(pos[0], [0.0, 0.0, 0.0]);
        assert_eq!(pos[1], [1.0, 1.0, 1.0]);
        assert_eq!(pos[2], [2.0, 2.0, 2.0]);
        assert_eq!(pos[3], [2.0, 2.0, 3.0]);
        assert_eq!(pos[4], [2.0, 3.0, 2.0]);
        assert_eq!(pos[5], [2.0, 3.0, 3.0]);
    }
}
