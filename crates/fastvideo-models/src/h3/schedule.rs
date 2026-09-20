//! `MiniMaxH3Scheduler` (diffusers `scheduling_minimax_h3.py`) and the FastH3
//! DMD ladder on top of it (FastVideo `minimax_h3_denoising.py:73-87`), as pure
//! host math.
//!
//! Three conventions differ from every other flow-match scheduler in this
//! crate, and getting any of them backwards still produces a finite tensor:
//!
//! 1. The transformer is conditioned on **`t = 1 - sigma` in `[0, 1]`**, where
//!    `t = 1` is clean. It never sees `999`, `874`, ...; those rungs only name
//!    unshifted sigmas.
//! 2. The velocity is **data-ward**: `x0 = x_t + sigma * v` (note the `+`).
//! 3. Video and audio run **two schedules** over one shared ladder, shifted by
//!    10 and 3, so one forward carries two distinct timesteps.
//!
//! All arithmetic is float32 in the order the reference evaluates it, so the
//! sigmas are bit-identical to torch's.

use super::config::{H3InferenceContract, MODALITY_NUM, TAG_AUDIO, TAG_TEXT, TAG_VIDEO};

/// One modality's schedule: `sigmas` has one more entry than `timesteps`, and
/// ends at exactly `0.0`.
#[derive(Debug, Clone, PartialEq)]
pub struct H3Schedule {
    pub shift: f64,
    pub sigmas: Vec<f32>,
    /// `1 - sigmas[..n-1]`, the value the transformer's time embedding consumes.
    pub timesteps: Vec<f32>,
}

/// Scalars of one Euler step, for callers that combine device tensors
/// themselves: `x0 = x + sigma_from_timestep * v`, then
/// `x_next = ratio * x + (1 - ratio) * x0`. On the last step `ratio` is `0`
/// and `x_next = x0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct H3StepCoeffs {
    /// `1 - timestep`, recovered from the float32 timestep rather than read off
    /// the sigma grid: `1 - (1 - sigma)` is not exact below 0.5 and the
    /// reference keeps the two apart (`scheduling_minimax_h3.py:260-269`).
    pub sigma_from_timestep: f32,
    /// `sigma_next / sigma`.
    pub ratio: f32,
}

/// `sigma' = s * sigma / (1 + (s - 1) * sigma)`, evaluated as torch does on a
/// float32 tensor: both Python scalars `s` and `s - 1` are rounded to float32
/// before they meet the tensor.
fn shift_sigma(shift: f64, base: f32) -> f32 {
    let s = shift as f32;
    let s_minus_1 = (shift - 1.0) as f32;
    (s * base) / (1.0f32 + s_minus_1 * base)
}

/// `torch.linspace(1.0, 0.0, n, dtype=float32)`: the first half counts up from
/// `start`, the second half counts down from `end`, with a float32 step.
fn linspace_one_to_zero(n: usize) -> Vec<f32> {
    let step = (0.0f32 - 1.0f32) / (n - 1) as f32;
    let half = n / 2;
    (0..n)
        .map(|i| {
            if i < half {
                1.0f32 + step * i as f32
            } else {
                0.0f32 - step * (n - 1 - i) as f32
            }
        })
        .collect()
}

impl H3Schedule {
    fn from_sigmas(shift: f64, sigmas: Vec<f32>) -> Result<Self, String> {
        let decreasing = sigmas.windows(2).all(|w| w[1] < w[0]);
        if sigmas.len() < 2 || !decreasing || sigmas[sigmas.len() - 1] != 0.0 {
            return Err(
                "sigmas must hold at least two strictly decreasing values ending at 0.0".into(),
            );
        }
        let timesteps = sigmas[..sigmas.len() - 1]
            .iter()
            .map(|&s| 1.0f32 - s)
            .collect();
        Ok(Self {
            shift,
            sigmas,
            timesteps,
        })
    }

    /// The distilled recipe (`_set_dmd_schedule`): `base = rungs / 1000` with a
    /// trailing `0`, pushed through this modality's shift and handed to the
    /// scheduler as explicit sigmas. `rung / 1000.0` is a float64 division that
    /// is then stored as float32, exactly as `torch.tensor([...], float32)`.
    pub fn from_dmd_rungs(rungs: &[u32], shift: f64) -> Result<Self, String> {
        if shift <= 0.0 || !shift.is_finite() {
            return Err(format!("shift must be positive and finite, got {shift}"));
        }
        let valid =
            rungs.iter().all(|&r| r > 0 && r <= 1000) && rungs.windows(2).all(|w| w[0] > w[1]);
        if rungs.is_empty() || !valid {
            return Err("DMD rungs must be strictly decreasing integers in (0, 1000]".into());
        }
        let sigmas = rungs
            .iter()
            .map(|&r| (f64::from(r) / 1000.0) as f32)
            .chain(std::iter::once(0.0f32))
            .map(|base| shift_sigma(shift, base))
            .collect();
        Self::from_sigmas(shift, sigmas)
    }

    /// The base (undistilled) model's grid, `set_timesteps(num_points)`:
    /// `linspace(1, 0, num_points)` through the shift, consecutive duplicates
    /// collapsed. `num_points` counts sigma-grid points, so it drives
    /// `num_points - 1` forwards. Not what FastH3 runs: its first rung is 0.999,
    /// not 1.0.
    pub fn uniform(num_points: usize, shift: f64) -> Result<Self, String> {
        if shift <= 0.0 || !shift.is_finite() {
            return Err(format!("shift must be positive and finite, got {shift}"));
        }
        if num_points < 2 {
            return Err(format!(
                "need at least two sigma grid points, got {num_points}"
            ));
        }
        let mut sigmas: Vec<f32> = linspace_one_to_zero(num_points)
            .into_iter()
            .map(|b| shift_sigma(shift, b))
            .collect();
        sigmas.dedup(); // torch.unique_consecutive
        Self::from_sigmas(shift, sigmas)
    }

    /// Transformer forwards this schedule drives.
    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    pub fn step_coeffs(&self, i: usize) -> Result<H3StepCoeffs, String> {
        if i >= self.num_steps() {
            return Err(format!(
                "H3 step {i} is past the end of a {}-step schedule",
                self.num_steps()
            ));
        }
        Ok(H3StepCoeffs {
            sigma_from_timestep: 1.0f32 - self.timesteps[i],
            ratio: self.sigmas[i + 1] / self.sigmas[i],
        })
    }

    /// Host reference for `MiniMaxH3Scheduler.step` on float32 latents.
    pub fn step(&self, i: usize, sample: &[f32], velocity: &[f32]) -> Result<Vec<f32>, String> {
        if sample.len() != velocity.len() {
            return Err("sample / velocity length mismatch".into());
        }
        let c = self.step_coeffs(i)?;
        Ok(sample
            .iter()
            .zip(velocity)
            .map(|(&x, &v)| {
                let denoised = x + c.sigma_from_timestep * v;
                c.ratio * x + (1.0f32 - c.ratio) * denoised
            })
            .collect())
    }

    /// `scale_noise`: `x_t = t * x0 + (1 - t) * noise`. T2AV never calls it (it
    /// noises keyframe anchors); it is here because it fixes the direction of
    /// `t` in one line.
    pub fn scale_noise(timestep: f32, clean: &[f32], noise: &[f32]) -> Result<Vec<f32>, String> {
        if clean.len() != noise.len() {
            return Err("clean / noise length mismatch".into());
        }
        Ok(clean
            .iter()
            .zip(noise)
            .map(|(&x, &n)| timestep * x + (1.0f32 - timestep) * n)
            .collect())
    }
}

/// The distinct timesteps of one forward and which of them each modality reads
/// (`build_row_timesteps`, `packing.py:489-500`, for a request with no
/// condition rows). Text rows take the **video** timestep.
#[derive(Debug, Clone, PartialEq)]
pub struct H3RowTimesteps {
    /// Sorted ascending, duplicates removed (`torch.unique(sorted=True)`): the
    /// transformer's `timestep` input.
    pub timesteps: Vec<f32>,
    pub video_index: usize,
    pub audio_index: usize,
}

impl H3RowTimesteps {
    pub fn new(video_timestep: f32, audio_timestep: f32) -> Self {
        let mut timesteps = vec![video_timestep, audio_timestep];
        timesteps.sort_by(f32::total_cmp);
        timesteps.dedup();
        let find = |t: f32| timesteps.iter().position(|&u| u == t).unwrap_or(0);
        let (video_index, audio_index) = (find(video_timestep), find(audio_timestep));
        Self {
            timesteps,
            video_index,
            audio_index,
        }
    }

    /// Row of the per-block AdaLN table for a modality tag:
    /// `timestep_index * 3 + tag` (`minimax_h3.py:926`).
    pub fn adaln_row(&self, tag: u8) -> usize {
        let index = if tag == TAG_AUDIO {
            self.audio_index
        } else {
            self.video_index
        };
        index * MODALITY_NUM + usize::from(tag)
    }

    /// The three `(timestep_index, tag)` rows a T2AV forward reads, in
    /// `[video, text, audio]` order. Nothing else in the 6-row table is used,
    /// which is what lets the AdaLN projections be evaluated once per ladder.
    pub fn adaln_rows(&self) -> [usize; 3] {
        [
            self.adaln_row(TAG_VIDEO),
            self.adaln_row(TAG_TEXT),
            self.adaln_row(TAG_AUDIO),
        ]
    }
}

/// Both schedules of a request, stepped in lockstep.
#[derive(Debug, Clone, PartialEq)]
pub struct H3JointSchedule {
    pub video: H3Schedule,
    pub audio: H3Schedule,
}

impl H3JointSchedule {
    pub fn from_contract(contract: &H3InferenceContract) -> Result<Self, String> {
        let video = H3Schedule::from_dmd_rungs(
            &contract.dmd_denoising_steps,
            contract.video_scheduler_shift,
        )?;
        let audio = H3Schedule::from_dmd_rungs(
            &contract.dmd_denoising_steps,
            contract.audio_scheduler_shift,
        )?;
        if contract.num_inference_steps != video.sigmas.len()
            || contract.transformer_forwards != video.num_steps()
        {
            return Err("contract grid-point / forward counts disagree with its ladder".into());
        }
        Ok(Self { video, audio })
    }

    pub fn fasth3_8step() -> Self {
        // The constants are validated by the tests below; this cannot fail.
        Self::from_contract(&H3InferenceContract::fasth3_8step())
            .unwrap_or_else(|e| unreachable!("{e}"))
    }

    pub fn fasth3_4step_vsa() -> Self {
        Self::from_contract(&H3InferenceContract::fasth3_4step_vsa())
            .unwrap_or_else(|e| unreachable!("{e}"))
    }

    pub fn fasth3_4step_dense() -> Self {
        Self::from_contract(&H3InferenceContract::fasth3_4step_dense())
            .unwrap_or_else(|e| unreachable!("{e}"))
    }

    pub fn num_steps(&self) -> usize {
        self.video.num_steps()
    }

    pub fn row_timesteps(&self, i: usize) -> Result<H3RowTimesteps, String> {
        if i >= self.num_steps() {
            return Err(format!(
                "H3 step {i} is past the end of a {}-step schedule",
                self.num_steps()
            ));
        }
        Ok(H3RowTimesteps::new(
            self.video.timesteps[i],
            self.audio.timesteps[i],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RUNGS: [u32; 8] = [999, 874, 749, 624, 500, 375, 250, 125];

    // Reference values: numpy float32 evaluation of `_set_dmd_schedule`
    // (`shift * base / (1 + (shift - 1) * base)`, `base = rung / 1000`), which
    // is elementwise IEEE arithmetic and therefore what torch produces too.
    const VIDEO_SIGMA_BITS: [u32; 9] = [
        0x3f7f_f970,
        0x3f7c_5ca0,
        0x3f77_b303,
        0x3f71_7377,
        0x3f68_ba2f,
        0x3f5b_6db7,
        0x3f44_ec4f,
        0x3f16_9697,
        0,
    ];
    const VIDEO_TIMESTEP_BITS: [u32; 8] = [
        0x38d2_0000,
        0x3c68_d800,
        0x3d04_cfd0,
        0x3d68_c890,
        0x3dba_2e88,
        0x3e12_4924,
        0x3e6c_4ec4,
        0x3ed2_d2d2,
    ];
    const AUDIO_SIGMA_BITS: [u32; 9] = [
        0x3f7f_ea23,
        0x3f74_4312,
        0x3f66_46ea,
        0x3f55_2e75,
        0x3f40_0000,
        0x3f24_9249,
        0x3f00_0000,
        0x3e99_999a,
        0,
    ];
    const AUDIO_TIMESTEP_BITS: [u32; 8] = [
        0x39ae_e800,
        0x3d3b_cee0,
        0x3dcd_c8b0,
        0x3e2b_462c,
        0x3e80_0000,
        0x3eb6_db6e,
        0x3f00_0000,
        0x3f33_3333,
    ];

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn video_ladder_is_bit_exact() {
        let s = H3Schedule::from_dmd_rungs(&RUNGS, 10.0).unwrap();
        assert_eq!(bits(&s.sigmas), VIDEO_SIGMA_BITS);
        assert_eq!(bits(&s.timesteps), VIDEO_TIMESTEP_BITS);
        assert_eq!(s.num_steps(), 8);
    }

    #[test]
    fn audio_ladder_is_bit_exact() {
        let s = H3Schedule::from_dmd_rungs(&RUNGS, 3.0).unwrap();
        assert_eq!(bits(&s.sigmas), AUDIO_SIGMA_BITS);
        assert_eq!(bits(&s.timesteps), AUDIO_TIMESTEP_BITS);
    }

    #[test]
    fn ladder_matches_the_closed_form() {
        // By hand: rung 500 -> base 0.5 -> 10*0.5/(1+9*0.5) = 5/5.5 = 10/11;
        // rung 125 -> 1.25/2.125 = 10/17. Audio: 1.5/2 = 0.75 and 0.375/1.25 = 0.3.
        let v = H3Schedule::from_dmd_rungs(&RUNGS, 10.0).unwrap();
        let a = H3Schedule::from_dmd_rungs(&RUNGS, 3.0).unwrap();
        assert!((f64::from(v.sigmas[4]) - 10.0 / 11.0).abs() < 1e-7);
        assert!((f64::from(v.sigmas[7]) - 10.0 / 17.0).abs() < 1e-7);
        assert_eq!(a.sigmas[4], 0.75);
        assert!((f64::from(a.sigmas[7]) - 0.3).abs() < 1e-7);
        // The first rung is 0.999, not 1.0: the first forward is *not* at t = 0.
        assert!(v.sigmas[0] < 1.0 && v.timesteps[0] > 0.0);
        assert!((f64::from(v.timesteps[0]) - 1.001e-4).abs() < 1e-6);
        // Video is always noisier than audio at the same rung (shift 10 > 3).
        for i in 0..8 {
            assert!(v.sigmas[i] > a.sigmas[i], "step {i}");
            assert!(v.timesteps[i] < a.timesteps[i], "step {i}");
        }
    }

    #[test]
    fn uniform_grid_differs_from_the_ladder_only_in_the_interior_rungs() {
        let u = H3Schedule::uniform(9, 10.0).unwrap();
        assert_eq!(u.sigmas.len(), 9);
        assert_eq!(u.sigmas[0], 1.0);
        assert_eq!(u.timesteps[0], 0.0);
        // linspace(1, 0, 9) is exact in float32, so these are closed forms:
        // 0.875 -> 8.75/8.875, 0.5 -> 10/11.
        assert_eq!(u.sigmas[1], 8.75f32 / 8.875f32);
        let ladder = H3Schedule::from_dmd_rungs(&RUNGS, 10.0).unwrap();
        // Rungs 500, 375, 250, 125 coincide with the uniform grid; 999/874/749/624 do not.
        assert_eq!(bits(&u.sigmas[4..]), bits(&ladder.sigmas[4..]));
        for i in 0..4 {
            assert_ne!(u.sigmas[i], ladder.sigmas[i], "rung {i}");
        }
        // The base model's 50-point, shift-12 grid has no float32 collisions to collapse.
        assert_eq!(H3Schedule::uniform(50, 12.0).unwrap().sigmas.len(), 50);
    }

    #[test]
    fn invalid_schedules_are_rejected() {
        assert!(H3Schedule::from_dmd_rungs(&[], 10.0).is_err());
        assert!(H3Schedule::from_dmd_rungs(&[500, 500], 10.0).is_err());
        assert!(H3Schedule::from_dmd_rungs(&[250, 500], 10.0).is_err());
        assert!(H3Schedule::from_dmd_rungs(&[1001], 10.0).is_err());
        assert!(H3Schedule::from_dmd_rungs(&[0], 10.0).is_err());
        assert!(H3Schedule::from_dmd_rungs(&RUNGS, 0.0).is_err());
        assert!(H3Schedule::uniform(1, 10.0).is_err());
    }

    #[test]
    fn step_is_the_x0_blend_with_a_plus_sign() {
        let s = H3Schedule::from_dmd_rungs(&RUNGS, 10.0).unwrap();
        let x = [0.5f32, -1.25, 2.0];
        let v = [0.1f32, 0.3, -0.7];

        let c = s.step_coeffs(3).unwrap();
        assert_eq!(c.ratio, s.sigmas[4] / s.sigmas[3]);
        assert_eq!(c.sigma_from_timestep, 1.0f32 - s.timesteps[3]);
        let got = s.step(3, &x, &v).unwrap();
        for k in 0..3 {
            let x0 = x[k] + c.sigma_from_timestep * v[k];
            assert_eq!(got[k], c.ratio * x[k] + (1.0f32 - c.ratio) * x0);
            // Same update in the more familiar Euler form, x + (sigma - sigma_next) * v.
            let euler = f64::from(x[k]) + f64::from(s.sigmas[3] - s.sigmas[4]) * f64::from(v[k]);
            assert!((f64::from(got[k]) - euler).abs() < 1e-6, "k={k}");
        }

        // Last step: sigma_next = 0, so the result is x0 itself.
        let last = s.step_coeffs(7).unwrap();
        assert_eq!(last.ratio, 0.0);
        let got = s.step(7, &x, &v).unwrap();
        for k in 0..3 {
            assert_eq!(got[k], x[k] + last.sigma_from_timestep * v[k]);
        }

        assert!(s.step(8, &x, &v).is_err());
        assert!(s.step(0, &x, &v[..2]).is_err());
    }

    #[test]
    fn a_perfect_velocity_lands_on_the_data() {
        // x_t = t*x0 + (1-t)*noise and v = x0 - noise, so x0 = x_t + sigma*v
        // at every step and the loop must finish on x0.
        let s = H3Schedule::from_dmd_rungs(&RUNGS, 10.0).unwrap();
        let (x0, noise) = ([0.25f32, -2.0, 1.5], [1.0f32, 0.5, -0.75]);
        let v: Vec<f32> = x0.iter().zip(&noise).map(|(a, b)| a - b).collect();
        let mut x = H3Schedule::scale_noise(s.timesteps[0], &x0, &noise).unwrap();
        for i in 0..s.num_steps() {
            x = s.step(i, &x, &v).unwrap();
        }
        for k in 0..3 {
            assert!((x[k] - x0[k]).abs() < 1e-5, "k={k}: {} vs {}", x[k], x0[k]);
        }
    }

    #[test]
    fn scale_noise_direction() {
        let (clean, noise) = ([2.0f32], [10.0f32]);
        assert_eq!(
            H3Schedule::scale_noise(1.0, &clean, &noise).unwrap(),
            vec![2.0]
        );
        assert_eq!(
            H3Schedule::scale_noise(0.0, &clean, &noise).unwrap(),
            vec![10.0]
        );
    }

    #[test]
    fn four_step_preview_ladder() {
        // Preview VSA / Dense: rungs [999, 749, 500, 250], video shift 12, audio 3.
        const RUNGS4: [u32; 4] = [999, 749, 500, 250];
        let v = H3Schedule::from_dmd_rungs(&RUNGS4, 12.0).unwrap();
        let a = H3Schedule::from_dmd_rungs(&RUNGS4, 3.0).unwrap();
        assert_eq!(v.num_steps(), 4);
        assert_eq!(a.num_steps(), 4);
        // Closed forms at half-integers: 0.5 -> 12*0.5/(1+11*0.5) = 6/6.5 = 12/13;
        // 0.25 -> 3/(1+2.75) = 0.8. Audio: 0.5 -> 0.75, 0.25 -> 0.5.
        assert!((f64::from(v.sigmas[2]) - 12.0 / 13.0).abs() < 1e-7);
        assert_eq!(v.sigmas[3], 0.8);
        assert_eq!(a.sigmas[2], 0.75);
        assert_eq!(a.sigmas[3], 0.5);
        assert_eq!(v.sigmas[4], 0.0);
        // First rung is still 0.999, not 1.0.
        assert!(v.sigmas[0] < 1.0 && v.timesteps[0] > 0.0);
        for i in 0..4 {
            assert!(v.sigmas[i] > a.sigmas[i], "step {i}");
        }
        let j = H3JointSchedule::fasth3_4step_vsa();
        assert_eq!(j.num_steps(), 4);
        assert_eq!(j.video.sigmas, v.sigmas);
        assert_eq!(j.audio.sigmas, a.sigmas);
        assert_eq!(
            H3JointSchedule::fasth3_4step_dense().video.sigmas,
            j.video.sigmas
        );
    }

    #[test]
    fn row_timesteps_and_adaln_rows() {
        let j = H3JointSchedule::fasth3_8step();
        assert_eq!(j.num_steps(), 8);
        for i in 0..8 {
            let r = j.row_timesteps(i).unwrap();
            // Video t < audio t at every rung, so video sorts first.
            assert_eq!(
                r.timesteps,
                vec![j.video.timesteps[i], j.audio.timesteps[i]]
            );
            assert_eq!((r.video_index, r.audio_index), (0, 1));
            // (t_video, video) = 0, (t_video, text) = 1, (t_audio, audio) = 1*3 + 2.
            assert_eq!(r.adaln_rows(), [0, 1, 5]);
        }
        assert!(j.row_timesteps(8).is_err());

        // Equal timesteps collapse to one entry, as torch.unique does.
        let same = H3RowTimesteps::new(0.25, 0.25);
        assert_eq!(same.timesteps, vec![0.25]);
        assert_eq!(same.adaln_rows(), [0, 1, 2]);
        // And the order follows the values, not the modality.
        let flipped = H3RowTimesteps::new(0.75, 0.5);
        assert_eq!((flipped.video_index, flipped.audio_index), (1, 0));
        assert_eq!(flipped.adaln_rows(), [3, 4, 2]);
    }
}
