//! LPIPS (Zhang et al. 2018, `lpips` 0.1.4, `LPIPS(net="alex")`, version
//! 0.1): the perceptual distance sol-engine's `tools/vision/lpips_judge.py`
//! scores frame pairs with, ported so the evaluation gate runs in-process on
//! an image with no Python.
//!
//! The network is torchvision AlexNet's five conv layers
//! (`alexnet-owt-7be5be79.pth`, the file `torchvision.models.alexnet(
//! pretrained=True)` downloads) and LPIPS's five 1x1 linear heads
//! (`lpips/weights/v0.1/alex.pth`). Both are read straight from the official
//! `.pth` files ([`read_pth`]: torch's zip format and its legacy tar-less
//! format, through a small pickle interpreter), fetched and hash-pinned by
//! `scripts/gpu/fetch-lpips.sh` and hash-checked again here.
//!
//! `lpips.LPIPS.forward(in0, in1)` with inputs in [-1, 1]
//! (`im2tensor(load_image(p))` = `rgb / 127.5 - 1`):
//!
//! 1. `ScalingLayer`: `(x - shift) / scale`, shift `[-.030, -.088, -.188]`,
//!    scale `[.458, .448, .450]`.
//! 2. AlexNet slices: relu1 = relu(conv1 11x11/4 pad 2); relu2 =
//!    relu(conv2 5x5 pad 2 (maxpool 3/2 (relu1))); relu3 = relu(conv3 3x3 pad 1
//!    (maxpool 3/2 (relu2))); relu4, relu5 = relu(conv 3x3 pad 1) in turn.
//! 3. Per layer: `normalize_tensor` (divide by the channel L2 norm + 1e-10),
//!    squared difference, the 1x1 linear head (dropout is identity in eval),
//!    spatial mean. The score is the sum over the five layers.
//!
//! Two backends: plain Rust on the CPU (the reference; tests) and the device
//! (the convolutions on cuDNN, the cheap elementwise rest on the host), which
//! the matrix uses.

use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde_json::{json, Value};

use crate::clipcmp::Rgb;

/// torchvision AlexNet (`AlexNet_Weights.IMAGENET1K_V1`).
pub const ALEXNET_FILE: &str = "alexnet-owt-7be5be79.pth";
pub const ALEXNET_URL: &str = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth";
pub const ALEXNET_SHA256: &str = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02";
/// LPIPS v0.1 AlexNet linear heads (richzhang/PerceptualSimilarity, the
/// file the `lpips` 0.1.4 wheel ships).
pub const LIN_FILE: &str = "lpips_v0.1_alex.pth";
pub const LIN_SHA256: &str = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0";

const SHIFT: [f32; 3] = [-0.030, -0.088, -0.188];
const SCALE: [f32; 3] = [0.458, 0.448, 0.450];
/// `(features.<i>, stride, padding, maxpool before)` for the five slices.
const ALEX: [(usize, usize, usize, bool); 5] = [
    (0, 4, 2, false),
    (3, 1, 2, true),
    (6, 1, 1, true),
    (8, 1, 1, false),
    (10, 1, 1, false),
];
const EPS: f32 = 1e-10;

// ---------------------------------------------------------------- .pth reader

// The reader lives in `fastvideo_loader::pth` (shared with the LongLive-Plug
// `.pt` LoRA loader).
pub use fastvideo_loader::pth::read_pth;

// ---------------------------------------------------------------- weights

#[derive(Clone, Debug)]
struct Conv {
    w: Vec<f32>,
    b: Vec<f32>,
    cout: usize,
    cin: usize,
    k: usize,
    stride: usize,
    pad: usize,
    pool_before: bool,
}

#[derive(Clone, Debug)]
pub struct LpipsWeights {
    convs: Vec<Conv>,
    lins: Vec<Vec<f32>>,
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let bytes = std::fs::read(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    Ok(crate::benchmark::sha256(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

impl LpipsWeights {
    /// `dir` holds [`ALEXNET_FILE`] and [`LIN_FILE`] (`fetch-lpips.sh`). Both
    /// must match their pinned SHA-256.
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let (alex, lin) = (dir.join(ALEXNET_FILE), dir.join(LIN_FILE));
        if !alex.is_file() || !lin.is_file() {
            anyhow::bail!(
                "{}: needs {ALEXNET_FILE} ({ALEXNET_URL}) and {LIN_FILE}; run scripts/gpu/fetch-lpips.sh",
                dir.display()
            );
        }
        for (p, want) in [(&alex, ALEXNET_SHA256), (&lin, LIN_SHA256)] {
            let got = sha256_file(p)?;
            if got != want {
                anyhow::bail!("{}: sha256 {got}, pinned {want}", p.display());
            }
        }
        let a = read_pth(&alex, |k| k.starts_with("features."))?;
        let l = read_pth(&lin, |k| k.starts_with("lin"))?;
        let mut convs = Vec::new();
        for &(i, stride, pad, pool_before) in &ALEX {
            let w = a
                .get(&format!("features.{i}.weight"))
                .ok_or_else(|| anyhow::anyhow!("alexnet: features.{i}.weight missing"))?;
            let b = a
                .get(&format!("features.{i}.bias"))
                .ok_or_else(|| anyhow::anyhow!("alexnet: features.{i}.bias missing"))?;
            let [cout, cin, k, k2] = w.shape[..] else {
                anyhow::bail!("alexnet: features.{i}.weight shape {:?}", w.shape);
            };
            anyhow::ensure!(
                k == k2 && b.data.len() == cout,
                "alexnet: features.{i} shapes"
            );
            convs.push(Conv {
                w: w.data.clone(),
                b: b.data.clone(),
                cout,
                cin,
                k,
                stride,
                pad,
                pool_before,
            });
        }
        let mut lins = Vec::new();
        for (j, c) in convs.iter().enumerate() {
            let t = l
                .get(&format!("lin{j}.model.1.weight"))
                .ok_or_else(|| anyhow::anyhow!("lpips: lin{j}.model.1.weight missing"))?;
            anyhow::ensure!(
                t.data.len() == c.cout,
                "lpips: lin{j} has {} weights for {} channels",
                t.data.len(),
                c.cout
            );
            lins.push(t.data.clone());
        }
        Ok(Self { convs, lins })
    }

    /// A random network of the real shapes (tests).
    #[cfg(test)]
    fn random(seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let chans = [3, 64, 192, 384, 256, 256];
        let ks = [11, 5, 3, 3, 3];
        let convs = ALEX
            .iter()
            .enumerate()
            .map(|(j, &(_, stride, pad, pool_before))| {
                let (cin, cout, k) = (chans[j], chans[j + 1], ks[j]);
                let s = (2.0 / (cin * k * k) as f32).sqrt();
                Conv {
                    w: (0..cout * cin * k * k)
                        .map(|_| rng.gen_range(-s..s))
                        .collect(),
                    b: (0..cout).map(|_| rng.gen_range(-0.1..0.1)).collect(),
                    cout,
                    cin,
                    k,
                    stride,
                    pad,
                    pool_before,
                }
            })
            .collect::<Vec<_>>();
        let lins = convs
            .iter()
            .map(|c| (0..c.cout).map(|_| rng.gen_range(0.0..0.2)).collect())
            .collect();
        Self { convs, lins }
    }
}

/// `im2tensor` then `ScalingLayer`: CHW float32.
fn input(img: &Rgb) -> Vec<f32> {
    let hw = img.w * img.h;
    let mut x = vec![0f32; 3 * hw];
    for p in 0..hw {
        for c in 0..3 {
            let v = f32::from(img.px[p * 3 + c]) / 127.5 - 1.0;
            x[c * hw + p] = (v - SHIFT[c]) / SCALE[c];
        }
    }
    x
}

// ---------------------------------------------------------------- host

fn conv_host(x: &[f32], h: usize, w: usize, c: &Conv) -> (Vec<f32>, usize, usize) {
    let oh = (h + 2 * c.pad - c.k) / c.stride + 1;
    let ow = (w + 2 * c.pad - c.k) / c.stride + 1;
    let mut y = vec![0f32; c.cout * oh * ow];
    y.par_chunks_mut(oh * ow)
        .enumerate()
        .for_each(|(o, plane)| {
            plane.fill(c.b[o]);
            for ci in 0..c.cin {
                let xin = &x[ci * h * w..(ci + 1) * h * w];
                for ky in 0..c.k {
                    for kx in 0..c.k {
                        let wv = c.w[((o * c.cin + ci) * c.k + ky) * c.k + kx];
                        for oy in 0..oh {
                            let iy = (oy * c.stride + ky) as isize - c.pad as isize;
                            if iy < 0 || iy >= h as isize {
                                continue;
                            }
                            let row = &xin[iy as usize * w..(iy as usize + 1) * w];
                            let out = &mut plane[oy * ow..(oy + 1) * ow];
                            for (ox, o) in out.iter_mut().enumerate() {
                                let ix = (ox * c.stride + kx) as isize - c.pad as isize;
                                if ix >= 0 && ix < w as isize {
                                    *o += wv * row[ix as usize];
                                }
                            }
                        }
                    }
                }
            }
            for v in plane.iter_mut() {
                *v = v.max(0.0);
            }
        });
    (y, oh, ow)
}

fn maxpool_host(x: &[f32], c: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    let (oh, ow) = ((h - 3) / 2 + 1, (w - 3) / 2 + 1);
    let mut y = vec![0f32; c * oh * ow];
    y.par_chunks_mut(oh * ow)
        .enumerate()
        .for_each(|(ch, plane)| {
            let xin = &x[ch * h * w..(ch + 1) * h * w];
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut m = f32::NEG_INFINITY;
                    for dy in 0..3 {
                        for dx in 0..3 {
                            m = m.max(xin[(oy * 2 + dy) * w + ox * 2 + dx]);
                        }
                    }
                    plane[oy * ow + ox] = m;
                }
            }
        });
    (y, oh, ow)
}

/// relu1..relu5 as `(data, channels, h, w)`.
fn features_host(net: &LpipsWeights, img: &Rgb) -> Vec<(Vec<f32>, usize, usize, usize)> {
    let (mut x, mut c, mut h, mut w) = (input(img), 3usize, img.h, img.w);
    let mut out = Vec::with_capacity(5);
    for conv in &net.convs {
        if conv.pool_before {
            let (p, ph, pw) = maxpool_host(&x, c, h, w);
            (x, h, w) = (p, ph, pw);
        }
        let (y, oh, ow) = conv_host(&x, h, w, conv);
        (x, c, h, w) = (y, conv.cout, oh, ow);
        out.push((x.clone(), c, h, w));
    }
    out
}

/// One layer's term: spatial mean of `sum_c lin_c (n(a)_c - n(b)_c)^2`.
fn layer_distance_host(a: &[f32], b: &[f32], lin: &[f32], c: usize, hw: usize) -> f64 {
    let total: f64 = (0..hw)
        .into_par_iter()
        .map(|p| {
            let (mut na, mut nb) = (0f32, 0f32);
            for ch in 0..c {
                na += a[ch * hw + p] * a[ch * hw + p];
                nb += b[ch * hw + p] * b[ch * hw + p];
            }
            let (na, nb) = (na.sqrt() + EPS, nb.sqrt() + EPS);
            let mut s = 0f32;
            for ch in 0..c {
                let d = a[ch * hw + p] / na - b[ch * hw + p] / nb;
                s += lin[ch] * d * d;
            }
            f64::from(s)
        })
        .sum();
    total / hw as f64
}

// ---------------------------------------------------------------- device

/// The five convolutions on the GPU (cuDNN, f32); bias + ReLU, the max pools
/// and the distance stay on the host, where they are a few MB of work. No
/// runtime-compiled kernel: NVRTC PTX can be newer than the pod's driver
/// accepts (CUDA_ERROR_UNSUPPORTED_PTX_VERSION), and the crate's embedded
/// cubins carry no LPIPS kernel.
#[cfg(feature = "cuda")]
struct DeviceNet {
    dev: std::sync::Arc<fastvideo_cudarc::wan::device::DeviceContext>,
    convs: Vec<cudarc::driver::CudaSlice<f32>>,
}

#[cfg(feature = "cuda")]
impl DeviceNet {
    fn new(net: &LpipsWeights) -> anyhow::Result<Self> {
        let dev = fastvideo_cudarc::wan::device::global_device()
            .ok_or_else(|| anyhow::anyhow!("lpips: no CUDA device"))?;
        let convs = net
            .convs
            .iter()
            .map(|c| Ok(dev.stream.memcpy_stod(&c.w)?))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self { dev, convs })
    }

    fn features(
        &self,
        net: &LpipsWeights,
        img: &Rgb,
    ) -> anyhow::Result<Vec<(Vec<f32>, usize, usize, usize)>> {
        let s = &self.dev.stream;
        let (mut x, mut c, mut h, mut w) = (input(img), 3usize, img.h, img.w);
        let mut out = Vec::with_capacity(5);
        for (j, conv) in net.convs.iter().enumerate() {
            if conv.pool_before {
                let (p, ph, pw) = maxpool_host(&x, c, h, w);
                (x, h, w) = (p, ph, pw);
            }
            let xd = s.memcpy_stod(&x)?;
            let (y, shape) = fastvideo_cudarc::wan::conv::cudnn_conv(
                &xd,
                &[1, c, h, w],
                &self.convs[j],
                &[conv.cout, conv.cin, conv.k, conv.k],
                &[conv.pad, conv.pad],
                &[conv.stride, conv.stride],
            )
            .map_err(|e| anyhow::anyhow!("lpips conv{}: {e}", j + 1))?;
            let mut y = s.memcpy_dtov(&y)?;
            let (oh, ow) = (shape[2], shape[3]);
            y.par_chunks_mut(oh * ow)
                .enumerate()
                .for_each(|(o, plane)| {
                    for v in plane.iter_mut() {
                        *v = (*v + conv.b[o]).max(0.0);
                    }
                });
            (x, c, h, w) = (y, conv.cout, oh, ow);
            out.push((x.clone(), c, h, w));
        }
        Ok(out)
    }

    fn distance(&self, net: &LpipsWeights, a: &Rgb, b: &Rgb) -> anyhow::Result<[f64; 5]> {
        let (fa, fb) = (self.features(net, a)?, self.features(net, b)?);
        let mut l = [0f64; 5];
        for (j, ((xa, c, h, w), (xb, _, _, _))) in fa.iter().zip(&fb).enumerate() {
            l[j] = layer_distance_host(xa, xb, &net.lins[j], *c, h * w);
        }
        Ok(l)
    }
}

// ---------------------------------------------------------------- API

/// A loaded LPIPS(alex) scorer.
pub struct Lpips {
    net: LpipsWeights,
    #[cfg(feature = "cuda")]
    dev: Option<DeviceNet>,
}

impl Lpips {
    /// `device`: the convolutions on the GPU (cuDNN); the CPU path otherwise.
    pub fn load(dir: &Path, device: bool) -> anyhow::Result<Self> {
        Self::from_weights(LpipsWeights::load(dir)?, device)
    }

    fn from_weights(net: LpipsWeights, device: bool) -> anyhow::Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let dev = if device {
                crate::gpu::init("cuda").map_err(|e| anyhow::anyhow!("lpips device: {e:?}"))?;
                Some(DeviceNet::new(&net)?)
            } else {
                None
            };
            Ok(Self { net, dev })
        }
        #[cfg(not(feature = "cuda"))]
        {
            if device {
                anyhow::bail!("lpips: built without the cuda feature");
            }
            Ok(Self { net })
        }
    }

    pub fn backend(&self) -> &'static str {
        #[cfg(feature = "cuda")]
        if self.dev.is_some() {
            return "cuda (cuDNN conv)";
        }
        "cpu"
    }

    /// `(score, per-layer terms)`; `score` is their sum, as
    /// `LPIPS.forward(retPerLayer=True)`.
    pub fn score(&self, a: &Rgb, b: &Rgb) -> anyhow::Result<(f64, [f64; 5])> {
        anyhow::ensure!(
            a.w == b.w && a.h == b.h,
            "lpips: shapes differ ({}x{} vs {}x{})",
            a.w,
            a.h,
            b.w,
            b.h
        );
        anyhow::ensure!(
            a.w >= 63 && a.h >= 63,
            "lpips: {}x{} is below AlexNet's minimum",
            a.w,
            a.h
        );
        #[cfg(feature = "cuda")]
        if let Some(d) = &self.dev {
            let l = d.distance(&self.net, a, b)?;
            return Ok((l.iter().sum(), l));
        }
        Ok(score_host(&self.net, a, b))
    }
}

fn score_host(net: &LpipsWeights, a: &Rgb, b: &Rgb) -> (f64, [f64; 5]) {
    let (fa, fb) = (features_host(net, a), features_host(net, b));
    let mut l = [0f64; 5];
    for (j, ((xa, c, h, w), (xb, _, _, _))) in fa.iter().zip(&fb).enumerate() {
        l[j] = layer_distance_host(xa, xb, &net.lins[j], *c, h * w);
    }
    (l.iter().sum(), l)
}

/// Python's `round` (half to even), for sol-engine's frame selection.
fn round_half_even(x: f64) -> usize {
    let r = x.round();
    let r = if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    };
    r.max(0.0) as usize
}

/// sol-engine `select_stratified_and_worst_pairs` (collect_run.py:599-637):
/// `stratified` chronological picks, `round(i * last / (limit - 1))`, then
/// the `worst` pairs by mean |pixel diff| not already chosen, at most
/// `total`. Returns pair indices in selection order.
pub fn select_pairs(mae: &[f64], stratified: usize, worst: usize, total: usize) -> Vec<usize> {
    let n = mae.len();
    let mut out: Vec<usize> = Vec::new();
    let add = |out: &mut Vec<usize>, i: usize| {
        if !out.contains(&i) && out.len() < total {
            out.push(i);
        }
    };
    let limit = stratified.min(total);
    if n <= limit {
        (0..n).for_each(|i| add(&mut out, i));
    } else if limit <= 1 {
        add(&mut out, 0);
    } else {
        let last = (n - 1) as f64;
        for i in 0..limit {
            add(
                &mut out,
                round_half_even(i as f64 * last / (limit - 1) as f64),
            );
        }
    }
    if worst > 0 && out.len() < total {
        let mut order: Vec<usize> = (0..n).collect();
        // Python's sorted(reverse=True) is stable: ties keep frame order.
        order.sort_by(|&x, &y| mae[y].total_cmp(&mae[x]));
        for &i in order.iter().take(worst) {
            add(&mut out, i);
        }
    }
    out
}

/// sol-engine's LPIPS pair budget (collect_run.py:83-85).
pub const LPIPS_MAX_PAIRS: usize = 48;
pub const LPIPS_STRATIFIED_PAIRS: usize = 32;
pub const LPIPS_WORST_CASE_PAIRS: usize = 16;

/// Score the selected pairs of two frame lists; the `lpips_judge.py`
/// `success_payload` keys plus the frame indices scored.
pub fn judge(
    scorer: &Lpips,
    baseline: &[PathBuf],
    candidate: &[PathBuf],
    pairs: &[usize],
) -> anyhow::Result<Value> {
    let mut scores = Vec::with_capacity(pairs.len());
    let t0 = std::time::Instant::now();
    for &i in pairs {
        let (a, b) = (Rgb::load(&baseline[i])?, Rgb::load(&candidate[i])?);
        scores.push(scorer.score(&a, &b)?.0);
    }
    if scores.is_empty() {
        return Ok(
            json!({"metric": "lpips", "status": "unavailable", "reason": "no frame pairs", "n": 0}),
        );
    }
    let mut sorted = scores.clone();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    };
    Ok(json!({
        "metric": "lpips",
        "status": "ok",
        "per_frame": scores,
        "frames": pairs,
        "mean": scores.iter().sum::<f64>() / n as f64,
        "median": median,
        "max": sorted[n - 1],
        "n": n,
        "backend": scorer.backend(),
        "seconds": t0.elapsed().as_secs_f64(),
        "notes": ["lower_is_better", "frames_paired_by_order", "net=alex version=0.1 (lpips 0.1.4)"],
    }))
}

// ---------------------------------------------------------------- pinned reference

/// Official `lpips` 0.1.4 (`LPIPS(net="alex")`, torch CPU float32) on the
/// fixture pairs (`crates/fastvideo-gpucheck/fixtures/lpips`), computed by
/// `scripts/gpu/lpips_ref.py` in the upstream PyTorch image. Filled from
/// `lpips-ref.json`; see `PINNED_SOURCE`.
pub const PINNED: &[(&str, &str, f64)] = &[
    ("a", "a", 0.0),
    ("a", "a_noise", 0.11691055446863174),
    ("a", "a_blur", 0.17645150423049927),
    ("a", "a_shift", 0.08166157454252243),
    ("a", "a_tone", 0.007270511705428362),
    ("a", "b", 0.5431303977966309),
    ("c", "c_poster", 0.19080646336078644),
];
pub const PINNED_SOURCE: &str = "lpips 0.1.4, torch 2.13.0+cu130 / torchvision 0.28.0+cu130 CPU, runpod/pytorch:1.3.3-cu1300-torch2130-ubuntu2404, upstream run 26d5b13-09260351 (weights sha256 = ALEXNET_SHA256 / LIN_SHA256)";
/// Our port against the pinned numbers: float32 with a different summation
/// order (and cuDNN's algorithms on the device).
pub const PINNED_ABS_TOL: f64 = 2e-4;
pub const PINNED_REL_TOL: f64 = 1e-3;

#[cfg(test)]
pub fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/lpips")
}

/// The fixture PNGs, embedded so the check runs from the runtime image
/// (which carries the binary, not the crate sources).
const FIXTURES: &[(&str, &[u8])] = &[
    ("a", include_bytes!("../fixtures/lpips/a.png")),
    ("a_noise", include_bytes!("../fixtures/lpips/a_noise.png")),
    ("a_blur", include_bytes!("../fixtures/lpips/a_blur.png")),
    ("a_shift", include_bytes!("../fixtures/lpips/a_shift.png")),
    ("a_tone", include_bytes!("../fixtures/lpips/a_tone.png")),
    ("b", include_bytes!("../fixtures/lpips/b.png")),
    ("c", include_bytes!("../fixtures/lpips/c.png")),
    ("c_poster", include_bytes!("../fixtures/lpips/c_poster.png")),
];

/// Fixture `name` from `dir`, or the embedded copy.
pub fn fixture(dir: Option<&Path>, name: &str) -> anyhow::Result<Rgb> {
    if let Some(d) = dir {
        return Rgb::load(&d.join(format!("{name}.png")));
    }
    let (_, bytes) = FIXTURES
        .iter()
        .find(|(n, _)| *n == name)
        .ok_or_else(|| anyhow::anyhow!("no embedded LPIPS fixture {name}"))?;
    let img = image::load_from_memory(bytes)?.into_rgb8();
    let (w, h) = img.dimensions();
    Ok(Rgb {
        w: w as usize,
        h: h as usize,
        px: img.into_raw(),
    })
}

/// `fv-gpucheck lpips`: the port against the pinned official numbers on the
/// fixture pairs (both backends when a device is present), or one ad-hoc pair.
pub fn run(
    report: &mut crate::report::Report,
    weights: &Path,
    fixtures: Option<&Path>,
    pair: Option<(&Path, &Path)>,
    device: &str,
) -> crate::report::StageResult<()> {
    let net = LpipsWeights::load(weights)?;
    let mut scorers = vec![Lpips::from_weights(net.clone(), false)?];
    if crate::gpu::on_gpu(device) {
        scorers.push(Lpips::from_weights(net, true)?);
    }
    if let Some((a, b)) = pair {
        let (ia, ib) = (Rgb::load(a)?, Rgb::load(b)?);
        for s in &scorers {
            let (v, l) = s.score(&ia, &ib)?;
            report.note(
                format!("pair/{}", s.backend()),
                json!({"a": a, "b": b, "lpips": v, "per_layer": l}),
            );
        }
        return Ok(());
    }
    report.set("pinned_source", PINNED_SOURCE);
    if PINNED.is_empty() {
        return Err(crate::report::StageError::Check(
            "no pinned LPIPS reference numbers".into(),
        ));
    }
    let mut rows = Vec::new();
    let mut worst = 0f64;
    for &(a, b, want) in PINNED {
        let (ia, ib) = (fixture(fixtures, a)?, fixture(fixtures, b)?);
        for s in &scorers {
            let (got, layers) = s.score(&ia, &ib)?;
            let err = (got - want).abs();
            let ok = err <= PINNED_ABS_TOL.max(PINNED_REL_TOL * want.abs());
            worst = worst.max(err);
            rows.push(json!({"a": a, "b": b, "backend": s.backend(), "ours": got, "official": want, "abs_err": err, "per_layer": layers, "ok": ok}));
            report.check(
                format!("lpips/{a}--{b}/{}", s.backend()),
                ok,
                json!({"ours": got, "official": want, "abs_err": err}),
                json!({"abs_tol": PINNED_ABS_TOL, "rel_tol": PINNED_REL_TOL}),
            )?;
        }
    }
    report.set("pairs", rows);
    report.set("max_abs_err", worst);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: usize, h: usize, f: impl Fn(usize, usize, usize) -> u8) -> Rgb {
        let mut px = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    px.push(f(x, y, c));
                }
            }
        }
        Rgb { w, h, px }
    }

    #[test]
    fn identical_frames_score_zero_and_perturbations_score_positive() {
        let net = LpipsWeights::random(3);
        let a = frame(72, 64, |x, y, c| ((x * 3 + y * 5 + c * 40) % 256) as u8);
        let b = frame(72, 64, |x, y, c| {
            ((x * 3 + y * 5 + c * 40 + (x % 7)) % 256) as u8
        });
        let (same, _) = score_host(&net, &a, &a);
        assert_eq!(same, 0.0);
        let (d, l) = score_host(&net, &a, &b);
        assert!(d > 0.0 && l.iter().all(|v| *v >= 0.0), "{d} {l:?}");
        assert!((l.iter().sum::<f64>() - d).abs() < 1e-12);
    }

    #[test]
    fn host_layers_match_a_naive_transcription() {
        // conv1 of a small random net at one output pixel, by hand.
        let net = LpipsWeights::random(5);
        let a = frame(67, 67, |x, y, c| ((x * 7 + y * 3 + c * 11) % 256) as u8);
        let x = input(&a);
        let feats = features_host(&net, &a);
        let (f1, c1, h1, w1) = &feats[0];
        assert_eq!((*c1, *h1, *w1), (64, 16, 16));
        let conv = &net.convs[0];
        let (o, oy, ox) = (9usize, 3usize, 5usize);
        let mut acc = f64::from(conv.b[o]);
        for ci in 0..3 {
            for ky in 0..11 {
                for kx in 0..11 {
                    let iy = (oy * 4 + ky) as isize - 2;
                    let ix = (ox * 4 + kx) as isize - 2;
                    if iy >= 0 && ix >= 0 && (iy as usize) < 67 && (ix as usize) < 67 {
                        acc += f64::from(conv.w[((o * 3 + ci) * 11 + ky) * 11 + kx])
                            * f64::from(x[ci * 67 * 67 + iy as usize * 67 + ix as usize]);
                    }
                }
            }
        }
        let got = f64::from(f1[o * 256 + oy * 16 + ox]);
        assert!((got - acc.max(0.0)).abs() < 1e-4, "{got} vs {acc}");
        // Shapes through the pools: 16 -> pool 7 -> conv2 7 -> pool 3.
        assert_eq!((feats[1].2, feats[2].2, feats[4].2), (7, 3, 3));
    }

    #[test]
    fn embedded_fixtures_match_the_files() {
        for (name, _) in FIXTURES {
            assert_eq!(
                fixture(None, name).unwrap(),
                fixture(Some(&fixtures_dir()), name).unwrap()
            );
        }
    }

    #[test]
    fn frame_selection_follows_collect_run() {
        // Fewer pairs than the budget: all, in order, then nothing new.
        assert_eq!(select_pairs(&[1.0, 3.0, 2.0], 32, 16, 48), vec![0, 1, 2]);
        // 121 frames: 32 stratified + the 16 worst not already chosen.
        let mae: Vec<f64> = (0..121)
            .map(|i| if i % 10 == 7 { 100.0 + i as f64 } else { 1.0 })
            .collect();
        let s = select_pairs(&mae, 32, 16, 48);
        assert!(s.len() <= 48);
        assert_eq!(&s[..3], &[0, 4, 8]);
        assert_eq!(s[31], 120);
        // round(i * 120 / 31): i=4 → 15.48 → 15, i=31 → 120.
        assert_eq!(s[4], 15);
        // Every large-diff frame is in; ties at the tail follow frame order.
        assert!((7..121).step_by(10).all(|i| s.contains(&i)), "{s:?}");
        assert_eq!(s[32], 117, "largest diff first");
        assert_eq!(round_half_even(2.5), 2);
        assert_eq!(round_half_even(3.5), 4);
    }

    #[test]
    fn legacy_and_zip_pth_parse() {
        // A zip-format pth built by hand: {"w": tensor [2, 2]}.
        fn pickle_zip() -> Vec<u8> {
            let mut p: Vec<u8> = vec![0x80, 2];
            p.extend(b"ccollections\nOrderedDict\nq\x00)Rq\x01(X\x01\x00\x00\x00wq\x02");
            p.extend(b"ctorch._utils\n_rebuild_tensor_v2\nq\x03((X\x07\x00\x00\x00storageq\x04ctorch\nFloatStorage\nq\x05X\x01\x00\x00\x000q\x06X\x03\x00\x00\x00cpuq\x07K\x05tq\x08QK\x01K\x02K\x02\x86q\tK\x02K\x01\x86q\n\x89ccollections\nOrderedDict\n)Rtq\x0bRq\x0cu.");
            p
        }
        fn zip(files: &[(&str, Vec<u8>)]) -> Vec<u8> {
            let mut out = Vec::new();
            let mut cd = Vec::new();
            for (name, data) in files {
                let off = out.len() as u32;
                let mut lh = vec![
                    0x50, 0x4b, 0x03, 0x04, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ];
                lh.extend((data.len() as u32).to_le_bytes());
                lh.extend((data.len() as u32).to_le_bytes());
                lh.extend((name.len() as u16).to_le_bytes());
                lh.extend(0u16.to_le_bytes());
                out.extend(&lh);
                out.extend(name.as_bytes());
                out.extend(data);
                let mut c = vec![
                    0x50, 0x4b, 0x01, 0x02, 20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ];
                c.extend((data.len() as u32).to_le_bytes());
                c.extend((data.len() as u32).to_le_bytes());
                c.extend((name.len() as u16).to_le_bytes());
                c.extend([0u8; 12]);
                c.extend(off.to_le_bytes());
                c.extend(name.as_bytes());
                cd.extend(c);
            }
            let cd_off = out.len() as u32;
            out.extend(&cd);
            let mut e = vec![0x50, 0x4b, 0x05, 0x06, 0, 0, 0, 0];
            e.extend((files.len() as u16).to_le_bytes());
            e.extend((files.len() as u16).to_le_bytes());
            e.extend((cd.len() as u32).to_le_bytes());
            e.extend(cd_off.to_le_bytes());
            e.extend([0u8; 2]);
            out.extend(e);
            out
        }
        let storage: Vec<u8> = [9.0f32, 1.0, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bytes = zip(&[
            ("archive/data.pkl", pickle_zip()),
            ("archive/data/0", storage),
        ]);
        let dir = std::env::temp_dir().join(format!("fv-lpips-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("t.pth");
        std::fs::write(&f, bytes).unwrap();
        let t = read_pth(&f, |_| true).unwrap();
        assert_eq!(t["w"].shape, vec![2, 2]);
        assert_eq!(
            t["w"].data,
            vec![1.0, 2.0, 3.0, 4.0],
            "offset 1 skips the first float"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// With FV_LPIPS_WEIGHTS=<dir of fetch-lpips.sh>: the CPU port against
    /// the pinned official numbers.
    #[test]
    fn pinned_official_numbers_when_weights_present() {
        let Ok(dir) = std::env::var("FV_LPIPS_WEIGHTS") else {
            eprintln!("FV_LPIPS_WEIGHTS unset: skipping the pinned LPIPS check");
            return;
        };
        let s = Lpips::load(Path::new(&dir), false).unwrap();
        assert!(!PINNED.is_empty());
        for &(a, b, want) in PINNED {
            let ia = fixture(None, a).unwrap();
            let ib = fixture(None, b).unwrap();
            let got = s.score(&ia, &ib).unwrap().0;
            assert!(
                (got - want).abs() <= PINNED_ABS_TOL.max(PINNED_REL_TOL * want),
                "{a}--{b}: {got} vs {want}"
            );
        }
    }
}
