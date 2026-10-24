//! Identity of the one chooser with the merge-base choosers it replaced
//! (plan B5, `docs/adr/0015-one-sdpa-entry.md`).
//!
//! [`merge_base`] transcribes, branch for branch, the dispatch of
//! `326c879a` (main before B5): `rt/nn.rs`
//! `scaled_dot_product_attention_masked`'s `run` closure,
//! `sdpa_block_causal`, `sdpa_vdn_window`, `sdpa_kv_window(_with)`;
//! `rt/attn.rs` `device_mma_sdpa`, `flash_kernel_for`, `auto_cudnn_or_v2`;
//! and `h3/transformer.rs` `dense_attention`. Both sides run against the
//! same [`World`]: a fake [`Probe`] over every setting and device fact the
//! choosers read, plus which kernels decline the call (a kernel that cannot
//! run a shape, or whose operands are not on the device, returns `None`).
//! Every kernel attempt is recorded with its arguments, and so is every
//! probe with an effect (loading attn_dc, building a cuDNN plan), so equal
//! traces mean the same kernels launched with the same arguments in the same
//! order.
//!
//! The one intended difference: `FASTVIDEO_SDPA=host|sparse` (unmasked) ran
//! a host-only backend, which refused a GPU run (`host_only_op`) and
//! computed on a CPU run; both are retired, so the call is refused
//! everywhere ([`Step::Retired`]). The tests check that mapping explicitly.

use std::cell::RefCell;

use super::*;

/// What a trace records.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Event {
    /// `attn_dc::dense()` asked (loads the module).
    DcProbe,
    /// `cudnn_sdpa::has_plan(bh, sq, sk, d)` asked (builds the plan).
    PlanProbe(usize, usize, usize, usize),
    /// A kernel tried.
    Try(Kernel),
    /// A retired backend refused the call.
    Refused(Backend),
}

/// A kernel and the arguments that change what it computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Fp8 {
        out_bf16: bool,
    },
    /// The merge base's host backends (`flash_style_sdpa_host`,
    /// `block_sparse_sdpa`).
    Host(Backend),
    FlashF32,
    Sage {
        out_bf16: bool,
    },
    /// `device_mma_sdpa_with(kernel, out_bf16)`.
    Mma {
        kernel: FlashKernel,
        out_bf16: bool,
    },
    /// `timed_cudnn_or_v2`.
    MmaTimed,
    Cublas,
    Composed {
        masked: bool,
    },
    Causal {
        out_bf16: bool,
    },
    Window {
        out_bf16: bool,
    },
    /// The f32 composed reference of the KV window.
    ComposedF32,
}

/// Kernel families that may decline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Family {
    Fp8,
    FlashF32,
    Sage,
    Mma,
    Cublas,
    Causal,
    Window,
}

#[derive(Clone, Debug)]
struct World {
    backend: Backend,
    sp_world: usize,
    mma_default: bool,
    sage: Option<usize>,
    fp8: bool,
    causal_flash: bool,
    choice: &'static str,
    sm_major: Option<i32>,
    sms: usize,
    dc: bool,
    auto: AutoMode,
    has_plan: bool,
    declines: Vec<Family>,
    events: RefCell<Vec<Event>>,
}

impl World {
    fn base() -> Self {
        Self {
            backend: Backend::Dense,
            sp_world: 1,
            mma_default: true,
            sage: None,
            fp8: false,
            causal_flash: true,
            choice: "auto",
            sm_major: Some(12),
            sms: 188,
            dc: false,
            auto: AutoMode::Rule,
            has_plan: true,
            declines: Vec::new(),
            events: RefCell::new(Vec::new()),
        }
    }

    fn take(&self) -> Vec<Event> {
        std::mem::take(&mut *self.events.borrow_mut())
    }

    /// Tries a kernel: records it, and says whether it ran (`Some`).
    fn attempt(&self, k: Kernel) -> Option<Kernel> {
        self.events.borrow_mut().push(Event::Try(k));
        let family = match k {
            Kernel::Fp8 { .. } => Some(Family::Fp8),
            Kernel::FlashF32 => Some(Family::FlashF32),
            Kernel::Sage { .. } => Some(Family::Sage),
            Kernel::Mma { .. } | Kernel::MmaTimed => Some(Family::Mma),
            Kernel::Cublas => Some(Family::Cublas),
            Kernel::Causal { .. } => Some(Family::Causal),
            Kernel::Window { .. } => Some(Family::Window),
            Kernel::Host(_) | Kernel::Composed { .. } | Kernel::ComposedF32 => None,
        };
        match family {
            Some(f) if self.declines.contains(&f) => None,
            _ => Some(k),
        }
    }
}

impl Probe for World {
    fn backend(&self) -> Backend {
        self.backend
    }
    fn sp_world(&self) -> usize {
        self.sp_world
    }
    fn mma_default(&self) -> bool {
        self.mma_default
    }
    fn sage_min_seq(&self) -> Option<usize> {
        self.sage
    }
    fn fp8_dense(&self) -> bool {
        self.fp8
    }
    fn causal_flash(&self) -> bool {
        self.causal_flash
    }
    fn kernel_choice(&self) -> String {
        self.choice.to_string()
    }
    fn sm_major(&self) -> Option<i32> {
        self.sm_major
    }
    fn sms(&self) -> usize {
        self.sms
    }
    fn dc_loaded(&self) -> bool {
        self.events.borrow_mut().push(Event::DcProbe);
        self.dc
    }
    fn auto_mode(&self) -> AutoMode {
        self.auto
    }
    fn cudnn_has_plan(&self, bh: usize, sq: usize, sk: usize, d: usize) -> bool {
        self.events
            .borrow_mut()
            .push(Event::PlanProbe(bh, sq, sk, d));
        self.has_plan
    }
}

/// The merge base's dispatch, transcribed. Settings and device facts are
/// read from the [`World`] at the points the old code read them.
mod merge_base {
    use super::*;

    /// `attn.rs` `flash_kernel_for(sq, bh, sms)`.
    fn flash_kernel_for(w: &World, sq: usize, bh: usize, sms: usize) -> FlashKernel {
        match w.choice {
            "v1" => FlashKernel::V1,
            "v2" => FlashKernel::V2,
            "v3" => FlashKernel::V3,
            "v3s" => FlashKernel::V3s,
            "cudnn" => FlashKernel::Cudnn,
            "dc" => FlashKernel::Dc,
            _ if w.dc_loaded() => FlashKernel::Dc,
            _ if flash_v2_default() && flash_v2_fills(sq, bh, sms) => FlashKernel::V2,
            _ => FlashKernel::V1,
        }
    }

    /// `attn.rs` `auto_cudnn_or_v2` (after its timed branch: one launch).
    fn auto_cudnn_or_v2(w: &World, dims: Option<Dims>) -> Option<Kernel> {
        if w.auto == AutoMode::Timed {
            return w.attempt(Kernel::MmaTimed);
        }
        let Some(Dims { b, h, sq, sk, d }) = dims else {
            return w.attempt(Kernel::Mma {
                kernel: FlashKernel::V2,
                out_bf16: true,
            });
        };
        let sm_major = w.sm_major.unwrap_or(0);
        let rule = pick(sm_major, sq, sk, d);
        let kernel = match rule {
            DensePick::Cudnn if w.cudnn_has_plan(b * h, sq, sk, d) => FlashKernel::Cudnn,
            _ => FlashKernel::V2,
        };
        w.attempt(Kernel::Mma {
            kernel,
            out_bf16: true,
        })
    }

    /// `attn.rs` `device_mma_sdpa`.
    pub(super) fn device_mma_sdpa(w: &World, dims: Option<Dims>, out_bf16: bool) -> Option<Kernel> {
        let kernel = match (dims, w.sm_major) {
            (Some(Dims { b, h, sq, .. }), Some(sm_major)) => {
                let sms = w.sms;
                match flash_kernel_for(w, sq, b * h, sms) {
                    FlashKernel::V2
                        if out_bf16 && cudnn_default(sm_major) && w.choice == "auto" =>
                    {
                        return auto_cudnn_or_v2(w, dims);
                    }
                    k => k,
                }
            }
            _ => flash_kernel_for(w, usize::MAX, 1, 1),
        };
        w.attempt(Kernel::Mma { kernel, out_bf16 })
    }

    /// `nn.rs` `scaled_dot_product_attention_masked`'s `run` closure on one
    /// (shard of the) query; `out_bf16` is `bf16_activations()` there.
    pub(super) fn run(w: &World, shape: &Shape, masked: bool, out_bf16: bool) -> Kernel {
        if !masked {
            if w.backend == Backend::Sparse {
                return w.attempt(Kernel::Host(Backend::Sparse)).expect("host");
            }
            if w.backend == Backend::Host {
                return w.attempt(Kernel::Host(Backend::Host)).expect("host");
            }
            if w.backend == Backend::Flash {
                if let Some(out) = w.attempt(Kernel::FlashF32) {
                    return out;
                }
            }
            if w.backend != Backend::Cublas && w.mma_default {
                let sage = w.sage;
                if sage.is_some()
                    && shape.sq >= sage.expect("sage")
                    && shape.sk >= sage.expect("sage")
                {
                    if let Some(out) = w.attempt(Kernel::Sage { out_bf16 }) {
                        return out;
                    }
                }
                if let Some(out) = device_mma_sdpa(w, shape.dims, out_bf16) {
                    return out;
                }
            }
            if let Some(out) = w.attempt(Kernel::Cublas) {
                return out;
            }
        }
        w.attempt(Kernel::Composed { masked })
            .expect("composed never declines")
    }

    /// `h3/transformer.rs` `dense_attention` (CUDA build), then
    /// `scaled_dot_product_attention(q, k, v, None)`.
    pub(super) fn h3_dense_attention(w: &World, shape: &Shape, out_bf16: bool) -> Kernel {
        if w.fp8 {
            if let Some(out) = w.attempt(Kernel::Fp8 { out_bf16 }) {
                return out;
            }
        }
        run(w, shape, false, out_bf16)
    }

    /// `nn.rs` `sdpa_block_causal`.
    pub(super) fn block_causal(w: &World, shape: &Shape, act16: bool) -> Kernel {
        if w.causal_flash && w.sp_world == 1 && w.mma_default {
            if let Some(out) = w.attempt(Kernel::Causal { out_bf16: act16 }) {
                return out;
            }
        }
        run(w, shape, true, act16)
    }

    /// `nn.rs` `sdpa_vdn_window` (the dense-mask size check aside).
    pub(super) fn vdn_window(w: &World, shape: &Shape, act16: bool) -> Kernel {
        if w.sp_world == 1 && w.mma_default {
            if let Some(out) = w.attempt(Kernel::Window { out_bf16: act16 }) {
                return out;
            }
        }
        run(w, shape, true, act16)
    }

    /// `nn.rs` `sdpa_kv_window_with(flash)`; `None` is `sdpa_kv_window`.
    pub(super) fn kv_window(w: &World, shape: &Shape, flash: Option<bool>, act16: bool) -> Kernel {
        let flash = match flash {
            Some(f) => f,
            None => w.causal_flash,
        };
        if flash {
            return run(w, shape, false, act16);
        }
        w.attempt(Kernel::ComposedF32).expect("composed")
    }
}

/// The new chooser driven the way `rt/sdpa.rs` drives it.
mod now {
    use super::*;

    fn mma(w: &World, dims: Option<Dims>, out_bf16: bool) -> Option<Kernel> {
        match mma_kernel(dims, out_bf16, w) {
            MmaKernel::Fixed(kernel) => w.attempt(Kernel::Mma { kernel, out_bf16 }),
            MmaKernel::Rule { kernel, .. } => w.attempt(Kernel::Mma {
                kernel,
                out_bf16: true,
            }),
            MmaKernel::Timed => w.attempt(Kernel::MmaTimed),
        }
    }

    pub(super) fn run(w: &World, shape: &Shape, masked: bool, out_bf16: bool) -> Kernel {
        let route = dense_route(shape, masked, out_bf16, w);
        run_route(&route, |step| -> Result<Option<Kernel>, ()> {
            Ok(match step {
                Step::Retired(b) => {
                    w.events.borrow_mut().push(Event::Refused(b));
                    Some(Kernel::Host(b))
                }
                Step::FlashF32 => w.attempt(Kernel::FlashF32),
                Step::Sage { out_bf16 } => w.attempt(Kernel::Sage { out_bf16 }),
                Step::Mma { out_bf16 } => mma(w, shape.dims, out_bf16),
                Step::Cublas => w.attempt(Kernel::Cublas),
                Step::Composed => w.attempt(Kernel::Composed { masked }),
            })
        })
        .expect("infallible")
        .expect("a route ends in Composed")
    }

    pub(super) fn h3_dense_attention(w: &World, shape: &Shape, out_bf16: bool) -> Kernel {
        if fp8_first(w) {
            if let Some(out) = w.attempt(Kernel::Fp8 { out_bf16 }) {
                return out;
            }
        }
        run(w, shape, false, out_bf16)
    }

    pub(super) fn block_causal(w: &World, shape: &Shape, act16: bool) -> Kernel {
        if block_causal_first(w) {
            if let Some(out) = w.attempt(Kernel::Causal { out_bf16: act16 }) {
                return out;
            }
        }
        run(w, shape, true, act16)
    }

    pub(super) fn vdn_window(w: &World, shape: &Shape, act16: bool) -> Kernel {
        if vdn_window_first(w) {
            if let Some(out) = w.attempt(Kernel::Window { out_bf16: act16 }) {
                return out;
            }
        }
        run(w, shape, true, act16)
    }

    pub(super) fn kv_window(w: &World, shape: &Shape, flash: Option<bool>, act16: bool) -> Kernel {
        if kv_window_flash(flash, w) {
            return run(w, shape, false, act16);
        }
        w.attempt(Kernel::ComposedF32).expect("composed")
    }
}

/// The old trace with the retired host backends mapped to the refusal that
/// replaces them.
fn retire(old: Vec<Event>) -> Vec<Event> {
    old.into_iter()
        .map(|e| match e {
            Event::Try(Kernel::Host(b)) => Event::Refused(b),
            e => e,
        })
        .collect()
}

fn dims(b: usize, h: usize, sq: usize, sk: usize, d: usize) -> Shape {
    Shape {
        sq,
        sk,
        dims: Some(Dims { b, h, sq, sk, d }),
    }
}

/// The shapes the families reach (and the class edges of the rules).
fn shapes() -> Vec<Shape> {
    vec![
        dims(1, 56, 37_710, 37_710, 128),   // H3 768p self-attention
        dims(1, 32, 124_440, 124_440, 128), // LTX 1080p 20 s
        dims(1, 32, 130_560, 130_560, 128), // LTX 4K 5 s (above the cuDNN row)
        dims(1, 32, 6_144, 6_144, 128),     // LTX 512p (sage edge)
        dims(1, 32, 6_143, 6_143, 128),     // below the sage edge
        dims(1, 40, 24_576, 512, 128),      // text cross-attention
        dims(1, 24, 24_576, 24_576, 128),   // the cuDNN row's lower edge
        dims(2, 16, 32_760, 32_760, 64),    // d = 64 (audio, Wan VAE)
        dims(1, 12, 4_096, 77, 80),         // d = 80: no fused kernel
        Shape {
            sq: 30_000,
            sk: 30_000,
            dims: None,
        }, // K/V disagree with Q
    ]
}

fn declines() -> Vec<Vec<Family>> {
    let all = [
        Family::Fp8,
        Family::FlashF32,
        Family::Sage,
        Family::Mma,
        Family::Cublas,
        Family::Causal,
        Family::Window,
    ];
    (0u32..1 << all.len())
        .map(|m| {
            all.iter()
                .enumerate()
                .filter(|(i, _)| m & (1 << i) != 0)
                .map(|(_, f)| *f)
                .collect()
        })
        .collect()
}

const BACKENDS: [Backend; 5] = [
    Backend::Dense,
    Backend::Cublas,
    Backend::Flash,
    Backend::Host,
    Backend::Sparse,
];
const CHOICES: [&str; 8] = ["auto", "v1", "v2", "v3", "v3s", "cudnn", "dc", "Auto "];
const SMS: [Option<i32>; 5] = [None, Some(8), Some(9), Some(10), Some(12)];

/// The fused-kernel pick over every setting and device fact it reads, every
/// shape and output type: the kernel, its arguments, and the effectful
/// probes (attn_dc load, cuDNN plan build) in order.
#[test]
fn mma_kernel_is_the_merge_base_device_mma_sdpa() {
    let mut cases = 0usize;
    for choice in CHOICES {
        for sm_major in SMS {
            for sms in [188usize, 132, 1] {
                for dc in [false, true] {
                    for auto in [AutoMode::Rule, AutoMode::Timed] {
                        for has_plan in [false, true] {
                            for out_bf16 in [false, true] {
                                for shape in shapes() {
                                    let w = World {
                                        choice,
                                        sm_major,
                                        sms,
                                        dc,
                                        auto,
                                        has_plan,
                                        ..World::base()
                                    };
                                    let old = merge_base::device_mma_sdpa(&w, shape.dims, out_bf16);
                                    let old_trace = w.take();
                                    let new = match mma_kernel(shape.dims, out_bf16, &w) {
                                        MmaKernel::Fixed(kernel) => {
                                            w.attempt(Kernel::Mma { kernel, out_bf16 })
                                        }
                                        MmaKernel::Rule {
                                            kernel,
                                            sm_major: s,
                                        } => {
                                            assert_eq!(Some(s), sm_major);
                                            w.attempt(Kernel::Mma {
                                                kernel,
                                                out_bf16: true,
                                            })
                                        }
                                        MmaKernel::Timed => w.attempt(Kernel::MmaTimed),
                                    };
                                    let new_trace = w.take();
                                    assert_eq!(old, new, "{w:?} {shape:?} out_bf16={out_bf16}");
                                    assert_eq!(old_trace, new_trace, "{w:?} {shape:?}");
                                    cases += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 8 * 5 * 3 * 2 * 2 * 2 * 2 * 10);
}

/// Every dense entry (unmasked and masked dense, the H3 FP8 dense block,
/// block-causal, VDN window, KV window by setting and by caller) over the
/// route settings, a cut of the fused-kernel facts, every shape, output
/// type and set of declining kernels (those the entry can try): identical
/// traces, the retired backends aside.
#[test]
fn every_entry_routes_as_the_merge_base_did() {
    let decline_sets = declines();
    let mut cases = 0usize;
    let mut retired = 0usize;
    for backend in BACKENDS {
        for sp_world in [1usize, 2] {
            for mma_default in [false, true] {
                for sage in [None, Some(6_144usize)] {
                    for fp8 in [false, true] {
                        for causal_flash in [false, true] {
                            for (choice, sm_major, dc, auto) in [
                                ("auto", Some(12), false, AutoMode::Rule),
                                ("auto", Some(12), false, AutoMode::Timed),
                                ("auto", Some(10), true, AutoMode::Rule),
                                ("v1", Some(12), false, AutoMode::Rule),
                                ("auto", None, false, AutoMode::Rule),
                            ] {
                                for declining in &decline_sets {
                                    for out_bf16 in [false, true] {
                                        for shape in shapes() {
                                            let w = World {
                                                backend,
                                                sp_world,
                                                mma_default,
                                                sage,
                                                fp8,
                                                causal_flash,
                                                choice,
                                                sm_major,
                                                dc,
                                                auto,
                                                declines: declining.clone(),
                                                ..World::base()
                                            };
                                            type Entry = fn(&World, &Shape, bool) -> Kernel;
                                            let entries: [(&str, Entry, Entry); 8] = [
                                                (
                                                    "dense",
                                                    |w, s, o| merge_base::run(w, s, false, o),
                                                    |w, s, o| now::run(w, s, false, o),
                                                ),
                                                (
                                                    "masked",
                                                    |w, s, o| merge_base::run(w, s, true, o),
                                                    |w, s, o| now::run(w, s, true, o),
                                                ),
                                                (
                                                    "h3_fp8",
                                                    merge_base::h3_dense_attention,
                                                    now::h3_dense_attention,
                                                ),
                                                (
                                                    "block_causal",
                                                    merge_base::block_causal,
                                                    now::block_causal,
                                                ),
                                                (
                                                    "vdn_window",
                                                    merge_base::vdn_window,
                                                    now::vdn_window,
                                                ),
                                                (
                                                    "kv_window",
                                                    |w, s, o| merge_base::kv_window(w, s, None, o),
                                                    |w, s, o| now::kv_window(w, s, None, o),
                                                ),
                                                (
                                                    "kv_window_flash",
                                                    |w, s, o| {
                                                        merge_base::kv_window(w, s, Some(true), o)
                                                    },
                                                    |w, s, o| now::kv_window(w, s, Some(true), o),
                                                ),
                                                (
                                                    "kv_window_ref",
                                                    |w, s, o| {
                                                        merge_base::kv_window(w, s, Some(false), o)
                                                    },
                                                    |w, s, o| now::kv_window(w, s, Some(false), o),
                                                ),
                                            ];
                                            for (name, old_entry, new_entry) in entries {
                                                // A kernel the entry never tries cannot
                                                // change its trace: those decline sets
                                                // repeat a case already run.
                                                let extra = match name {
                                                    "h3_fp8" => Some(Family::Fp8),
                                                    "block_causal" => Some(Family::Causal),
                                                    "vdn_window" => Some(Family::Window),
                                                    _ => None,
                                                };
                                                if declining.iter().any(|f| {
                                                    matches!(
                                                        f,
                                                        Family::Fp8
                                                            | Family::Causal
                                                            | Family::Window
                                                    ) && Some(*f) != extra
                                                }) {
                                                    continue;
                                                }
                                                let old = old_entry(&w, &shape, out_bf16);
                                                let old_trace = retire(w.take());
                                                let new = new_entry(&w, &shape, out_bf16);
                                                let new_trace = w.take();
                                                assert_eq!(
                                                    old_trace, new_trace,
                                                    "{name}: {w:?} {shape:?} out_bf16={out_bf16}"
                                                );
                                                assert_eq!(old, new, "{name}: {w:?} {shape:?}");
                                                if matches!(old, Kernel::Host(_)) {
                                                    retired += 1;
                                                }
                                                cases += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // 5 entries try the 4 dense-route kernels (16 decline sets), 3 also
    // their own first kernel (32).
    assert_eq!(
        cases,
        5 * 2 * 2 * 2 * 2 * 2 * 5 * 2 * 10 * (5 * 16 + 3 * 32)
    );
    assert!(retired > 0, "the retired backends are in the table");
}

/// The retired backends are refused only where the merge base ran a host
/// backend: unmasked calls under `FASTVIDEO_SDPA=host|sparse`. A masked call
/// under either setting keeps the composed path, as before.
#[test]
fn only_unmasked_host_and_sparse_calls_are_refused() {
    for backend in BACKENDS {
        for masked in [false, true] {
            let w = World {
                backend,
                ..World::base()
            };
            let shape = shapes()[0];
            let route = dense_route(&shape, masked, true, &w);
            let refused = route.iter().any(|s| matches!(s, Step::Retired(_)));
            assert_eq!(
                refused,
                !masked && matches!(backend, Backend::Host | Backend::Sparse),
                "{backend:?} masked={masked}: {route:?}"
            );
            assert!(matches!(
                route.last(),
                Some(Step::Composed | Step::Retired(_))
            ));
        }
    }
}
