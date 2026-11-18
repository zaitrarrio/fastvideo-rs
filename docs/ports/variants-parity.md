# Variants parity with fastwan-rs and h3-rs (2026-11-18)

What the two consumer-GPU ports (`zaitrarrio/fastwan-rs`, the FastWan /
Self-Forcing Wan 2.1 1.3B path; `zaitrarrio/h3-rs`, the MiniMax H3 cookbook)
run that this repo did not, and what this port brings over. The reference
facts are quoted from those repositories' `CLAUDE.md` and `docs/runs/`
(their paths are given as `fastwan-rs:` / `h3-rs:`); nothing here was
measured in this repo unless a `docs/` path of this repo is cited.

## Ported in this change

First GPU run (2026-11-18, one RTX PRO 6000 Server Edition): every arm of the
`variants` tier passed; the numbers are in docs/perf/variants-2026-11-18.md.

| Variant / feature | Reference | Here | Status |
|---|---|---|---|
| **History Forcing** (ziplab, arXiv 2611.02782; Wan2.1-T2V-1.3B DMD generator, same recipe as Self-Forcing, trained window 21 / sink 3; weights carry **no stated licence**, evaluation only) | fastwan-rs: `generate --causal --cache-frames 21 --sink-frames 3`, `scripts/convert_self_forcing.py`; 7.9 fps on an RTX 4090 at 832×480, more prompt-faithful and dynamic than Self-Forcing at 21/3, over-saturates and shows light streaks (`fastwan-rs:docs/runs/2026-11-17-runpod-4090-history-forcing`) | the SF-Wan causal engine with a `LongLiveConfig::history_forcing()` preset (window 21, sink 3, no LoRA, no re-cache), the checkpoint read from `dmd/model.pt` directly (`fastvideo_loader::pth`) or from the torch-free converter's safetensors; family variant `history-forcing-1.3b`; `fv-gpucheck wan stream ... preset=history-forcing`; docs/ports/wan.md "History Forcing" | GPU run: the `variants` tier (`lab/cells/history-forcing.toml`). Tree not on R2 yet (owner publishes; the cell stages it from the Hub with `fv-gpucheck fetch`) |
| **Real-ESRGAN `realesr-animevideov3` ×4 upscaler** (SRVGGNetCompact, BSD-3, 2.5 MB) with ×2 / ×3 | fastwan-rs: `upscale.rs` (4090 CUDA: 81 frames 480×272 → 1920×1088 in 0.87 s), `docs/runs/2026-11-12-upscalers`; h3-rs: `esrgan.rs`, `finish.rs` (Server Edition: 272p ×4 1.7 s, 480p ×4 4.4 s, 768p ×2 18.5 s mostly host Lanczos) | `rt::esrgan` on cuDNN conv2d + fused bias/PReLU + pack kernels, `fv-cli generate --upscale 2|3|4`, `fv-gpucheck wan gen / h3 gen --upscale N`; the `.pth` read directly; fetched by `fetch-tae.sh`; docs/ports/upscaler.md | GPU run: the `variants` tier (`lab/cells/upscale.toml`). The 2026-09 decision not to port a diffusion VSR (docs/serve/h3-1080p-and-upscaler.md) stands: this is the cheap pixel upscaler |
| **LongLive-Plug on H3 as an unmerged LoRA branch** (`h3-plug4` = h3-rs's draft tier; `h3-cfg`) | h3-rs: `linear::LowRank`, `Backend::linear_add`; the merge keeps a third of the delta in bf16 and a tenth in MXFP8 (|ΔW|/|W| ≈ 2·10⁻⁴) and the MXFP8 clips turned stylised (`h3-rs:docs/runs/2026-11-15-runpod-6000s-longlive-plug`); plug 4-step 34.7 s after load on the Server Edition with MXFP8 + MXFP8 decoder + pv8 | `Linear` branch `y = Wx + B(Ax)` on every forward path, `H3LoraFuse` mode Branch (default for the Plug recipes, `FASTVIDEO_H3_LORA_BRANCH`), served variants `h3-plug-4step` and `h3-plug-cfg` in `families/h3.toml`; docs/techniques.md "LongLive-Plug recipes" | GPU run: the `variants` tier (`lab/cells/h3-plug.toml`: branch vs merged vs bf16 branch) |

## Already here before this change

| Variant | Reference | Here |
|---|---|---|
| Self-Forcing Wan 1.3B (causal DMD, 4 steps) | fastwan-rs `generate --causal` (12.9 fps on a 4090) | `sfwan21-1.3b` (window 21, sink 15, TAEHV per block; docs/ports/wan.md) |
| LongLive 1.0 (1.3B, window 12 / sink 3, KV re-cache) | not in fastwan-rs | `longlive-1.3b` (non-commercial weights; docs/serve/research-longlive.md) |
| LongLive-Plug Wan 14B / 5B 4-step | h3-rs: none | `wan14b-plug-4step` (served as `wan14b-turbo`), `wan5b-plug-4step` (opt-in) |
| LightX2V MiniMax-H3 Turbo 4 / 8-step (T2VA, FL2VA) | h3-rs `h3-turbo4`, `h3-turbo8` | `h3-lightx2v-turbo`, `h3-lightx2v-turbo-v01`, `h3-larry-turbo` (docs/ports/h3-turbo-loras.md) |
| LightX2V Turbo Ref2V | h3-rs `h3-turbo4-ref`, `h3-turbo8-ref` | `h3-ref2v-turbo` (`sol-h3-ref2va`, docs/ports/h3-ref2v.md) |
| FastH3 V1 / V2 / Trim (bf16, NVFP4), TAEH3 twins, the draft tier | h3-rs `fasth3-v1`, `fasth3-v2`, `fasth3-trim` | `fasth3-4step-vsa*`, `fasth3-8step-dense`, `fasth3-v2-nvfp4*`, `fasth3-trim-*`; draft = `fasth3-4step-vsa-480p-taeh3` |
| MXFP8 / INT8 Q·Kᵀ / FP8 P·V attention | h3-rs `--weights mxfp8`, `--qk8`, `--pv8` | `FASTVIDEO_H3_QUANT=mxfp8`, Sage / FP8 attention routes (docs/techniques.md) |
| int8 / int4 (HQQ) weight-only Wan 1.3B | fastwan-rs `--weights int8 | int4-hqq-g64[-ffn]` | not as such; the NVFP4 W4A4 route (`FASTVIDEO_NVFP4`) and MXFP8 cover the memory case on datacenter cards. Not ported: consumer-card VRAM is not this repo's target (docs/scope.md) |

## Not ported (follow-ups, in the order h3-rs ranks them)

| Variant | Reference | What it would take here |
|---|---|---|
| **Pika-H3-Flex** (`Pika-Labs/Pika-H3-Flex`: rank-128 LoRA per task, renoise knots 1000/750/500/250/100/40, 480p / 768p buckets; FL2VA, Ref2VA) | h3-rs `pika-flex` (`variant.rs`, `schedule::StepRule::Renoise`; Server Edition FL2VA 6 × 12.9 s, Ref2VA 6 × 18.6 s) | a Plug-like recipe (adapter per task, run as a branch as above), the renoise step rule (predict x0, re-noise to the next knot: already the `H3FreshNoise4` sampler's shape with six knots), Pika's bucket rule for the canvas, the two adapter trees in `weights-manifest.tsv` / R2. About the size of the Plug variants' work |
| **Open-H3-Regenerate-2K** (`Efficient-Large-Model/Open-H3-Regenerate-2K`: 768p draft → 2560×1440 through the 3-D latent upscaler, a rank-576 LoRA and "Sharp curve" branches, 3 Euler forwards on explicit sigmas, base-VAE decode) | h3-rs `regen-2k` (`regen.rs`, `upscaler.rs`; 446–450 s on the Server Edition; its `vae_fast/` Sol decoder is not implemented there either) | the 3-D latent upscaler already loads for `sol-h3-spark` (`h3/spark.rs`); missing: the Regen LoRA (rank 576, as a branch), the sharp-curve branch modules, the sigma schedule and the draft-conditioning contract (`regen_canvases`). Larger than Pika; the 2K decode alone is 30 s |
| **FastH3-Omni** (`FastVideo/FastVideo-FastH3-Omni-8-Step-V1`: Ref2VA with Parallel Decoding Distillation, 32 fused heads, 8 forwards, VSA 0.9 tile 128, reference regions at keep rate 0.1) | h3-rs `fasth3-omni` (`Sampler::Pdd`, `vsa::ref_keep_rate`; Omni Ref2VA 8 × 14.1 s) | the H3 pipeline refuses `pdd_steps` transformers today (`h3/pipeline.rs` `transformer_config`); needs the PDD head layout, the PDD plan over the fine grid, and VSA reference regions. The biggest item |
| History Forcing's **MemRoPE** demo cache | not implemented in fastwan-rs either (its vendored code is CC-BY-NC-SA) | out of scope while the licence stands |
| fastwan-rs's browser / WebGPU path, phones, Prompt Studio | fastwan-rs `crates/fastwan-web`, `web/` | out of scope: this repo serves datacenter GPUs (docs/scope.md) |
