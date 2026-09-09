# Decision log

Project code: FVID

### FVID · 2026-09-09 · FVID-2026-09-09-luminal
- Trigger: third Rust ML backend named “luminar”
- Options: Luminal (graph compiler) vs some other crate
- Decision: Luminal (https://github.com/luminal-ai/luminal)
- Reason: user confirmed that spelling
- Reversibility: cheap
- Executed by: Executor
- ADR: ADR-0001
- Verification: checked against source

### FVID · 2026-09-09 · FVID-2026-09-09-wan-inference
- Trigger: FastVideo is a full post-training + inference monorepo
- Options: 1.3B-only inference; all Wan/FastWan inference; inference + training
- Decision: inference-first for every Wan/FastWan family in FastVideo’s registry
- Reason: user chose all Wan families, still no training
- Reversibility: cheap
- Executed by: Executor
- ADR: none
- Verification: checked against FastVideo `fastvideo/models/wan/definition.py`

### FVID · 2026-09-09 · FVID-2026-09-09-tensor-backend
- Trigger: need one model implementation for three runtimes
- Options: write models three times; Burn-only + export; custom TensorBackend trait
- Decision: custom `TensorBackend` trait; do not use deprecated `burn-candle`
- Reason: Burn, Candle, and Luminal are distinct runtimes; Luminal is graph-based
- Reversibility: costly
- Executed by: Executor
- ADR: ADR-0001
- Verification: pending

### FVID · 2026-09-09 · FVID-2026-09-09-vast-gpu
- Trigger: CPU `--tiny` is not a real inference bar; need GPU for Wan 1.3B
- Options: local Metal; wait for Luminal GPU; CUDA on Vast.ai
- Decision: real GPU runs on Vast CUDA hardware (Candle `--features cuda`, BF16)
- Reason: user directed GPU runs onto Vast
- Reversibility: cheap
- Executed by: Executor
- ADR: ADR-0002
- Verification: gates green (tiny CUDA F32 generate on RTX 4090 instance 50416610)

