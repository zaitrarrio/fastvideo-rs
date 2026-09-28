# Working rules for this repository

## Model weights and network volumes

- New weights always go on **both** Runpod network volumes: US `s2k01690bi`
  (US-CA-2) and EU `jg48s6o1w0` (EUR-IS-1). Nothing may live on only one
  volume or only on a pod's container disk.
- Writes are add-only: new folders, written under a temporary name, verified
  (sha256) and then renamed. Never modify or delete existing volume data, and
  never delete a volume.
- Record every new tree in `scripts/gpu/weights-manifest.tsv` and
  `scripts/gpu/verify-weights.sh`; see `docs/gaps/2026-09-27-volume-sync.md`
  for the copy-and-verify method.
- Large new downloads need the owner's approval.

## GPU pods

- Only touch pods and endpoints you created; give each a wall-clock backstop,
  never leave one idle while building or debugging, and delete it when done.
- Stop before the Runpod balance would drop below $8.

## Builds

- Compile and test on the shared **build pod** (a small Runpod CPU pod with a
  50 GB network volume), not in this container, whose disk is small. Each
  agent gets its own worktree directory and `CARGO_TARGET_DIR` on that volume.
  See `scripts/dev/build-pod.sh` and `docs/dev/build-pod.md`.
