# Working rules for this repository

## Model weights and network volumes

- New weights go on the EU Runpod network volume `jg48s6o1w0`
  (`fv-weights-h3-ltx-hy`, EUR-IS-1), and never only on a pod's container
  disk. **EU only (owner decision, 2026-10-06):** Runpod deleted the US volume
  `s2k01690bi` (`fv-weights-b200-us`, US-CA-2) on about 2026-10-05 while the
  account balance was negative, and the owner chose not to rebuild it for
  now. The earlier "both volumes" rule (US and EU, eventually in sync) is
  suspended until the owner rebuilds US; do not create, mount or copy to a US
  volume without the owner. The rebuild plan (from EU or the Hub) is in
  `docs/ops/runpod-volumes.md` §5 and `scripts/gpu/rebuild-volume.sh`.
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
  200 GB network volume `fv-build`), not in this container, whose disk is small. Each
  agent gets its own worktree directory and `CARGO_TARGET_DIR` on that volume.
  See `scripts/dev/build-pod.sh` and `docs/dev/build-pod.md`.

## Merging to main

- Sub-agents push their branch and open a PR; they never merge to main.
- The coordinating session reviews each PR and merges it after pulling the
  latest main, merging it into the branch and getting CI green on that head.
  Merges use `--no-ff` (or a fast-forward of an already-merged branch).
