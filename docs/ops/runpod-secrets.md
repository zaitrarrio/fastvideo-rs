# Runpod secrets the GPU pods reference

The lab's pods never carry a secret's value. Their create body names a Runpod
secret of the owner's account as `{{ RUNPOD_SECRET_<name> }}`, and Runpod puts
the value in the pod's environment when the pod starts
(docs.runpod.io, "Manage secrets"). The repository, the pod bodies, the
goldens and the fetched run directories hold only the reference.

| Secret | Pod env | Used by | What it holds |
|---|---|---|---|
| `fv_r2_access_key_id`, `fv_r2_secret_access_key` | `FV_WEIGHTS_R2_ACCESS_KEY_ID`, `FV_WEIGHTS_R2_SECRET_ACCESS_KEY` | `fv-lab pod up` for a `weights = "volume+r2"` tier (docs/dev/fv-lab.md, ADR-0019 §16); the upstream reference pods when a selected target has an `r2` tree (`fv-lab oracle`, `lab/upstream.toml`: `sfwan13`) or `UP_STEPS` has an `r2:<tree>` step (`runpod-http.sh upstream`; `scripts/gpu/upstream/pod.sh` hands the key to the plan step alone, ADR-0019 §21); (`runpod-http.sh run` with `FV_R2_TREES`, the script runner's merged weight root, `scripts/gpu/r2-stage.sh`, ADR-0019 §21.8, went with the script path, §22.20) | the R2 weights store's read key (docs/ops/r2-weights.md) |
| `fv_hf_token` | `HF_TOKEN` | the upstream reference pods: `fv-lab oracle` for a stack with `hub_token = true` (`lab/upstream.toml`: `sol-ltx25`), and `runpod-http.sh upstream` when `UP_STEPS` has a `weights:ltx25*` step (or `UP_HUB_TOKEN=1`) | a Hugging Face read token of an account that has accepted the Lightricks LTX-2.5 licence |

## What a pod does with the value, and what it cannot hide

Runpod starts the container with the value in its environment. The scripts
that take the R2 key (`runpod-http.sh`'s start command for an upstream
`r2:<tree>` step, fv-lab's upstream start command, `upstream/pod.sh`; and
`FV_R2_TREES`'s start command with `r2-stage.sh` until they went, ADR-0019
§22.20) copy it into shell variables and `unset` it
on their first line, and hand it back to the one command that needs it (the
`r2-weights.py plan` step) as an env prefix. What that achieves, exactly:

- the key is in **no child's environment**: no watchdog, file server, step,
  cell or matrix has it, and it is in no argv, log, status file or served run
  directory (`scripts/gpu/tests/runpod-r2.test.sh`, `upstream-r2.test.sh`);
- it is **still on the pod**: the container's initial environment (PID 1's
  `/proc/<pid>/environ`, and the start shell's own initial environ) keeps it,
  readable by root on the pod, so anything running as root there could read it;
- the R2 key is the store's **read-only** key: what it exposes is the
  published weights, not the bucket's writes;
- `HF_TOKEN` (`fv_hf_token`, a Hub read token) has the same exposure, and
  fv-lab's own `volume+r2` runtime pods keep the R2 key in their start
  command's and runner's environment (the runner removes it from every arm's;
  ADR-0019 §16.4).

ADR-0019 §21.2 and §21.8 record this.

The serving templates and endpoints reference their own `fv_*` secrets
(`scripts/serve/runpod-templates.sh`, docs/serve/images.md).

## `fv_hf_token`: the owner's one-time step

`weights:ltx25` and `weights:ltx25-dev` (`scripts/gpu/upstream/pod.sh`) rebuild
the single-file LTX-2.5 packs from the volume's Diffusers tree with range
reads of `huggingface.co/Lightricks/LTX-2.5`, a gated repository. Without a
token every read answers 401, the oracle cells find no transformer and the
LTX references are missing (ADR-0019 §20.7, "First attempt", 2026-10-26). The
EU volume `bz0ofkfx45` has no `hf/token`; a token on a shared volume would be
readable by every pod that mounts it, and the secret keeps it in the account.

1. On huggingface.co, signed in as the account the token will belong to, open
   `Lightricks/LTX-2.5` and accept its licence.
2. Create a **read** access token for that account (Settings, Access Tokens).
3. In the Runpod console, Secrets, create a secret named exactly
   `fv_hf_token` with the token as its value. (Or `POST /secrets` of the
   Runpod REST API; a name already taken answers 409.)

Nothing changes in the repository. The next upstream pod that needs it gets
it. To check a run, read the upstream run's `box.txt`
(`artifacts/runpod/upstream/<tag>/box.txt`): `hf_token=env` means the pod had
the token from its env; its `steps.tsv` then has `weights:ltx25 0`.

## A reference to a secret that does not exist

**UNVERIFIED.** Runpod's documentation ("Manage secrets", "Environment
variables", read 2026-10-26) says the reference is substituted when the pod
starts, and that deleting a secret still in use makes those deployments fail.
It does not say what a reference to a name that was never created does: the
create refused, the variable empty, or the reference left as text. No run here
has tried one. Every outcome is handled:

- **Create refused**: `fv-lab oracle` fails before anything runs (fv-deploy's
  `up` returns the error; the pods already up are released); `runpod-http.sh
  upstream` dies with `pod create failed: …`.
- **Empty value, or the text `{{ RUNPOD_SECRET_fv_hf_token }}` left as it
  is**: `upstream/pod.sh` takes a blank value, or one that still holds `{{`, as
  no token and falls back to `/workspace/hf/token` as before (an empty file is
  no token either); `box.txt` says `hf_token=none (env: unfilled secret
  reference)` or `hf_token=none (env: blank)` (the file's path in place of
  `none` when the file had a token; `scripts/gpu/tests/hub-token.test.sh`).
  `weights:ltx25` then fails with 401, the oracle cells write `ORACLE_DONE`
  without a dump, and `fv-lab oracle` reports those targets missing
  (`ORACLE_DONE without oracle-dump.tar; cell exit 2`), releases the upstream
  pod without starting a runtime pod and exits non-zero. The R2 runner treats a
  value holding `{{` the same way (`weights::R2Key::from_env`).

The first run after the secret is created settles which of these Runpod does
for the R2 secrets' case too; record it here.
