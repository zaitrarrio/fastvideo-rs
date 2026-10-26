# Runpod secrets the GPU pods reference

The lab's pods never carry a secret's value. Their create body names a Runpod
secret of the owner's account as `{{ RUNPOD_SECRET_<name> }}`, and Runpod puts
the value in the pod's environment when the pod starts
(docs.runpod.io, "Manage secrets"). The repository, the pod bodies, the
goldens and the fetched run directories hold only the reference.

| Secret | Pod env | Used by | What it holds |
|---|---|---|---|
| `fv_r2_access_key_id`, `fv_r2_secret_access_key` | `FV_WEIGHTS_R2_ACCESS_KEY_ID`, `FV_WEIGHTS_R2_SECRET_ACCESS_KEY` | `fv-lab pod up` for a `weights = "volume+r2"` tier (docs/dev/fv-lab.md, ADR-0019 §16) | the R2 weights store's read key (docs/ops/r2-weights.md) |
| `fv_hf_token` | `HF_TOKEN` | the upstream reference pods: `fv-lab oracle` for a stack with `hub_token = true` (`lab/upstream.toml`: `sol-ltx25`), and `runpod-http.sh upstream` when `UP_STEPS` has a `weights:ltx25*` step (or `UP_HUB_TOKEN=1`) | a Hugging Face read token of an account that has accepted the Lightricks LTX-2.5 licence |

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
  no token and falls back to `/workspace/hf/token` as before; `box.txt` says
  `hf_token=none (unfilled secret reference)` or `hf_token=none`.
  `weights:ltx25` then fails with 401, the oracle cells write `ORACLE_DONE`
  without a dump, and `fv-lab oracle` reports those targets missing
  (`ORACLE_DONE without oracle-dump.tar; cell exit 2`), releases the upstream
  pod without starting a runtime pod and exits non-zero. The R2 runner treats a
  value holding `{{` the same way (`weights::R2Key::from_env`).

The first run after the secret is created settles which of these Runpod does
for the R2 secrets' case too; record it here.
