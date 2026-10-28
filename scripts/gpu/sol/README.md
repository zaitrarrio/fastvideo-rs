# Sol-engine benchmark inputs (LingBot-Video, Cosmos3)

Copied from NVlabs/Sana branch `sol-engine` (read 2026-10-06) so a GPU pod
needs no checkout of it:

| file | source |
|---|---|
| `lingbot-t2v-val3.json` | `models/lingbot_video/prompts/t2v_val3.json` (the 3-prompt set of `config/lingbot_video/*`; prompts from robbyant/lingbot-video, Apache-2.0) |
| `lingbot-t2v-val3.txt` | the same three captions as the reference runner feeds them: `json.dumps(caption, ensure_ascii=False, separators=(",", ":"))` (`utils.caption_from_sample`), one per line, generated with Python so key order and escaping match |
| `cosmos3-default.txt`, `cosmos3-negative.txt` | `models/cosmos3/prompts/{default,negative}.txt` |

Who reads them now:
- `lingbot-t2v-val3.txt`: the LingBot arms of the `sol-bench-<set>` cells
  (`--prompts {root}/scripts/gpu/sol/lingbot-t2v-val3.txt`, the image's checkout on the
  pod) and `fv-gpucheck sol`'s tests (`sol_stage.rs`).
- `lingbot-t2v-val3.json`: provenance only, the source of the `.txt`.
- `cosmos3-default.txt`, `cosmos3-negative.txt`: provenance only. The Cosmos3
  cells carry both prompts inline, as the deleted `scripts/gpu/sol-bench-pod.sh`
  passed them (`$(cat …)`). `scripts/gpu/tests/fv-lab-experiment-goldens.sh`
  reads them from the history, with the script (docs/adr/0019-fv-lab.md
  §23.5, §23.10).
