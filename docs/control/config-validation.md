# fv-control: typed, validated configuration

Owner requirement (2026-10-07): every configuration the dashboard accepts is
validated and typed; enums are enums, not open text; naming rules are known
before the user types; the user cannot misconfigure the system through the UI.

This page is the audit of every place the UI accepts configuration, the rule
for each field and where the server enforces it.

## How it works

- **One source of truth.** Every configuration is a zod schema in
  `control/src/schemas.ts` (the serverless spec in
  `control/src/serverless/spec.ts`). The closed sets and naming rules they use
  live in `control/src/enums.ts`. `GET /api/schemas` serves all of them as
  JSON Schema with UI hints: `x-rule` (the rule shown before typing),
  `x-unit`, `x-dynamic` (the live list a control offers) and `x-name` (the
  uniqueness check).
- **Bound controls.** `control/ui/fields.ts` makes each control from the
  field's JSON Schema. `controlKind()` picks the control: an enum becomes a
  segmented control (≤ 4 values) or a select, a closed live list a select, an
  array of enums ordered chips, a number an `<input type=number>` with the
  schema's min, max and step plus its unit, a name a text input with its
  pattern, its rule shown and a uniqueness check, and a record the env editor.
  The form model checks each field locally on every edit. It sends the draft
  to the server's validator, debounced, for the cross-field and live rules,
  shows each problem inline at its field and in a summary with a link to the
  field. Forms with no validator of their own use `POST
  /api/schemas/<name>/validate`, which runs the schema's refinements on a
  draft. The form model keeps Save, Launch and Create disabled until the form
  is valid.
  A value the live list no longer offers (a channel that is gone, a GPU type
  Runpod dropped) stays visible and is flagged "not available".
- **The server never trusts the client.** Every write path parses its input
  with the same zod schema (`parseOr400`, `normalizeSpec`,
  `normalizeEndpointSpec`, `saveDoc`) and answers 400 with path-anchored
  `issues`. A taken name answers 409.
- **Live checks** need Runpod or D1, so they run on the server only
  (`liveSpecIssues` in `cluster/editor.ts`, `checkName` in `names.ts`):
  - a `max_gpu_dph` below every GPU type a pool may use, which would get each
    pod deleted right after create;
  - a GPU type Runpod does not offer in Secure Cloud;
  - a name that is already taken.

  The forms show these results inline because they come back from the
  validators.
- **Image preflight before a save or launch.** `POST /api/preflight`, given
  `{spec}`, `{launch}` or `{endpoint}`, resolves each image to a digest and
  runs `checkImages` (`cluster/preflight.ts`). The standalone Launch and the
  serverless Create run it first and refuse when it fails. Start still runs
  it for clusters.
- **Coverage tests.** `test/unit/config-validation.test.ts` checks four things:
  - it walks every node of every schema and fails if an enum would be
    rendered as anything but a select or segmented control, or an array of
    enums as anything but chips;
  - every field in the forms' manifests (`ui/forms/common.ts FORM_FIELDS`)
    exists in its schema;
  - every schema property is offered or listed in `notOffered` with a reason;
  - the cluster page offers every spec field.

  `test/ui/config-validation.mjs` runs `FVEditor.auditForms()` in Chromium on
  each form. It fails on any control without a schema path and on any enum
  rendered as a text input.

## Runpod enums: the staging failures of 2026-10-07

Serverless creates of `scale2`, `scale` and `testing` failed on Runpod. Their
`gpu_types` were `["RTX 6000 PRO","H100"]` or `["H100","RTX PRO 6000"]`,
because the form took free text and the spec accepted any 3–80 characters.

Runpod REST v1 refused the create with
`At /endpoints/properties/gpuTypeIds/items/enum: value must be one of …`.
Only `…"problems":["At /endp` was visible, because fv-control cut errors at
300 and 500 characters.

The fix:

- **GPU types are an enum of Runpod's GPU type ids** in the serverless spec,
  cluster pools and standalone pods (`RUNPOD_GPU_TYPES`, REST v1 openapi).
  - The forms offer them as chips from the live catalog, with price and
    stock, and flag a type that is missing from the catalog.
  - A wrong value is refused with close matches, e.g.
    `"RTX 6000 PRO" is not a Runpod GPU type id; did you mean "NVIDIA RTX PRO 6000 Blackwell Server Edition" …?`
    (`enums.ts closeMatches`).
- **The other enums Runpod validates are enums here too, with the same
  hint:**
  - `dataCenterIds` (`data_centers`, the build-pod `volumes` keys);
  - `cpuFlavorIds`;
  - `allowedCudaVersions`;
  - `scalerType`.
- **GPU type against data centre.** `placementIssues` in
  `serverless/spec.ts` checks the GPU types against the data centres the
  workers may use: the volume's data centre (EUR-IS-1 for the EU volume) or
  `data_centers`.
  - It uses Runpod's stock for each type and data centre.
  - It refuses the create and flags the field when none of the types has
    stock there, for example H100 only on the EU volume.
  - It warns for each type that has no stock.
  - Clusters and standalone pods get the same warning per pool
    (`liveSpecIssues`).
- **Readable Runpod errors.** `runpoderr.ts` turns each entry of `problems`
  into this order: the field, the values fv-control sent, did-you-mean
  matches, then the full allowed list. The REST callers use it
  (`runpod.ts`, `serverless/runpod-sls.ts`), and `last_error` keeps 4000
  characters instead of 500.
- **Tests.** `test/unit/config-validation.test.ts` reproduces the three
  staging specs, the Runpod error text and the placement check.

## Naming rules

| Name | Rule | Unique among | Reserved | Sources |
|---|---|---|---|---|
| Cluster, standalone pod | `^[a-z](?:[a-z0-9-]{0,29}[a-z0-9])?$`: 1–31 characters, a DNS label (RFC 1123) that starts with a letter and does not end with `-` | Clusters and standalone pods together (D1 `clusters.name UNIQUE`, `GET /api/names/cluster`) | `new`, `validate` (`#/clusters/new`, `/api/clusters/new/price`, `POST /api/clusters/validate`) | `enums.ts NAME_RE`. The name goes into Runpod pod names `fv-ctl-<cluster>-<pool>-<MMDDHHMMSS>` (at most 81 characters; Runpod REST v1 allows 191), Durable Object names and URLs. |
| Pool id | Same as cluster | Within the cluster (schema `superRefine`) | `gateway` (the retired role; `pool:gateway` in restart) | as above |
| Serverless endpoint | Same as cluster | Live endpoints (D1 `serverless_endpoints_live_name`, `GET /api/names/endpoint`) | `defaults`, `policy`, `validate`, `tick` (routes under `/api/serverless/`) | The Runpod endpoint is `fvc-<name>` and the template `fvc-<name>-<stamp>`, both within 191 characters. |
| API token | `^[A-Za-z0-9._-]{1,40}$` | Active tokens (not revoked, not expired); new server rule, 409 | — | `TokenCreateZ` |
| User API key (cluster) | `^[A-Za-z0-9 ._-]{1,60}$` | — | — | `MintKeyZ` (fv-serve admin keys) |
| Env key | `^[A-Za-z_][A-Za-z0-9_]{0,127}$` | One per level (object keys) | `RESERVED_KEYS` (the controller's own keys, `enums.ts`); for serverless also `SLS_RESERVED` (`FV_SERVE_MODE`, `PORT`, the Runpod secret references, …) | `envvars.ts validateVar`, `EnvSetZ`, `StandaloneLaunchZ.env`, `EndpointSpecZ.env` |
| Release channel | `^[a-z][a-z0-9-]{0,30}$` and one of the known channels (`stable`, `latest`, the fv-jobs heads) | — | — | `ReleaseDispatchZ` plus the route check |

## Field audit

Columns:

- **Type/control:** what the UI renders now.
- **Allowed:** the enum, pattern or range.
- **Deps:** cross-field rules.
- **Server:** where the rule is enforced.
- **Before:** what the UI had before this change. "free" means free text.

### Cluster configuration (`#/clusters/new`, `#/cluster/<id>/config`, JSON tab, raw document editor) — schema `cluster-spec`

| Field | Type/control | Allowed | Deps | Server | Before |
|---|---|---|---|---|---|
| name | text + rule + uniqueness | name rule; not `new`/`validate` | fixed after define | `ClusterSpecZ`, `checkSpec`, POST 409 | text (no reserved or trailing `-` check) |
| control_plane | segmented | edge, direct | `pools[].family` only on edge | zod enum | segmented |
| auth | segmented | keys, none | direct only | zod enum | segmented |
| image source | segmented | channel / sha / ref, exactly one | — | zod refine | segmented |
| image.channel | select (live channels, stale flagged) | channel pattern | — | zod | **free** (datalist) |
| image.sha | text + suggestions (commits with images: fv-jobs history + GHCR `*-sha-*` tags) | 7–40 hex | preflight | zod | free (datalist) |
| image.ref | text, pattern | OCI ref | preflight | zod (was normalizeSpec only) | free |
| regions | chips (live, available only) | `eu` (US volume deleted) | unique | zod enum + `regionProblem` | chips |
| cap_s | number (minutes, ×60), presets | 300 s–7 d, integer | warning > 6 h | zod | number |
| max_gpu_dph | number $/hr | 0.1–50 | ≥ cheapest GPU of each pool (live) | zod + `liveSpecIssues` | number |
| min_start | number $ | 8–10000 | ≥ balance_floor | zod | number |
| balance_floor | number $ | 8–10000 | — | zod | number |
| min_balance | number $ | 8–10000 | warning > floor + 50 | zod | number |
| auto_stop_idle_min | toggle (policy) + number | 5–1440 or null | — | zod | same |
| log_shipping | toggle | bool | — | zod | toggle |
| log_level | select (from the schema) | trace…error | — | zod enum | select |
| pools[].id | text, pattern | name rule; not `gateway` | unique | zod | text |
| pools[].variant | select (live variants) | `VARIANTS` | cpu ⇔ compute CPU (unless image override) | zod enum (was regex) | select |
| pools[].count | stepper / number | 0–8 | warning at 0 | zod | number |
| pools[].compute | segmented | GPU, CPU | GPU: no cpu_flavors/vcpu; CPU: no gpu_types | zod refine | segmented |
| pools[].gpu_types | chips (live: $/hr, stock) | Runpod GPU type ids (REST openapi enum), unique, 1–12 | Secure-Cloud offered (live) | zod enum + live | chips (items free strings server-side) |
| pools[].regions | chips | available regions | unique | zod | chips |
| pools[].cpu_flavors | chips | cpu3c…cpu5m | CPU only | zod | chips |
| pools[].vcpu | select | 2, 4, 8, 16, 32 (Runpod CPU sizes) | CPU only | zod literal | **number 1–32** |
| pools[].container_disk_gb | number GB | 5–500 | — | zod | number |
| pools[].volume | select default/on/off | bool | — | zod | select |
| pools[].image | text, pattern | OCI ref | — | zod pattern (new) | free |
| pools[].family | select | h3, ltx, wan, sfwan, fake | edge only | zod enum (was regex) | **free** |
| pools[].config / config_toml | segmented file/inline; path text (suggestions) / textarea | `/…​.toml` / ≤ 32 KiB | exactly one (was "at least one") | zod | free path |
| pools[].models[].id | select (catalog) — fills family and recipe | catalog model ids | — | zod enum | **free** (datalist) |
| pools[].models[].family | select | h3, ltx2, wan | = the model's family | zod refine | **free** |
| pools[].models[].recipe | select (servable recipes of that family) | servable recipes | recipe's family = family | zod enum + refine | **free** |
| pools[].fake_models | chips | `FAKE_MODELS` | the cpu variant: fake only | zod enum | **free** (comma list) |
| pools[].max_queued | number | 0–10000 | — | zod | number |
| pools[].job_timeout_s / stale_after_s | number s | 10–86400 | stale ≤ job timeout | zod refine | number |
| pools[].provider | select (live: off providers say why) | runpod, gmi, brev | gmi/brev: no gpu_types, cpu_flavors, vcpu, volume, regions; compute GPU | zod enum + `providerIssues` (configured, PUBLIC_URL) | new (docs/serve/deploy-gmi-brev.md) |
| pools[].provider_gpu | select (GMI_PRODUCTS / BREV_INSTANCE_TYPES) | `[A-Za-z0-9][A-Za-z0-9._:-]{0,127}` | gmi/brev: required; the owner's allow-list; price ≤ max_gpu_dph (live) | zod + live | new |
| pools[].provider_region | text | GMI IDC id | gmi only | zod | new |
| pools[].weights_source | select | volume (Runpod), hub, none | gmi/brev: hub or none; models need hub | zod refine | new |
| pools[].hub_download_approved | toggle | bool | hub: required, and the Worker's FV_HUB_DOWNLOADS_APPROVED=1 | `providerIssues` | new |
| JSON tab | CodeMirror, schema lint + server issues at their paths | — | Save off while the JSON does not parse or has issues | — | Save stayed on while the JSON did not parse |

### Standalone pod launch (`#/standalone`) — schema `standalone-launch`, `POST /api/standalone/validate`

| Field | Type/control | Allowed | Deps | Server | Before |
|---|---|---|---|---|---|
| name | text + rule + uniqueness | name rule | unique with clusters | zod + 409 | text (pattern attr only) |
| preset | select (live presets) or "custom" | `POOL_PRESET_IDS` | excludes variant/config/models | zod enum | select |
| variant | select | `VARIANTS` | custom only | zod enum | **free** (datalist) |
| config | text, suggestions | `/…​.toml` | custom: required | zod | free |
| models / fake_models | chips (catalog / fake models) | catalog / `FAKE_MODELS` | cpu variant: fake | zod | **free** (comma list); GPU custom could not set models |
| channel / sha / image | segmented source + select / suggestions / text | patterns | at most one | zod refine | **free** |
| compute | segmented | GPU, CPU | cpu variant → CPU | zod | select |
| gpu_types | chips (live) | GPU enum | GPU only | zod | **free** (datalist) |
| cpu_flavors, vcpu | chips, select | flavors; 2–32 sizes | CPU only | zod | absent |
| region | select | eu | — | zod enum (EU only message) | select |
| volume | toggle | bool | CPU off by default | zod | checkbox |
| container_disk_gb | number GB | 5–500 | — | zod | absent |
| deadline_min | number min | 5–10080 | — | zod (was manual) | number (no inline error) |
| idle_stop_min | number or policy | 5–1440 / null | — | zod | number |
| max_gpu_dph | number $/hr | 0.1–50 | ≥ the cheapest candidate GPU (live) | zod + live | absent |
| auth, log_level | segmented, select | enums | — | zod | absent |
| env | env editor | key rule, not reserved, typed known keys, secrets masked | — | zod + `validateVar` | **textarea `KEY=value`** |
| provider | select (live; an off provider shows its reason) | runpod, gmi, brev | gmi/brev hide the Runpod fields (compute, region, gpu_types, volume, disk) | zod + `providerIssues` | new |
| provider_gpu, provider_region | select (the owner's allow-list), text (gmi) | product / IDC patterns | gmi/brev only; brev: no region | zod + live | new |
| weights_source, weights_download_approved | select, toggle | hub, none; bool | approval only with hub, plus FV_HUB_DOWNLOADS_APPROVED=1 | zod + `providerIssues` | new |
| (Launch) | disabled until valid; image preflight first | — | — | `up` also refuses | always enabled |

### Serverless (`#/serverless`) — schema `serverless-endpoint`, `POST /api/serverless/validate`

| Field | Type/control | Allowed | Deps | Server | Before |
|---|---|---|---|---|---|
| name | text + rule + uniqueness, shows `fvc-<name>` | name rule; not defaults/policy/validate/tick | unique live | zod + 409 | text |
| variant | select (live) — loads the variant's defaults | `VARIANTS` | cpu ⇔ CPU | zod enum (was regex) | select |
| compute | segmented | GPU, CPU | — | zod | absent (implied) |
| mode | segmented | queue, lb | lb: GPU only, REQUEST_COUNT, no config_toml | zod refine | select |
| image.* | segmented source + select / suggestions / text | patterns | exactly one | zod | free |
| config | text, suggestions | `/…​.toml` | not with config_toml | zod | free |
| gpu_types | chips (live) | GPU enum, unique, 1–12 | GPU only | zod enum | **textarea** |
| gpu_count | number | 1–8 | GPU only | zod | absent |
| allowed_cuda | chips | CUDA enum | GPU only | zod | absent |
| cpu_flavors / vcpu | chips / select | serverless flavors / 2–32 sizes | CPU only | zod | absent |
| network_volume | select (known volumes, none) | `jg48s6o1w0` or null | DCs = volume's | zod refine | select |
| data_centers | chips (Runpod DCs) | Runpod DC enum, unique | with a volume: its DC | zod enum (was regex) | **free** (comma list) |
| workers_min / workers_max | number | 0–4 / 0–8 | min ≤ max; policy `max_workers` | zod + `checkLimits` | number |
| idle_timeout_s | number s | 5–3600 (Runpod REST: 5–3600) | — | zod (min was 1) | number |
| execution_timeout_s | number s | 10–86400 | — | zod | number |
| scaler_type / scaler_value | segmented / number (unit by type) | QUEUE_DELAY, REQUEST_COUNT / integer 1–500 (Runpod: integer) | lb → REQUEST_COUNT | zod (was 0.5–3600 float) | select / number step 0.5 |
| flashboot | toggle | bool | — | zod | checkbox |
| container_disk_gb | number GB | 5–200 | — | zod | absent |
| deadline_min / deadline_action | number or none / segmented | 5–10080 / scale0, delete | — | zod | number / select |
| env | env editor (plain values) | key rule; not `SLS_RESERVED`; typed known keys | — | zod (reserved message, typed values new) | absent |
| JSON tab | schema lint + server issues; Create off while invalid | — | — | — | textarea, no validation |
| Endpoint spec editor | CodeMirror with schema + server validation; Save off while invalid or unchanged | — | — | PUT validates | **textarea** |
| Scale card | bound numbers (serverless-scale) | 0–4 / 0–8, min ≤ max | — | `SlsScaleZ` (new) | numbers |
| Policy (new UI) | doc panel (serverless-policy) | margin 0–1000, endpoints 0–50, workers 0–200 | — | `SlsPolicyZ` (PUT used to clamp silently) | API only |

### Build pod policy (dashboard card, Settings) — schema `build-pods-policy`, document `build-pods/default`

| Field | Type/control | Allowed | Server | Before |
|---|---|---|---|---|
| enabled, regions_only, runner, wake_on_queue | toggles | bool | zod | **one JSON textarea for the whole policy** |
| max_pods | number | 0–8 integer | zod | textarea |
| max_dph_per_pod, daily_usd_max, balance_margin | number | 0.05–5, 0–500, 0–1000 | zod | textarea |
| flavors | chips | CPU flavor enum | zod | textarea |
| vcpus | chips | 2, 4, 8, 16, 32, 64 | zod literal | textarea |
| disk_gb / disk_gb_fallback | number GB | 20–500; fallback ≤ disk_gb | zod refine | textarea |
| regions | list (pattern) | DC or prefix | zod | textarea |
| volumes | record DC → volume id | Runpod DC enum → `[a-z0-9]{6,40}` | zod | textarea |
| idle_min, max_h, max_grace_min, evict_hours, backstop_margin_min | number | ranges of `normalizePolicy` | zod | textarea |
| labels, wake_workflows | list (pattern) | label / workflow patterns | zod | textarea |
| server_ref, image, cache.* | text (patterns) | git ref, image with tag/digest or "", bucket, https URL | zod | textarea |

`PUT /api/build-pods/policy` used to drop or clamp bad values silently. It now
answers 400 with the issues.

### Env editors (account, cluster, pool, pod levels; doc kind `env`)

| Part | Rule | Server | Before |
|---|---|---|---|
| key | rule shown; pattern, reserved controller keys and duplicates refused as you type | `EnvSetZ` key (message at the key), `validateDoc` | text, no inline check |
| value of a known engine key | select or 0/1 (`ENV_VALUE_TYPES`: `FASTVIDEO_ATTN_SAGE`, `_FLASH_KERNEL`, `_NVFP4`, `_FP8`, `_H3_QUANT`, `_LTX2_TEXT_FP8`, `_TAE_DIR`, `_LTX_OFFLOAD`, `_DIT_OFFLOAD`, `_WAN_AUDIO`, `FV_LONGLIVE_*`; values from the parsers in `crates/`) | `envValueProblem` in `validateVar`, `EnvSetZ`, the launch and serverless env | free text |
| secret | write-only password field, masked in the JSON tab | unchanged | same |
| Review & save | off while invalid; each issue also at its field | `saveDoc` | always on (validated on click) |

### Settings, releases, small dialogs

| Form | Field | Control | Allowed | Server | Before |
|---|---|---|---|---|---|
| API tokens | name | text + rule + uniqueness | token rule, unique active | zod + 409 (new) | free |
| | scope | segmented | read, admin, ci | zod enum | select **without ci** |
| | ttl_days | number days | 1–365 | zod | number |
| Policies / attribution | every field | doc panel (generated from the schema: units, rules, inline errors) | ranges; auto_stop_idle_min ≥ idle_min; unique prefixes; no spaces | `PoliciesZ`, `AttributionZ` | doc panel without inline errors |
| Releases | action | segmented | promote, rollback | zod | two buttons |
| | channel | select (live channels) | known channels | zod + route | **free** |
| | target | text + suggestions | sha / digest / tag; required for promote | zod refine (was validateDispatch only) | free |
| | to | select (release history) | release id; rollback only | zod | absent |
| | notes, dry_run, templates, allow_partial | text ≤ 200, toggles | — | zod | dry run only |
| Extend (cluster, standalone; serverless +30 is fixed) | minutes | dialog, number min | 1–10080 integer | `ExtendZ` (new; cluster extend took any number) | **prompt()** |
| Scale (cluster page) | pool, count | dialog: select, number | pool of the spec; 0–8 | `ScaleZ` + pool existence (new) | **prompt()** |
| Roll | target, pools | dialog: text + suggestions (channels, commits), chips | channel / sha / image ref | `RollZ` (new) | free text |
| Mint user key | name | dialog, rule | key rule | `MintKeyZ` | **prompt()** |

### Log explorer filters (`#/logs`) — schema `log-query`

| Field | Control | Allowed | Server | Before |
|---|---|---|---|---|
| sources, levels | chips | `LOG_SOURCES`, `LOG_LEVELS` | `LogQueryZ` in `parseLogQuery` | chips |
| cluster, pool, pod | selects (live) | ids | `LogQueryZ` | selects |
| from / to / jump | text, checked as you type (`aria-invalid` and a message) | UTC date-time, HH:MM, unix ms, relative 15m/2h/7d | `TIME_RE` + `parseTime` | checked on Enter only (toast) |
| q (regex mode) | compiled as you type, message inline, not sent while invalid; ≤ 300 characters | — | `compileRegex` | red text only |
| order | select | asc, desc | zod | select |
| context | number | 0–200 | view only | number |

## Counts

The tables above list 112 rows. A row is one field, or one group of fields
with the same rule.

| | Before | Now |
|---|---|---|
| Rows that were free text, an unvalidated textarea or `prompt()` | 41, including the build-pod policy's JSON textarea, which counted as 13 rows; 18 of them had a closed set (an enum or a live list) | 0 enums or closed live lists rendered as free text; the coverage tests enforce this |
| Server-side checks | Mixed: the build-pod and serverless policies were silently clamped, the extend, scale and roll parameters were unchecked, the standalone launch had manual checks only | Every row is parsed by a zod schema on its write path |

Free text remains only for values with a pattern and an inline check:

- names;
- config paths, with suggestions;
- image refs;
- commit shas, with suggestions;
- the roll target;
- release notes;
- build-pod refs, labels and region prefixes;
- the values of env keys that fv-serve does not know.

## Not done / known limits

- **Stock does not block a save.** GPU stock per data centre is a warning
  (the price and stock card), because stock moves by the minute.
- **No server-side image preflight on define or save.** Start, Launch and
  Create run it.
- **The GPU type and data centre enums are a copy of Runpod's REST v1
  openapi (2026-10).** A new Runpod GPU needs a line in `enums.ts`. The live
  list flags types that are in the enum but missing from Runpod's catalog.
