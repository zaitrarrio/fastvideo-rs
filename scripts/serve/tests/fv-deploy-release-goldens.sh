#!/usr/bin/env bash
# The request goldens of fv-deploy's `templates`, `release` and `reconcile`
# (docs/adr/0018-fv-deploy.md §10): what scripts/serve/runpod-templates.sh,
# scripts/serve/release.sh and scripts/serve/lib/registry.sh send today, for
# fixed scenarios against scripts/serve/tests/mock_api.py (D1 on SQLite,
# Runpod REST, a GHCR-like registry, GitHub workflow_dispatch). The Rust code
# must send the same (crates/fastvideo-deploy/tests/release_goldens.rs runs
# the same scenarios against the same mock).
#
#   fv-deploy-release-goldens.sh <out.json>
#
# Called by fv-deploy-goldens.sh (which writes or --checks the fixture
# crates/fastvideo-deploy/fixtures/deploy/release.json). Per scenario: the
# command, whether it succeeded, its stdout, the retags (`docker buildx
# imagetools create`, a stub here), the Runpod writes (method, path, body),
# the D1 writes (INSERT / UPDATE: the SQL with its whitespace collapsed, the
# params) and the workflow dispatches. Normalized so the fixture does not
# change from run to run: image digests become <build.image> (<build.image:12>
# for the 12-character short form), unix-millisecond times <now>, created
# template ids <new template>. Needs bash, curl, jq, python3. No network.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVE="$(cd "$HERE/.." && pwd)"
OUT="${1:?out.json}"

T="$(mktemp -d)"
cleanup() { [[ -n "${MOCK_PID:-}" ]] && kill "$MOCK_PID" 2>/dev/null; wait 2>/dev/null; rm -rf "$T"; }
trap cleanup EXIT
python3 "$HERE/mock_api.py" "$T/port" &
MOCK_PID=$!
for _ in $(seq 100); do [[ -s "$T/port" ]] && break; sleep 0.1; done
[[ -s "$T/port" ]] || { echo "release goldens: the mock did not start" >&2; exit 1; }
M="http://127.0.0.1:$(cat "$T/port")"

# The scripts' environment (as release.test.sh): everything points at the mock.
unset CLOUDFLARE_API_KEY CLOUDFLARE_API_TOKEN FV_CF_ACCOUNT_ID FV_D1_DATABASE_ID GITHUB_TOKEN GITHUB_ACTIONS CLAUDECODE \
  FV_RELEASE_CHANNEL FV_TEMPLATE_CHANNEL RUNPOD_REGISTRY_AUTH_ID FV_SERVE_REPO GITHUB_SERVER_URL GITHUB_RUN_ID GITHUB_STEP_SUMMARY \
  FV_RELEASE_REF FV_GITHUB_REPO FV_REGISTRY
export FV_ENV_FILE=/dev/null
export FV_D1_API_BASE="$M/client/v4" FV_CF_API_TOKEN=test-cf-token
export RUNPOD_API_BASE="$M/v1" RUNPOD_GRAPHQL="$M/graphql" RUNPOD_API_KEY=test-runpod-key
export FV_REGISTRY_API="$M" FV_GITHUB_API="$M" GH_TOKEN=test-gh-token
export FV_DEPLOYED_BY=test
REPO=ghcr.io/zaitrarrio/fastvideo-rs-serve

mkdir -p "$T/bin"
cat >"$T/bin/docker" <<EOF
#!/usr/bin/env bash
tag=""; src=""
while (( \$# )); do case "\$1" in --tag|-t) tag="\$2"; shift ;; *@sha256:*) src="\$1" ;; esac; shift; done
printf '%s\t%s\n' "\$tag" "\$src" >>"$T/docker.log"
curl -sS -X POST -d "{\"tag\":\"\${tag##*:}\",\"digest\":\"\${src##*@}\"}" "$M/__tag" >/dev/null
EOF
chmod +x "$T/bin/docker"
export FV_DOCKER="$T/bin/docker"
: >"$T/docker.log"

hook() { curl -sS -X POST -d "$2" "$M/$1"; }
build() { hook __build "$(jq -nc --arg s "$1" --arg k "$2" '{sha: $s, keys: ($k | split(" "))}')"; }

# The builds (the same as release.test.sh) and the templates on an older image.
A=aaaaaaa111111111111111111111111111111111
B=bbbbbbb222222222222222222222222222222222
C=ccccccc333333333333333333333333333333333
DA="$(build "$A" "debug cuda cpu")"
DB="$(build "$B" "debug cuda cpu")"
DC="$(build "$C" "debug cuda")"
tpls='{}'
for t in cuda-sls cuda-pod cpu-pod; do
  tpls="$(jq -c --arg id "tpl-$t" --arg n "fv-serve-$t" '. + {($id): {id: $id, name: $n, imageName: "ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:0ld", env: {}}}' <<<"$tpls")"
done
hook __seed "{\"templates\": $tpls}" >/dev/null

# digest -> symbol, longest first (the full form before the short one).
SYMS="$(jq -nc --argjson a "$DA" --argjson b "$DB" --argjson c "$DC" '
  [{A: $a}, {B: $b}, {C: $c}] | map(to_entries[0] | .key as $n | .value | to_entries[]
    | {d: (.value | ltrimstr("sha256:")), n: "\($n).\(.key)"})
  | [(.[] | {re: ("sha256:" + .d), to: ("<" + .n + ">")}), (.[] | {re: .d[0:12], to: ("<" + .n + ":12>")})]')"

SCEN='[]'
# scenario <name> <prog: release|templates|registry> <args...>: runs it, records its effects.
scenario() {
  local name="$1" prog="$2" since docker_n out rc=0 reqs
  shift 2
  since="$(curl -sS "$M/__log" | jq -r .next)"
  docker_n="$(wc -l <"$T/docker.log")"
  case "$prog" in
    release) out="$(bash "$SERVE/release.sh" "$@" 2>"$T/$name.err")" || rc=$? ;;
    templates) out="$(env "${ENVS[@]}" bash "$SERVE/runpod-templates.sh" "$@" 2>"$T/$name.err")" || rc=$? ;;
    registry) out="$(registry_calls "$@" 2>"$T/$name.err")" || rc=$? ;;
  esac
  reqs="$(curl -sS "$M/__log?since=$since" | jq -c .requests)"
  SCEN="$(jq -c --arg name "$name" --arg prog "$prog" --argjson args "$(printf '%s\n' "$@" | jq -R . | jq -sc .)" \
    --argjson env "$(printf '%s\n' "${ENVS[@]}" | jq -R 'select(length > 0)' | jq -sc .)" \
    --arg out "$out" --argjson rc "$rc" --argjson reqs "$reqs" \
    --arg docker "$(tail -n +"$((docker_n + 1))" "$T/docker.log")" '. + [{
      name: $name, prog: $prog, args: $args, env: $env, ok: ($rc == 0),
      stdout: ($out | split("\n") | map(select(length > 0))),
      retags: ($docker | split("\n") | map(select(length > 0) | split("\t"))),
      runpod: [$reqs[] | select((.p | startswith("/v1/")) and .m != "GET") | [.m, .p, .body]],
      d1: [$reqs[] | select(.p | endswith("/query")) | .body
           | {sql: (.sql | gsub("\\s+"; " ") | ltrimstr(" ") | rtrimstr(" ")), params: (.params // [])}
           | select(.sql | test("^(INSERT|UPDATE) ")) | [.sql, .params]],
      dispatch: [$reqs[] | select(.p | endswith("/dispatches")) | .body]}]' <<<"$SCEN")"
  ENVS=()
}
ENVS=()

# The deployments writes of runpod-pod.sh and runpod-endpoint.sh, made with the scripts' own registry functions and
# fixed values (the scripts call them so after a create, a ready and a delete).
registry_calls() {
  # shellcheck source=../../gpu/lib.sh
  source "$SERVE/../gpu/lib.sh"
  # shellcheck source=../lib/registry.sh
  source "$SERVE/lib/registry.sh"
  case "$1" in
    pod)
      FV_DEPLOY_SCRIPT=runpod-pod.sh fv_deploy_created pod pod-golden-1 name=fv-serve-smoke-1025120000 image="$2" \
        gpu="NVIDIA RTX PRO 6000 Blackwell Server Edition" cost_per_hr=1.84 variant=cuda dc=EU-RO-1 \
        meta='{"config":"/etc/fv/runpod.toml"}'
      FV_DEPLOY_SCRIPT=runpod-pod.sh fv_deploy_ready pod pod-golden-1
      FV_DEPLOY_SCRIPT=runpod-pod.sh fv_deploy_deleted pod pod-golden-1 ;;
    endpoint)
      FV_DEPLOY_SCRIPT=runpod-endpoint.sh fv_deploy_created endpoint ep-golden-1 name=fv-serve-ep-1025120000 image="$2" \
        variant=cuda pool="" dc=EU-RO-1 gpu="NVIDIA RTX PRO 6000 Blackwell Server Edition" \
        meta='{"type":"queue","template":"tpl-golden-1","shared_template":false,"volume":"bz0ofkfx45","config":"/etc/fv/runpod.toml"}'
      FV_DEPLOY_SCRIPT=runpod-endpoint.sh fv_deploy_ready endpoint ep-golden-1
      FV_DEPLOY_SCRIPT=runpod-endpoint.sh fv_deploy_deleted endpoint ep-golden-1 ;;
  esac
}

ref() { echo "$REPO@$(jq -r ".$2" <<<"$1")"; }

# ---- release promote / rollback ----------------------------------------------------------------------------------
scenario promote-dry-run release promote aaaaaaa stable --local --dry-run
scenario promote-dispatch release promote aaaaaaa stable --notes "first stable"
scenario promote release promote aaaaaaa stable --local --notes "first stable"
scenario promote-again release promote aaaaaaa stable --local
scenario promote-latest release promote bbbbbbb latest --local
scenario promote-missing-image release promote ccccccc stable --local
scenario promote-digest release promote "$(ref "$DB" cuda)" stable --local --no-templates
scenario rollback-dry-run release rollback stable --local --dry-run
scenario rollback release rollback stable --local
scenario rollback-again release rollback stable --local
scenario rollback-to release rollback stable --local --to 3
scenario rollback-dispatch release rollback stable --to 1
# A release from before the one CUDA image (variant keys, no `cuda`): refused, then --allow-partial.
OLD="$(jq -nc --arg d "$(ref "$DA" debug)" --arg v "$(ref "$DA" cuda)" --arg c "$(ref "$DA" cpu)" '{debug: $d, "h3-turbo": $v, cpu: $c}')"
OLD_SQL="INSERT INTO releases (channel, git_sha, digests, action, promoted_at, promoted_by) VALUES ('stable', '0000000aaaa', '$OLD', 'promote', 1, 'test')"
hook __sql "$(jq -nc --arg s "$OLD_SQL" '{sql: $s}')" >/dev/null
OLDID="$(hook __sql '{"sql": "SELECT MAX(id) AS n FROM releases"}' | jq -r '.[0].n')"
scenario rollback-partial-refused release rollback stable --local --to "$OLDID"
scenario rollback-partial release rollback stable --local --to "$OLDID" --allow-partial
scenario rollback-back release rollback stable --local --to 3

# ---- templates ---------------------------------------------------------------------------------------------------
ENVS=(FV_RELEASE_CHANNEL=stable)
scenario templates-same templates sync cuda "$(ref "$DB" cuda)"
curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "$M/v1/templates/tpl-cpu-pod" >/dev/null
ENVS=(FV_RELEASE_CHANNEL=stable)
scenario templates-create templates sync cpu "$(ref "$DB" cpu)"
ENVS=(FV_RELEASE_CHANNEL=latest RUNPOD_REGISTRY_AUTH_ID=auth-golden)
scenario templates-update templates sync cuda "$(ref "$DA" cuda)"
scenario templates-unknown templates sync gpu "$(ref "$DA" cuda)"
scenario templates-tag templates sync cuda "$REPO:cuda-stable"

# ---- reconcile ---------------------------------------------------------------------------------------------------
# A stray fv pod nobody recorded, someone else's pod, an fv endpoint on a serve template, and a row whose pod is gone.
hook __seed "{\"pods\": {\"stray1\": {\"id\": \"stray1\", \"name\": \"fv-serve-smoke-x\", \"imageName\": \"$(ref "$DB" cuda)\",
  \"desiredStatus\": \"RUNNING\", \"createdAt\": \"2026-09-29 01:22:30.836 +0000 UTC\", \"costPerHr\": 0.5,
  \"machine\": {\"gpuDisplayName\": \"RTX PRO 6000\", \"dataCenterId\": \"EU-RO-1\"}, \"env\": {\"FV_ADMIN_TOKEN\": \"never-print-me\"}},
  \"other1\": {\"id\": \"other1\", \"name\": \"someone-else\", \"imageName\": \"daydreamlive/scope\", \"desiredStatus\": \"RUNNING\"}},
  \"endpoints\": {\"ep1\": {\"id\": \"ep1\", \"name\": \"fv-serve-ep-x\", \"templateId\": \"tpl-cuda-sls\",
  \"createdAt\": \"2026-09-30 10:00:00.000 +0000 UTC\", \"gpuTypeIds\": [\"NVIDIA RTX PRO 6000 Blackwell Server Edition\"], \"dataCenterIds\": [\"EU-RO-1\"]}}}" >/dev/null
hook __sql "{\"sql\": \"INSERT INTO deployments (id, kind, runpod_id, created_at, updated_at, created_by, status) VALUES ('pod:ghost', 'pod', 'ghost', 1, 1, 'test', 'ready')\"}" >/dev/null
scenario reconcile-dry-run release reconcile --dry-run
scenario reconcile-adopt release reconcile --adopt
hook __seed "{\"pods\": {\"stray1\": {\"id\": \"stray1\", \"name\": \"fv-serve-smoke-x\", \"imageName\": \"$(ref "$DA" cuda)\", \"desiredStatus\": \"RUNNING\"}}}" >/dev/null
scenario reconcile-drift release reconcile
scenario reconcile-fix release reconcile --fix

# ---- the deployments rows of the deploy scripts ------------------------------------------------------------------
scenario registry-pod registry pod "$(ref "$DB" cuda)"
scenario registry-endpoint registry endpoint "$(ref "$DB" cuda)"

# Normalized: digests as symbols, millisecond times as <now>, created template ids as <new template>.
jq -S --argjson syms "$SYMS" '
  def sym: reduce $syms[] as $s (.; gsub($s.re; $s.to));
  {mock: "scripts/serve/tests/mock_api.py",
   builds: {A: "aaaaaaa", B: "bbbbbbb", C: "ccccccc (debug cuda: no cpu)"},
   scenarios: (. | walk(if type == "string" then sym elif type == "number" and . >= 1000000000000 then "<now>" else . end))}' \
  <<<"$SCEN" >"$OUT"
if grep -Eq 'test-cf-token|test-runpod-key|test-gh-token|never-print-me' "$OUT" "$T"/*.err; then
  echo "release goldens: a secret reached the output" >&2
  exit 1
fi
