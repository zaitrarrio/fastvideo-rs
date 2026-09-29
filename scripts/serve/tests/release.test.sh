#!/usr/bin/env bash
# Tests for the release and deployment scripts (docs/serve/releases.md)
# against scripts/serve/tests/mock_api.py: D1 (SQLite), Runpod REST and
# GraphQL, a GHCR-like registry, GitHub workflow_dispatch and fv-serve pods.
# No network, no Docker (a stub stands in for `docker buildx imagetools`).
#
#   bash scripts/serve/tests/release.test.sh      # needs bash, curl, jq, python3
# shellcheck disable=SC2016 # the checks pass values to `bash -c` programs as $0, $1
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVE="$(cd "$HERE/.." && pwd)"
for t in curl jq python3; do command -v "$t" >/dev/null || { echo "release.test: skipped (no $t)"; exit 0; }; done

T="$(mktemp -d)"
cleanup() { [[ -n "${MOCK_PID:-}" ]] && kill "$MOCK_PID" 2>/dev/null; wait 2>/dev/null; rm -rf "$T"; }
trap cleanup EXIT
python3 "$HERE/mock_api.py" "$T/port" &
MOCK_PID=$!
for _ in $(seq 50); do [[ -s "$T/port" ]] && break; sleep 0.1; done
M="http://127.0.0.1:$(cat "$T/port")"

# Everything points at the mock; real credentials never reach it.
unset CLOUDFLARE_API_KEY CLOUDFLARE_API_TOKEN FV_CF_ACCOUNT_ID FV_D1_DATABASE_ID GITHUB_TOKEN GITHUB_ACTIONS CLAUDECODE FV_RELEASE_CHANNEL
export FV_ENV_FILE=/dev/null
export FV_D1_API_BASE="$M/client/v4" FV_CF_API_TOKEN=test-cf-token
export RUNPOD_API_BASE="$M/v1" RUNPOD_GRAPHQL="$M/graphql" RUNPOD_API_KEY=test-runpod-key
export FV_REGISTRY_API="$M" FV_GITHUB_API="$M" GH_TOKEN=test-gh-token
export FV_POD_URL_TEMPLATE="$M/pod/{pod}" FV_CLUSTER_STATE="$T/cluster.json" FV_SERVE_LEDGER="$T/ledger.tsv"
export FV_DEPLOYED_BY=test FV_POD_CAP_S=1 FV_ROLL_WAIT_S=30 FV_DRAIN_WAIT_S=5
REPO=ghcr.io/zaitrarrio/fastvideo-rs-serve
KEYS="debug h3-turbo h3-max ltx wan wan5b sfwan gateway"

# `docker buildx imagetools create --tag <repo>:<tag> <repo>@<digest>` moves the mock's tag.
mkdir -p "$T/bin"
cat >"$T/bin/docker" <<EOF
#!/usr/bin/env bash
tag=""; src=""
while (( \$# )); do case "\$1" in --tag|-t) tag="\$2"; shift ;; *@sha256:*) src="\$1" ;; esac; shift; done
echo "docker \$tag <- \$src" >>"$T/docker.log"
curl -sS -X POST -d "{\"tag\":\"\${tag##*:}\",\"digest\":\"\${src##*@}\"}" "$M/__tag" >/dev/null
EOF
chmod +x "$T/bin/docker"
export FV_DOCKER="$T/bin/docker"

PASS=0 FAIL=0
ok() { PASS=$((PASS + 1)); printf 'ok   %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf 'FAIL %s\n' "$1"; [[ -z "${2:-}" ]] || printf '%s\n' "$2" | sed 's/^/     /' | head -30; }
check() { local name="$1"; shift; if "$@"; then ok "$name"; else bad "$name"; fi; }
state() { curl -sS "$M/__state"; }
sql() { curl -sS -X POST -d "$(jq -nc --arg s "$1" '{sql: $s}')" "$M/__sql"; }
tag() { state | jq -r --arg t "$1" '.tags[$t] // empty'; }
rel() { bash "$SERVE/release.sh" "$@" 2>>"$T/stderr.log"; }
build() { curl -sS -X POST -d "$(jq -nc --arg s "$1" --arg k "$2" '{sha: $s, keys: ($k | split(" "))}')" "$M/__build"; }

A=aaaaaaa1111111111111111111111111111111111
A="${A:0:40}"
B="bbbbbbb2222222222222222222222222222222222"; B="${B:0:40}"
C="ccccccc3333333333333333333333333333333333"; C="${C:0:40}"
DA="$(build "$A" "$KEYS")"; DB="$(build "$B" "$KEYS")"; build "$C" "debug h3-turbo" >/dev/null
# The shared templates, on an older image.
tpls='{}'
for v in $KEYS; do
  [[ "$v" == debug ]] && continue
  for f in sls pod; do
    [[ "$v" == gateway && "$f" == sls ]] && continue
    tpls="$(jq -c --arg id "tpl-$v-$f" --arg n "fv-serve-$v-$f" '. + {($id): {id: $id, name: $n, imageName: "ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:0ld", env: {}}}' <<<"$tpls")"
  done
done
curl -sS -X POST -d "{\"templates\": $tpls}" "$M/__seed" >/dev/null

# --- resolve -------------------------------------------------------------------
out="$(rel resolve "${A:0:7}")"
check "resolve: a short sha gives every image, the full sha from the labels" \
  test "$(jq -r '.sha, (.digests | length), .digests["h3-turbo"]' <<<"$out" | tr '\n' ' ')" = "$A 8 $REPO@$(jq -r '.["h3-turbo"]' <<<"$DA") "
out="$(rel resolve "$REPO@$(jq -r .ltx <<<"$DB")")"
check "resolve: a digest finds its build through the revision label" test "$(jq -r .short <<<"$out")" = "${B:0:7}"

# --- promote -------------------------------------------------------------------
out="$(rel promote "${A:0:7}" stable --dry-run)"
check "promote --dry-run: prints the retags and template changes" \
  bash -c 'grep -q "retag  :stable -> " <<<"$0" && grep -q "retag  :h3-turbo-stable -> " <<<"$0" && grep -q "template fv-serve-wan-sls: .*0ld -> " <<<"$0"' "$out"
check "promote --dry-run: changes nothing" \
  bash -c '[[ -z "$(jq -r ".tags.stable // empty" <<<"$0")" && ! -e "$1/docker.log" ]]' "$(state)" "$T"

rel promote "${A:0:7}" stable >/dev/null
check "promote (not in CI, no --local): dispatches release.yml" \
  test "$(state | jq -c '.dispatches[-1] | [.workflow, .ref, .inputs.action, .inputs.target, .inputs.channel]')" = '["release.yml","main","promote","aaaaaaa","stable"]'
check "promote dispatch: no tag moved" test -z "$(tag stable)"

out="$(rel promote "${A:0:7}" stable --local --notes "first stable")"
check "promote --local: release 1 recorded" bash -c 'grep -q "release 1: stable = aaaaaaa (promote)" <<<"$0"' "$out"
check "promote: :stable and :<variant>-stable point at the build" \
  test "$(tag stable) $(tag h3-turbo-stable) $(tag gateway-stable)" = "$(jq -r '.debug, .["h3-turbo"], .gateway' <<<"$DA" | xargs)"
check "promote: stable does not move the legacy :<variant> tags" test -z "$(tag h3-turbo)"
check "promote: every template boots the build, with its identity env" \
  test "$(state | jq -r --arg d "$(jq -r .wan <<<"$DA")" '[.templates[] | select(.name | startswith("fv-serve-wan-"))
    | (.imageName | endswith($d)) and .env.FV_IMAGE_DIGEST == $d and .env.FV_RELEASE_CHANNEL == "stable"] | all')" = true
check "promote: the D1 row" \
  test "$(sql "SELECT channel, git_sha, action, notes, templates_updated, promoted_by FROM releases WHERE id = 1" | jq -c '.[0]')" \
  = "{\"channel\":\"stable\",\"git_sha\":\"$A\",\"action\":\"promote\",\"notes\":\"first stable\",\"templates_updated\":1,\"promoted_by\":\"test\"}"
out="$(rel promote "${A:0:7}" stable --local)"
check "promote again: a no-op" bash -c 'grep -q "already aaaaaaa" <<<"$0"' "$out"

if rel promote "$REPO:${B:0:7}" stable --local >/dev/null 2>&1; then bad "promote: an unknown tag is refused"; else ok "promote: an unknown tag is refused"; fi
rel promote "${B:0:7}" stable --local >/dev/null
check "promote B: release 2" test "$(sql "SELECT MAX(id) AS n FROM releases" | jq -r '.[0].n')" = 2
rel promote "${B:0:7}" latest --local >/dev/null
check "promote to latest: :latest, :<v>-latest and :<v>; templates untouched" \
  bash -c '[[ "$(jq -r ".tags.latest, .tags[\"wan-latest\"], .tags.wan" <<<"$0" | xargs)" == "$1" ]] && [[ "$(jq -r "[.templates[] | .env.FV_RELEASE_CHANNEL] | unique[]" <<<"$0")" == stable ]]' \
  "$(state)" "$(jq -r '.debug, .wan, .wan' <<<"$DB" | xargs)"

out="$(bash "$SERVE/release.sh" promote "${C:0:7}" stable --local 2>&1)"
check "promote: a build missing variants is refused" bash -c '[[ "$0" == *"has no image for: h3-max ltx wan wan5b sfwan gateway"* ]]' "$out"
check "promote: refused, nothing moved" test "$(tag stable)" = "$(jq -r .debug <<<"$DB")"

# --- list / history ------------------------------------------------------------
out="$(rel list --verify)"
check "list: the head of each channel, the tags verified" \
  bash -c 'grep -q "^stable   release 2  *bbbbbbb" <<<"$0" && grep -q "^latest   release 3" <<<"$0" && ! grep -q DIFFERS <<<"$0" && grep -c ":.* ok" <<<"$0" | grep -q 16' "$out"
out="$(rel history --channel stable)"
check "history: newest first" bash -c '[[ "$(awk "NR>1 {print \$1}" <<<"$0" | xargs)" == "2 1" ]]' "$out"

# --- rollback ------------------------------------------------------------------
rel rollback stable >/dev/null
check "rollback (not in CI): dispatched" test "$(state | jq -r '.dispatches[-1].inputs.action')" = rollback
out="$(rel rollback stable --local --dry-run)"
check "rollback --dry-run: plans the previous release" bash -c 'grep -q "retag  :stable -> .*'"$(jq -r .debug <<<"$DA")"'" <<<"$0"' "$out"
out="$(rel rollback stable --local)"
check "rollback: release 4 re-promotes release 1" bash -c 'grep -q "release 4: stable = aaaaaaa (rollback)" <<<"$0"' "$out"
check "rollback: the tags and templates are back on A" \
  test "$(tag stable) $(state | jq -r '.templates["tpl-ltx-pod"].imageName')" = "$(jq -r .debug <<<"$DA") $REPO@$(jq -r .ltx <<<"$DA")"
check "rollback: the row names its source and B is marked rolled back" \
  test "$(sql "SELECT (SELECT source_release FROM releases WHERE id = 4) AS s, (SELECT rolled_back_at IS NOT NULL FROM releases WHERE id = 2) AS r" | jq -c '.[0]')" = '{"s":1,"r":1}'
out="$(bash "$SERVE/release.sh" rollback stable --local 2>&1)"
if [[ "$out" == *"no earlier release"* ]]; then ok "rollback twice: nothing earlier than A"; else bad "rollback twice: nothing earlier than A" "$out"; fi
out="$(rel rollback stable --local --to 2)"
check "rollback --to: an explicit release" bash -c 'grep -q "stable = bbbbbbb (rollback)" <<<"$0"' "$out"

# --- CI record -----------------------------------------------------------------
printf 'h3-turbo\t%s\t1000\nwan\t%s\t900\n' "$REPO@$(jq -r '.["h3-turbo"]' <<<"$DA")" "$REPO@$(jq -r .wan <<<"$DA")" >"$T/variants.tsv"
out="$(rel record-build latest "$A" "$T/variants.tsv" "$REPO@$(jq -r .debug <<<"$DA")")"
check "record-build: the main build as a latest release" \
  test "$(sql "SELECT action, digests FROM releases WHERE id = (SELECT MAX(id) FROM releases)" | jq -c '.[0] | [.action, (.digests | fromjson | keys)]')" = '["build",["debug","h3-turbo","wan"]]'

# --- deploy scripts write the registry ------------------------------------------
line="$(FV_SERVE_IMAGE="$REPO@$(jq -r '.["h3-turbo"]' <<<"$DA")" FV_SERVE_CONFIG=/etc/fv/runpod.toml bash "$SERVE/runpod-pod.sh" up 2>>"$T/stderr.log")"
POD="${line%% *}"
row="$(sql "SELECT kind, status, variant, git_sha, created_by, digest FROM deployments WHERE runpod_id = '$POD'" | jq -c '.[0]')"
check "runpod-pod.sh up: a deployments row with the build's sha and variant" \
  test "$row" = "{\"kind\":\"pod\",\"status\":\"creating\",\"variant\":\"h3-turbo\",\"git_sha\":\"$A\",\"created_by\":\"test/runpod-pod.sh\",\"digest\":\"$(jq -r '.["h3-turbo"]' <<<"$DA")\"}"
check "runpod-pod.sh up: the pod env names its image" \
  test "$(state | jq -r --arg p "$POD" '.pods[$p].env.FV_IMAGE_DIGEST')" = "$(jq -r '.["h3-turbo"]' <<<"$DA")"
bash "$SERVE/runpod-pod.sh" down "$POD" >/dev/null 2>>"$T/stderr.log"
check "runpod-pod.sh down: the row is deleted" \
  test "$(sql "SELECT status, deleted_at IS NOT NULL AS d FROM deployments WHERE runpod_id = '$POD'" | jq -c '.[0]')" = '{"status":"deleted","d":1}'
FV_REGISTRY=0 FV_SERVE_IMAGE="$REPO@$(jq -r .debug <<<"$DA")" bash "$SERVE/runpod-pod.sh" up >/dev/null 2>>"$T/stderr.log"
check "FV_REGISTRY=0: nothing recorded" test "$(sql "SELECT COUNT(*) AS n FROM deployments" | jq -r '.[0].n')" = 1

# --- deployed / reconcile --------------------------------------------------------
# A stray fv pod nobody recorded, and a row for a pod that is gone.
curl -sS -X POST "$M/__seed" -d "{\"pods\": {\"stray1\": {\"id\": \"stray1\", \"name\": \"fv-serve-smoke-x\", \"imageName\": \"$REPO@$(jq -r .ltx <<<"$DB")\",
  \"desiredStatus\": \"RUNNING\", \"createdAt\": \"2026-09-29 01:22:30.836 +0000 UTC\", \"env\": {\"FV_ADMIN_TOKEN\": \"never-print-me\"}},
  \"other1\": {\"id\": \"other1\", \"name\": \"someone-else\", \"imageName\": \"daydreamlive/scope\", \"desiredStatus\": \"RUNNING\"}}}" >/dev/null
sql "INSERT INTO deployments (id, kind, runpod_id, created_at, updated_at, created_by, status) VALUES ('pod:ghost', 'pod', 'ghost', 1, 1, 'test', 'ready')" >/dev/null
out="$(rel deployed --json)"
got="$(jq -c '[.[] | select(.id == "stray1") | {key, sha: .sha[0:7], channels, follows, drift, known}]' <<<"$out")"
want='[{"key":"ltx","sha":"bbbbbbb","channels":["stable"],"follows":"stable","drift":false,"known":false}]'
if [[ "$got" == "$want" ]]; then ok "deployed: fv pods only, with sha, image key, channels and drift"; else bad "deployed: fv pods only, with sha, image key, channels and drift" "$got"; fi
check "deployed: never someone else's pods" test "$(jq '[.[] | select(.id == "other1")] | length' <<<"$out")" = 0
out="$(rel deployed)"
check "deployed: the table" bash -c 'grep -q "^pod  *stray1 .* ltx  *bbbbbbb .* ok (stable) unregistered" <<<"$0"' "$out"
out="$(rel reconcile --dry-run)"
check "reconcile --dry-run: the gone row and the unknown pod, nothing marked" \
  bash -c 'grep -q "^gone      pod ghost" <<<"$0" && grep -q "^unknown   pod stray1" <<<"$0" && grep -q "not marked" <<<"$0"' "$out"
check "reconcile --dry-run: the row stays" test "$(sql "SELECT status FROM deployments WHERE id = 'pod:ghost'" | jq -r '.[0].status')" = ready
out="$(rel reconcile --adopt)"
check "reconcile: gone rows marked, unknown pods adopted" \
  test "$(sql "SELECT (SELECT status FROM deployments WHERE id = 'pod:ghost') AS g, (SELECT created_by FROM deployments WHERE id = 'pod:stray1') AS a" | jq -c '.[0]')" \
  = '{"g":"gone","a":"reconcile/release.sh"}'
check "reconcile: the templates follow stable (B)" bash -c '! grep -q "^template" <<<"$0"' "$out"
curl -sS -X POST "$M/__seed" -d '{"pods": {"stray1": {"id": "stray1", "name": "fv-serve-smoke-x", "imageName": "ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:0ther", "desiredStatus": "RUNNING"}}}' >/dev/null
out="$(rel reconcile --fix)"
check "reconcile: a changed image is flagged and --fix records it" \
  bash -c 'grep -q "^drifted   pod stray1" <<<"$0" && [[ "$1" == "sha256:0ther" ]]' "$out" "$(sql "SELECT digest FROM deployments WHERE id = 'pod:stray1'" | jq -r '.[0].digest')"

# --- redeploy (rolling) ------------------------------------------------------------
now=$(date +%s)
curl -sS -X POST "$M/__seed" -d "{\"pods\": {
  \"gw1\": {\"id\": \"gw1\", \"name\": \"fv-cluster-gw-1\", \"imageName\": \"$REPO@$(jq -r .debug <<<"$DA")\", \"desiredStatus\": \"RUNNING\", \"env\": {}},
  \"old-h3\": {\"id\": \"old-h3\", \"name\": \"fv-cluster-h3-turbo-1\", \"imageName\": \"$REPO@$(jq -r .debug <<<"$DA")\", \"desiredStatus\": \"RUNNING\", \"env\": {}},
  \"old-wan\": {\"id\": \"old-wan\", \"name\": \"fv-cluster-wan-1\", \"imageName\": \"$REPO@$(jq -r .debug <<<"$DA")\", \"desiredStatus\": \"RUNNING\", \"env\": {}}}}" >/dev/null
jq -n --arg img "$REPO@$(jq -r .debug <<<"$DA")" --argjson dl "$((now + 6000))" '{image: $img, internal_token: "test-internal-token",
  url_signing_key: "k", admin_token: "fvadm_x", auth: "none", deadline: $dl, gateway_url: "https://gw1-8000.proxy.runpod.net",
  gateway: {pod: "gw1", cpu: "cpu3c", dph: 0.1, created: 1, dc: "EUR-IS-1", image: $img},
  workers: {"h3-turbo": {pod: "old-h3", gpu: "g", dc: "EUR-IS-1", dph: 1, created: 1, url: "https://old-h3-8000.proxy.runpod.net"},
            wan: {pod: "old-wan", gpu: "g", dc: "EUR-IS-1", dph: 1, created: 1, url: "https://old-wan-8000.proxy.runpod.net", image: $img}}}' >"$T/cluster.json"
npods="$(state | jq '.pods | length')"
out="$(rel redeploy h3-turbo stable --dry-run)"
check "redeploy --dry-run: the plan (stable is B now)" bash -c 'grep -q "h3-turbo: .* -> '"$(jq -r .debug <<<"$DB" | cut -c8-19)"'" <<<"$0" && grep -q "dry run" <<<"$0"' "$out"
check "redeploy --dry-run: no pod created" test "$(state | jq '.pods | length')" = "$npods"
rel redeploy h3-turbo stable >"$T/roll.out" 2>&1
NEW="$(jq -r '.workers["h3-turbo"].pod' "$T/cluster.json")"
check "redeploy: the pool's worker is a new pod on B" \
  test "$NEW/$(jq -r '.workers["h3-turbo"].image' "$T/cluster.json")" = "$NEW/$REPO@$(jq -r .debug <<<"$DB")"
check "redeploy: the old worker was drained, then deleted" \
  test "$(state | jq -c '[.drains, (.pods | has("old-h3")), (.pods | has("old-wan"))]')" = '[["old-h3"],false,true]'
check "redeploy: the gateway saw both workers, then only the new one" \
  test "$(state | jq -r '.pods.gw1.env.FV_POOL_H3_TURBO_URLS')" = "https://$NEW-8000.proxy.runpod.net"
check "redeploy: two gateway restarts, no image change" \
  test "$(state | jq -c '[.patches[] | select(.id == "gw1") | .keys]')" = '[["env"],["env"]]'
check "redeploy: the registry has the new pod ready and the old one deleted" \
  test "$(sql "SELECT runpod_id, status, channel FROM deployments WHERE pool = 'h3-turbo' OR runpod_id = 'old-h3' ORDER BY created_at" | jq -c 'map([.status, .channel])')" = '[["ready","stable"]]'
check "redeploy: the state has no roll leftovers" test "$(jq -c '[.rolling, (.retired | length)]' "$T/cluster.json")" = '[null,0]'
rel redeploy gateway stable >/dev/null 2>>"$T/stderr.log"
check "redeploy gateway: same pod, new image" \
  test "$(state | jq -r '.pods.gw1.imageName') $(jq -r .gateway.image "$T/cluster.json")" = "$REPO@$(jq -r .debug <<<"$DB") $REPO@$(jq -r .debug <<<"$DB")"
out="$(bash "$SERVE/release.sh" redeploy all stable 2>&1)"
check "redeploy all: only what is behind (wan)" bash -c 'grep -q "  wan: " <<<"$0" && ! grep -q "  h3-turbo: " <<<"$0" && ! grep -q "  gateway: " <<<"$0"' "$out"

# --- secrets -------------------------------------------------------------------
all="$(cat "$T"/*.out "$T/stderr.log" 2>/dev/null)"
check "no secret in any output" bash -c '! grep -Eq "test-cf-token|test-runpod-key|test-gh-token|test-internal-token|never-print-me" <<<"$0"' "$all"
check "no secret in the ledger" bash -c '! grep -Eq "test-|never-print-me" "$0"' "$T/ledger.tsv"

printf '\nrelease.test: %d passed, %d failed\n' "$PASS" "$FAIL"
[[ "$FAIL" == 0 ]] || { echo "--- stderr (tail)"; tail -40 "$T/stderr.log"; exit 1; }
