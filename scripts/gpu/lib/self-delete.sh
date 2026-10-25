# fv_self_delete LOG: a Runpod pod deletes itself (docs/adr/0023-lab-report-and-ci-bridge.md §4).
# The one definition of the in-pod self-delete: local launchers embed this file in the start
# commands they send; scripts that run on a pod carry the same function text inline.
# scripts/gpu/tests/self-delete.test.sh checks every copy against this file and that no
# in-pod self-delete uses REST. Returns 0 on success; the caller retries otherwise.
fv_self_delete() {
  # GraphQL podTerminate: the pod-scoped key may not DELETE through REST (HTTP 403,
  # docs/adr/0023-lab-report-and-ci-bridge.md §4). Success: HTTP 200 and no "errors".
  # The key reaches curl on stdin (-H @-), never in its argv; the answer is logged, then removed.
  _fvsd_code=$(printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY" | curl -sS --max-time 30 -o "$1.resp" \
    -w '%{http_code}' -X POST https://api.runpod.io/graphql -H 'Content-Type: application/json' -H @- \
    --data "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" 2>>"$1")
  { cat "$1.resp"; echo " http=$_fvsd_code"; } >>"$1" 2>/dev/null
  _fvsd_ok=1
  [ "$_fvsd_code" = 200 ] && ! grep -q '"errors"' "$1.resp" 2>/dev/null && _fvsd_ok=0
  rm -f "$1.resp"
  return "$_fvsd_ok"
}
