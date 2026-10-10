// The host half of a Brev pod's boot (docs/serve/deploy-gmi-brev.md §6.3):
// the script that runs on the VM (root or passwordless sudo, UNVERIFIED) and
// starts our image with Docker. Everything else a pod reports comes from
// inside the container (PROVIDER_BOOT) once it runs; a VM whose host part
// failed (no Docker, no NVIDIA toolkit, sudo asking for a password, a pull or
// run error) said nothing at all. So the host reports too, with the cluster's
// ingest token to POST /ingest/v1/endpoint (never failing the script):
//   host              the host's facts, first thing: user, uid, sudo, docker,
//                     docker info, nvidia-smi -L, the NVIDIA container
//                     toolkit, disk free, kernel, OS
//   host_failed       the failing step and the last 20 lines of its log
//                     (~/workspace/fv/host.log); fails the boot
//   container_started docker run succeeded
//   heartbeat         every 30 s from a detached loop, with the container's
//                     state and its last log line, until the boot logs its
//                     tunnel (or the container exits: host_failed with its log)
// Docker missing: installed (docker.io + the NVIDIA container toolkit, per
// NVIDIA's apt instructions) only with FV_BREV_INSTALL_DOCKER=1 in the pod's
// env; else host_failed says so.
import { HttpError } from "./util";

const shq = (s: string) => `'${s.replace(/'/g, `'\\''`)}'`;

/** Shared by the host script and its heartbeat: JSON-safe one-line text, and the report (10 s, never fails). */
const HOST_LIB = `esc() { printf '%s' "$1" | tr '\\n\\r\\t' '   ' | tr -d '\\000-\\037' | sed -e 's/\\\\/\\\\\\\\/g' -e 's/"/\\\\"/g' | cut -c1-1900; }
hreport() { # phase detail
  command -v curl >/dev/null 2>&1 || return 0
  curl -sS --max-time 10 -X POST -H @"$W/fv/report-header" -H "content-type: application/json" \\
    -d "{\\"pod\\":\\"$R_POD\\",\\"phase\\":\\"$1\\",\\"url\\":\\"\\",\\"detail\\":\\"$(esc "$2")\\"}" "$R_URL" >/dev/null 2>&1 || true
}
tail20() { tail -n 20 "$1" 2>/dev/null | awk 1 ORS=' | '; }
`;

/** The heartbeat loop (a file of its own, run detached): every 30 s until the boot logs its tunnel or the container exits. */
const HEARTBEAT = `n=0
while [ "$n" -lt 480 ]; do
  n=$((n + 1))
  st="$($S docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}}' fv-serve 2>/dev/null || echo "no container")"
  last="$($S docker logs --tail 1 fv-serve 2>&1 | tail -n 1)"
  if $S docker logs fv-serve 2>&1 | grep -q '\\[fv-boot\\] tunnel '; then hreport heartbeat "tunnel up; heartbeat stops"; exit 0; fi
  case "$st" in
    exited*|dead*) hreport host_failed "container $st: $($S docker logs --tail 20 fv-serve 2>&1 | awk 1 ORS=' | ')"; exit 0 ;;
  esac
  hreport heartbeat "container: $st; last: $last"
  sleep 30
done
`;

/** NVIDIA's documented install: docker.io, then nvidia-container-toolkit from its apt repo, configured for Docker. */
const INSTALL_DOCKER = `install_docker() {
  $S apt-get update -qq && $S env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io curl gpg || return 1
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | $S gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg || return 1
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \\
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \\
    | $S tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null || return 1
  $S apt-get update -qq && $S env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-container-toolkit || return 1
  $S nvidia-ctk runtime configure --runtime=docker && $S systemctl restart docker
}
`;

/**
 * The host script of one Brev launch: report the host's facts, check root / sudo and Docker (install it only with
 * FV_BREV_INSTALL_DOCKER=1), write the env file (0600) and the boot, pull and run our image with the GPU and the
 * weights dir, report container_started, heartbeat until the tunnel, and a deadline watchdog. FV_HOST_WS (tests)
 * overrides the workspace dir.
 */
export function brevHostScript(image: string, env: Record<string, string>, boot: string): string {
  for (const [k, v] of Object.entries(env)) if (/[\r\n]/.test(v)) throw new HttpError(400, `brev: env ${k} has a line break (docker --env-file takes one line per variable)`);
  const envFile = Object.entries(env)
    .map(([k, v]) => `${k}=${v}`)
    .join("\n");
  const deadline = Number(env.FV_CLUSTER_DEADLINE || "0");
  const install = env.FV_BREV_INSTALL_DOCKER === "1";
  return `#!/usr/bin/env bash
set -u
umask 077
W="\${FV_HOST_WS:-/home/ubuntu/workspace}"
mkdir -p "$W/fv" "$W/weights" 2>/dev/null || { W="\${HOME:-/tmp}/workspace"; mkdir -p "$W/fv" "$W/weights"; }
exec > >(tee -a "$W/fv/host.log") 2>&1
echo "[fv-host] start $(date -u +%FT%TZ)"
R_URL=${shq(env.FV_ENDPOINT_REPORT_URL || "")}
R_POD=${shq(env.FV_POD_ID || "")}
printf 'Authorization: Bearer %s\\n' ${shq(env.FV_ENDPOINT_REPORT_TOKEN || "")} > "$W/fv/report-header"
S=""; [ "$(id -u)" = 0 ] || S="sudo -n"
printf %s ${shq(btoa(HOST_LIB))} | base64 -d > "$W/fv/host-lib.sh"
. "$W/fv/host-lib.sh"
# ---- the host's facts, first
if [ "$(id -u)" = 0 ]; then SUDO=root; elif sudo -n true 2>/dev/null; then SUDO=yes; else SUDO=NO; fi
DK="$(command -v docker || echo none)"
DI="$($S docker info --format '{{.ServerVersion}} runtimes={{range $k, $v := .Runtimes}}{{$k}},{{end}}' 2>&1 | head -n 1)"
GPU="$(nvidia-smi -L 2>&1 | head -n 1)"
if command -v nvidia-ctk >/dev/null 2>&1; then CTK="$(nvidia-ctk --version 2>/dev/null | head -n 1)"; elif dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then CTK=package; else CTK=none; fi
DISK="$(df -h "$W" 2>/dev/null | awk 'NR==2 {print $4 " free of " $2}')"
OS="$( (. /etc/os-release 2>/dev/null; echo "\${ID:-?} \${VERSION_ID:-?}") )"
hreport host "user=$(whoami) uid=$(id -u) sudo=$SUDO docker=$DK info=[$DI] gpu=[$GPU] toolkit=[$CTK] disk=[$DISK] kernel=$(uname -r) os=[$OS] ws=$W"
die() { echo "[fv-host] FAILED: $1"; hreport host_failed "$1: $(tail20 "$W/fv/host.log")"; exit 1; }
step() { # what, command...
  what="$1"; shift
  echo "[fv-host] $what"
  "$@" || die "$what (exit $?)"
}
[ "$SUDO" = NO ] && die "not root and sudo -n needs a password (user $(whoami))"
${INSTALL_DOCKER}if [ "$DK" = none ]; then
  ${install ? 'step "install docker + the NVIDIA container toolkit (FV_BREV_INSTALL_DOCKER=1)" install_docker' : 'die "docker not found on the host (set FV_BREV_INSTALL_DOCKER=1 in the pod env to install it)"'}
fi
# ---- the heartbeat (detached): a silent pod is then told apart from a dead script
printf '%s\\n' "W=\\"$W\\"" "R_URL='$R_URL'" "R_POD='$R_POD'" "S=\\"$S\\"" > "$W/fv/heartbeat.sh"
cat "$W/fv/host-lib.sh" >> "$W/fv/heartbeat.sh"
printf %s ${shq(btoa(HEARTBEAT))} | base64 -d >> "$W/fv/heartbeat.sh"
nohup setsid bash "$W/fv/heartbeat.sh" >/dev/null 2>&1 &
# ---- the launch
step "write the env file and the boot" sh -c "printf %s ${shq(btoa(envFile))} | base64 -d > '$W/fv/env' && printf %s ${shq(btoa(boot))} | base64 -d > '$W/fv/boot.sh'"
step "docker pull" $S docker pull -q ${shq(image)}
$S docker rm -f fv-serve >/dev/null 2>&1 || true
step "docker run" $S docker run -d --name fv-serve --restart no --gpus all --network host --env-file "$W/fv/env" \\
  -v "$W/weights:/workspace/weights" -v "$W/fv/boot.sh:/fv-boot.sh:ro" \\
  --entrypoint bash ${shq(image)} /fv-boot.sh
hreport container_started "$($S docker inspect -f '{{.Id}}' fv-serve 2>/dev/null | cut -c1-12) ${image.replace(/[^A-Za-z0-9@:/._-]/g, "")}"
nohup setsid bash -c 'while [ "$(date +%s)" -lt ${deadline} ]; do sleep 30; done; '"$S"' docker rm -f fv-serve; '"$S"' shutdown -h now' >/dev/null 2>&1 &
echo "[fv-host] done"
`;
}
