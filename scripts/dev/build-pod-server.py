#!/usr/bin/env python3
"""fv-build pod service: authenticated sync + allowlisted build runner.

Runs on the shared CPU build pod (docs/dev/build-pod.md), started by the pod's
start command from scripts/dev/build-pod.sh. Python standard library only.

Layout. Source snapshots, target dirs and CARGO_HOME live on the container
disk (FV_BUILD_LOCAL, default /root/fvb): cargo on the network volume spent
18 min on a cold `cargo check` that sccache and a local disk make minutes, and
30 s on a no-op one (fingerprint stats over the network filesystem). They are
lost when the pod stops; FV_BUILD_TARGETS=volume keeps them on the volume.
  <local>/worktrees/<agent>/   the agent's synced source snapshot
  <local>/target/<agent>/      its CARGO_TARGET_DIR
  <local>/cargo/               CARGO_HOME; registry/cache and git/db link to the volume
On the network volume (FV_BUILD_ROOT, default /workspace/fv-build):
  cargo/registry/cache, cargo/git/db   downloaded crates and git checkouts (shared)
  rustup/              RUSTUP_HOME (toolchains, shared)
  sccache/             SCCACHE_DIR (shared; FV_BUILD_SCCACHE_SIZE, default 40G)
  cuda-13.4/           CUDA 13.4 nvcc/NVRTC/crt/cudart/cccl/tileiras (redist)
  tools/               sccache binary
  jobs/<id>.log        job logs; logs/pod.log service log; ledger.tsv

Auth: every endpoint but GET /healthz needs "Authorization: Bearer <token>";
the pod only knows sha256(token) (FV_BUILD_TOKEN_SHA256).

Endpoints (all JSON unless noted):
  GET  /healthz                         {ok, ready, phase, boot, timers, self_stop, jobs}  (no auth)
  GET  /v1/status                       setup state, jobs, idle timer, disk
  GET  /v1/agents[?sizes=1]             agent dirs (sizes via du, slow)
  GET  /v1/agents/<a>/manifest          text: path\\tsize\\tmtime per file
  PUT  /v1/agents/<a>/files             body: tar.gz of changed files
  POST /v1/agents/<a>/delete            body: NUL-separated relative paths
  POST /v1/agents/<a>/jobs              {argv:[...], env:{K:V}} -> {id}
  GET  /v1/jobs/<id>?offset=N&wait=S    {state, exit, next, data}  (long poll)
  POST /v1/jobs/<id>/cancel
  GET  /v1/agents/<a>/artifact?path=P[&gz=1]   file under target/<a> (binary)
  POST /v1/agents/<a>/clean             {what: target|worktree|all}
  POST /v1/evict                        run an eviction pass now -> {evicted}
  GET  /v1/log?lines=N                  text: tail of logs/pod.log (self-stop attempts etc.)
  POST /v1/stop                         stop the pod now

Only these commands run (no shell): cargo {check,build,test,clippy,fmt,doc,
tree,metadata}, and `bash <script>` for the scripts in SCRIPTS. Environment
overrides must match ENV_ALLOW. Build scripts still execute code, so the token
is the real boundary; the allowlist keeps agents on the build path.

Auto-stop (StopPolicy, Stopper below; docs/dev/build-pod.md "Costs and money
guards"):
  idle  FV_BUILD_IDLE_MIN (default 20) minutes with no queued or running job
        and no work request (sync, job submit, artifact fetch, clean). Status,
        health, agent listings, manifests and job-log polls are reads: they do
        not keep the pod alive (monitors poll /v1/status).
  cap   FV_BUILD_MAX_HOURS (default 8) after boot: stop at once when no job is
        active; otherwise new jobs are refused, active jobs get a warning in
        their log, and the pod stops FV_BUILD_MAX_GRACE_MIN (default 30) later
        even if they are still running.
The stop goes through the Runpod API with the pod-scoped key Runpod injects
(RUNPOD_API_KEY, RUNPOD_POD_ID): REST stop, REST terminate, then GraphQL
podStop / podTerminate (everything worth keeping lives on the volume). Every
failure is logged with its HTTP status and body, and the whole sequence is
retried (1, 2, 4 ... 10 min apart) until the pod is gone; after an accepted
call it is retried 10 min later should the pod still be running. All calls send
an explicit User-Agent: Cloudflare in front of rest.runpod.io and
api.runpod.io answers Python's default "Python-urllib/3.x" with 403 "error
code: 1010", which is why no self-stop ever worked before 2026-10-02.

Eviction (the container disk fills up with agents' target dirs): a background
pass every FV_BUILD_EVICT_INTERVAL_S (60) seconds, and after every job,
removes the target dir and snapshot of each agent unused for more than
FV_BUILD_EVICT_HOURS (default 6; last sync, job or fetch), then, while the
disk has less than FV_BUILD_EVICT_FREE_GB (default 40) free, target dirs in
least-recently-used order. An agent with a job running or queued is never
touched. The release agent (FV_BUILD_EVICT_PROTECT, default fv-release: the
`build-pod.sh release-artifacts` flow syncs, runs, then fetches its tarballs
from target/fv-release/) is also kept for FV_BUILD_EVICT_HOLD_MIN (default 60)
after its last use, and goes last under disk pressure. Only target/<agent>
and worktrees/<agent> are ever evicted: the volume (release-cache/, caches,
toolchains) never is. Each eviction is logged and listed in /v1/status; an
evicted target dir just means the next job builds cold (sccache on the volume
refills it). Sync and job requests that find the disk full (after an
eviction pass) get HTTP 507 with a "build pod disk full" error instead of a
500.
"""

import collections
import errno
import gzip
import hashlib
import hmac
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.environ.get("FV_BUILD_ROOT", "/workspace/fv-build")
PORT = int(os.environ.get("FV_BUILD_PORT", "8000"))
TOKEN_SHA = os.environ.get("FV_BUILD_TOKEN_SHA256", "").strip().lower()
IDLE_S = float(os.environ.get("FV_BUILD_IDLE_MIN", "20")) * 60
MAX_S = float(os.environ.get("FV_BUILD_MAX_HOURS", "8")) * 3600
MAX_GRACE_S = float(os.environ.get("FV_BUILD_MAX_GRACE_MIN", "30")) * 60
# Cloudflare (rest.runpod.io, api.runpod.io) rejects urllib's default
# "Python-urllib/3.x" with 403 / error code 1010.
USER_AGENT = "fv-build-pod/1 (+scripts/dev/build-pod-server.py)"
RUNPOD_REST = os.environ.get("FV_BUILD_RUNPOD_REST", "https://rest.runpod.io/v1")
RUNPOD_GRAPHQL = os.environ.get("FV_BUILD_RUNPOD_GRAPHQL", "https://api.runpod.io/graphql")
MAX_JOBS = int(os.environ.get("FV_BUILD_MAX_JOBS", "4"))
EVICT_S = float(os.environ.get("FV_BUILD_EVICT_HOURS", "6")) * 3600  # 0 disables
EVICT_FREE_GB = float(os.environ.get("FV_BUILD_EVICT_FREE_GB", "40"))  # 0 disables
EVICT_INTERVAL_S = float(os.environ.get("FV_BUILD_EVICT_INTERVAL_S", "60"))
# Agents a multi-request client flow needs between its jobs: the release build
# (build-pod.sh release-artifacts) syncs, runs, then fetches its tarballs from
# target/fv-release/release-artifacts/. Kept within EVICT_HOLD_S of their last
# use (sync, job, fetch) as if busy, and evicted last under disk pressure.
EVICT_PROTECT = frozenset(os.environ.get("FV_BUILD_EVICT_PROTECT", "fv-release").split())
EVICT_HOLD_S = float(os.environ.get("FV_BUILD_EVICT_HOLD_MIN", "60")) * 60
# Sync and job requests are refused (507) below this much free disk, after an
# eviction pass could not free it.
MIN_FREE_GB = float(os.environ.get("FV_BUILD_MIN_FREE_GB", "2"))
SCCACHE_SIZE = os.environ.get("FV_BUILD_SCCACHE_SIZE", "40G")
SCCACHE_VER = os.environ.get("FV_BUILD_SCCACHE_VERSION", "v0.10.0")
CUDA_REDIST = os.environ.get("FV_BUILD_CUDA_REDIST", "13.4.2")
CUDA_DIR = os.path.join(ROOT, "cuda-13.4")
# Browser / client-compat extras (tests/compat/run.sh, tests/console/run.sh,
# FV_SERVE_UI=1): Node on the volume, Playwright's Chromium on the volume,
# ffmpeg + python3-venv + Chromium's system libraries in the container.
NODE_VER = os.environ.get("FV_BUILD_NODE_VERSION", "v22.23.3")
NODE_DIR = os.path.join(ROOT, "node-" + NODE_VER)
PLAYWRIGHT_VER = os.environ.get("FV_BUILD_PLAYWRIGHT_VERSION", "1.56.1")  # tests/compat/package.json
PW_DIR = os.path.join(ROOT, "playwright-" + PLAYWRIGHT_VER)
PW_BROWSERS = os.path.join(ROOT, "pw-browsers")
# Where snapshots, target dirs and CARGO_HOME live (see the docstring).
TARGETS_ON = os.environ.get("FV_BUILD_TARGETS", "local")
LOCAL = ROOT if TARGETS_ON == "volume" else os.environ.get("FV_BUILD_LOCAL", "/root/fvb")
WT_BASE = os.path.join(LOCAL, "worktrees")
TARGET_BASE = os.path.join(LOCAL, "target")
CARGO_HOME = os.path.join(LOCAL, "cargo")
LAST_USE = os.path.join(LOCAL, ".last-use")  # <agent>: mtime = last sync / job / fetch
MAX_UPLOAD = 512 << 20
MAX_ARTIFACT = 2 << 30
BOOT = time.time()
with open(os.path.abspath(__file__), "rb") as _f:
    SERVER_SHA = hashlib.sha256(_f.read()).hexdigest()[:12]

AGENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
CARGO_SUBCOMMANDS = {"check", "build", "test", "clippy", "fmt", "doc", "tree", "metadata"}
SCRIPTS = {
    "scripts/serve/check.sh",
    "scripts/gpu/lint.sh",
    "tests/compat/run.sh",
    "tests/console/run.sh",
}
# Jobs that wait for the browser extras (the second setup phase).
EXTRAS_SCRIPTS = {"tests/compat/run.sh", "tests/console/run.sh"}
ENV_ALLOW = re.compile(
    r"^(CUDARC_CUDA_VERSION|FV_[A-Z0-9_]+|RUST_LOG|RUST_BACKTRACE|RUSTFLAGS|RUSTDOCFLAGS"
    r"|CARGO_PROFILE_[A-Z0-9_]+|CARGO_INCREMENTAL|CARGO_BUILD_JOBS|CARGO_TERM_COLOR)$"
)
ENV_DENY = {"FV_BUILD_TOKEN_SHA256", "FV_BUILD_ROOT"}

state_lock = threading.Lock()
last_activity = time.time()
jobs = {}  # id -> Job
job_slots = threading.Semaphore(MAX_JOBS)
agent_locks = {}
setup_state = {"phase": "starting", "error": None, "ready": False, "log": []}
setup_done = threading.Event()
extras_state = {"phase": "pending", "error": None, "ready": False}
extras_done = threading.Event()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
        with open(os.path.join(ROOT, "logs", "pod.log"), "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def ledger(event):
    try:
        with open(os.path.join(ROOT, "ledger.tsv"), "a") as f:
            pod = os.environ.get("RUNPOD_POD_ID", "?")
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{pod}\t{event}\n")
    except OSError:
        pass


def touch():
    global last_activity
    with state_lock:
        last_activity = time.time()


# --------------------------------------------------------------------------- setup


def sh(cmd, **kw):
    """Run a setup command; on failure the error carries its output's tail
    (the pod's stdout is not reachable over the proxy)."""
    setup_state["log"].append(" ".join(cmd) if isinstance(cmd, list) else cmd)
    kw.pop("stdout", None)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw)
    if r.returncode != 0:
        tail = r.stdout.decode("utf-8", "replace")[-1500:]
        log(f"{' '.join(cmd)} exited {r.returncode}:\n{tail}")
        raise RuntimeError(f"{' '.join(cmd[:3])} exited {r.returncode}: {tail[-600:]}")
    return r


def job_env(agent=None):
    # Nothing Runpod-specific reaches jobs: besides the pod-scoped API key,
    # RUNPOD_POD_ID / RUNPOD_PUBLIC_IP / RUNPOD_TCP_PORT_* switch fv-serve's
    # WebRTC ICE and worker identity into "serving on a Runpod pod" mode,
    # which local tests (director, console) must not see.
    env = {k: v for k, v in os.environ.items() if not k.startswith("RUNPOD_")}
    env.pop("FV_BUILD_TOKEN_SHA256", None)
    env.update(
        {
            "CARGO_HOME": CARGO_HOME,
            "RUSTUP_HOME": os.path.join(ROOT, "rustup"),
            "CUDA_HOME": CUDA_DIR,
            "CUDA_PATH": CUDA_DIR,
            "CUDA_TOOLKIT_PATH": CUDA_DIR,
            "NVCC": os.path.join(CUDA_DIR, "bin", "nvcc"),
            "CUDARC_CUDA_VERSION": "13000",
            # Same release overrides as the CI builder (docker/gpucheck.Dockerfile):
            # what the published images ship, at a fraction of fat-LTO time.
            "CARGO_PROFILE_RELEASE_LTO": "off",
            "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "16",
            "CARGO_PROFILE_RELEASE_PANIC": "unwind",
            "CARGO_TERM_COLOR": "never",
            "SCCACHE_DIR": os.path.join(ROOT, "sccache"),
            "SCCACHE_CACHE_SIZE": SCCACHE_SIZE,
            "SCCACHE_IDLE_TIMEOUT": "0",
            "PLAYWRIGHT_BROWSERS_PATH": PW_BROWSERS,
            "NODE_PATH": os.path.join(PW_DIR, "node_modules"),
        }
    )
    path = [
        os.path.join(CARGO_HOME, "bin"),
        "/usr/local/cargo/bin",
        os.path.join(CUDA_DIR, "bin"),
        os.path.join(ROOT, "tools"),
        os.path.join(NODE_DIR, "bin"),
    ]
    env["PATH"] = ":".join(path + [env.get("PATH", "/usr/local/bin:/usr/bin:/bin")])
    lib = os.path.join(CUDA_DIR, "lib64")
    env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    sccache = os.path.join(ROOT, "tools", "sccache")
    if os.path.exists(sccache) and os.environ.get("FV_BUILD_SCCACHE", "1") == "1":
        env["RUSTC_WRAPPER"] = sccache
    if shutil.which("mold"):
        env["CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS"] = "-C link-arg=-fuse-ld=mold"
    if agent:
        env["CARGO_TARGET_DIR"] = os.path.join(TARGET_BASE, agent)
    return env


def fetch(url, dest):
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(tmp, dest)


def install_cuda():
    marker = os.path.join(CUDA_DIR, ".fv-redist-" + CUDA_REDIST)
    if os.path.exists(marker):
        return
    setup_state["phase"] = "cuda"
    base = "https://developer.download.nvidia.com/compute/cuda/redist/"
    with urllib.request.urlopen(base + f"redistrib_{CUDA_REDIST}.json", timeout=60) as r:
        index = json.load(r)
    stage = CUDA_DIR + ".staging"
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    dl = os.path.join(ROOT, "tmp")
    os.makedirs(dl, exist_ok=True)
    for comp in ("cuda_nvcc", "cuda_crt", "cuda_cudart", "cuda_nvrtc", "libnvvm", "cccl", "cuda_tileiras"):
        rel = index[comp]["linux-x86_64"]["relative_path"]
        want = index[comp]["linux-x86_64"]["sha256"]
        path = os.path.join(dl, os.path.basename(rel))
        fetch(base + rel, path)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != want:
            raise RuntimeError(f"sha256 mismatch for {comp}")
        sh(["tar", "-xJf", path, "-C", stage, "--strip-components=1"])
        os.remove(path)
    lib = os.path.join(stage, "lib")
    if os.path.isdir(lib) and not os.path.exists(os.path.join(stage, "lib64")):
        os.symlink("lib", os.path.join(stage, "lib64"))
    shutil.rmtree(CUDA_DIR, ignore_errors=True)
    os.replace(stage, CUDA_DIR)
    open(marker, "w").close()
    log(f"cuda redist {CUDA_REDIST} installed at {CUDA_DIR}")


def install_sccache():
    dest = os.path.join(ROOT, "tools", "sccache")
    ver_marker = dest + "." + SCCACHE_VER
    if os.path.exists(ver_marker):
        return
    setup_state["phase"] = "sccache"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    name = f"sccache-{SCCACHE_VER}-x86_64-unknown-linux-musl"
    url = f"https://github.com/mozilla/sccache/releases/download/{SCCACHE_VER}/{name}.tar.gz"
    tgz = os.path.join(ROOT, "tmp", name + ".tar.gz")
    os.makedirs(os.path.dirname(tgz), exist_ok=True)
    try:
        fetch(url, tgz)
        with tarfile.open(tgz) as t:
            member = t.getmember(f"{name}/sccache")
            with t.extractfile(member) as src, open(dest + ".part", "wb") as out:
                shutil.copyfileobj(src, out)
        os.chmod(dest + ".part", 0o755)
        os.replace(dest + ".part", dest)
        open(ver_marker, "w").close()
    except Exception as e:  # sccache is optional
        log(f"sccache install skipped: {e}")
    finally:
        if os.path.exists(tgz):
            os.remove(tgz)


def setup():
    try:
        for d in ("rustup", "sccache", "tools", "jobs", "logs", "tmp"):
            os.makedirs(os.path.join(ROOT, d), exist_ok=True)
        link_cargo_caches()
        setup_state["phase"] = "apt"
        need = [p for p, b in (("cmake", "cmake"), ("clang", "clang"), ("mold", "mold"), ("pkg-config", "pkg-config")) if not shutil.which(b)]
        if need:
            env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
            sh(["apt-get", "update", "-qq"], env=env, stdout=subprocess.DEVNULL)
            sh(["apt-get", "install", "-y", "-qq", "--no-install-recommends", *need, "libssl-dev", "xz-utils"],
               env=env, stdout=subprocess.DEVNULL)
        install_cuda()
        install_sccache()
        setup_state["phase"] = "rustup"
        env = job_env()
        marker = os.path.join(ROOT, "rustup", ".fv-stable-ok")
        if not os.path.exists(marker):
            # --no-self-update: rustup lives in the image, not in CARGO_HOME/bin on
            # the volume, and its self-update step fails the install otherwise.
            # Idempotent, so a boot interrupted mid-install just completes it.
            sh(["rustup", "toolchain", "install", "stable", "--profile", "minimal", "-c", "rustfmt", "-c", "clippy",
                "--no-self-update"], env=env)
            open(marker, "w").close()
        sh(["rustup", "default", "stable"], env=env, stdout=subprocess.DEVNULL)
        prune_targets()
        setup_state["phase"] = "ready"
        setup_state["ready"] = True
        log(f"setup complete after {time.time() - BOOT:.0f}s")
    except Exception as e:
        setup_state["phase"] = "failed"
        setup_state["error"] = str(e)
        log(f"setup failed: {e}")
    finally:
        setup_done.set()
    if setup_state["ready"]:
        setup_extras()
    else:
        extras_state.update(phase="skipped", error="setup failed")
        extras_done.set()


def install_node():
    marker = os.path.join(NODE_DIR, ".fv-ok")
    if os.path.exists(marker):
        return
    extras_state["phase"] = "node"
    base = f"https://nodejs.org/dist/{NODE_VER}/"
    name = f"node-{NODE_VER}-linux-x64.tar.xz"
    with urllib.request.urlopen(base + "SHASUMS256.txt", timeout=60) as r:
        sums = {ln.split()[1]: ln.split()[0] for ln in r.read().decode().splitlines() if len(ln.split()) == 2}
    path = os.path.join(ROOT, "tmp", name)
    fetch(base + name, path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != sums.get(name):
        raise RuntimeError(f"sha256 mismatch for {name}")
    stage = NODE_DIR + ".staging"
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    sh(["tar", "-xJf", path, "-C", stage, "--strip-components=1"])
    os.remove(path)
    shutil.rmtree(NODE_DIR, ignore_errors=True)
    os.replace(stage, NODE_DIR)
    open(marker, "w").close()
    log(f"node {NODE_VER} installed at {NODE_DIR}")


def setup_extras():
    """Second phase, after the build toolchain is ready: what the browser and
    client-compat jobs need. Build jobs never wait for it."""
    try:
        t0 = time.time()
        env = dict(job_env(), DEBIAN_FRONTEND="noninteractive")
        extras_state["phase"] = "apt"
        sh(["apt-get", "update", "-qq"], env=env, stdout=subprocess.DEVNULL)
        sh(["apt-get", "install", "-y", "-qq", "--no-install-recommends", "ffmpeg", "python3-venv"],
           env=env, stdout=subprocess.DEVNULL)
        install_node()
        extras_state["phase"] = "playwright"
        if not os.path.exists(os.path.join(PW_DIR, ".fv-ok")):
            os.makedirs(PW_DIR, exist_ok=True)
            sh(["npm", "install", "--prefix", PW_DIR, "--no-audit", "--no-fund", "--loglevel=error",
                f"playwright@{PLAYWRIGHT_VER}"], env=dict(env, PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD="1"),
               stdout=subprocess.DEVNULL)
            open(os.path.join(PW_DIR, ".fv-ok"), "w").close()
        pw = os.path.join(PW_DIR, "node_modules", ".bin", "playwright")
        extras_state["phase"] = "chromium"
        sh([pw, "install", "chromium"], env=env, stdout=subprocess.DEVNULL)  # no-op when present
        sh([pw, "install-deps", "chromium"], env=env, stdout=subprocess.DEVNULL)
        extras_state.update(phase="ready", ready=True)
        log(f"extras ready after {time.time() - t0:.0f}s (ffmpeg, node, playwright chromium)")
    except Exception as e:
        extras_state.update(phase="failed", error=str(e))
        log(f"extras setup failed: {e}")
    finally:
        extras_done.set()


def link_cargo_caches():
    """CARGO_HOME on the container disk, with the download caches (registry
    .crate files, git databases) on the volume: crates download once, while
    the many small files cargo extracts and stats stay local."""
    for d in (WT_BASE, TARGET_BASE, CARGO_HOME, os.path.join(LOCAL, "trash")):
        os.makedirs(d, exist_ok=True)
    if CARGO_HOME == os.path.join(ROOT, "cargo"):
        return
    for rel in (("registry", "cache"), ("git", "db")):
        vol = os.path.join(ROOT, "cargo", *rel)
        loc = os.path.join(CARGO_HOME, *rel)
        os.makedirs(vol, exist_ok=True)
        os.makedirs(os.path.dirname(loc), exist_ok=True)
        if not os.path.islink(loc):
            shutil.rmtree(loc, ignore_errors=True)
            os.symlink(vol, loc)


def prune_targets():
    """An eviction pass, and the snapshots and target dirs an earlier
    volume-mode pod left on the volume."""
    evictor.run_once()
    if LOCAL != ROOT:
        # Snapshots and target dirs a volume-mode pod left behind are dead
        # weight for a local-mode one (tens of GB after a few check.sh runs).
        for d in ("target", "worktrees"):
            old = os.path.join(ROOT, d)
            if os.path.isdir(old) and os.listdir(old):
                log(f"removing volume-mode {d}/ (targets are local now)")
                trash = os.path.join(ROOT, "tmp", f"trash-{d}-{uuid.uuid4().hex[:6]}")
                os.replace(old, trash)
                threading.Thread(target=shutil.rmtree, args=(trash, True), daemon=True).start()


# --------------------------------------------------------------------------- eviction


def agent_lock(agent):
    return agent_locks.setdefault(agent, threading.Lock())


def agent_busy(agent):
    return any(j.agent == agent and j.state in ("running", "queued") for j in list(jobs.values()))


def mark_used(agent):
    """Record that `agent` used its dirs now (sync, job start/end, fetch)."""
    try:
        os.makedirs(LAST_USE, exist_ok=True)
        p = os.path.join(LAST_USE, agent)
        with open(p, "a"):
            pass
        os.utime(p)
    except OSError:
        pass


def last_use(agent):
    """Epoch of the agent's last use: its stamp, else (a pre-stamp or
    restarted server) its old worktree stamp or the dirs' mtimes."""
    for p in (os.path.join(LAST_USE, agent), os.path.join(WT_BASE, agent, ".fv-build-last-run"),
              os.path.join(WT_BASE, agent), os.path.join(TARGET_BASE, agent)):
        try:
            return os.path.getmtime(p)
        except OSError:
            pass
    return 0.0


def move_to_trash(d, agent):
    """Rename `d` out of the way (same filesystem, instant); returns the
    trash path to delete, or None when `d` does not exist."""
    if not os.path.isdir(d):
        return None
    trash = os.path.join(LOCAL, "trash" if LOCAL != ROOT else "tmp", f"trash-{agent}-{uuid.uuid4().hex[:6]}")
    os.replace(d, trash)
    return trash


KIND_DIR = {"target": "target", "worktree": "worktrees"}


class LocalFS:
    """The container disk as the Evictor sees it (tests use a fake)."""

    def path(self, agent, kind):
        return os.path.join(TARGET_BASE if kind == "target" else WT_BASE, agent)

    def agents(self):
        names = set()
        for d in (WT_BASE, TARGET_BASE):
            try:
                names.update(n for n in os.listdir(d) if AGENT_RE.match(n))
            except OSError:
                pass
        return names

    def has(self, agent, kind):
        return os.path.isdir(self.path(agent, kind))

    def last_use(self, agent):
        return last_use(agent)

    def busy(self, agent):
        return agent_busy(agent)

    def free_bytes(self):
        return shutil.disk_usage(LOCAL).free

    def remove(self, agent, kind):
        """Delete the dir unless the agent has a job in flight (whose runner
        holds the agent lock). Synchronous, so free_bytes() sees the result.
        Only ever a direct child of target/ or worktrees/ on the container
        disk (never the volume's caches or release-cache/)."""
        base = TARGET_BASE if kind == "target" else WT_BASE
        if not AGENT_RE.match(agent) or os.path.dirname(os.path.realpath(self.path(agent, kind))) != os.path.realpath(base):
            return False
        lock = agent_lock(agent)
        if not lock.acquire(blocking=False):
            return False
        try:
            if agent_busy(agent):
                return False
            trash = move_to_trash(self.path(agent, kind), agent)
        finally:
            lock.release()
        if trash:
            shutil.rmtree(trash, ignore_errors=True)
        agent_sizes.pop(agent, None)
        return True


class Evictor:
    """Frees the container disk. One pass: (1) the target dir and snapshot of
    every agent idle for more than `evict_s`; (2) while free space is under
    `min_free` bytes, target dirs in least-recently-used order (snapshots are
    ~0.1 GB, not worth a rebuild's sync). Agents with a job in flight are
    skipped, and fs.remove re-checks that under the agent lock. A `protect`ed
    agent used within `hold_s` is skipped too (a release build between its
    job and its fetches), and goes last in the LRU order."""

    def __init__(self, fs, clock=time.time, evict_s=EVICT_S, min_free=EVICT_FREE_GB * 1e9, log=log,
                 protect=EVICT_PROTECT, hold_s=EVICT_HOLD_S):
        self.fs, self.clock, self.evict_s, self.min_free, self.log = fs, clock, evict_s, min_free, log
        self.protect, self.hold_s = frozenset(protect), hold_s
        self.recent = collections.deque(maxlen=50)
        self.last_pass = None
        self.warned = float("-inf")
        self.lock = threading.Lock()

    def held(self, agent, now):
        """A protected agent used within the hold (its flow may continue)."""
        return agent in self.protect and now - self.fs.last_use(agent) < self.hold_s

    def keep(self, agent, now):
        return self.fs.busy(agent) or self.held(agent, now)

    def run_once(self):
        with self.lock:
            done = []
            now = self.clock()
            if self.evict_s > 0:
                for agent in sorted(self.fs.agents()):
                    idle = now - self.fs.last_use(agent)
                    if idle <= self.evict_s or self.keep(agent, now):
                        continue
                    for kind in ("target", "worktree"):
                        if self.fs.has(agent, kind):
                            self._evict(agent, kind, f"unused {idle / 3600:.1f} h > {self.evict_s / 3600:g} h", done)
            if self.min_free > 0 and self.fs.free_bytes() < self.min_free:
                lru = sorted((a for a in self.fs.agents() if self.fs.has(a, "target") and not self.keep(a, now)),
                             key=lambda a: (a in self.protect, self.fs.last_use(a), a))
                for agent in lru:
                    free = self.fs.free_bytes()
                    if free >= self.min_free:
                        break
                    self._evict(agent, "target",
                                f"disk pressure: {free / 1e9:.1f} GB free < {self.min_free / 1e9:g} GB", done)
                if self.fs.free_bytes() < self.min_free and (done or now - self.warned > 600):
                    self.warned = now
                    self.log(f"eviction: still {self.fs.free_bytes() / 1e9:.1f} GB free "
                             f"(< {self.min_free / 1e9:g} GB); the rest is in use")
            self.last_pass = self.clock()
            return done

    def _evict(self, agent, kind, reason, done):
        before = self.fs.free_bytes()
        if not self.fs.remove(agent, kind):
            self.log(f"eviction skipped {KIND_DIR[kind]}/{agent}: job in flight")
            return
        freed = max(0, self.fs.free_bytes() - before)
        ev = {"t": round(self.clock()), "agent": agent, "what": kind, "reason": reason,
              "freed_gb": round(freed / 1e9, 1)}
        self.recent.append(ev)
        done.append(ev)
        self.log(f"evicted {KIND_DIR[kind]}/{agent} ({reason}); freed {freed / 1e9:.1f} GB")


evictor = Evictor(LocalFS())
evict_wake = threading.Event()


def evict_loop():
    while True:
        evict_wake.wait(EVICT_INTERVAL_S)
        evict_wake.clear()
        try:
            evictor.run_once()
        except Exception as e:  # keep the loop alive
            log(f"eviction pass failed: {e!r}")


# Per-agent dir sizes for /v1/status (du is too slow to run per request).
agent_sizes = {}


def agent_sizes_loop():
    while True:
        for a in sorted(LocalFS().agents()):
            t = du(os.path.join(TARGET_BASE, a), timeout=300)
            w = du(os.path.join(WT_BASE, a), timeout=60)
            agent_sizes[a] = {"target_gb": round(t / 1e9, 1) if t is not None else None,
                              "worktree_gb": round(w / 1e9, 2) if w is not None else None, "t": time.time()}
        time.sleep(300)


class DiskFull(Exception):
    pass


def disk_full_msg(free=None):
    if free is None:
        free = shutil.disk_usage(LOCAL).free
    return (f"build pod disk full: {free / 1e9:.1f} GB free on the container disk after an eviction pass "
            f"(dirs of agents with a job in flight are never evicted). See build-pod.sh status (agents, sizes); "
            f"free space with build-pod.sh clean <agent> target, or retry when running jobs finish")


def ensure_free(need_bytes):
    """Raise DiskFull unless `need_bytes` are free, running an eviction pass
    first when they are not."""
    if shutil.disk_usage(LOCAL).free >= need_bytes:
        return
    evictor.run_once()
    free = shutil.disk_usage(LOCAL).free
    if free < need_bytes:
        log(f"refusing request: {free / 1e9:.1f} GB free < {need_bytes / 1e9:.1f} GB")
        raise DiskFull(disk_full_msg(free))


# --------------------------------------------------------------------------- jobs


class Job:
    def __init__(self, agent, argv, env_over):
        self.id = time.strftime("%m%d%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
        self.agent = agent
        self.argv = argv
        self.env_over = env_over
        self.state = "queued"
        self.exit = None
        self.started = None
        self.ended = None
        self.proc = None
        self.log_path = os.path.join(ROOT, "jobs", self.id + ".log")
        self.cond = threading.Condition()
        open(self.log_path, "wb").close()

    def info(self):
        dur = None
        if self.started:
            dur = round((self.ended or time.time()) - self.started, 1)
        return {"id": self.id, "agent": self.agent, "argv": self.argv, "state": self.state,
                "exit": self.exit, "seconds": dur}

    def write(self, b):
        with open(self.log_path, "ab") as f:
            f.write(b)
        with self.cond:
            self.cond.notify_all()

    def run(self):
        lock = agent_lock(self.agent)
        setup_done.wait()
        if not setup_state["ready"]:
            self.write(f"build pod setup failed: {setup_state['error']}\n".encode())
            self.finish(125)
            return
        if needs_extras(self.argv, self.env_over):
            if not extras_done.is_set():
                self.write(f"waiting for the browser extras (phase {extras_state['phase']})\n".encode())
            extras_done.wait()
            if not extras_state["ready"]:
                self.write(f"browser extras setup failed: {extras_state['error']}\n".encode())
                self.finish(125)
                return
        with lock, job_slots:
            if self.state == "cancelled":
                self.finish(130)
                return
            wt = os.path.join(WT_BASE, self.agent)
            if not os.path.isdir(wt):
                # Evicted between the sync and this job (disk pressure).
                self.write(f"agent {self.agent} has no worktree (evicted?); run again, which syncs first\n".encode())
                self.finish(125)
                return
            env = job_env(self.agent)
            env.update(self.env_over)
            # An evicted target dir is simply rebuilt from scratch.
            os.makedirs(env["CARGO_TARGET_DIR"], exist_ok=True)
            self.state = "running"
            self.started = time.time()
            touch()
            mark_used(self.agent)
            self.write(f"$ {' '.join(self.argv)}   [agent {self.agent}, CARGO_TARGET_DIR={env['CARGO_TARGET_DIR']}]\n".encode())
            try:
                self.proc = subprocess.Popen(self.argv, cwd=wt, env=env, stdin=subprocess.DEVNULL,
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                for chunk in iter(lambda: self.proc.stdout.read1(65536), b""):
                    self.write(chunk)
                    touch()
                rc = self.proc.wait()
            except OSError as e:
                self.write(f"failed to start: {e}\n".encode())
                rc = 127
            self.write(f"\n[exit {rc} after {time.time() - self.started:.1f}s]\n".encode())
            try:
                free = shutil.disk_usage(LOCAL).free
                if rc != 0 and free < MIN_FREE_GB * 1e9 * 2:
                    self.write(f"[{disk_full_msg(free)}]\n".encode())
            except OSError:
                pass
            mark_used(self.agent)
            self.finish(rc)
            evict_wake.set()

    def finish(self, rc):
        self.exit = rc
        self.ended = time.time()
        if self.state != "cancelled":
            self.state = "done"
        touch()
        with self.cond:
            self.cond.notify_all()

    def cancel(self):
        if self.state == "queued":
            self.state = "cancelled"
        elif self.state == "running" and self.proc:
            self.state = "cancelled"
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def needs_extras(argv, env):
    return (len(argv) >= 2 and argv[0] == "bash" and os.path.normpath(argv[1]) in EXTRAS_SCRIPTS) \
        or env.get("FV_SERVE_UI") == "1"


def validate_command(argv, env):
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        return "argv must be a non-empty list of strings"
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        return "env must map strings to strings"
    for k in env:
        if k in ENV_DENY or not ENV_ALLOW.match(k):
            return f"env {k} is not allowed"
    if argv[0] == "cargo":
        sub = next((a for a in argv[1:] if not a.startswith("+")), None)
        if sub not in CARGO_SUBCOMMANDS:
            return f"cargo {sub} is not allowed (allowed: {', '.join(sorted(CARGO_SUBCOMMANDS))})"
        for a in argv[1:]:
            if a.startswith(("--target-dir", "--manifest-path")):
                return f"argument {a} is not allowed"
        return None
    if argv[0] == "bash" and len(argv) >= 2 and os.path.normpath(argv[1]) in SCRIPTS:
        return None
    return f"not allowed: {' '.join(argv[:3])} (cargo {'/'.join(sorted(CARGO_SUBCOMMANDS))} or bash one of {sorted(SCRIPTS)})"


# --------------------------------------------------------------------------- files


def safe_rel(p):
    p = p.replace("\\", "/")
    if not p or p.startswith("/") or "\x00" in p:
        return None
    parts = [x for x in p.split("/") if x not in ("", ".")]
    if not parts or any(x == ".." for x in parts):
        return None
    return "/".join(parts)


def manifest(wt):
    out = []
    for dirpath, dirnames, filenames in os.walk(wt):
        rel_dir = os.path.relpath(dirpath, wt)
        for name in filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
            full = os.path.join(dirpath, name)
            rel = name if rel_dir == "." else os.path.join(rel_dir, name)
            if rel == ".fv-build-last-run":
                continue
            st = os.lstat(full)
            out.append(f"{rel}\t{st.st_size}\t{int(st.st_mtime)}")
    return "\n".join(sorted(out)) + ("\n" if out else "")


def extract(tar_path, wt):
    n = 0
    real_wt = os.path.realpath(wt)
    with tarfile.open(tar_path, "r:*") as t:
        for m in t:
            rel = safe_rel(m.name)
            if rel is None:
                raise ValueError(f"unsafe path {m.name!r}")
            if not (m.isfile() or m.isdir() or m.issym()):
                raise ValueError(f"unsupported member type {m.name!r}")
            if m.issym():
                target = os.path.normpath(os.path.join(os.path.dirname(os.path.join(real_wt, rel)), m.linkname))
                if not (target == real_wt or target.startswith(real_wt + os.sep)):
                    raise ValueError(f"symlink escapes worktree: {m.name!r}")
            dest = os.path.join(real_wt, rel)
            parent = os.path.realpath(os.path.dirname(dest))
            if not (parent == real_wt or parent.startswith(real_wt + os.sep)):
                raise ValueError(f"path escapes worktree: {m.name!r}")
            if not m.isdir() and (os.path.islink(dest) or os.path.isfile(dest)):
                os.unlink(dest)
            elif not m.isdir() and os.path.isdir(dest):
                shutil.rmtree(dest)
            m.name = rel
            m.uid = m.gid = 0
            m.uname = m.gname = ""
            m.mode = (m.mode & 0o755) | 0o600
            t.extract(m, real_wt, set_attrs=True)
            n += 1
    return n


def delete_paths(wt, paths):
    n = 0
    real_wt = os.path.realpath(wt)
    for p in paths:
        rel = safe_rel(p)
        if rel is None:
            continue
        full = os.path.join(real_wt, rel)
        if not os.path.realpath(os.path.dirname(full)).startswith(real_wt):
            continue
        try:
            if os.path.islink(full) or os.path.isfile(full):
                os.unlink(full)
                n += 1
        except OSError:
            pass
        d = os.path.dirname(full)
        while d.startswith(real_wt + os.sep):
            try:
                os.rmdir(d)
            except OSError:
                break
            d = os.path.dirname(d)
    return n


def disk_gb(path):
    try:
        d = shutil.disk_usage(path)
        return {"total_gb": round(d.total / 1e9, 1), "free_gb": round(d.free / 1e9, 1)}
    except OSError:
        return None


def cgroup_limits():
    """(vCPUs, memory MiB) from the container's cgroup (v2, then v1); the
    host's numbers when unlimited."""
    try:  # the CPUs this container may run on (what cargo sizes -j by)
        vcpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        vcpus = os.cpu_count()
    mem = None
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            q, period = f.read().split()
        if q != "max":
            vcpus = min(vcpus, round(int(q) / int(period), 1))
    except (OSError, ValueError):
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
                q = int(f.read())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                period = int(f.read())
            if q > 0:
                vcpus = min(vcpus, round(q / period, 1))
        except (OSError, ValueError):
            pass
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                v = f.read().strip()
            if v != "max" and int(v) < 1 << 50:
                mem = int(v) >> 20
                break
        except (OSError, ValueError):
            pass
    if mem is None:
        try:
            with open("/proc/meminfo") as f:
                mem = int(f.readline().split()[1]) // 1024
        except OSError:
            pass
    return vcpus, mem


# The volume's quota is not visible from the pod (statfs shows the whole
# shared filesystem behind it), so status reports our usage from a periodic du.
VOLUME_GB = float(os.environ.get("FV_BUILD_VOLUME_GB", "200"))
volume_usage = {}


def volume_usage_loop():
    while True:
        t = time.time()
        n = du(ROOT, timeout=900)
        if n is not None:
            volume_usage.update(used_gb=round(n / 1e9, 1), t=t)
        time.sleep(900)


def du(path, timeout=60):
    try:
        r = subprocess.run(["du", "-sb", path], capture_output=True, text=True, timeout=timeout)
        return int(r.stdout.split()[0])
    except Exception:
        return None


# --------------------------------------------------------------------------- self-stop


def runpod_call(method, url, body=None):
    """One Runpod API call with the pod-scoped key; returns the parsed JSON
    (or None). Raises RuntimeError carrying the HTTP status and the start of
    the body (never the key) on failure, including GraphQL errors."""
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RuntimeError("no RUNPOD_API_KEY in the pod environment")
    data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
    req = urllib.request.Request(url, method=method, data=data, headers={
        "Authorization": "Bearer " + key, "content-type": "application/json", "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            text = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:300].strip()
        except Exception:
            detail = ""
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {detail}") from None
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"network: {e}") from None
    try:
        out = json.loads(text) if text.strip() else None
    except ValueError:
        out = None
    if isinstance(out, dict) and out.get("errors"):
        raise RuntimeError("graphql: " + "; ".join(str(x.get("message", x)) for x in out["errors"])[:300])
    return out


def stop_calls(pod):
    """The stop sequence, first accepted call wins. REST stop keeps the pod
    record (a later `up` starts or recreates it); terminate drops it; the
    GraphQL mutations are what runpodctl uses with the same key."""
    q = json.dumps(pod)
    return [
        ("REST stop", "POST", f"{RUNPOD_REST}/pods/{pod}/stop", None),
        ("REST terminate", "DELETE", f"{RUNPOD_REST}/pods/{pod}", None),
        ("GraphQL podStop", "POST", RUNPOD_GRAPHQL,
         {"query": f"mutation {{ podStop(input: {{podId: {q}}}) {{ id desiredStatus }} }}"}),
        ("GraphQL podTerminate", "POST", RUNPOD_GRAPHQL,
         {"query": f"mutation {{ podTerminate(input: {{podId: {q}}}) }}"}),
    ]


class StopPolicy:
    """When the pod stops itself. Pure (the clock is an argument), so tests
    drive it with a fake one."""

    def __init__(self, boot, idle_s, max_s, grace_s):
        self.boot, self.idle_s, self.max_s, self.grace_s = boot, idle_s, max_s, grace_s

    def past_cap(self, now):
        return now - self.boot >= self.max_s

    def decide(self, now, last_activity, active):
        """A reason to stop now, or None. `active`: queued + running jobs."""
        up = now - self.boot
        if up >= self.max_s:
            if not active:
                return f"wall-clock cap {self.max_s / 3600:g}h"
            if up >= self.max_s + self.grace_s:
                return f"wall-clock cap {self.max_s / 3600:g}h + {self.grace_s / 60:g} min grace ({active} job(s) still active)"
            return None
        if not active and now - last_activity >= self.idle_s:
            return f"idle {(now - last_activity) / 60:.0f} min"
        return None

    def timers(self, now, last_activity, active):
        """Seconds idle, and until the idle and the cap stop (None: not
        counting down: jobs are active)."""
        up = now - self.boot
        idle = 0 if active else now - last_activity
        cap_at = self.max_s + (self.grace_s if active else 0)
        return {
            "uptime_s": round(up),
            "idle_s": round(idle),
            "idle_stop_in_s": None if active else max(0, round(self.idle_s - idle)),
            "max_stop_in_s": max(0, round(cap_at - up)),
        }


class Stopper:
    """Runs the stop sequence; on failure retries 1, 2, 4 ... 10 min apart,
    and 10 min after an accepted call if the pod is still running then."""

    RETRY_MIN_S, RETRY_MAX_S, AFTER_OK_S = 60, 600, 600

    def __init__(self, call=runpod_call, pod=None):
        self.call = call
        self.pod = pod
        self.lock = threading.Lock()
        self.attempts = 0
        self.failures = 0
        self.next_at = 0.0
        self.last = {}

    def due(self, now):
        return now >= self.next_at

    def info(self):
        return {"attempts": self.attempts, "next_at": round(self.next_at) or None, **self.last}

    def attempt(self, reason, now):
        if not self.lock.acquire(blocking=False):
            return False
        try:
            pod = self.pod or os.environ.get("RUNPOD_POD_ID")
            self.attempts += 1
            log(f"self-stop ({reason}) pod={pod} attempt {self.attempts}")
            ledger(f"self-stop {reason}")
            errors = []
            if pod:
                for name, method, url, body in stop_calls(pod):
                    try:
                        self.call(method, url, body)
                    except Exception as e:
                        errors.append(f"{name}: {e}")
                        log(f"self-stop: {name} failed: {e}")
                        continue
                    ledger(f"self-stop-ok {name}")
                    log(f"self-stop: {name} accepted")
                    self.failures = 0
                    self.next_at = now + self.AFTER_OK_S
                    self.last = {"reason": reason, "at": round(now), "ok": name, "error": None}
                    return True
            else:
                errors.append("no RUNPOD_POD_ID")
            self.failures += 1
            delay = min(self.RETRY_MAX_S, self.RETRY_MIN_S * 2 ** (self.failures - 1))
            self.next_at = now + delay
            self.last = {"reason": reason, "at": round(now), "ok": None, "error": "; ".join(errors)[-600:]}
            ledger(f"self-stop-failed retry-in={delay:.0f}s")
            log(f"self-stop failed (attempt {self.attempts}); retrying in {delay:.0f}s")
            return False
        finally:
            self.lock.release()


POLICY = StopPolicy(BOOT, IDLE_S, MAX_S, MAX_GRACE_S)
STOPPER = Stopper()


def active_jobs():
    return [j for j in list(jobs.values()) if j.state in ("running", "queued")]


def public_jobs():
    """The active jobs for /healthz: id, agent, state and seconds only."""
    return [{k: i[k] for k in ("id", "agent", "state", "seconds")} for i in (j.info() for j in active_jobs())]


def public_timers(now):
    act = active_jobs()
    return {**POLICY.timers(now, last_activity, len(act)), "jobs_active": len(act), "idle_stop_s": IDLE_S,
            "max_s": MAX_S, "max_grace_s": MAX_GRACE_S}


def note_jobs(act, msg):
    for j in act:
        try:
            j.write(f"\n[build pod] {msg}\n".encode())
        except OSError:
            pass


cap_warned = set()


def watch_tick(now):
    act = active_jobs()
    if POLICY.past_cap(now):
        new = [j for j in act if j.id not in cap_warned]
        if new:
            left = POLICY.boot + POLICY.max_s + POLICY.grace_s - now
            note_jobs(new, f"the pod passed its {POLICY.max_s / 3600:g} h cap: it stops in {max(0, left) / 60:.0f} min "
                           "even if this job is still running; new jobs are refused (build-pod.sh up after it is gone)")
            log(f"past the {POLICY.max_s / 3600:g} h cap with {len(act)} active job(s); hard stop in {max(0, left) / 60:.0f} min")
            cap_warned.update(j.id for j in new)
    reason = POLICY.decide(now, last_activity, len(act))
    if reason and STOPPER.due(now):
        if act:
            note_jobs(act, f"stopping the pod now ({reason}); this job is killed")
        STOPPER.attempt(reason, now)
    return reason


def watchdog():
    while True:
        time.sleep(30)
        try:
            watch_tick(time.time())
        except Exception as e:  # never let the watchdog thread die
            log(f"watchdog: {e!r}")


def tail_file(path, lines):
    lines = max(1, min(lines, 5000))
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - lines * 300))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    return "\n".join(data.splitlines()[-lines:]) + "\n"


# --------------------------------------------------------------------------- http


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "fv-build/1"

    def log_message(self, fmt, *args):
        pass

    def send_json(self, code, obj):
        body = (json.dumps(obj) + "\n").encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, code, text):
        body = text.encode()
        self.send_response(code)
        self.send_header("content-type", "text/plain; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authed(self):
        h = self.headers.get("authorization", "")
        if not TOKEN_SHA or not h.startswith("Bearer "):
            return False
        got = hashlib.sha256(h[7:].strip().encode()).hexdigest()
        return hmac.compare_digest(got, TOKEN_SHA)

    def body(self, limit=1 << 20):
        n = int(self.headers.get("content-length") or 0)
        if n > limit:
            raise ValueError(f"body too large ({n} > {limit})")
        return self.rfile.read(n) if n else b""

    def route(self, method):
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        parts = [p for p in u.path.split("/") if p]
        if method == "GET" and parts == ["healthz"]:
            # Timers without auth, for external backstops and fv-control's
            # build pod card (docs/dev/build-pod.md): the last self-stop
            # attempt and the active jobs (no argv, logs or paths).
            return self.send_json(200, {"ok": True, "ready": setup_state["ready"], "phase": setup_state["phase"],
                                        "boot": round(BOOT), **public_timers(time.time()),
                                        "self_stop": STOPPER.info(), "jobs": public_jobs()})
        if not self.authed():
            return self.send_json(401, {"error": "unauthorized"})
        # No touch() here: only job events and work requests (sync, job
        # submit, artifact, clean) count as activity, never status polls.
        if parts[:1] != ["v1"]:
            return self.send_json(404, {"error": "not found"})
        parts = parts[1:]
        if method == "GET" and parts == ["status"]:
            return self.status()
        if method == "POST" and parts == ["stop"]:
            threading.Thread(target=STOPPER.attempt, args=("requested", time.time()), daemon=True).start()
            return self.send_json(202, {"stopping": True})
        if method == "GET" and parts == ["log"]:
            return self.send_text(200, tail_file(os.path.join(ROOT, "logs", "pod.log"), int(q.get("lines", "200"))))
        if method == "GET" and parts == ["agents"]:
            return self.agents(q.get("sizes") == "1")
        if method == "POST" and parts == ["evict"]:
            return self.send_json(200, {"evicted": evictor.run_once()})
        if len(parts) >= 2 and parts[0] == "jobs":
            job = jobs.get(parts[1])
            if not job:
                return self.send_json(404, {"error": "no such job"})
            if method == "GET" and len(parts) == 2:
                return self.job_poll(job, int(q.get("offset", "0")), min(float(q.get("wait", "20")), 25))
            if method == "POST" and parts[2:] == ["cancel"]:
                job.cancel()
                return self.send_json(200, job.info())
        if len(parts) >= 3 and parts[0] == "agents":
            agent = parts[1]
            if not AGENT_RE.match(agent):
                return self.send_json(400, {"error": "bad agent name"})
            wt = os.path.join(WT_BASE, agent)
            action = parts[2]
            if action in ("manifest", "files", "delete", "jobs", "artifact"):
                mark_used(agent)
            if method == "GET" and action == "manifest":
                return self.send_text(200, manifest(wt) if os.path.isdir(wt) else "")
            if method == "PUT" and action == "files":
                touch()
                return self.put_files(agent, wt)
            if method == "POST" and action == "delete":
                touch()
                paths = [p for p in self.body(64 << 20).decode().split("\0") if p]
                with agent_lock(agent):
                    n = delete_paths(wt, paths) if os.path.isdir(wt) else 0
                return self.send_json(200, {"deleted": n})
            if method == "POST" and action == "jobs":
                return self.new_job(agent, wt)
            if method == "GET" and action == "artifact":
                touch()
                return self.artifact(agent, q.get("path", ""), q.get("gz") == "1")
            if method == "POST" and action == "clean":
                touch()
                return self.clean(agent, json.loads(self.body() or b"{}").get("what", "all"))
        return self.send_json(404, {"error": "not found"})

    def status(self):
        running = [j.info() for j in jobs.values() if j.state in ("running", "queued")]
        recent = sorted((j.info() for j in jobs.values() if j.state not in ("running", "queued")),
                        key=lambda i: i["id"])[-10:]
        vcpus, mem = cgroup_limits()
        self.send_json(200, {
            "pod": os.environ.get("RUNPOD_POD_ID"), "setup": {k: setup_state[k] for k in ("phase", "ready", "error")},
            "extras": dict(extras_state),
            # The container sees the host's CPUs and RAM and the whole shared
            # filesystem behind the volume; report the pod's own limits.
            "vcpus": vcpus, "host_cpus": os.cpu_count(), "mem_mib": mem,
            **POLICY.timers(time.time(), last_activity, len(running)), "idle_stop_s": IDLE_S, "max_s": MAX_S,
            "max_grace_s": MAX_GRACE_S, "self_stop": STOPPER.info(),
            "volume": {"size_gb": VOLUME_GB, "used_gb": volume_usage.get("used_gb"),
                       "measured_s_ago": round(time.time() - volume_usage["t"]) if "t" in volume_usage else None},
            "local_disk": disk_gb(LOCAL),
            "sccache": os.path.exists(os.path.join(ROOT, "tools", "sccache")), "mold": bool(shutil.which("mold")),
            "server_sha": SERVER_SHA, "self_stop_key": bool(os.environ.get("RUNPOD_API_KEY")),
            "eviction": {
                "idle_hours": EVICT_S / 3600, "free_gb_floor": EVICT_FREE_GB, "interval_s": EVICT_INTERVAL_S,
                "protect": sorted(EVICT_PROTECT), "hold_min": EVICT_HOLD_S / 60,
                "last_pass_s_ago": round(time.time() - evictor.last_pass) if evictor.last_pass else None,
                "recent": list(evictor.recent)[-10:],
            },
            "agents": self.agent_rows(),
            "jobs_active": running, "jobs_recent": recent,
        })

    def agent_rows(self):
        now = time.time()
        rows = []
        for a in sorted(LocalFS().agents()):
            idle = now - last_use(a)
            sz = agent_sizes.get(a, {})
            rows.append({
                "agent": a, "idle_h": round(idle / 3600, 2), "busy": agent_busy(a), "held": evictor.held(a, now),
                "target": os.path.isdir(os.path.join(TARGET_BASE, a)), "target_gb": sz.get("target_gb"),
                "worktree": os.path.isdir(os.path.join(WT_BASE, a)), "worktree_gb": sz.get("worktree_gb"),
                "evict_in_h": round(max(0.0, EVICT_S - idle) / 3600, 2) if EVICT_S > 0 else None,
            })
        return rows

    def agents(self, sizes):
        out = []
        for a in sorted(LocalFS().agents()):
            e = {"agent": a, "last_run": int(last_use(a)) or None}
            if sizes:
                e["worktree_bytes"] = du(os.path.join(WT_BASE, a))
                e["target_bytes"] = du(os.path.join(TARGET_BASE, a))
            out.append(e)
        self.send_json(200, {"agents": out})

    def put_files(self, agent, wt):
        tmp = os.path.join(LOCAL, f"sync-{agent}-{uuid.uuid4().hex[:8]}.tar")
        try:
            n = int(self.headers.get("content-length") or 0)
            if n > MAX_UPLOAD:
                return self.send_json(413, {"error": f"upload over {MAX_UPLOAD >> 20} MiB"})
            # The tarball, plus its extracted files (source compresses ~4x).
            ensure_free(MIN_FREE_GB * 1e9 + 5 * n)
            with open(tmp, "wb") as f:
                left = n
                while left:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            os.makedirs(wt, exist_ok=True)
            with agent_lock(agent):
                count = extract(tmp, wt)
            return self.send_json(200, {"extracted": count})
        except (ValueError, tarfile.TarError) as e:
            return self.send_json(400, {"error": str(e)})
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def new_job(self, agent, wt):
        req = json.loads(self.body() or b"{}")
        argv, env = req.get("argv"), req.get("env", {})
        err = validate_command(argv, env)
        if err:
            return self.send_json(403, {"error": err})
        if POLICY.past_cap(time.time()):
            return self.send_json(503, {"error": f"build pod is past its {MAX_S / 3600:g} h cap and stopping; "
                                                  "run build-pod.sh up for a fresh pod once it is gone"})
        touch()
        if not os.path.isdir(wt):
            return self.send_json(409, {"error": f"agent {agent} has no worktree; sync first"})
        ensure_free(MIN_FREE_GB * 1e9)
        job = Job(agent, argv, env)
        jobs[job.id] = job
        threading.Thread(target=job.run, daemon=True).start()
        return self.send_json(202, job.info())

    def job_poll(self, job, offset, wait):
        deadline = time.time() + wait
        while True:
            size = os.path.getsize(job.log_path)
            if size > offset or job.state in ("done", "cancelled") or time.time() >= deadline:
                break
            with job.cond:
                job.cond.wait(timeout=max(0.1, min(2.0, deadline - time.time())))
        with open(job.log_path, "rb") as f:
            f.seek(offset)
            data = f.read(1 << 20)
        info = job.info()
        info.update({"next": offset + len(data), "data": data.decode("utf-8", "replace"),
                     "finished": job.state in ("done", "cancelled") and offset + len(data) >= os.path.getsize(job.log_path)})
        return self.send_json(200, info)

    def artifact(self, agent, rel_path, gz):
        rel = safe_rel(rel_path)
        base = os.path.realpath(os.path.join(TARGET_BASE, agent))
        if rel is None:
            return self.send_json(400, {"error": "bad path"})
        full = os.path.realpath(os.path.join(base, rel))
        if not full.startswith(base + os.sep) or not os.path.isfile(full):
            return self.send_json(404, {"error": f"no file target/{agent}/{rel}"})
        size = os.path.getsize(full)
        if size > MAX_ARTIFACT:
            return self.send_json(413, {"error": "artifact too large"})
        self.send_response(200)
        self.send_header("content-type", "application/octet-stream")
        self.send_header("x-fv-size", str(size))
        with open(full, "rb") as f:
            if gz:
                self.send_header("transfer-encoding", "chunked")
                self.end_headers()

                class Chunked:
                    def __init__(s, w):
                        s.w = w

                    def write(s, b):
                        if b:
                            s.w.write(b"%x\r\n" % len(b) + b + b"\r\n")
                        return len(b)

                    def flush(s):
                        pass

                ch = Chunked(self.wfile)
                with gzip.GzipFile(fileobj=ch, mode="wb", compresslevel=3, mtime=0) as z:
                    shutil.copyfileobj(f, z, 1 << 20)
                self.wfile.write(b"0\r\n\r\n")
            else:
                self.send_header("content-length", str(size))
                self.end_headers()
                shutil.copyfileobj(f, self.wfile, 1 << 20)

    def clean(self, agent, what):
        if what not in ("target", "worktree", "all"):
            return self.send_json(400, {"error": "what must be target, worktree or all"})
        if any(j.agent == agent and j.state in ("running", "queued") for j in jobs.values()):
            return self.send_json(409, {"error": f"agent {agent} has a job in flight"})
        dirs = []
        if what in ("target", "all"):
            dirs.append(os.path.join(TARGET_BASE, agent))
        if what in ("worktree", "all"):
            dirs.append(os.path.join(WT_BASE, agent))
        for d in dirs:
            trash = move_to_trash(d, agent)
            if trash:
                threading.Thread(target=shutil.rmtree, args=(trash, True), daemon=True).start()
        return self.send_json(200, {"cleaned": [os.path.relpath(d, LOCAL) for d in dirs]})

    def handle_one(self, method):
        try:
            self.route(method)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # keep the service alive
            if not isinstance(e, DiskFull):
                log(f"{method} {self.path}: {e!r}")
            try:
                if isinstance(e, DiskFull):
                    self.send_json(507, {"error": str(e)})
                elif isinstance(e, OSError) and e.errno == errno.ENOSPC:
                    self.send_json(507, {"error": disk_full_msg()})
                else:
                    self.send_json(500, {"error": str(e)})
            except Exception:
                pass

    def do_GET(self):
        self.handle_one("GET")

    def do_POST(self):
        self.handle_one("POST")

    def do_PUT(self):
        self.handle_one("PUT")


def main():
    if not TOKEN_SHA or not re.fullmatch(r"[0-9a-f]{64}", TOKEN_SHA):
        print("FV_BUILD_TOKEN_SHA256 must be a sha256 hex digest", file=sys.stderr)
        sys.exit(2)
    os.makedirs(os.path.join(ROOT, "tmp"), exist_ok=True)
    for d in ("jobs", "logs"):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    for d in (WT_BASE, TARGET_BASE):
        os.makedirs(d, exist_ok=True)
    # Leftovers from a clean that the previous pod did not finish.
    os.makedirs(os.path.join(LOCAL, "trash"), exist_ok=True)
    for d in {os.path.join(ROOT, "tmp"), os.path.join(LOCAL, "trash"), LOCAL}:
        for name in os.listdir(d):
            full = os.path.join(d, name)
            if not name.startswith(("trash-", "sync-")):
                continue
            if os.path.isdir(full) and not os.path.islink(full):
                threading.Thread(target=shutil.rmtree, args=(full, True), daemon=True).start()
            else:
                os.remove(full)
    ledger("service-start")
    log(f"fv-build service on :{PORT} root={ROOT} local={LOCAL} idle={IDLE_S / 60:g}min cap={MAX_S / 3600:g}h")
    if os.environ.get("FV_BUILD_SKIP_SETUP") == "1":
        setup_state.update(phase="ready", ready=True)
        setup_done.set()
        extras_state.update(phase="ready", ready=True)
        extras_done.set()
    else:
        threading.Thread(target=setup, daemon=True).start()
    threading.Thread(target=volume_usage_loop, daemon=True).start()
    threading.Thread(target=agent_sizes_loop, daemon=True).start()
    threading.Thread(target=evict_loop, daemon=True).start()
    if os.environ.get("FV_BUILD_NO_WATCHDOG") != "1":
        threading.Thread(target=watchdog, daemon=True).start()
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
