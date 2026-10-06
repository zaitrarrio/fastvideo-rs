#!/usr/bin/env python3
"""Unit tests for the build pod's server (scripts/dev/build-pod-server.py).

Self-stop: idle / cap decisions on a fake clock, the stop fallback chain and
its retry schedule, the User-Agent the Runpod calls send, and that status /
health polls do not count as activity.

Eviction and disk-full handling: the Evictor against a fake filesystem and
clock (idle and LRU passes, busy agents, the release agent's hold); LocalFS,
job start and the HTTP 507 path against a temp dir.

Image and caches: setup installs nothing and fails naming a missing tool;
jobs compile through sccache; sccache stats; deps seeds (key, stripping path
packages, extraction into a missing target dir, one seed build per key).

Standard library only:

    python3 scripts/dev/test_build_pod_server.py
"""

import http.server
import importlib.util
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from collections import namedtuple
from http.server import ThreadingHTTPServer
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp(prefix="fvb-test-")
TOKEN = "t" * 16
os.environ.update({
    "FV_BUILD_ROOT": os.path.join(TMP, "vol"),
    "FV_BUILD_LOCAL": os.path.join(TMP, "local"),
    "FV_BUILD_TOKEN_SHA256": __import__("hashlib").sha256(TOKEN.encode()).hexdigest(),
    "FV_BUILD_SKIP_SETUP": "1",
    "FV_BUILD_NO_WATCHDOG": "1",
    "FV_BUILD_TARGETS": "local",
})
for k in ("RUNPOD_API_KEY", "RUNPOD_POD_ID"):
    os.environ.pop(k, None)
_spec = importlib.util.spec_from_file_location("bps", os.path.join(HERE, "build-pod-server.py"))
bps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bps)
srv = bps  # the eviction tests' name for the module
for _d in (os.path.join(TMP, "vol", "logs"), os.path.join(TMP, "vol", "jobs"), bps.WT_BASE, bps.TARGET_BASE):
    os.makedirs(_d, exist_ok=True)

GB = 1e9
H = 3600
M = 60


class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.p = bps.StopPolicy(boot=1000.0, idle_s=20 * M, max_s=8 * H, grace_s=30 * M)

    def test_idle_needs_no_jobs_and_the_full_window(self):
        t0 = 1000.0
        self.assertIsNone(self.p.decide(t0 + 19 * M, t0, 0))
        self.assertTrue(self.p.decide(t0 + 20 * M, t0, 0).startswith("idle 20 min"))
        # A queued or running job holds the idle stop off however long it runs.
        self.assertIsNone(self.p.decide(t0 + 5 * H, t0, 1))

    def test_idle_counts_from_the_last_activity(self):
        t0 = 1000.0
        self.assertIsNone(self.p.decide(t0 + 2 * H, t0 + 2 * H - 10 * M, 0))
        self.assertIsNotNone(self.p.decide(t0 + 2 * H, t0 + 2 * H - 21 * M, 0))

    def test_cap_without_jobs_stops_at_once(self):
        t = 1000.0 + 8 * H
        self.assertIsNone(self.p.decide(t - 1, t - 1, 0))
        self.assertTrue(self.p.decide(t, t, 0).startswith("wall-clock cap 8h"))

    def test_cap_with_jobs_waits_for_the_grace_then_stops_anyway(self):
        t = 1000.0 + 8 * H
        self.assertTrue(self.p.past_cap(t))
        self.assertIsNone(self.p.decide(t + 29 * M, t, 2))
        r = self.p.decide(t + 30 * M, t, 2)
        self.assertIn("grace", r)
        self.assertIn("2 job(s)", r)

    def test_timers(self):
        t0 = 1000.0
        x = self.p.timers(t0 + H, t0 + H - 5 * M, 0)
        self.assertEqual((x["uptime_s"], x["idle_s"], x["idle_stop_in_s"], x["max_stop_in_s"]), (H, 5 * M, 15 * M, 7 * H))
        busy = self.p.timers(t0 + H, t0, 1)
        self.assertEqual((busy["idle_s"], busy["idle_stop_in_s"], busy["max_stop_in_s"]), (0, None, 7 * H + 30 * M))
        late = self.p.timers(t0 + 9 * H, t0, 0)
        self.assertEqual((late["idle_stop_in_s"], late["max_stop_in_s"]), (0, 0))


class StopperTest(unittest.TestCase):
    def test_falls_back_rest_stop_terminate_then_graphql(self):
        seen = []

        def call(method, url, body):
            seen.append((method, url, body and body["query"].split("(")[0]))
            if len(seen) < 3:
                raise RuntimeError("HTTP 403 Forbidden: error code: 1010")
            return {"data": {}}

        s = bps.Stopper(call=call, pod="pod1")
        self.assertTrue(s.attempt("idle 20 min", 100.0))
        self.assertEqual(seen, [
            ("POST", bps.RUNPOD_REST + "/pods/pod1/stop", None),
            ("DELETE", bps.RUNPOD_REST + "/pods/pod1", None),
            ("POST", bps.RUNPOD_GRAPHQL, "mutation { podStop"),
        ])
        self.assertEqual(s.info()["ok"], "GraphQL podStop")
        # Still alive 10 min after an accepted call: try again.
        self.assertFalse(s.due(100.0 + 599))
        self.assertTrue(s.due(100.0 + 600))

    def test_failures_back_off_and_keep_retrying(self):
        def call(method, url, body):
            raise RuntimeError("HTTP 403 Forbidden: error code: 1010")

        s = bps.Stopper(call=call, pod="pod1")
        now, gaps = 0.0, []
        for _ in range(6):
            self.assertFalse(s.attempt("cap", now))
            gaps.append(s.next_at - now)
            now = s.next_at
        self.assertEqual(gaps, [60, 120, 240, 480, 600, 600])
        info = s.info()
        self.assertEqual(info["attempts"], 6)
        self.assertIn("1010", info["error"])
        self.assertIn("GraphQL podTerminate", info["error"])

    def test_graphql_errors_count_as_failures(self):
        os.environ["RUNPOD_API_KEY"] = "k"
        try:
            with _FakeRunpod({"errors": [{"message": "not authorized"}]}) as srv:
                with self.assertRaisesRegex(RuntimeError, "not authorized"):
                    bps.runpod_call("POST", srv.url + "/graphql", {"query": "x"})
        finally:
            os.environ.pop("RUNPOD_API_KEY")

    def test_calls_send_an_explicit_user_agent_and_report_the_body(self):
        os.environ["RUNPOD_API_KEY"] = "k"
        try:
            with _FakeRunpod({}, status=403, body=b"error code: 1010") as srv:
                with self.assertRaisesRegex(RuntimeError, "HTTP 403.*error code: 1010"):
                    bps.runpod_call("POST", srv.url + "/pods/x/stop")
                self.assertTrue(srv.seen[-1]["ua"].startswith("fv-build-pod/"))
                self.assertNotIn("Python-urllib", srv.seen[-1]["ua"])
                self.assertEqual(srv.seen[-1]["auth"], "Bearer k")
        finally:
            os.environ.pop("RUNPOD_API_KEY")


class _FakeRunpod:
    def __init__(self, reply, status=200, body=None):
        self.reply, self.status, self.body, self.seen = reply, status, body, []

    def __enter__(self):
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def handle_any(self):
                n = int(self.headers.get("content-length") or 0)
                self.rfile.read(n)
                outer.seen.append({"ua": self.headers.get("user-agent", ""), "auth": self.headers.get("authorization")})
                b = outer.body if outer.body is not None else json.dumps(outer.reply).encode()
                self.send_response(outer.status)
                self.send_header("content-length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            do_GET = do_POST = do_DELETE = handle_any

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self.srv.shutdown()


class WatchTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.orig = (bps.POLICY, bps.STOPPER, bps.last_activity)
        bps.POLICY = bps.StopPolicy(boot=0.0, idle_s=20 * M, max_s=8 * H, grace_s=30 * M)
        bps.STOPPER = bps.Stopper(call=lambda *a: self.calls.append(a), pod="pod1")
        bps.last_activity = 0.0
        bps.jobs.clear()
        bps.cap_warned.clear()

    def tearDown(self):
        bps.POLICY, bps.STOPPER, bps.last_activity = self.orig
        bps.jobs.clear()

    def _job(self, state):
        j = bps.Job.__new__(bps.Job)
        j.id, j.state, j.agent = f"j{len(bps.jobs)}", state, "a"
        j.log_path = os.path.join(TMP, j.id + ".log")
        j.cond = threading.Condition()
        open(j.log_path, "wb").close()
        bps.jobs[j.id] = j
        return j

    def test_idle_stop_fires_once_then_rechecks_after_ten_minutes(self):
        self.assertIsNone(bps.watch_tick(19 * M))
        self.assertTrue(bps.watch_tick(20 * M).startswith("idle"))
        self.assertEqual(len(self.calls), 1)
        bps.watch_tick(25 * M)
        self.assertEqual(len(self.calls), 1)
        bps.watch_tick(30 * M)
        self.assertEqual(len(self.calls), 2)

    def test_public_jobs_carry_no_argv(self):
        j = self._job("running")
        j.argv, j.started, j.ended, j.exit = ["cargo", "test", "--secret-ish"], 0.0, None, None
        self._job("done")
        out = bps.public_jobs()
        self.assertEqual([x["id"] for x in out], [j.id])
        self.assertEqual(set(out[0]), {"id", "agent", "state", "seconds"})

    def test_cap_warns_running_jobs_then_kills_them_after_the_grace(self):
        j = self._job("running")
        bps.last_activity = 8 * H
        self.assertIsNone(bps.watch_tick(8 * H + 1))
        with open(j.log_path) as f:
            self.assertIn("passed its 8 h cap", f.read())
        self.assertEqual(self.calls, [])
        self.assertIn("grace", bps.watch_tick(8 * H + 30 * M))
        self.assertEqual(len(self.calls), 1)
        with open(j.log_path) as f:
            self.assertIn("this job is killed", f.read())


class HttpTest(unittest.TestCase):
    """Status and health polls must not reset the idle timer."""

    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), bps.Handler)
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def get(self, path, auth=True):
        req = urllib.request.Request(self.url + path, headers={"Authorization": "Bearer " + TOKEN} if auth else {})
        with urllib.request.urlopen(req, timeout=10) as r:
            body = r.read().decode()
            return json.loads(body) if r.headers.get("content-type", "").startswith("application/json") else body

    def test_polls_do_not_touch(self):
        bps.last_activity = 12345.0
        st = self.get("/v1/status")
        self.get("/healthz", auth=False)
        self.get("/v1/agents")
        self.get("/v1/log?lines=5")
        self.assertEqual(bps.last_activity, 12345.0)
        for k in ("idle_s", "idle_stop_in_s", "max_stop_in_s", "max_grace_s", "self_stop"):
            self.assertIn(k, st)
        hz = self.get("/healthz", auth=False)
        for k in ("idle_s", "idle_stop_in_s", "max_stop_in_s", "jobs_active", "uptime_s", "self_stop", "jobs"):
            self.assertIn(k, hz)
        self.assertIn("attempts", hz["self_stop"])
        self.assertIsInstance(hz["jobs"], list)

    def test_job_submit_past_the_cap_is_refused(self):
        orig = bps.POLICY
        bps.POLICY = bps.StopPolicy(boot=0.0, idle_s=20 * M, max_s=1, grace_s=30 * M)
        try:
            os.makedirs(os.path.join(bps.WT_BASE, "agent1"), exist_ok=True)
            req = urllib.request.Request(self.url + "/v1/agents/agent1/jobs", method="POST",
                                         data=json.dumps({"argv": ["cargo", "check"]}).encode(),
                                         headers={"Authorization": "Bearer " + TOKEN})
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(req, timeout=10)
            self.assertEqual(cm.exception.code, 503)
        finally:
            bps.POLICY = orig


class FakeFS:
    """agent -> {"target": bytes | None, "worktree": bytes | None}; free space
    grows by what remove() deletes."""

    def __init__(self, free_gb, agents):
        self.free = free_gb * GB
        self.dirs = {a: {"target": t * GB if t is not None else None,
                         "worktree": w * GB if w is not None else None} for a, (t, w, _) in agents.items()}
        self.used = {a: u for a, (_, _, u) in agents.items()}
        self.busy_set = set()
        self.refuse = set()  # agents whose remove() loses the race with a new job
        self.removed = []

    def agents(self):
        return {a for a, d in self.dirs.items() if any(v is not None for v in d.values())}

    def has(self, agent, kind):
        return self.dirs.get(agent, {}).get(kind) is not None

    def last_use(self, agent):
        return self.used[agent]

    def busy(self, agent):
        return agent in self.busy_set

    def free_bytes(self):
        return self.free

    def remove(self, agent, kind):
        if agent in self.busy_set or agent in self.refuse:
            return False
        self.free += self.dirs[agent][kind]
        self.dirs[agent][kind] = None
        self.removed.append((agent, kind))
        return True


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def evictor(fs, clock, hours=6, free_gb=40, protect=("fv-release",), hold_min=60):
    logs = []
    ev = srv.Evictor(fs, clock=clock, evict_s=hours * H, min_free=free_gb * GB, log=logs.append,
                     protect=protect, hold_s=hold_min * M)
    return ev, logs


class ReleaseHoldTest(unittest.TestCase):
    """The release agent (build-pod.sh release-artifacts: sync, job, then
    fetches from target/fv-release/) is not evicted while its flow runs."""

    def test_between_the_job_and_the_fetch_it_is_held(self):
        c = Clock()
        # The release job just finished (no job in flight); the disk is short.
        fs = FakeFS(10, {"fv-release": (60, 0.1, c.t - 2 * M), "other": (20, 0.1, c.t - 30 * M)})
        ev, _ = evictor(fs, c)
        ev.run_once()
        self.assertEqual(fs.removed, [("other", "target")])
        self.assertTrue(fs.has("fv-release", "target"))

    def test_while_its_job_runs_it_is_kept_whatever_its_age(self):
        c = Clock()
        fs = FakeFS(5, {"fv-release": (60, 0.1, c.t - 9 * H)})
        fs.busy_set.add("fv-release")
        ev, _ = evictor(fs, c, hold_min=0)
        self.assertEqual(ev.run_once(), [])

    def test_after_the_hold_it_goes_last_under_pressure(self):
        c = Clock()
        # fv-release is the least recently used, but goes after the others.
        fs = FakeFS(10, {"fv-release": (60, 0.1, c.t - 3 * H), "a": (20, 0.1, c.t - 2 * H),
                         "b": (20, 0.1, c.t - 1 * H)})
        ev, _ = evictor(fs, c, free_gb=45)
        ev.run_once()
        self.assertEqual(fs.removed, [("a", "target"), ("b", "target")])
        ev2, _ = evictor(fs, c, free_gb=100)
        ev2.run_once()
        self.assertEqual(fs.removed[-1], ("fv-release", "target"))

    def test_idle_eviction_still_applies_after_the_hold(self):
        c = Clock()
        fs = FakeFS(150, {"fv-release": (60, 0.1, c.t - 7 * H)})
        ev, _ = evictor(fs, c)
        ev.run_once()
        self.assertEqual(fs.removed, [("fv-release", "target"), ("fv-release", "worktree")])

    def test_other_agents_are_not_held(self):
        c = Clock()
        fs = FakeFS(10, {"recent": (30, 0.1, c.t - 1 * M)})
        ev, _ = evictor(fs, c)
        ev.run_once()
        self.assertEqual(fs.removed, [("recent", "target")])


class EvictorTest(unittest.TestCase):
    def test_idle_agents_lose_target_and_worktree(self):
        c = Clock()
        fs = FakeFS(150, {"old": (30, 0.1, c.t - 7 * H), "fresh": (20, 0.1, c.t - 1 * H),
                          "edge": (10, 0.1, c.t - 6 * H)})
        ev, logs = evictor(fs, c)
        done = ev.run_once()
        self.assertEqual(fs.removed, [("old", "target"), ("old", "worktree")])
        self.assertEqual([(e["agent"], e["what"]) for e in done], fs.removed)
        self.assertEqual(done[0]["freed_gb"], 30.0)
        self.assertIn("unused 7.0 h > 6 h", done[0]["reason"])
        self.assertTrue(any(line.startswith("evicted target/old") for line in logs))
        self.assertTrue(any(line.startswith("evicted worktrees/old") for line in logs))
        self.assertEqual(list(ev.recent), done)
        self.assertEqual(ev.last_pass, c.t)

    def test_idle_but_busy_agent_is_kept(self):
        c = Clock()
        fs = FakeFS(150, {"long-job": (30, 0.1, c.t - 9 * H)})
        fs.busy_set.add("long-job")
        ev, _ = evictor(fs, c)
        self.assertEqual(ev.run_once(), [])
        self.assertEqual(fs.removed, [])

    def test_agent_becomes_idle_as_the_clock_advances(self):
        c = Clock()
        fs = FakeFS(150, {"a": (5, 0.1, c.t)})
        ev, _ = evictor(fs, c)
        self.assertEqual(ev.run_once(), [])
        c.t += 6 * H + 1
        self.assertEqual(len(ev.run_once()), 2)

    def test_pressure_evicts_lru_targets_until_the_floor(self):
        c = Clock()
        fs = FakeFS(10, {"a": (20, 0.1, c.t - 3 * H), "b": (15, 0.1, c.t - 2 * H),
                         "c": (25, 0.1, c.t - 1 * H), "d": (30, 0.1, c.t - 60)})
        ev, _ = evictor(fs, c)
        done = ev.run_once()
        # 10 + 20 (a) = 30 < 40, + 15 (b) = 45 >= 40: stop; c and d stay.
        self.assertEqual(fs.removed, [("a", "target"), ("b", "target")])
        self.assertTrue(all("disk pressure" in e["reason"] for e in done))
        self.assertTrue(fs.has("a", "worktree"), "pressure keeps snapshots")
        self.assertEqual(fs.free, 45 * GB)

    def test_pressure_never_touches_a_busy_agent(self):
        c = Clock()
        fs = FakeFS(5, {"lru-busy": (50, 0.1, c.t - 5 * H), "b": (10, 0.1, c.t - 1 * H),
                        "c": (10, 0.1, c.t - 30)})
        fs.busy_set.add("lru-busy")
        ev, logs = evictor(fs, c)
        ev.run_once()
        self.assertEqual(fs.removed, [("b", "target"), ("c", "target")])
        self.assertTrue(fs.has("lru-busy", "target"))
        self.assertTrue(any("still 25.0 GB free" in line for line in logs))

    def test_remove_refused_by_a_racing_job_is_logged_not_recorded(self):
        c = Clock()
        fs = FakeFS(150, {"racy": (30, 0.1, c.t - 8 * H)})
        fs.refuse.add("racy")
        ev, logs = evictor(fs, c)
        self.assertEqual(ev.run_once(), [])
        self.assertEqual(list(ev.recent), [])
        self.assertTrue(any("skipped target/racy: job in flight" in line for line in logs))

    def test_idle_pass_runs_before_pressure(self):
        c = Clock()
        fs = FakeFS(5, {"stale": (40, 0.1, c.t - 10 * H), "recent": (30, 0.1, c.t - 60)})
        ev, _ = evictor(fs, c)
        ev.run_once()
        # Evicting the stale agent already reaches 45 GB: the recent one stays.
        self.assertEqual(fs.removed, [("stale", "target"), ("stale", "worktree")])

    def test_disabled(self):
        c = Clock()
        fs = FakeFS(1, {"a": (30, 0.1, c.t - 100 * H)})
        ev, _ = evictor(fs, c, hours=0, free_gb=0)
        self.assertEqual(ev.run_once(), [])

    def test_low_disk_warning_is_rate_limited(self):
        c = Clock()
        fs = FakeFS(5, {"busy": (50, 0.1, c.t)})
        fs.busy_set.add("busy")
        ev, logs = evictor(fs, c)
        ev.run_once()
        ev.run_once()
        self.assertEqual(sum("still" in line for line in logs), 1)
        c.t += 601
        ev.run_once()
        self.assertEqual(sum("still" in line for line in logs), 2)


FakeUsage = namedtuple("FakeUsage", "total used free")


class LocalFSTest(unittest.TestCase):
    """The real LocalFS and job start on a temp dir."""

    def setUp(self):
        for d in (srv.WT_BASE, srv.TARGET_BASE, srv.LAST_USE, os.path.join(srv.LOCAL, "trash"),
                  os.path.join(srv.ROOT, "jobs"), os.path.join(srv.ROOT, "logs")):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
        srv.jobs.clear()

    def make_agent(self, agent, age_h):
        for base in (srv.WT_BASE, srv.TARGET_BASE):
            os.makedirs(os.path.join(base, agent, "debug"))
            with open(os.path.join(base, agent, "debug", "x"), "w") as f:
                f.write("x")
        srv.mark_used(agent)
        t = time.time() - age_h * H
        os.utime(os.path.join(srv.LAST_USE, agent), (t, t))

    def test_idle_eviction_and_busy_guard(self):
        self.make_agent("old", 7)
        self.make_agent("running", 7)
        self.make_agent("new", 1)
        job = srv.Job("running", ["true"], {})
        job.state = "running"
        srv.jobs[job.id] = job
        ev = srv.Evictor(srv.LocalFS(), evict_s=6 * H, min_free=0, log=lambda m: None)
        done = ev.run_once()
        self.assertEqual({(e["agent"], e["what"]) for e in done}, {("old", "target"), ("old", "worktree")})
        self.assertFalse(os.path.exists(os.path.join(srv.TARGET_BASE, "old")))
        self.assertFalse(os.path.exists(os.path.join(srv.WT_BASE, "old")))
        self.assertTrue(os.path.isdir(os.path.join(srv.TARGET_BASE, "running")))
        self.assertTrue(os.path.isdir(os.path.join(srv.TARGET_BASE, "new")))
        self.assertEqual(os.listdir(os.path.join(srv.LOCAL, "trash")), [], "removal is synchronous")

    def test_volume_and_release_cache_are_never_evicted(self):
        # The release build's caches on the volume (release-cache/) and the
        # volume's other trees are outside eviction's reach, under any
        # pressure, and a protected agent is kept within its hold.
        cache = os.path.join(srv.ROOT, "release-cache", "curand")
        os.makedirs(cache, exist_ok=True)
        self.make_agent("fv-release", 0.1)
        self.make_agent("someone", 3)
        ev = srv.Evictor(srv.LocalFS(), evict_s=6 * H, min_free=1e18, log=lambda m: None)
        done = ev.run_once()
        self.assertEqual([(e["agent"], e["what"]) for e in done], [("someone", "target")])
        self.assertTrue(os.path.isdir(cache))
        self.assertTrue(os.path.isdir(os.path.join(srv.TARGET_BASE, "fv-release")))
        self.assertFalse(srv.LocalFS().remove("..", "target"))
        self.assertFalse(srv.LocalFS().remove("release-cache/..", "target"))
        self.assertTrue(os.path.isdir(cache))

    def test_eviction_does_not_count_as_activity(self):
        self.make_agent("idle", 9)
        srv.last_activity = 12345.0
        srv.Evictor(srv.LocalFS(), evict_s=6 * H, min_free=0, log=lambda m: None).run_once()
        self.assertFalse(os.path.exists(os.path.join(srv.TARGET_BASE, "idle")))
        self.assertEqual(srv.last_activity, 12345.0, "the self-stop idle timer is untouched")

    def test_remove_refuses_while_the_agent_lock_is_held(self):
        self.make_agent("locked", 9)
        with srv.agent_lock("locked"):
            self.assertFalse(srv.LocalFS().remove("locked", "target"))
        self.assertTrue(srv.LocalFS().remove("locked", "target"))

    def test_job_after_target_eviction_builds_cold(self):
        self.make_agent("cold", 0)
        self.assertTrue(srv.LocalFS().remove("cold", "target"))
        srv.setup_state.update(ready=True)
        srv.setup_done.set()
        job = srv.Job("cold", ["true"], {})
        job.run()
        self.assertEqual(job.exit, 0)
        self.assertTrue(os.path.isdir(os.path.join(srv.TARGET_BASE, "cold")), "target dir recreated")

    def test_job_after_worktree_eviction_fails_clearly(self):
        self.make_agent("gone", 0)
        srv.LocalFS().remove("gone", "worktree")
        srv.setup_state.update(ready=True)
        srv.setup_done.set()
        job = srv.Job("gone", ["true"], {})
        job.run()
        self.assertEqual(job.exit, 125)
        with open(job.log_path) as f:
            self.assertIn("has no worktree (evicted?)", f.read())


class HttpDiskFullTest(unittest.TestCase):
    """sync and run answer 507 "build pod disk full", not 500."""

    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def call(self, method, path, body=b""):
        req = urllib.request.Request(self.base + path, data=body if method != "GET" else None, method=method,
                                     headers={"Authorization": "Bearer " + TOKEN})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def test_full_disk(self):
        os.makedirs(os.path.join(srv.WT_BASE, "a"), exist_ok=True)
        full = FakeUsage(200 * GB, 199.5 * GB, 0.5 * GB)
        with mock.patch.object(srv.shutil, "disk_usage", return_value=full), \
                mock.patch.object(srv.evictor, "run_once", return_value=[]) as evict:
            code, body = self.call("PUT", "/v1/agents/a/files", b"\0" * 1024)
            self.assertEqual(code, 507)
            self.assertIn("build pod disk full: 0.5 GB free", body["error"])
            code, body = self.call("POST", "/v1/agents/a/jobs",
                                   json.dumps({"argv": ["cargo", "check"], "env": {}}).encode())
            self.assertEqual(code, 507)
            self.assertIn("disk full", body["error"])
            self.assertEqual(evict.call_count, 2, "an eviction pass runs before refusing")
            self.assertFalse(any(j.agent == "a" for j in srv.jobs.values()), "no job was queued")

    def test_enospc_during_extract_is_507(self):
        import errno
        with mock.patch.object(srv, "extract", side_effect=OSError(errno.ENOSPC, "No space left on device")):
            code, body = self.call("PUT", "/v1/agents/b/files", b"x")
        self.assertEqual(code, 507)
        self.assertIn("build pod disk full", body["error"])

    def test_status_exposes_eviction(self):
        code, body = self.call("GET", "/v1/status")
        self.assertEqual(code, 200)
        self.assertEqual(body["eviction"]["idle_hours"], 6)
        self.assertEqual(body["eviction"]["free_gb_floor"], 40)
        self.assertEqual((body["eviction"]["protect"], body["eviction"]["hold_min"]), (["fv-release"], 60))
        self.assertIsInstance(body["agents"], list)



class ImageSetupTest(unittest.TestCase):
    """The pod installs nothing: setup checks the image and fails with what is
    missing; jobs compile through sccache."""

    def tearDown(self):
        bps.setup_state.update(phase="ready", ready=True, error=None)
        bps.setup_done.set()
        bps.extras_state.update(phase="ready", ready=True, error=None)
        bps.extras_done.set()

    def test_a_missing_tool_fails_the_setup_and_names_it(self):
        bps.setup_done.clear()
        bps.setup_state.update(phase="starting", ready=False, error=None)
        with mock.patch.object(bps, "REQUIRED_TOOLS", ("cargo", "fv-no-such-tool")), \
                mock.patch.object(bps, "prune_targets"), mock.patch.object(bps, "sccache_start") as start:
            bps.setup()
        self.assertEqual(bps.setup_state["phase"], "failed")
        self.assertIn("missing from the image", bps.setup_state["error"])
        self.assertIn("fv-no-such-tool", bps.setup_state["error"])
        self.assertEqual(bps.extras_state["phase"], "skipped")
        start.assert_not_called()
        job = bps.Job("x", ["true"], {})
        os.makedirs(os.path.join(bps.WT_BASE, "x"), exist_ok=True)
        job.run()
        self.assertEqual(job.exit, 125)
        with open(job.log_path) as f:
            self.assertIn("fv-no-such-tool", f.read())

    def test_jobs_compile_through_sccache(self):
        which = lambda n, path=None: "/usr/local/bin/" + n if n in ("sccache", "mold") else None
        with mock.patch.object(bps.shutil, "which", side_effect=which):
            env = bps.job_env("a")
        self.assertEqual(env["RUSTC_WRAPPER"], "/usr/local/bin/sccache")
        self.assertEqual(env["CMAKE_C_COMPILER_LAUNCHER"], "/usr/local/bin/sccache")
        self.assertEqual(env["SCCACHE_DIR"], os.path.join(bps.ROOT, "sccache"))
        self.assertEqual(env["CARGO_TARGET_DIR"], os.path.join(bps.TARGET_BASE, "a"))
        self.assertNotIn("FV_BUILD_SERVER_B64", env)
        with mock.patch.object(bps.shutil, "which", return_value=None):
            self.assertNotIn("RUSTC_WRAPPER", bps.job_env("a"))

    def test_eviction_floor_scales_with_the_disk(self):
        self.assertEqual(bps.scaled_floor_gb(40, 200), 40)
        self.assertEqual(bps.scaled_floor_gb(40, 80), 16)
        self.assertEqual(bps.scaled_floor_gb(0, 80), 0)

    def test_sccache_stats(self):
        out = json.dumps({"stats": {"compile_requests": 10, "requests_executed": 9,
                                    "cache_hits": {"counts": {"Rust": 6, "C/C++": 1}, "adv_counts": {"x": 99}},
                                    "cache_misses": {"counts": {"Rust": 3}}, "requests_not_cacheable": 1,
                                    "cache_errors": {"counts": {}}},
                          "cache_location": "Local disk: /v/sccache", "cache_size": 2e9, "max_cache_size": 40e9})
        with mock.patch.object(bps, "sccache_cmd", return_value=(0, out)):
            st = bps.sccache_stats()
        self.assertEqual((st["hits"], st["misses"], st["hit_rate"]), (7, 3, 0.7))
        self.assertEqual((st["cache_size_gb"], st["max_cache_size_gb"], st["errors"]), (2.0, 40.0, 0))
        with mock.patch.object(bps, "sccache_cmd", return_value=(2, "")):
            self.assertIsNone(bps.sccache_stats())


class SeedTest(unittest.TestCase):
    """Deps seeds: keyed by Cargo.lock, path packages stripped, extracted into
    a missing target dir, built once per key."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(dir=TMP)
        self.wt = os.path.join(self.dir, "wt")
        os.makedirs(self.wt)
        with open(os.path.join(self.wt, "Cargo.lock"), "w") as f:
            f.write("version = 4\n")

    def touch(self, *parts, d=False):
        p = os.path.join(self.dir, *parts)
        os.makedirs(p if d else os.path.dirname(p), exist_ok=True)
        if not d:
            open(p, "w").close()
        return p

    def test_key_follows_cargo_lock(self):
        k1 = bps.seed_key(self.wt)
        self.assertRegex(k1, r"^[0-9a-f]{16}$")
        self.assertEqual(bps.seed_key(self.wt), k1)
        with open(os.path.join(self.wt, "Cargo.lock"), "a") as f:
            f.write("[[package]]\nname = \"itoa\"\n")
        self.assertNotEqual(bps.seed_key(self.wt), k1)
        self.assertIsNone(bps.seed_key(os.path.join(self.dir, "none")))

    def test_strip_removes_path_packages_only(self):
        h, h2, r = "0123456789abcdef", "fedcba9876543210", "1111222233334444"
        keep = [self.touch("t", "release", ".fingerprint", f"serde-{r}", "lib-serde"),
                self.touch("t", "release", ".fingerprint", f"fastvideo-serve-extra-{h}", "x"),  # registry look-alike
                self.touch("t", "release", "build", f"ring-{r}", "output"),
                self.touch("t", "release", "deps", f"libserde-{r}.rlib"),  # same name as our tests/serde.rs
                self.touch("t", "release", "deps", f"serde-{r}.d"),
                self.touch("t", "release", "deps", f"liblibc-{r}.rlib")]
        gone = [self.touch("t", "release", ".fingerprint", f"fastvideo-serve-{h}", "lib"),
                self.touch("t", "release", ".fingerprint", f"fastvideo-serve-{h2}", "run-build-script"),
                self.touch("t", "release", "build", f"fastvideo-serve-{h}", "out", "x"),
                self.touch("t", "release", "deps", f"libfastvideo_serve-{h}.rlib"),
                self.touch("t", "release", "deps", f"fastvideo_serve-{h}.d"),
                self.touch("t", "release", "deps", f"serde-{h2}"),  # the tests/serde.rs test binary
                self.touch("t", "release", "fv-serve"),
                self.touch("t", "release", "fv-serve.d"),
                self.touch("t", "release", "incremental", "x", "y"),
                self.touch("t", "debug", ".fingerprint", f"fastvideo-serve-{h}", "lib")]
        bps.strip_path_packages(os.path.join(self.dir, "t"), {"fastvideo-serve"})
        for p in keep:
            self.assertTrue(os.path.exists(p), p)
        for p in gone:
            self.assertFalse(os.path.exists(p), p)

    def test_no_seed_builds_cold(self):
        lines = []
        with mock.patch.object(bps, "SEED_DIR", os.path.join(self.dir, "seeds")):
            self.assertFalse(bps.seed_target(self.wt, os.path.join(self.dir, "tgt"), lines.append))
        self.assertIn(b"none for this Cargo.lock", lines[0])
        self.assertFalse(os.path.exists(os.path.join(self.dir, "tgt")))

    @unittest.skipUnless(shutil.which("zstd"), "needs zstd")
    def test_seed_round_trip(self):
        seeds = os.path.join(self.dir, "seeds")
        os.makedirs(seeds)
        src = os.path.join(self.dir, "src")
        self.touch("src", "release", "deps", "libserde-0123456789abcdef.rlib")
        key = bps.seed_key(self.wt)
        tar, meta = os.path.join(seeds, key + ".tar.zst"), os.path.join(seeds, key + ".json")
        import subprocess
        subprocess.run(["tar", "-I", "zstd", "-cf", tar, "-C", src, "."], check=True)
        with open(meta, "w") as f:
            json.dump({"bytes": 1, "unpacked_bytes": 1}, f)
        lines = []
        tgt = os.path.join(self.dir, "tgt")
        with mock.patch.object(bps, "SEED_DIR", seeds):
            self.assertTrue(bps.seed_target(self.wt, tgt, lines.append))
            self.assertEqual([s["key"] for s in bps.seed_list()], [key])
        self.assertTrue(os.path.exists(os.path.join(tgt, "release", "deps", "libserde-0123456789abcdef.rlib")))

    def test_seed_build_is_skipped_when_present_or_busy(self):
        seeds = os.path.join(self.dir, "seeds")
        os.makedirs(seeds)
        wt = os.path.join(bps.WT_BASE, "seeder")
        os.makedirs(wt, exist_ok=True)
        shutil.copy(os.path.join(self.wt, "Cargo.lock"), wt)
        key = bps.seed_key(wt)
        started = []
        with mock.patch.object(bps, "SEED_DIR", seeds), \
                mock.patch.object(bps.threading, "Thread", side_effect=lambda target, daemon: mock.Mock(start=lambda: started.append(target))):
            self.assertIsNone(bps.maybe_build_seed(bps.SEED_AGENT))
            job = bps.maybe_build_seed("seeder")
            self.assertIsInstance(job, bps.SeedJob)
            self.assertEqual(job.key, key)
            self.assertIsNone(bps.maybe_build_seed("seeder"), "one seed build at a time")
            job.state = "done"
            for p in bps.seed_paths(key):
                open(p, "w").write("{}")
            self.assertIsNone(bps.maybe_build_seed("seeder"), "the seed exists")
            self.assertIsNotNone(bps.maybe_build_seed("seeder", force=True))
        self.assertEqual(len(started), 2)
        for j in list(bps.seed_building.values()):
            j.state = "done"


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
