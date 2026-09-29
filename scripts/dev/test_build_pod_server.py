#!/usr/bin/env python3
"""Tests for build-pod-server.py's disk eviction and disk-full handling.

    python3 scripts/dev/test_build_pod_server.py

Standard library only. The Evictor runs against a fake filesystem and clock;
LocalFS, job start and the HTTP 507 path run against a temp dir.
"""

import hashlib
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

TMP = tempfile.mkdtemp(prefix="fvb-test-")
TOKEN = "test-token"
os.environ.update(
    FV_BUILD_ROOT=os.path.join(TMP, "volume"),
    FV_BUILD_LOCAL=os.path.join(TMP, "local"),
    FV_BUILD_TOKEN_SHA256=hashlib.sha256(TOKEN.encode()).hexdigest(),
    FV_BUILD_TARGETS="local",
)
_spec = importlib.util.spec_from_file_location(
    "build_pod_server", os.path.join(os.path.dirname(os.path.abspath(__file__)), "build-pod-server.py"))
srv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(srv)

GB = 1e9
H = 3600


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


def evictor(fs, clock, hours=6, free_gb=40):
    logs = []
    ev = srv.Evictor(fs, clock=clock, evict_s=hours * H, min_free=free_gb * GB, log=logs.append)
    return ev, logs


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
        self.assertIsInstance(body["agents"], list)


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
