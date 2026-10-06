#!/usr/bin/env python3
"""Unit tests for the build pod's self-stop (scripts/dev/build-pod-server.py):
idle / cap decisions on a fake clock, the stop fallback chain and its retry
schedule, the User-Agent the Runpod calls send, and that status / health polls
do not count as activity. Standard library only:

    python3 scripts/dev/test_build_pod_server.py
"""

import http.server
import importlib.util
import json
import os
import tempfile
import threading
import unittest
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp(prefix="fvb-test-")
TOKEN = "t" * 16
os.environ.update({
    "FV_BUILD_ROOT": os.path.join(TMP, "vol"),
    "FV_BUILD_LOCAL": os.path.join(TMP, "local"),
    "FV_BUILD_TOKEN_SHA256": __import__("hashlib").sha256(TOKEN.encode()).hexdigest(),
    "FV_BUILD_SKIP_SETUP": "1",
    "FV_BUILD_NO_WATCHDOG": "1",
})
for k in ("RUNPOD_API_KEY", "RUNPOD_POD_ID"):
    os.environ.pop(k, None)
_spec = importlib.util.spec_from_file_location("bps", os.path.join(HERE, "build-pod-server.py"))
bps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bps)
for _d in (os.path.join(TMP, "vol", "logs"), os.path.join(TMP, "vol", "jobs"), bps.WT_BASE, bps.TARGET_BASE):
    os.makedirs(_d, exist_ok=True)

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


if __name__ == "__main__":
    unittest.main()
