"""Tests of r2-weights.py (publisher) and r2-fetch.py (pod fetcher) against fake_r2 (an in-memory S3 with Range).

python3 -m unittest discover -s scripts/gpu/tests -p 'test_r2*.py' -v
"""
import contextlib
import datetime
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fake_r2  # noqa: E402


def load(name, mod):
    spec = importlib.util.spec_from_file_location(mod, HERE.parent / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


W = load("r2-weights.py", "r2_weights")
F = load("r2-fetch.py", "r2_fetch_t")
PART = 64 << 10  # small parts so a few hundred KB exercise multipart and many ranges


def args_ns(**kw):
    d = dict(prefix="test/", files=3, window=3, dry_run=False)
    d.update(kw)
    return type("A", (), d)()


class Base(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
        os.environ["NO_PROXY"] += ",127.0.0.1"
        self.srv, self.st, self.base = fake_r2.serve()
        self.s3 = W.S3(self.base, "fv-weights-test", fake_r2.ACCESS, fake_r2.SECRET)
        self.tmp = Path(tempfile.mkdtemp(prefix="r2w-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(self.srv.shutdown)

    def make_tree(self, files):
        src = self.tmp / "src"
        for rel, data in files.items():
            p = src / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        return src

    def publish(self, dest, rev, src, **kw):
        pub = W.Publisher(self.s3, args_ns(**kw))
        pub.part_size = PART
        files = W.dir_sources(dest, src)
        return pub.publish(dest, rev, {"kind": "dir", "path": str(src)}, files)

    def plan(self, *dests, ttl=600):
        trees = []
        for d in dests:
            m = W.load_manifest(self.s3, "test/", d)
            trees.append({"dest": d, "rev": m["rev"], "part_size": m["part_size"], "total_bytes": m["total_bytes"],
                          "files": [dict(f, url=W.presign(self.base, self.s3.bucket, W.obj_key("test/", d, m["rev"], f["path"]), fake_r2.ACCESS, fake_r2.SECRET, ttl)) for f in m["files"]]})
        return {"format": F.PLAN_FORMAT, "expires_at": int(time.time() + ttl), "trees": trees}

    def fetch(self, plan, root, *extra, plan_url=None):
        pf = self.tmp / "plan.json"
        pf.write_text(json.dumps(plan))
        a = ["--root", str(root), "--concurrency", "6", "--retries", "4", "--min-free-gb", "0", "--status", str(self.tmp / "status.json")]
        a += ["--plan-url", plan_url, "--token-env", ""] if plan_url else ["--plan", str(pf)]
        return F.main(a + list(extra))

    def status(self):
        return json.loads((self.tmp / "status.json").read_text())

    def range_gets(self):
        """The fetcher's ranged GETs of tree objects (not the publisher's manifest reads)."""
        return [(k, a) for k, a in self.st.gets if a is not None and "/weights/" in k]


def rnd(n, seed):
    out = bytearray()
    h = hashlib.sha256(str(seed).encode()).digest()
    while len(out) < n:
        h = hashlib.sha256(h).digest()
        out += h * 64
    return bytes(out[:n])


TREE = {
    "config.json": b'{"a": 1}',
    "tokenizer/tok.json": rnd(5000, 1),
    "text_encoder/model-00001-of-00002.safetensors": rnd(300_000, 2),
    "text_encoder/model-00002-of-00002.safetensors": rnd(150_001, 3),
    "transformer/diffusion_pytorch_model.safetensors": rnd(700_000, 4),
    "vae/empty.bin": b"",
}


class SigV4(unittest.TestCase):
    """AWS's published SigV4 examples (docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html, …-query-string-auth.html)."""

    K, S = "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    WHEN = datetime.datetime(2013, 5, 24, tzinfo=datetime.timezone.utc)

    def test_header_auth_get_object(self):
        h = W.sign_headers("GET", "examplebucket.s3.amazonaws.com", "/test.txt", {}, {"range": "bytes=0-9"}, W.EMPTY_SHA, self.K, self.S, self.WHEN, "us-east-1")
        self.assertTrue(h["authorization"].endswith("Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"))

    def test_presigned_get(self):
        u = W.presign_path("https", "examplebucket.s3.amazonaws.com", "/test.txt", self.K, self.S, 86400, self.WHEN, "us-east-1")
        self.assertTrue(u.endswith("X-Amz-Signature=aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404"))


class Publish(Base):
    def test_publish_dir_manifest_and_idempotent(self):
        src = self.make_tree(TREE)
        m = self.publish("demo", "eu-20261011", src)
        self.assertEqual(m["format"], "fv-weights-manifest/1")
        self.assertEqual(m["total_bytes"], sum(len(v) for v in TREE.values()))
        by = {f["path"]: f for f in m["files"]}
        self.assertEqual(set(by), set(TREE))
        for p, data in TREE.items():
            self.assertEqual(by[p]["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(len(by[p]["parts"]), -(-len(data) // PART))
            if data:
                self.assertEqual(by[p]["parts"][0], hashlib.sha256(data[:PART]).hexdigest())
            self.assertEqual(self.st.objects[("fv-weights-test", f"test/weights/demo/eu-20261011/{p}")][0], data)
        # second run: the manifest exists, nothing is written
        n = len(self.st.objects)
        self.publish("demo", "eu-20261011", src)
        self.assertEqual(len(self.st.objects), n)
        self.assertEqual(self.st.bad_auth, 0)

    def test_resume_skips_published_files(self):
        src = self.make_tree(TREE)
        pub = W.Publisher(self.s3, args_ns())
        pub.part_size = PART
        files = W.dir_sources("demo", src)
        big = [f for f in files if f.path.startswith("transformer/")][0]
        pub.publish_file("demo", "r1", big)  # an earlier run that stopped after one file
        pub2 = W.Publisher(self.s3, args_ns())
        pub2.part_size = PART
        m = pub2.publish("demo", "r1", {"kind": "dir"}, W.dir_sources("demo", src))
        self.assertEqual(pub2.skipped, 1)
        self.assertEqual(len(m["files"]), len(TREE))

    def test_an_object_without_its_record_is_adopted_only_when_its_bytes_match(self):
        src = self.make_tree(TREE)
        big = "transformer/diffusion_pytorch_model.safetensors"
        self.st.objects[("fv-weights-test", f"test/weights/demo/r1/{big}")] = (TREE[big], {})  # a run stopped before the record
        m = self.publish("demo", "r1", src)
        self.assertEqual(next(f for f in m["files"] if f["path"] == big)["sha256"], hashlib.sha256(TREE[big]).hexdigest())
        self.assertIn(("fv-weights-test", f"test/parts/demo/r1/{big}.json"), self.st.objects)
        bad = bytearray(TREE["config.json"])
        bad[0] ^= 1
        self.st.objects[("fv-weights-test", "test/weights/demo/r2/config.json")] = (bytes(bad), {})
        with self.assertRaises(W.Fail):
            self.publish("demo", "r2", src)
        self.assertEqual(self.st.objects[("fv-weights-test", "test/weights/demo/r2/config.json")][0], bytes(bad))

    def test_existing_object_with_other_size_is_never_replaced(self):
        src = self.make_tree(TREE)
        self.st.objects[("fv-weights-test", "test/weights/demo/r1/config.json")] = (b"other bytes!", {})
        with self.assertRaises(W.Fail):
            self.publish("demo", "r1", src)
        self.assertEqual(self.st.objects[("fv-weights-test", "test/weights/demo/r1/config.json")][0], b"other bytes!")
        self.assertNotIn(("fv-weights-test", "test/manifests/demo/r1.json"), self.st.objects)

    def test_hash_mismatch_aborts_before_publishing(self):
        src = self.make_tree(TREE)
        files = W.dir_sources("demo", src)
        big = [f for f in files if f.path.startswith("transformer/")][0]
        big.sha256 = "0" * 64  # what the Hub says the bytes must hash to
        pub = W.Publisher(self.s3, args_ns())
        pub.part_size = PART
        with self.assertRaises(W.Fail):
            pub.publish_file("demo", "r1", big)
        self.assertNotIn(("fv-weights-test", "test/weights/demo/r1/" + big.path), self.st.objects)
        self.assertEqual(self.st.uploads, {})  # aborted

    def test_r2_rejects_a_part_whose_body_does_not_match_its_signature(self):
        uid = self.s3.mp_create("test/x", {})
        with self.assertRaises(W.S3Error) as e:
            self.s3.mp_part("test/x", uid, 1, b"abc", hashlib.sha256(b"abd").hexdigest())
        self.assertEqual(e.exception.code, "XAmzContentSHA256Mismatch")

    def test_recorded_hash_is_checked(self):
        src = self.make_tree({"a.bin": b"hello"})
        f = W.dir_sources("demo", src)[0]
        f.recorded = ("md5", [hashlib.md5(b"other").hexdigest()])
        pub = W.Publisher(self.s3, args_ns())
        with self.assertRaises(W.Fail):
            pub.publish_file("demo", "r1", f)
        f.recorded = ("md5", [hashlib.md5(b"hello").hexdigest()])
        self.assertEqual(pub.publish_file("demo", "r1", f)["sha256"], hashlib.sha256(b"hello").hexdigest())

    def test_compare_dir_by_part_hashes(self):
        src = self.make_tree(TREE)
        self.publish("demo", "r1", src)
        a = lambda **kw: type("A", (), dict(dict(prefix="test/", dest="demo", dir=str(src), rev="r1", sizes_only=False, workers=4), **kw))()  # noqa: E731
        self.assertEqual(W.cmd_compare_dir(self.s3, a()), 0)
        big = src / "transformer/diffusion_pytorch_model.safetensors"
        b = bytearray(big.read_bytes())
        b[PART * 5 + 3] ^= 1
        big.write_bytes(bytes(b))
        self.assertEqual(W.cmd_compare_dir(self.s3, a(sizes_only=True)), 0)
        self.assertEqual(W.cmd_compare_dir(self.s3, a()), 1)

    def test_verify(self):
        src = self.make_tree(TREE)
        m = self.publish("demo", "r1", src)
        self.assertEqual(W.verify_tree(self.s3, "test/", m, True, 4), [])
        key = ("fv-weights-test", "test/weights/demo/r1/transformer/diffusion_pytorch_model.safetensors")
        data, meta = self.st.objects[key]
        self.st.objects[key] = (data[:-1] + b"\0", meta)
        self.assertEqual(W.verify_tree(self.s3, "test/", m, False, 4), [])  # sizes still match
        bad = W.verify_tree(self.s3, "test/", m, True, 4)
        self.assertEqual(len(bad), 1)
        self.assertIn("transformer/", bad[0])


class Fetch(Base):
    def setUp(self):
        super().setUp()
        self.src = self.make_tree(TREE)
        self.publish("demo", "r1", self.src)
        self.root = self.tmp / "weights"

    def assert_tree(self, dest="demo", files=TREE, rev="r1"):
        d = self.root / dest
        for p, data in files.items():
            self.assertEqual((d / p).read_bytes(), data, p)
        self.assertTrue((d / ".complete").is_file())
        self.assertEqual((d / ".fv-revision").read_text(), rev)
        self.assertFalse((self.root / ".r2-partial").exists())
        lines = (d / "sha256.txt").read_text().splitlines()
        self.assertEqual(len(lines), len(files))

    def test_fetch_whole_tree_then_present(self):
        plan = self.plan("demo")
        self.assertEqual(self.fetch(plan, self.root), 0)
        self.assert_tree()
        n = len(self.range_gets())
        self.assertEqual(n, sum(-(-len(v) // PART) for v in TREE.values()))
        self.assertEqual(self.fetch(plan, self.root), 0)  # present: no GET
        self.assertEqual(len(self.range_gets()), n)

    def test_order_small_files_and_text_encoder_before_transformer(self):
        self.assertEqual(self.fetch(self.plan("demo"), self.root, "--concurrency", "1"), 0)
        order = [k.split("/r1/", 1)[1] for k, _ in self.range_gets()]
        first_tr = next(i for i, p in enumerate(order) if p.startswith("transformer/"))
        last_te = max(i for i, p in enumerate(order) if p.startswith("text_encoder/"))
        self.assertLess(last_te, first_tr)
        self.assertEqual(order[0], "config.json")

    def test_corrupt_parts_are_fetched_again(self):
        plan = self.plan("demo")
        self.st.faults["corrupt"] = 3
        self.assertEqual(self.fetch(plan, self.root), 0)
        self.assert_tree()
        self.assertGreaterEqual(self.status()["corrupt_parts"], 3)

    def test_cut_connections_and_500s_are_retried(self):
        plan = self.plan("demo")
        self.st.faults["cut"] = 3
        self.st.faults["err500"] = 2
        self.assertEqual(self.fetch(plan, self.root), 0)
        self.assert_tree()
        self.assertGreaterEqual(self.status()["retries"], 5)

    def test_gives_up_after_retries(self):
        plan = self.plan("demo")
        self.st.faults["corrupt"] = 1000
        self.assertEqual(self.fetch(plan, self.root), 1)
        self.assertFalse((self.root / "demo").exists())

    def test_resume_fetches_only_bad_parts(self):
        plan = self.plan("demo")
        t = plan["trees"][0]
        stage = self.root / ".r2-partial" / "demo@r1"
        big = "transformer/diffusion_pytorch_model.safetensors"
        data = bytearray(TREE[big])
        data[PART * 3 + 10] ^= 0xFF  # part 3 bad
        data[PART * 7:PART * 8] = b"\0" * PART  # part 7 never written
        (stage / "transformer").mkdir(parents=True)
        (stage / big).write_bytes(bytes(data))
        self.assertEqual(self.fetch(plan, self.root), 0)
        self.assert_tree()
        got = sorted(a // PART for k, a in self.range_gets() if k.endswith(big))
        self.assertEqual(got, [3, 7])
        nparts = sum(len(f["parts"]) for f in t["files"])
        self.assertEqual(self.status()["resumed_parts"], -(-len(data) // PART) - 2)
        self.assertEqual(len(self.range_gets()), nparts - (-(-len(data) // PART)) + 2)

    def test_expired_urls_are_refreshed_from_the_plan_url(self):
        # presigned URLs dated before the next whole second are refused (X-Amz-Date has 1 s resolution)
        self.st.reject_before = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0) + datetime.timedelta(seconds=1)
        old = self.plan("demo")
        time.sleep(1.2)
        # the first plan (from --plan-url) is stale too: hand it out once, then fresh ones
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            return old if calls["n"] == 1 else self.plan("demo")
        self.st.plan_fn = fn
        self.assertEqual(self.fetch(None, self.root, plan_url=self.base + "/plan"), 0)
        self.assert_tree()
        self.assertGreaterEqual(self.status()["refreshes"], 1)

    def test_stale_tree_is_replaced_and_nested_tree_goes_in_after_its_parent(self):
        self.make_tree({})
        child = {"model.safetensors": rnd(200_000, 9), "manifest.json": b"{}"}
        csrc = self.tmp / "child"
        for p, d in child.items():
            (csrc / p).parent.mkdir(parents=True, exist_ok=True)
            (csrc / p).write_bytes(d)
        self.publish("demo/text_encoder_fp8", "eu-1", csrc)
        (self.root / "demo").mkdir(parents=True)
        (self.root / "demo" / "leftover.bin").write_bytes(b"x")  # no .complete: stale
        self.assertEqual(self.fetch(self.plan("demo/text_encoder_fp8", "demo"), self.root), 0)
        self.assert_tree()
        self.assertFalse((self.root / "demo" / "leftover.bin").exists())
        self.assertEqual((self.root / "demo" / "text_encoder_fp8" / "model.safetensors").read_bytes(), child["model.safetensors"])
        self.assertEqual((self.root / "demo" / "text_encoder_fp8" / ".fv-revision").read_text(), "eu-1")

    def test_whole_file_mismatch_fails(self):
        plan = self.plan("demo")
        f = [x for x in plan["trees"][0]["files"] if x["path"].startswith("transformer/")][0]
        f["sha256"] = "1" * 64  # parts fine, whole file not: refetched once, then a failure
        self.assertEqual(self.fetch(plan, self.root), 1)
        self.assertFalse((self.root / "demo").exists())

    def test_unsafe_paths_are_refused(self):
        plan = self.plan("demo")
        plan["trees"][0]["files"][0]["path"] = "../escape"
        self.assertEqual(self.fetch(plan, self.root), 1)
        self.assertFalse((self.tmp / "escape").exists())


class HubViaDir(Base):
    """publish-dir of a Hub tree (the volume copy): the Hub listing at the pinned revision decides the files and hashes."""

    def setUp(self):
        super().setUp()
        self.rev = "a" * 40
        self.files = {"cfg.json": b'{"k": 1}', "w/model.bin": rnd(200_000, 7)}
        self.orig = (W.hub_rows, W.hub_list)
        W.hub_rows = lambda: {"demo": ("org/demo", self.rev, ["*.bin", "cfg.json", "w/*"])}

        def blob(b):
            return hashlib.sha1(f"blob {len(b)}\0".encode() + b).hexdigest()
        self.listing = [("cfg.json", len(self.files["cfg.json"]), None, blob(self.files["cfg.json"])), ("w/model.bin", 200_000, hashlib.sha256(self.files["w/model.bin"]).hexdigest(), "x")]
        W.hub_list = lambda repo, rev: list(self.listing)

    def tearDown(self):
        W.hub_rows, W.hub_list = self.orig
        super().tearDown()

    def pub(self, root):
        rev, source, files = W.hub_dir_sources("demo", root, self.s3.conns)
        pub = W.Publisher(self.s3, args_ns())
        pub.part_size = PART
        return pub, pub.publish("demo", rev, source, files)

    def test_plain_layout_checked_against_the_hub_and_carries_the_pinned_revision(self):
        src = self.make_tree(dict(self.files, **{"w/extra.bin": b"not at this revision"}))
        pub, m = self.pub(src)
        self.assertEqual((m["rev"], m["source"]["kind"]), (self.rev, "hub-via-dir"))
        self.assertEqual(sorted(f["path"] for f in m["files"]), ["cfg.json", "w/model.bin"])
        self.assertEqual(pub.checked, {"cfg.json": ["hub-git-blob-sha1"], "w/model.bin": ["hub-lfs-sha256"]})

    def test_hf_cache_layout_read_through_the_snapshot(self):
        snap = self.tmp / "src" / "models--org--demo" / "snapshots" / self.rev
        for p, b in self.files.items():
            (snap / p).parent.mkdir(parents=True, exist_ok=True)
            (snap / p).write_bytes(b)
        (self.tmp / "src" / "README").write_bytes(b"x")
        _, m = self.pub(self.tmp / "src")
        self.assertEqual(len(m["files"]), 2)

    def test_a_volume_file_that_differs_from_the_hub_is_refused(self):
        src = self.make_tree(dict(self.files, **{"w/model.bin": rnd(200_000, 8)}))
        with self.assertRaises(W.Fail):
            self.pub(src)
        self.assertNotIn(("fv-weights-test", f"test/manifests/demo/{self.rev}.json"), self.st.objects)

    def test_dot_files_the_hub_lists_are_kept(self):
        def blob(b):
            return hashlib.sha1(f"blob {len(b)}\0".encode() + b).hexdigest()
        ga = b"*.bin filter=lfs\n"
        self.listing.append((".gitattributes", len(ga), None, blob(ga)))
        W.hub_rows = lambda: {"demo": ("org/demo", self.rev, ["*"])}  # a whole-repo glob, as longlive-plug/minimax-h3-few-step
        src = self.make_tree(dict(self.files, **{".gitattributes": ga, ".complete": b"x"}))
        _, m = self.pub(src)
        self.assertEqual(sorted(f["path"] for f in m["files"]), [".gitattributes", "cfg.json", "w/model.bin"])

    def test_a_missing_file_is_refused(self):
        src = self.make_tree({"cfg.json": self.files["cfg.json"]})
        with self.assertRaises(W.Fail):
            self.pub(src)


class Archive(Base):
    def test_archive_copies_everything_with_an_index_and_is_idempotent(self):
        src = self.make_tree({"a.txt": b"a", ".hidden/b.log": b"bb", "c/d.bin": rnd(150_000, 3)})
        a = type("A", (), dict(prefix="test/", files=2, window=2, dry_run=False, name="runs", dir=str(src), label="eu-volume"))()
        self.assertEqual(W.cmd_archive(self.s3, a), 0)
        idx = json.loads(self.st.objects[("fv-weights-test", "test/archive/eu-volume/runs.index.json")][0])
        self.assertEqual(sorted(f["path"] for f in idx["files"]), [".hidden/b.log", "a.txt", "c/d.bin"])
        self.assertEqual(self.st.objects[("fv-weights-test", "test/archive/eu-volume/runs/c/d.bin")][0], rnd(150_000, 3))
        n = len(self.st.objects)
        self.assertEqual(W.cmd_archive(self.s3, a), 0)
        self.assertEqual(len(self.st.objects), n)
        self.assertFalse(any(k.startswith("test/manifests/") for _, k in self.st.objects))

    def test_compare_archive_against_the_directory(self):
        src = self.make_tree({"a.txt": b"a", "empty": b"", ".hidden/b.log": b"bb", "c/d.bin": rnd(150_000, 4)})
        a = type("A", (), dict(prefix="test/", files=2, window=2, dry_run=False, name="libs", dir=str(src), label="eu-volume"))()
        self.assertEqual(W.cmd_archive(self.s3, a), 0)
        c = type("C", (), dict(prefix="test/", name="libs", dir=str(src), label="eu-volume", sizes_only=False, workers=4))()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(W.cmd_compare_archive(self.s3, c), 0)
        self.assertIn("4/4 files equal", out.getvalue())
        # same size, one byte changed on disk: the part hash catches it
        p = Path(src) / "c/d.bin"
        b = bytearray(p.read_bytes())
        b[100] ^= 1
        p.write_bytes(bytes(b))
        with contextlib.redirect_stdout(io.StringIO()) as o:
            self.assertEqual(W.cmd_compare_archive(self.s3, c), 1)
        self.assertIn("DIFF c/d.bin: part 0 differs", o.getvalue())
        p.write_bytes(rnd(150_000, 4))
        (Path(src) / "new.txt").write_bytes(b"n")
        with contextlib.redirect_stdout(io.StringIO()) as o:
            self.assertEqual(W.cmd_compare_archive(self.s3, c), 1)
        self.assertIn("new.txt: on disk, not in the archive", o.getvalue())


if __name__ == "__main__":
    unittest.main()
