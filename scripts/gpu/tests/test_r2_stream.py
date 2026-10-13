"""Tests of r2-fetch.py --stream (the load overlapping the download, docs/ops/r2-weights.md §11) against fake_r2.

python3 -m unittest discover -s scripts/gpu/tests -p 'test_r2*.py' -v

What the loader relies on (crates/fastvideo-loader/src/stream.rs reads the same maps): a file's progress map exists
before the file does; a part's flag is set only once its bytes are on disk and matched the manifest's SHA-256; the
map goes only when the whole file is verified; a failed fetch marks every remaining map failed; a rerun resumes the
tree in place. Plus the order (ungated files, headers, text encoder, DiT, VAE) and the loader's `want`.
"""
import json
import os
import threading
import time
import unittest

import test_r2_weights as T

F = T.F
PART = T.PART
TREE = T.TREE
MAGIC = b"FVPARTS1"


def read_map(path):
    """(state, want, flags) of a progress map, or None when it is not there."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        return None
    assert raw[:8] == MAGIC, raw[:8]
    n = int.from_bytes(raw[24:32], "little")
    return raw[40], int.from_bytes(raw[32:40], "little"), raw[48:48 + n]


class Watcher(threading.Thread):
    """What a loader would see, polled as fast as Python goes: for every part a map says is checked, the bytes on
    disk must be the source's; a file with no map (once it exists) must be whole and right."""

    def __init__(self, root, files, dest="demo"):
        super().__init__(daemon=True)
        self.d = root / dest
        self.files = files
        self.stop = threading.Event()
        self.bad = []
        self.seen_partial = 0
        self.seen_whole = 0

    def run(self):
        while not self.stop.is_set():
            for rel, src in self.files.items():
                p = self.d / rel
                m = read_map(str(p) + ".fvparts")
                if m is None:
                    if p.is_file() and read_map(str(p) + ".fvparts") is None:
                        data = p.read_bytes()
                        if data != src:
                            self.bad.append((rel, "no map but the file is not whole", len(data)))
                        else:
                            self.seen_whole += 1
                    continue
                state, _, flags = m
                if not any(flags):
                    continue  # the data file may not even exist yet (the map comes first)
                with open(p, "rb") as f:
                    for i, x in enumerate(flags):
                        if x != 1:
                            continue
                        f.seek(i * PART)
                        got = f.read(min(PART, len(src) - i * PART))
                        if got != src[i * PART:(i + 1) * PART]:
                            self.bad.append((rel, i, "flagged before its bytes were right"))
                        elif state == 0 and 0 in flags:
                            self.seen_partial += 1


class Stream(T.Base):
    def setUp(self):
        super().setUp()
        self.src = self.make_tree(TREE)
        self.publish("demo", "r1", self.src)
        self.root = self.tmp / "weights"
        self.ready = self.tmp / "ready"

    def fetch_stream(self, plan, *extra):
        return self.fetch(plan, self.root, "--stream", "--stream-ready", str(self.ready), *extra)

    def order(self):
        return [(k.split("/r1/", 1)[1], a // PART) for k, a in self.range_gets()]

    def assert_whole(self, dest="demo", files=TREE):
        d = self.root / dest
        for p, data in files.items():
            self.assertEqual((d / p).read_bytes(), data, p)
            self.assertFalse(os.path.exists(str(d / p) + ".fvparts"), p)
        self.assertTrue((d / ".complete").is_file())
        self.assertFalse((d / ".fv-streaming").exists())
        self.assertFalse((self.root / ".r2-partial").exists())

    def test_in_place_maps_gone_and_ready(self):
        self.assertEqual(self.fetch_stream(self.plan("demo")), 0)
        self.assert_whole()
        self.assertEqual((self.root / "demo" / ".fv-revision").read_text(), "r1")
        self.assertTrue(self.ready.is_file())
        st = self.status()
        self.assertTrue(st["stream"] and st["ready"])
        n = len(self.range_gets())
        self.ready.unlink()
        self.assertEqual(self.fetch_stream(self.plan("demo")), 0)  # present: kept, no GET, ready at once
        self.assertEqual(len(self.range_gets()), n)
        self.assertTrue(self.ready.is_file())

    def test_order_ungated_files_then_headers_then_text_encoder_dit_vae(self):
        extra = {"vae/diffusion_pytorch_model.safetensors": T.rnd(140_000, 11)}
        self.publish("vdemo", "r1", self.make_tree(dict(TREE, **extra), "src2"))
        self.assertEqual(self.fetch(self.plan("vdemo"), self.root, "--stream", "--concurrency", "1", "--get-parts", "2"), 0)
        got = self.order()
        te1, te2 = "text_encoder/model-00001-of-00002.safetensors", "text_encoder/model-00002-of-00002.safetensors"
        tr, vae = "transformer/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors"
        self.assertEqual(got, [
            ("config.json", 0), ("tokenizer/tok.json", 0),  # what fv-serve reads unguarded: first
            (te1, 0), (te2, 0), (tr, 0), (vae, 0),  # every header
            (te1, 1), (te1, 3), (te2, 1),  # the text encoder, in offset order
            (tr, 1), (tr, 3), (tr, 5), (tr, 7), (tr, 9),  # the DiT
            (vae, 1),  # the VAE
        ])
        self.assert_whole("vdemo", dict(TREE, **extra))

    def test_a_part_is_flagged_only_after_its_hash_passed(self):
        # corrupt and cut GETs on a slow fetch: whatever a map says is checked must already be the right bytes
        self.st.get_delay_s = 0.02
        self.st.faults["corrupt"] = 5
        self.st.faults["cut"] = 2
        w = Watcher(self.root, TREE)
        w.start()
        try:
            rc = self.fetch_stream(self.plan("demo"), "--get-parts", "1", "--concurrency", "3")
        finally:
            w.stop.set()
            w.join()
        self.assertEqual(rc, 0)
        self.assertEqual(w.bad, [])
        self.assertGreater(w.seen_partial, 0, "the watcher never saw a file mid-fetch: the test proves nothing")
        self.assertGreaterEqual(self.status()["corrupt_parts"], 1)
        self.assertGreaterEqual(self.status()["retries"], 5)
        self.assert_whole()

    def test_a_dead_fetch_marks_its_maps_failed_and_a_rerun_resumes_in_place(self):
        self.st.die_after = 6
        self.assertEqual(self.fetch_stream(self.plan("demo"), "--concurrency", "1", "--get-parts", "1", "--retries", "1"), 1)
        d = self.root / "demo"
        self.assertEqual((d / ".fv-streaming").read_text(), "r1")
        self.assertFalse((d / ".complete").exists())
        tr = "transformer/diffusion_pytorch_model.safetensors"
        state, _, flags = read_map(str(d / tr) + ".fvparts")
        self.assertEqual(state, 2, "a loader waiting on this file must see the fetch failed")
        self.assertEqual(flags[0], 1)  # its header part came before the fetch died
        self.assertEqual(sum(flags), 1)
        self.assertEqual(self.status()["state"], "failed")
        n = len(self.range_gets())
        self.st.die_after = None
        self.assertEqual(self.fetch_stream(self.plan("demo"), "--concurrency", "1", "--get-parts", "1"), 0)
        self.assert_whole()
        total = sum(-(-len(v) // PART) for v in TREE.values())
        self.assertEqual(self.status()["resumed_parts"], 6)
        self.assertEqual(len(self.range_gets()) - n, total - 6)

    def test_the_loaders_want_moves_that_files_gets_first(self):
        self.st.get_delay_s = 0.25
        tr = "transformer/diffusion_pytorch_model.safetensors"
        mp = self.root / "demo" / (tr + ".fvparts")

        def loader():
            # wait for fv-serve's start signal, then block on part 5 of the DiT as a view would
            t = time.time()
            while not self.ready.exists() and time.time() - t < 20:
                time.sleep(0.01)
            with open(mp, "r+b") as f:
                f.seek(32)
                f.write((5 + 1).to_bytes(8, "little"))
        th = threading.Thread(target=loader, daemon=True)
        th.start()
        self.assertEqual(self.fetch_stream(self.plan("demo"), "--concurrency", "1", "--get-parts", "2"), 0)
        th.join()
        got = self.order()
        self.assertLess(got.index((tr, 5)), got.index(("text_encoder/model-00002-of-00002.safetensors", 1)), got)
        self.assertLess(got.index((tr, 5)), got.index((tr, 1)), got)
        self.assertGreaterEqual(self.status()["boosts"], 1)
        self.assert_whole()

    def test_ready_waits_for_the_ready_trees(self):
        aux = {"taew.safetensors": T.rnd(150_000, 12)}
        self.publish("auxiliary/tae", "r1", self.make_tree(aux, "aux"))
        self.st.get_delay_s = 0.05
        seen = {}

        def watch():
            t = time.time()
            while not self.ready.exists() and time.time() - t < 30:
                time.sleep(0.005)
            seen["aux"] = (self.root / "auxiliary" / "tae" / ".complete").is_file()
            seen["tr"] = (self.root / "demo" / ".complete").is_file()
        th = threading.Thread(target=watch, daemon=True)
        th.start()
        self.assertEqual(self.fetch_stream(self.plan("demo", "auxiliary/tae"), "--concurrency", "1"), 0)
        th.join()
        self.assertEqual(seen, {"aux": True, "tr": False})
        self.assert_whole("auxiliary/tae", aux)

    def test_stream_is_refused_on_an_add_only_cache(self):
        self.assertEqual(self.fetch_stream(self.plan("demo"), "--add-only", "--pod", "w1"), 1)
        self.assertFalse((self.root / "demo").exists())

    def test_without_stream_the_staging_path_is_unchanged(self):
        self.assertEqual(self.fetch(self.plan("demo"), self.root, "--stream-ready", str(self.ready)), 0)
        self.assert_whole()
        self.assertTrue(self.ready.is_file())
        self.assertFalse(self.status()["stream"])


if __name__ == "__main__":
    unittest.main()
