#!/usr/bin/env python3
"""Drive the real Python `fal-client` against a local fv fal server.

Run by `tests/queue_compat.rs` (design §7.5): the Rust test serves the fal
router over plain HTTP on 127.0.0.1:<upstream> backed by the fake engine,
with `public_base = https://127.0.0.1:<tls>`. `fal_client` only speaks
https (`https://{FAL_QUEUE_RUN_HOST}/`, fal §12.2), so this script

1. makes a throwaway self-signed CA/cert for 127.0.0.1 with `openssl`,
2. terminates TLS on 127.0.0.1:<tls> and forwards bytes to the upstream,
3. sets `SSL_CERT_FILE` (httpx trusts it), `FAL_QUEUE_RUN_HOST=127.0.0.1:<tls>`
   and `FAL_RUN_HOST=127.0.0.1:<tls>/run` before importing `fal_client`,
4. exercises submit, status(with_logs), result, get_handle, subscribe,
   run (sync), cancel, the error mapping, and the async client.

Exits non-zero on the first failed check; prints a JSON summary on success.
"""

import argparse
import asyncio
import json
import os
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import uuid

APP = "minimax/h3-max/text-to-video"


def make_cert(d):
    cert, key = os.path.join(d, "cert.pem"), os.path.join(d, "key.pem")
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
            "-keyout", key, "-out", cert, "-subj", "/CN=127.0.0.1",
            "-addext", "subjectAltName=IP:127.0.0.1,DNS:localhost",
            "-addext", "basicConstraints=critical,CA:TRUE",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return cert, key


def start_tls_forwarder(tls_port, upstream_port, cert, key):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    ready = threading.Event()

    async def pipe(r, w):
        try:
            while True:
                b = await r.read(65536)
                if not b:
                    break
                w.write(b)
                await w.drain()
        except Exception:
            pass
        finally:
            try:
                w.close()
            except Exception:
                pass

    async def handle(cr, cw):
        try:
            ur, uw = await asyncio.open_connection("127.0.0.1", upstream_port)
        except Exception:
            cw.close()
            return
        await asyncio.gather(pipe(cr, uw), pipe(ur, cw))

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        srv = loop.run_until_complete(asyncio.start_server(handle, "127.0.0.1", tls_port, ssl=ctx))
        ready.set()
        loop.run_until_complete(srv.serve_forever())

    threading.Thread(target=run, daemon=True).start()
    if not ready.wait(10):
        raise RuntimeError("TLS forwarder did not start")


def check(cond, what):
    if not cond:
        raise AssertionError(what)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=int, required=True)
    ap.add_argument("--tls-port", type=int, required=True)
    ap.add_argument("--key", required=True)
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="fv-fal-compat-")
    cert, key = make_cert(tmp)
    start_tls_forwarder(args.tls_port, args.upstream, cert, key)

    host = f"127.0.0.1:{args.tls_port}"
    os.environ["SSL_CERT_FILE"] = cert
    os.environ["FAL_KEY"] = args.key
    os.environ["FAL_QUEUE_RUN_HOST"] = host
    os.environ["FAL_RUN_HOST"] = f"{host}/run"
    for k in ("NO_PROXY", "no_proxy"):
        os.environ[k] = "127.0.0.1,localhost"

    import httpx
    import fal_client
    from fal_client import Completed, FalClientHTTPError, InProgress, Queued

    summary = {"fal_client": fal_client.__version__, "checks": []}

    def ok(name):
        summary["checks"].append(name)

    c = fal_client.SyncClient(key=args.key)

    # submit -> status(with_logs) polling -> result
    h = c.submit(APP, {"prompt": "compat t2v", "seed": 5})
    uuid.UUID(h.request_id)
    check(h.status_url == f"https://{host}/minimax/h3-max/requests/{h.request_id}/status", h.status_url)
    seen = set()
    deadline = time.time() + 120
    while True:
        st = h.status(with_logs=True)
        seen.add(type(st).__name__)
        if isinstance(st, Queued):
            check(isinstance(st.position, int), "Queued.position is an int")
        elif isinstance(st, InProgress):
            check(isinstance(st.logs, list), "InProgress.logs is a list")
        else:
            check(isinstance(st, Completed), f"unexpected status {st!r}")
            break
        check(time.time() < deadline, "timed out polling")
        time.sleep(0.05)
    check(isinstance(st.logs, list) and len(st.logs) > 0, f"Completed logs with_logs=True: {st.logs}")
    check(st.error is None and st.error_type is None, f"no error: {st}")
    check("inference_time" in st.metrics, f"metrics: {st.metrics}")
    ok("submit/status(with_logs)")
    st0 = h.status(with_logs=False)
    check(isinstance(st0, Completed) and st0.logs == [], f"with_logs=False: {st0}")
    ok("status(with_logs=False)")

    res = h.get()
    v = res["video"]
    check(v["content_type"] == "video/mp4", v)
    check(v["file_name"].endswith("_minimax-h3.mp4"), v)
    check(res["expanded_prompt"] is None and "inference" in res["timings"], res)
    body = httpx.get(v["url"], verify=cert).content  # unauthenticated download
    check(len(body) == v["file_size"], f"downloaded {len(body)} != {v['file_size']}")
    check(body[4:8] == b"ftyp", "an MP4 container")
    ok("result + download")

    # Handles rebuilt from a stored request id use the app-only path form.
    h2 = c.get_handle(APP, h.request_id)
    check(isinstance(h2.status(), Completed), "get_handle status")
    check(c.result(APP, h.request_id)["video"]["url"] == v["url"], "result() by id")
    check(isinstance(fal_client.status(APP, h.request_id, with_logs=True), Completed), "module-level status")
    ok("get_handle/result/status by id")

    # subscribe with queue updates
    updates = []
    res = c.subscribe(APP, {"prompt": "compat subscribe"}, with_logs=True, on_queue_update=updates.append, interval=0.05)
    check("video" in res and updates and isinstance(updates[-1], Completed), f"subscribe: {updates}")
    ok("subscribe")

    # run: the sync endpoint via FAL_RUN_HOST=<host>/run
    res = c.run(APP, {"prompt": "compat run"})
    check(res["video"]["url"].startswith(f"https://{host}/files/"), res)
    ok("run (sync)")

    # other endpoints of the app
    res = c.run("minimax/h3-max/image-to-video", {"prompt": "no image means t2v"})
    check("video" in res, res)
    res = c.run("minimax/h3-turbo/text-to-video", {"prompt": "turbo"})
    check("video" in res, res)
    ok("image-to-video + turbo app")

    # cancel: the second of two back-to-back jobs is still queued
    a = c.submit(APP, {"prompt": "compat cancel a"})
    b = c.submit(APP, {"prompt": "compat cancel b"})
    b.cancel()  # 202 CANCELLATION_REQUESTED
    st = b.status()
    while not isinstance(st, Completed):  # a running job stops at its next step
        time.sleep(0.05)
        st = b.status()
    check(isinstance(st, Completed) and st.error_type == "client_cancelled", f"cancelled: {st}")
    try:
        b.cancel()
        check(False, "second cancel must fail")
    except FalClientHTTPError as e:
        check(e.status_code == 400, f"ALREADY_COMPLETED is 400: {e.status_code}")
    try:
        b.get()
        check(False, "a cancelled result is an error")
    except FalClientHTTPError as e:
        check(e.status_code == 499 and e.error_type == "client_cancelled", f"{e.status_code} {e.error_type}")
    a.get()
    ok("cancel")

    # error mapping
    try:
        c.submit(APP, {"prompt": "p", "duration": 99})
        check(False, "duration 99 must be refused")
    except FalClientHTTPError as e:
        check(e.status_code == 422, e.status_code)
        check(isinstance(e.message, list) and e.message[0]["loc"] == ["body", "duration"], e.message)
    try:
        fal_client.status(APP, str(uuid.uuid4()))
        check(False, "unknown id")
    except FalClientHTTPError as e:
        check(e.status_code == 404, e.status_code)
    try:
        fal_client.SyncClient(key="wrong").submit(APP, {"prompt": "p"})
        check(False, "bad key")
    except FalClientHTTPError as e:
        check(e.status_code == 401 and e.error_type == "unauthorized", f"{e.status_code} {e.error_type}")
    try:
        c.run(APP, {"prompt": "[fake:fail]"})
        check(False, "engine failure")
    except FalClientHTTPError as e:
        check(e.status_code == 500 and e.error_type == "internal_server_error", f"{e.status_code} {e.error_type}")
    ok("errors")

    # webhook_url is accepted (delivery is covered by the Rust tests)
    h = c.submit(APP, {"prompt": "compat webhook"}, webhook_url="https://hooks.example.com/fal")
    h.get()
    ok("submit(webhook_url)")

    # async client
    async def run_async():
        ac = fal_client.AsyncClient(key=args.key)
        ah = await ac.submit(APP, {"prompt": "compat async"})
        async for st in ah.iter_events(with_logs=True, interval=0.05):
            pass
        check(isinstance(st, Completed), st)
        r = await ah.get()
        check("video" in r, r)
        r = await ac.run(APP, {"prompt": "compat async run"})
        check("video" in r, r)

    asyncio.run(run_async())
    ok("AsyncClient submit/iter_events/get/run")

    print(json.dumps(summary))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"COMPAT FAILED: {e}", file=sys.stderr)
        sys.exit(1)
