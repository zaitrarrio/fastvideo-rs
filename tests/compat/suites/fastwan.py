#!/usr/bin/env python3
"""The FastWan Video API as its streaming client drives it (design §4.2,
research-minimax-fastvideo §2.2): aiohttp with an optional Bearer key,
`GET /health` (truthy `model_loaded`), `GET /` (`model`), `POST /generate`
`{prompt,width,height,num_frames,fps,seed}` -> `prompt_id`, poll
`GET /status/{id}` through `queued|processing` to `completed|failed`,
`GET /video/{id}` raw MP4, best-effort `DELETE /video/{id}`. Rejections are
400/413/415/422 with FastAPI `{"detail": ...}`; anything else is "server
unreachable" to that client.
"""

import asyncio
import json

import aiohttp

from common import Suite, is_mp4

REJECTING = {400, 413, 415, 422}
PENDING = {"queued", "processing"}
# The client's defaults are FASTWAN_SIZE 1280x704 and FASTWAN_FPS 24 (a
# FastWan-5B deployment), 49..121 frames on 4k+1. fv-serve's fake engine
# serves a 480p Wan (832x480 pixel budget, 16 fps), so this runs as the client
# configured with FASTWAN_SIZE=832x480 FASTWAN_FPS=16; the defaults are 400s
# (rejections) on this model, checked below.
WIDTH, HEIGHT, FPS = 832, 480, 16


class Rejected(Exception):
    pass


class Unreachable(Exception):
    pass


async def error_detail(resp):
    text = await resp.text(errors="replace")
    try:
        body = json.loads(text)
    except ValueError:
        return text[:200]
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str):
        return detail[:300]
    if isinstance(detail, list):
        return "; ".join(str(i.get("msg", i)) if isinstance(i, dict) else str(i) for i in detail)[:300]
    return text[:200]


async def call(http, base, method, path, body=None, raw=False):
    try:
        async with http.request(method, base + path, json=body, timeout=aiohttp.ClientTimeout(total=300)) as r:
            if r.status == 200:
                return await (r.read() if raw else r.json())
            status, detail = r.status, await error_detail(r)
    except (aiohttp.ClientError, TimeoutError, ValueError) as e:
        raise Unreachable(f"{method} {path}: {e}") from e
    if status in REJECTING:
        raise Rejected(status, detail)
    raise Unreachable(status, detail)


async def generate(http, base, prompt, frames, seed):
    job = await call(http, base, "POST", "/generate", {
        "prompt": prompt, "width": WIDTH, "height": HEIGHT, "num_frames": frames, "fps": FPS, "seed": seed,
    })
    job_id = str(job.get("prompt_id") or "")
    statuses = [job.get("status")]
    while (status := job.get("status")) != "completed":
        if status == "failed":
            return job_id, statuses, job, None
        if status not in PENDING:
            raise AssertionError(f"unknown status {status!r}")
        await asyncio.sleep(0.1)
        job = await call(http, base, "GET", f"/status/{job_id}")
        statuses.append(job.get("status"))
    video = await call(http, base, "GET", f"/video/{job_id}", raw=True)
    return job_id, statuses, job, video


async def run(s, a):
    headers = {"Authorization": f"Bearer {a.key}"}
    async with aiohttp.ClientSession(headers=headers) as http:
        health = await call(http, a.base, "GET", "/health")
        s.check(isinstance(health, dict) and health.get("model_loaded"), f"/health {health}")
        service = await call(http, a.base, "GET", "/")
        s.check(service.get("model") == "fake-wan", f"GET / {service}")
        s.info["model"] = service["model"]
        s.ok("GET /health model_loaded + GET / model")

        job_id, statuses, job, video = await generate(http, a.base, "compat: a lantern on a rope bridge", 81, 1000)
        s.check(job_id and video is not None, f"generate: {job}")
        s.check(set(statuses) <= PENDING | {"completed"}, f"statuses {statuses}")
        s.check(is_mp4(video), f"GET /video is an MP4 ({video[:16]!r})")
        s.ok("generate -> status -> video")

        # Two clips back to back, at the client's frame bounds.
        (j1, _, _, v1), (j2, _, _, v2) = await asyncio.gather(
            generate(http, a.base, "compat: min clip", 49, 1001),
            generate(http, a.base, "compat: max clip", 121, 1002),
        )
        s.check(v1 and v2 and j1 != j2, "two concurrent jobs")
        s.ok("49 and 121 frames concurrently")

        # Best-effort delete, then the job is gone.
        await call(http, a.base, "DELETE", f"/video/{job_id}")
        try:
            await call(http, a.base, "GET", f"/status/{job_id}")
            s.check(False, "deleted job still has a status")
        except Unreachable as e:
            s.check(e.args[0] == 404, f"deleted: {e}")
        s.ok("DELETE /video")

        # A failed job: `error` is a string.
        _, _, job, video = await generate(http, a.base, "[fake:fail] compat", 81, 1003)
        s.check(job["status"] == "failed" and isinstance(job.get("error"), str) and video is None, f"failed {job}")
        s.ok("failed job with a string error")

        # Rejections: off-grid frames and odd sizes are 4xx with a string detail.
        for body, what in (
            ({"prompt": "p", "width": WIDTH, "height": HEIGHT, "num_frames": 50, "fps": FPS, "seed": 1}, "off-grid frames"),
            ({"prompt": "", "width": WIDTH, "height": HEIGHT, "num_frames": 81, "fps": FPS, "seed": 1}, "empty prompt"),
            ({"prompt": "p", "width": 1280, "height": 704, "num_frames": 81, "fps": FPS, "seed": 1}, "over the pixel budget"),
            ({"prompt": "p", "width": WIDTH, "height": HEIGHT, "num_frames": 81, "fps": 24, "seed": 1}, "fps 24 on a 16 fps Wan"),
            ({"prompt": "p", "width": 833, "height": HEIGHT, "num_frames": 81, "fps": FPS, "seed": 1}, "odd width"),
        ):
            try:
                await call(http, a.base, "POST", "/generate", body)
                s.check(False, f"{what} accepted")
            except Rejected as e:
                s.check(isinstance(e.args[1], str) and e.args[1], f"{what}: detail {e.args}")
        s.ok("400-class rejections with FastAPI detail")


def main(s, a):
    asyncio.run(run(s, a))


if __name__ == "__main__":
    Suite("fastwan").run(main)
