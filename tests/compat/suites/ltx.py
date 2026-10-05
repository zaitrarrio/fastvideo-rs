#!/usr/bin/env python3
"""LTX API through its documented Python `requests` snippets with the host
swapped (design §4.5, research-ltx-api §2): `POST /v2/text-to-video` ->
202 `{id, created_at}`, poll `GET /v2/text-to-video/{id}` to `completed`,
download `result.video_url` with no key; the V1 sync body; `/v1/upload`
then `PUT upload_url` with `required_headers` and the `ltx://` URI as
`image_uri`; the error bodies (401, 400 incl. a retake without
`video_uri`, 403 on the HDR / reframe stubs, 404) and `x-request-id`.
"""

import re

import requests

from common import Suite, is_mp4, png, poll

TERMINAL = {"completed", "failed"}


def main(s, a):
    url = a.base
    headers = {"Authorization": f"Bearer {a.key}", "Content-Type": "application/json"}

    # --- the async (V2) quickstart snippet ---
    payload = {
        "prompt": "compat: a red fox in the snow",
        "model": "ltx-2-5-fast",
        "duration": 6,
        "resolution": "1280x720",
    }
    response = requests.post(f"{url}/v2/text-to-video", json=payload, headers=headers, timeout=30)
    s.check(response.status_code == 202, f"submit {response.status_code} {response.text}")
    s.check(re.fullmatch(r"[0-9a-f]{32}", response.headers.get("x-request-id", "")), response.headers)
    job = response.json()
    s.check(set(job) == {"id", "created_at"}, job)

    def status():
        r = requests.get(f"{url}/v2/text-to-video/{job['id']}", headers=headers, timeout=30)
        s.check(r.status_code == 200, f"status {r.status_code} {r.text}")
        return r.json()

    seen = set()
    st = poll(status, lambda j: seen.add(j["status"]) or j["status"] in TERMINAL)
    s.check(st["status"] == "completed" and st["completed_at"] and st["id"] == job["id"], st)
    s.check(seen <= {"pending", "processing", "completed"}, seen)
    video = requests.get(st["result"]["video_url"], timeout=60)  # no Authorization
    s.check(video.status_code == 200 and is_mp4(video.content), f"download {video.status_code}")
    r = requests.get(f"{url}/v2/image-to-video/{job['id']}", headers=headers, timeout=30)
    s.check(r.status_code == 404, f"wrong endpoint segment: {r.status_code}")
    s.ok("v2 text-to-video: submit, poll, download")

    # --- the sync (V1) snippet: the body is the MP4 ---
    response = requests.post(f"{url}/v1/text-to-video", json={**payload, "model": "ltx-2-3-fast", "generate_audio": False}, headers=headers, timeout=300)
    s.check(response.status_code == 200, f"v1 {response.status_code} {response.text[:200]}")
    s.check(response.headers["content-type"].startswith("video/mp4") and is_mp4(response.content), response.headers)
    s.ok("v1 text-to-video (sync, silent)")

    # --- upload, then image-to-video with the ltx:// URI ---
    up = requests.post(f"{url}/v1/upload", headers={"Authorization": f"Bearer {a.key}"}, timeout=30)
    s.check(up.status_code == 200, f"upload {up.status_code} {up.text}")
    u = up.json()
    s.check(u["storage_uri"].startswith("ltx://uploads/") and u["expires_at"], u)
    put = requests.put(u["upload_url"], data=png(1280, 720), headers={"Content-Type": "image/png", **u["required_headers"]}, timeout=30)
    s.check(put.status_code in (200, 201, 204), f"PUT {put.status_code} {put.text}")
    r = requests.post(
        f"{url}/v2/image-to-video",
        json={"prompt": "compat: the frame comes alive", "model": "ltx-2-3-fast", "duration": 6, "resolution": "1280x720", "image_uri": u["storage_uri"]},
        headers=headers,
        timeout=30,
    )
    s.check(r.status_code == 202, f"i2v {r.status_code} {r.text}")
    i2v = r.json()["id"]
    st = poll(lambda: requests.get(f"{url}/v2/image-to-video/{i2v}", headers=headers, timeout=30).json(), lambda j: j["status"] in TERMINAL)
    s.check(st["status"] == "completed", st)
    s.ok("v1/upload + PUT required_headers + v2 image-to-video(ltx://)")

    # data URI image input
    import base64

    uri = "data:image/png;base64," + base64.b64encode(png(1280, 720, (20, 160, 60))).decode()
    r = requests.post(
        f"{url}/v2/image-to-video",
        json={"prompt": "compat: data uri", "model": "ltx-2-3-pro", "duration": 6, "resolution": "1920x1080", "image_uri": uri},
        headers=headers,
        timeout=30,
    )
    s.check(r.status_code == 202, f"i2v data uri {r.status_code} {r.text}")
    st = poll(lambda: requests.get(f"{url}/v2/image-to-video/{r.json()['id']}", headers=headers, timeout=30).json(), lambda j: j["status"] in TERMINAL)
    s.check(st["status"] == "completed", st)
    s.ok("v2 image-to-video(data URI, 1920x1080)")

    # a failed job
    r = requests.post(f"{url}/v2/text-to-video", json={**payload, "prompt": "[fake:fail] compat"}, headers=headers, timeout=30)
    st = poll(lambda: requests.get(f"{url}/v2/text-to-video/{r.json()['id']}", headers=headers, timeout=30).json(), lambda j: j["status"] in TERMINAL)
    s.check(st["status"] == "failed" and st["error"]["type"] and st["error"]["message"] and st["completed_at"], st)
    s.ok("failed job error {type, message}")

    # errors: {"type":"error","error":{"type","message"}}
    def err(r, http, typ):
        b = r.json()
        s.check(r.status_code == http and b["type"] == "error" and b["error"]["type"] == typ and b["error"]["message"], f"{r.status_code} {b}")

    err(requests.post(f"{url}/v2/text-to-video", json=payload, headers={"Authorization": "Bearer wrong"}, timeout=30), 401, "authentication_error")
    err(requests.post(f"{url}/v2/text-to-video", json={k: v for k, v in payload.items() if k != "duration"}, headers=headers, timeout=30), 400, "invalid_request_error")
    err(requests.post(f"{url}/v2/text-to-video", json={**payload, "model": "ltx-2-fast"}, headers=headers, timeout=30), 400, "invalid_request_error")
    err(requests.post(f"{url}/v2/text-to-video", json={**payload, "camera_motion": "dolly_in"}, headers=headers, timeout=30), 400, "invalid_request_error")
    # Retake is served (LTX-2.5 distilled): an empty EditVideoRequest misses
    # the required video_uri (OAS) -> 400. The 403 permission_error contract
    # covers the endpoints with no engine path (stubs: HDR, reframe).
    r = requests.post(f"{url}/v2/retake", json={}, headers=headers, timeout=30)
    err(r, 400, "invalid_request_error")
    s.check("video_uri" in r.json()["error"]["message"], r.json())
    for ep in ("video-to-video-hdr", "video-to-video-reframe"):
        err(requests.post(f"{url}/v2/{ep}", json={"video_uri": "https://example.com/v.mp4"}, headers=headers, timeout=30), 403, "permission_error")
    err(requests.get(f"{url}/v2/text-to-video/00000000-0000-0000-0000-000000000000", headers=headers, timeout=30), 404, "not_found_error")
    s.ok("error bodies (401, 400, 403, 404)")


if __name__ == "__main__":
    Suite("ltx").run(main)
