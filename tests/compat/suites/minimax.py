#!/usr/bin/env python3
"""MiniMax Video Generation V2 as MiniMax documents it (design §4.3,
research-minimax-fastvideo §1): Python `requests` with
`Authorization: Bearer`, `POST /v2/video_generation` -> `task_id`, the
query loop on `GET /v2/query/video_generation/{task_id}` until `succeeded`,
then download `task.content.url` with no key. Plus I2V with a data URI,
list, delete, the OaiError envelope, and `callback_url` (challenge echo,
then `{"task": ...}` on each status change).
"""

import base64
import json

import requests

from common import Receiver, Suite, is_mp4, png, poll

TERMINAL = {"succeeded", "failed", "cancelled"}


def main(s, a):
    api = a.base
    h = {"Authorization": f"Bearer {a.key}", "Content-Type": "application/json"}

    def create(body):
        return requests.post(f"{api}/v2/video_generation", headers=h, json=body, timeout=30)

    def query(task_id):
        r = requests.get(f"{api}/v2/query/video_generation/{task_id}", headers=h, timeout=30)
        r.raise_for_status()
        return r.json()["task"]

    # The documented t2v flow.
    r = create({
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": "compat: a paper lantern drifts over a night river"}],
        "resolution": "768P",
        "duration": 5,
        "ratio": "16:9",
    })
    s.check(r.status_code == 200, f"create {r.status_code} {r.text}")
    body = r.json()
    s.check(set(body) == {"task_id"} and body["task_id"].isdigit(), f"create body {body} (no base_resp in V2)")
    task_id = body["task_id"]
    seen = set()
    task = poll(lambda: query(task_id), lambda t: seen.add(t["status"]) or t["status"] in TERMINAL)
    s.check(task["status"] == "succeeded", f"task {task}")
    s.check(seen <= {"queued", "running", "succeeded"}, f"statuses {seen}")
    s.check(task["model"] == "MiniMax-H3" and task["resolution"] == "768P" and task["duration"] == 5, task)
    s.check(task["ratio"] == "16:9" and task.get("task_type") == "generation", task)
    s.check("error" not in task and task["usage"]["output_seconds"] > 0, task)
    video = requests.get(task["content"]["url"], timeout=60)  # no Authorization
    s.check(video.status_code == 200 and is_mp4(video.content), f"download {video.status_code}")
    s.ok("create -> query loop -> download content.url")

    # I2V: first_frame image as a data URI; ratio is ignored (adaptive).
    uri = "data:image/png;base64," + base64.b64encode(png(768, 1344, (40, 90, 200))).decode()
    r = create({
        "model": "MiniMax-H3-Max",
        "content": [
            {"type": "text", "text": "compat: the blue wall ripples"},
            {"type": "image_url", "image_url": {"url": uri}, "role": "first_frame"},
        ],
        "resolution": "768P",
        "duration": 6,
        "extra": {"prompt_expansion_mode": "disabled"},
    })
    s.check(r.status_code == 200, f"i2v create {r.status_code} {r.text}")
    t2 = poll(lambda: query(r.json()["task_id"]), lambda t: t["status"] in TERMINAL)
    s.check(t2["status"] == "succeeded" and t2["usage"]["input_image_count"] == 1, t2)
    s.ok("i2v first_frame data URI (MiniMax-H3-Max)")

    # List with filters.
    r = requests.get(f"{api}/v2/query/video_generation", headers=h, params={"page_num": 1, "page_size": 10, "filter.status": "succeeded"}, timeout=30)
    s.check(r.status_code == 200, r.text)
    lst = r.json()
    ids = [t["id"] for t in lst["items"]]
    s.check(task_id in ids and lst["total"] >= 2, f"list {lst['total']} {ids}")
    s.check(all(t["status"] == "succeeded" for t in lst["items"]), "filter.status")
    s.ok("list with filter.status")

    # Delete a succeeded record.
    r = requests.delete(f"{api}/v2/video_generation/{task_id}", headers=h, timeout=30)
    s.check(r.status_code == 200 and r.json()["action"] == "deleted", f"delete {r.status_code} {r.text}")
    r = requests.get(f"{api}/v2/query/video_generation/{task_id}", headers=h, timeout=30)
    s.check(r.status_code == 404, f"deleted then queried: {r.status_code}")
    s.ok("delete succeeded -> deleted")

    # A failed task carries error {code, message}.
    r = create({"model": "MiniMax-H3", "content": [{"type": "text", "text": "[fake:fail] compat"}], "resolution": "768P", "duration": 5, "ratio": "16:9"})
    tf = poll(lambda: query(r.json()["task_id"]), lambda t: t["status"] in TERMINAL)
    s.check(tf["status"] == "failed" and tf["error"]["code"] and tf["error"]["message"], tf)
    s.ok("failed task error")

    # The OaiError envelope.
    def err(r, http, typ, code=None):
        b = r.json()
        s.check(r.status_code == http, f"{r.status_code} != {http}: {b}")
        s.check(b["type"] == "error" and b["error"]["type"] == typ and b["error"]["http_code"] == str(http), b)
        if code:
            s.check(b["error"]["message"].endswith(f"({code})"), b)
        s.check(isinstance(b.get("request_id"), str), b)

    err(create({"model": "MiniMax-H3", "content": [], "resolution": "768P", "duration": 5, "ratio": "16:9"}), 400, "bad_request_error", 2013)
    err(create({"model": "MiniMax-H3", "content": [{"type": "text", "text": "p"}], "resolution": "768P", "duration": 5, "ratio": "adaptive"}), 400, "bad_request_error", 2013)
    err(create({"model": "MiniMax-H3", "content": [{"type": "text", "text": "p"}], "resolution": "2K", "duration": 5, "ratio": "16:9"}), 400, "bad_request_error", 2013)
    err(requests.post(f"{api}/v2/video_generation", json={"model": "MiniMax-H3"}, timeout=30), 401, "authorized_error", 1004)
    r = requests.get(f"{api}/v2/query/video_generation/123456789012345678", headers=h, timeout=30)
    s.check(r.status_code == 404 and r.json()["error"]["type"] == "bad_request_error", r.text)
    s.ok("OaiError envelopes (2013, 1004, unknown task)")

    # callback_url: challenge echo within 3 s, then the task on each change.
    def reply(_path, _headers, raw):
        b = json.loads(raw or b"{}")
        if "challenge" in b:
            return 200, {"challenge": b["challenge"]}
        return 200, {}

    rx = Receiver(a.hook_host, reply)
    try:
        r = create({
            "model": "MiniMax-H3",
            "content": [{"type": "text", "text": "compat: callback"}],
            "resolution": "768P",
            "duration": 5,
            "ratio": "9:16",
            "callback_url": rx.url + "/minimax",
        })
        s.check(r.status_code == 200, f"create with callback_url {r.status_code} {r.text}")
        tid = r.json()["task_id"]
        got = rx.wait(lambda g: any(json.loads(b).get("task", {}).get("status") == "succeeded" for _, _, b in g))
        bodies = [json.loads(b) for _, _, b in got]
        s.check("challenge" in bodies[0], f"first POST is the challenge: {bodies[0]}")
        tasks = [b["task"] for b in bodies[1:]]
        s.check(all(t["id"] == tid for t in tasks), "callback task ids")
        statuses = [t["status"] for t in tasks]
        s.check(statuses[-1] == "succeeded" and len(set(statuses)) == len(statuses), f"callback statuses {statuses}")
        s.check(tasks[-1]["content"]["url"].startswith("http"), tasks[-1])
        s.info["callback_statuses"] = statuses
    finally:
        rx.close()
    s.ok("callback_url: challenge, then task bodies")


if __name__ == "__main__":
    Suite("minimax").run(main)
