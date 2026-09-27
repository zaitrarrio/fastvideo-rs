#!/usr/bin/env python3
"""FastVideo `/v1/videos` through the real `openai` Python library (design
§4.1, §7.5): `client.videos.create/retrieve/list/download_content/delete`,
`create_and_poll`, a multipart `input_reference` upload, `/v1/models`, and
the OpenAI error classes. The server is fv-serve on the fake engine.
"""

import warnings

import openai
from openai import OpenAI

from common import Suite, is_mp4, png

warnings.filterwarnings("ignore", category=DeprecationWarning)

MODEL = "h3-turbo"  # tier id; FastVideo accepts every served name too


def main(s, a):
    c = OpenAI(base_url=f"{a.base}/v1", api_key=a.key, max_retries=0)
    s.info["openai"] = openai.__version__

    ids = [m.id for m in c.models.list()]
    s.check("h3-turbo" in ids and "fake-h3-turbo" in ids, f"models {ids}")
    s.check(c.models.retrieve("fake-h3-max").owned_by == "fastvideo", "models.retrieve")
    s.ok("models.list/retrieve")

    # create (the SDK always sends multipart/form-data) -> retrieve -> download
    v = c.videos.create(model=MODEL, prompt="compat: a paper boat on a pond", seconds="8", size="1344x768")
    s.check(v.object == "video" and v.status in ("queued", "in_progress"), f"create: {v}")
    s.check(v.model == MODEL and v.size == "1344x768" and v.seconds == "8", f"echo: {v.model} {v.size} {v.seconds}")
    first = v.id
    while v.status in ("queued", "in_progress"):
        v = c.videos.retrieve(v.id)
    s.check(v.status == "completed" and v.progress == 100 and v.completed_at, f"completed: {v}")
    s.check(v.error is None, f"no error: {v.error}")
    s.ok("videos.create/retrieve")

    content = c.videos.download_content(v.id)
    body = content.read()
    s.check(is_mp4(body), f"download_content is an MP4 ({body[:16]!r})")
    s.check(content.response.headers["content-type"].startswith("video/mp4"), content.response.headers)
    s.ok("videos.download_content")

    # create_and_poll (the SDK's own polling loop)
    v2 = c.videos.create_and_poll(model=MODEL, prompt="compat: poll helper", seconds="8", size="1344x768", poll_interval_ms=100)
    s.check(v2.status == "completed", f"create_and_poll: {v2.status}")
    s.ok("videos.create_and_poll")

    # I2V: the input_reference file part, as the SDK uploads it
    v3 = c.videos.create(
        model=MODEL,
        prompt="compat: the red wall fades to dusk",
        seconds="8",
        input_reference=("frame.png", png(1344, 768), "image/png"),
    )
    v3 = c.videos.poll(v3.id, poll_interval_ms=100)
    s.check(v3.status == "completed", f"input_reference: {v3.status} {v3.error}")
    s.ok("videos.create(input_reference=file)")

    # extra_body carries FastVideo's own fields
    v4 = c.videos.create(model=MODEL, prompt="compat: seeded", size="1344x768", extra_body={"seed": 42, "num_frames": 141})
    v4 = c.videos.poll(v4.id, poll_interval_ms=100)
    s.check(v4.status == "completed", f"extra_body: {v4.status} {v4.error}")
    s.ok("videos.create(extra_body)")

    # list (newest first) and pagination
    page = c.videos.list(limit=2)
    s.check(len(page.data) == 2 and page.data[0].id == v4.id, f"list: {[x.id for x in page.data]}")
    s.check(page.has_more, "has_more")
    listed = [x.id for x in c.videos.list(limit=2)]  # auto-pagination walks every page
    s.check(first in listed, "auto-pagination reaches the first video")
    asc = c.videos.list(order="asc", limit=100)
    times = [x.created_at for x in asc.data]
    s.check(times == sorted(times) and first in [x.id for x in asc.data], "order=asc")
    s.ok("videos.list + pagination")

    # A failed generation is 200 with status failed; content is refused.
    vf = c.videos.create(model=MODEL, prompt="[fake:fail] compat", seconds="8", size="1344x768")
    vf = c.videos.poll(vf.id, poll_interval_ms=100)
    s.check(vf.status == "failed" and vf.error and vf.error.code == "generation_failed", f"failed: {vf}")
    try:
        c.videos.download_content(vf.id)
        s.check(False, "content of a failed video")
    except openai.APIStatusError as e:
        s.check(e.status_code == 422, f"failed content: {e.status_code}")
    s.ok("failed generation")

    # delete
    d = c.videos.delete(first)
    s.check(d.deleted and d.id == first and d.object == "video.deleted", f"delete: {d}")
    try:
        c.videos.retrieve(first)
        s.check(False, "deleted video still retrievable")
    except openai.NotFoundError:
        pass
    s.ok("videos.delete")

    # errors map onto the SDK's exception classes
    try:
        c.videos.create(model=MODEL, prompt="p", seconds="8", size="999x777")
        s.check(False, "bad size accepted")
    except openai.BadRequestError as e:
        s.check(e.status_code == 400 and isinstance(e.body, dict) and e.body.get("message"), f"400 body: {e.body}")
    try:
        c.videos.create(model="no-such-model", prompt="p")
        s.check(False, "unknown model accepted")
    except openai.BadRequestError:
        pass
    # FastVideo itself has no auth, so /v1/videos is open by design (serve-kit
    # AuthPolicy::Open): any key, or none, works.
    OpenAI(base_url=f"{a.base}/v1", api_key="any-key", max_retries=0).videos.list(limit=1)
    try:
        c.videos.retrieve("video_gen_" + "0" * 32)
        s.check(False, "unknown id")
    except openai.NotFoundError:
        pass
    s.ok("errors: BadRequestError / NotFoundError; open auth")


if __name__ == "__main__":
    Suite("openai").run(main)
