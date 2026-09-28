#!/usr/bin/env python3
"""reactor_sdk 1.6.0 (Python, local mode) against our Reactor runtime.

Design §7.5: connect, get_state, set_autoplay, enqueue, receive `main_video`
and `main_audio` frames and assert 48 kHz; the video-only variant asserts
there is no audio track. A causal variant drives set_prompt.

    python reactor_sdk_compat.py --url http://127.0.0.1:8080 --mode av|video|causal

Exit status 0 = green. Prints one JSON summary line.
"""

import argparse
import asyncio
import json
import sys
import os
import time

import reactor_sdk
from reactor_sdk import Reactor


def check(cond, what):
    if not cond:
        raise AssertionError(what)


async def wait_for(pred, timeout, step=0.1):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        await asyncio.sleep(step)
    return pred()


async def run(url, mode):
    expect_audio = mode == "av"
    r = Reactor("fake", local=True, api_url=url)
    video = {"n": 0, "size": None, "first": None, "last": None, "times": []}
    # Real engines need longer than the fake one (WP-18 GPU E2E).
    clip_timeout = float(os.environ.get("FV_REACTOR_CLIP_TIMEOUT_S", "40"))
    t_connect = time.monotonic()
    audio = {"n": 0, "rates": set(), "channels": set(), "samples": 0}
    messages = []
    r.on("message", lambda m: messages.append(m))

    vt = r.track("main_video")

    @vt.on_raw_frame
    def on_video(bgra, width, height, *_rest):
        video["n"] += 1
        video["size"] = (width, height)
        now = time.monotonic()
        video["first"] = video["first"] or now
        video["last"] = now
        video["times"].append(now)

    if expect_audio:
        at = r.track("main_audio")

        @at.on_raw_frame
        def on_audio(pcm, num_samples, sample_rate, num_channels):
            audio["n"] += 1
            audio["rates"].add(sample_rate)
            audio["channels"].add(num_channels)
            audio["samples"] += num_samples

    await asyncio.wait_for(r.connect(), 30)
    check(str(r.status.value if hasattr(r.status, "value") else r.status) == "ready", f"status {r.status}")
    names = sorted(t.name for t in r.tracks)
    want = ["main_audio", "main_video"] if expect_audio else ["main_video"]
    check(names == want, f"tracks {names} != {want}")

    schema = await asyncio.wait_for(r.request_schema(), 10)
    tracks = [t["name"] for t in schema["x-reactor"]["tracks"]]
    check(sorted(tracks) == want, f"x-reactor.tracks {tracks}")

    summary = {"mode": mode, "sdk": reactor_sdk.__version__, "tracks": names}
    if mode in ("av", "video"):
        st = await asyncio.wait_for(r.send_command("get_state", {}), 10)
        check(st and st["type"] == "state_update", f"get_state -> {st}")
        check("enqueue" in st["data"]["valid_commands"], f"valid_commands {st['data']}")
        ap = await asyncio.wait_for(r.send_command("set_autoplay", {"enabled": True}), 10)
        check(ap == {"type": "autoplay_accepted", "data": {"enabled": True}}, f"set_autoplay -> {ap}")
        secs = st["data"]["clip_seconds_min"]
        q = await asyncio.wait_for(
            r.send_command("enqueue", {"prompt": "a lighthouse at dusk", "metadata": "m1", "seconds": secs}), 10
        )
        check(q and q["type"] == "clip_queued", f"enqueue -> {q}")
        clip = q["data"]["clip"]
        check(clip["metadata"] == "m1" and clip["ready"] is False, f"clip {clip}")
        # A bodyless ack comes back as None.
        play = await asyncio.wait_for(r.send_command("play", {"clip_id": ""}), 10)
        check(play is None, f"play -> {play}")
        # A contract violation is an error on v1.
        try:
            await asyncio.wait_for(r.send_command("set_seed", {"seed": -3}), 10)
            raise AssertionError("set_seed(-3) was accepted")
        except reactor_sdk.ReactorError as e:
            summary["invalid_command_error"] = type(e).__name__
        got = await wait_for(
            lambda: any(m.get("type") == "clip_finished" for m in messages if isinstance(m, dict)), clip_timeout
        )
        check(got, f"no clip_finished; messages={[m.get('type') for m in messages if isinstance(m, dict)]}")
        summary["clip_frames"] = clip["frames"]
    else:
        ack = await asyncio.wait_for(r.send_command("set_prompt", {"prompt": "a forest road"}), 10)
        check(ack is None, f"set_prompt -> {ack}")
        await wait_for(lambda: video["n"] >= 30, 20)
        ack = await asyncio.wait_for(r.send_command("set_paused", {"paused": True}), 10)
        check(ack is None, f"set_paused -> {ack}")

    await wait_for(lambda: video["n"] >= 30, 10)
    check(video["n"] >= 30, f"only {video['n']} video frames")
    if expect_audio:
        await wait_for(lambda: audio["n"] >= 100, 10)
        check(audio["n"] > 0, "no audio frames")
        check(audio["rates"] == {48000}, f"audio rates {audio['rates']}")
        summary["audio"] = {
            "frames": audio["n"],
            "rates": sorted(audio["rates"]),
            "channels": sorted(audio["channels"]),
            "samples": audio["samples"],
        }
    else:
        check(audio["n"] == 0, "audio on a video-only session")
    types = sorted({m.get("type") for m in messages if isinstance(m, dict)})
    summary.update({"video_frames": video["n"], "video_size": video["size"], "message_types": types})
    if video["first"] and video["n"] > 1 and video["last"] > video["first"]:
        summary["first_frame_s"] = round(video["first"] - t_connect, 2)
        summary["fps"] = round((video["n"] - 1) / (video["last"] - video["first"]), 2)
        # Arrival gaps: the largest shows where playout waited on generation.
        t = video["times"]
        gaps = sorted(b - a for a, b in zip(t, t[1:]))
        summary["frame_gap_ms"] = {"median": round(1000 * gaps[len(gaps) // 2], 1), "max": round(1000 * gaps[-1], 1)}
    await r.disconnect()
    r.close()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--mode", choices=["av", "video", "causal"], required=True)
    a = ap.parse_args()
    try:
        s = asyncio.run(run(a.url, a.mode))
    except Exception as e:  # noqa: BLE001 - report every failure the same way
        print(json.dumps({"mode": a.mode, "ok": False, "error": f"{type(e).__name__}: {e}"}))
        sys.exit(1)
    s["ok"] = True
    print(json.dumps(s))


if __name__ == "__main__":
    main()
