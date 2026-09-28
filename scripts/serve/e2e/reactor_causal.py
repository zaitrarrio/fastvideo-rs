#!/usr/bin/env python3
"""Reactor causal mode on a real SF-Wan server (WP-18): reactor_sdk 1.6.0
local mode connects to fv-serve's Reactor runtime, then drives the causal
command set (design §5.7) with timings:

- connect (HTTP session + WebRTC) and the first decoded video frame;
- delivered fps over a steady window;
- set_prompt: ack latency, and the next `state_update` carrying the prompt;
- set_paused(true): frames during the pause; set_paused(false): frames resume;
- reset: `state_update` with the block index back near 0;
- get_state.

    python reactor_causal.py --url http://127.0.0.1:8000 --out reactor.json [--steady 30]

Prints one JSON summary line (also written to --out).
"""

import argparse
import asyncio
import json
import sys
import time

import reactor_sdk
from reactor_sdk import Reactor

P1 = "A drone shot gliding over a winding river through an autumn forest, golden afternoon light, slow steady forward camera motion, highly detailed"
P2 = "A drone shot gliding over snowy mountain peaks at dawn, pink sky, slow steady forward camera motion, highly detailed"


def check(cond, what):
    if not cond:
        raise AssertionError(what)


async def wait_for(pred, timeout, step=0.05):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        await asyncio.sleep(step)
    return pred()


async def run(url, steady):
    s = {"sdk": reactor_sdk.__version__}
    frames = []  # monotonic receive times
    size = {}
    msgs = []  # (t, message)
    r = Reactor("sf-wan", local=True, api_url=url)
    r.on("message", lambda m: msgs.append((time.monotonic(), m)))
    vt = r.track("main_video")

    @vt.on_raw_frame
    def on_video(_bgra, width, height, *_rest):
        frames.append(time.monotonic())
        size["wh"] = (width, height)

    def states():
        return [(t, m) for t, m in msgs if isinstance(m, dict) and m.get("type") == "state_update"]

    t0 = time.monotonic()
    await asyncio.wait_for(r.connect(), 60)
    s["connect_s"] = round(time.monotonic() - t0, 3)
    s["tracks"] = sorted(t.name for t in r.tracks)
    # A prompt first: generation starts with an audience and a prompt.
    t = time.monotonic()
    ack = await asyncio.wait_for(r.send_command("set_prompt", {"prompt": P1}), 15)
    s["set_prompt_ack_s"] = round(time.monotonic() - t, 3)
    check(ack is None, f"set_prompt -> {ack}")
    check(await wait_for(lambda: len(frames) > 0, 60), "no first frame in 60 s")
    s["first_frame_after_connect_s"] = round(frames[0] - t0, 3)
    s["first_frame_after_prompt_s"] = round(frames[0] - t, 3)
    s["video_size"] = size.get("wh")

    # Steady window.
    await asyncio.sleep(3)
    n0, ts = len(frames), time.monotonic()
    await asyncio.sleep(steady)
    n1, te = len(frames), time.monotonic()
    s["steady"] = {"seconds": round(te - ts, 2), "frames": n1 - n0, "fps": round((n1 - n0) / (te - ts), 3)}

    # Prompt switch: ack, and the state_update that names the new prompt.
    t = time.monotonic()
    ack = await asyncio.wait_for(r.send_command("set_prompt", {"prompt": P2}), 15)
    s["switch_ack_s"] = round(time.monotonic() - t, 3)
    got = await wait_for(lambda: any(m["data"].get("prompt") == P2 for tt, m in states() if tt >= t), 15)
    if got:
        tt = min(tt for tt, m in states() if tt >= t and m["data"].get("prompt") == P2)
        s["switch_state_update_s"] = round(tt - t, 3)

    # Pause / resume.
    t = time.monotonic()
    ack = await asyncio.wait_for(r.send_command("set_paused", {"paused": True}), 15)
    check(ack is None, f"set_paused(true) -> {ack}")
    await asyncio.sleep(1.0)
    n0 = len(frames)
    await asyncio.sleep(5.0)
    s["paused_frames_in_5s"] = len(frames) - n0
    pst = [m["data"] for tt, m in states() if tt >= t]
    s["paused_state"] = pst[-1] if pst else None
    t = time.monotonic()
    n0 = len(frames)
    ack = await asyncio.wait_for(r.send_command("set_paused", {"paused": False}), 15)
    check(ack is None, f"set_paused(false) -> {ack}")
    await wait_for(lambda: len(frames) > n0 + 5, 20)
    s["resume_to_5_frames_s"] = round(time.monotonic() - t, 3) if len(frames) > n0 + 5 else None

    # get_state, then reset (block index restarts).
    st = await asyncio.wait_for(r.send_command("get_state", {}), 15)
    s["get_state"] = st
    before = (st or {}).get("data", {}).get("block_index")
    t = time.monotonic()
    ack = await asyncio.wait_for(r.send_command("reset", {}), 15)
    s["reset_ack"] = ack
    await asyncio.sleep(2.0)
    st2 = await asyncio.wait_for(r.send_command("get_state", {}), 15)
    after = (st2 or {}).get("data", {}).get("block_index")
    s["reset"] = {"block_index_before": before, "block_index_2s_after": after, "ack_s": round(time.monotonic() - t, 3)}
    n0 = len(frames)
    await asyncio.sleep(5.0)
    s["frames_5s_after_reset"] = len(frames) - n0

    s["frames_total"] = len(frames)
    s["message_types"] = sorted({m.get("type") for _, m in msgs if isinstance(m, dict)})
    await r.disconnect()
    r.close()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steady", type=float, default=30)
    a = ap.parse_args()
    try:
        s = asyncio.run(run(a.url, a.steady))
        s["ok"] = True
    except Exception as e:  # noqa: BLE001
        s = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    line = json.dumps(s)
    with open(a.out, "w") as f:
        f.write(line + "\n")
    print(line, flush=True)
    sys.exit(0 if s["ok"] else 1)


if __name__ == "__main__":
    main()
