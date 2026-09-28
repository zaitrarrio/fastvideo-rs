#!/usr/bin/env python3
"""A WHEP viewer for the GPU E2E (WP-18): aiortc reads a MediaMTX path over
WHEP, decodes the H.264 and writes one JSON line per decoded frame (wall
clock, RTP-derived pts, mean R/G/B), so the stream's delivered fps, gaps and
a prompt switch's visible latency can be computed afterwards on the same
clock as the driver's command log.

    python whep_viewer.py --url http://127.0.0.1:8889/sfwan/whep --seconds 300 --out frames.jsonl

Prints one JSON summary line (connect, first frame, frames, fps).
"""

import argparse
import asyncio
import json
import time

import aiohttp
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError


async def run(url, seconds, out_path):
    # No STUN: the viewer runs next to MediaMTX (host candidates only).
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    pc.addTransceiver("video", direction="recvonly")
    t_start = time.time()
    summary = {"url": url, "t_start": t_start}
    frames = {"n": 0, "first": None, "last": None}
    got_track = asyncio.get_event_loop().create_future()

    @pc.on("track")
    def on_track(track):
        if track.kind == "video" and not got_track.done():
            got_track.set_result(track)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)  # aiortc gathers before returning (non-trickle)
    async with aiohttp.ClientSession() as http:
        async with http.post(url, data=pc.localDescription.sdp, headers={"content-type": "application/sdp"}) as r:
            answer = await r.text()
            summary["whep_status"] = r.status
            if r.status not in (200, 201):
                raise RuntimeError(f"WHEP {r.status}: {answer[:300]}")
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type="answer"))
    summary["answered_s"] = time.time() - t_start
    track = await asyncio.wait_for(got_track, 30)
    deadline = t_start + seconds
    with open(out_path, "w") as out:
        while time.time() < deadline:
            try:
                f = await asyncio.wait_for(track.recv(), 10)
            except (asyncio.TimeoutError, MediaStreamError) as e:
                summary["ended"] = type(e).__name__
                break
            now = time.time()
            a = f.to_ndarray(format="rgb24")
            # A coarse subsample keeps the per-frame cost well under 1 ms.
            m = a[::8, ::8].reshape(-1, 3).mean(axis=0)
            frames["n"] += 1
            frames["first"] = frames["first"] or now
            frames["last"] = now
            out.write(json.dumps({"t": round(now, 4), "pts": f.pts, "w": f.width, "h": f.height,
                                  "rgb": [round(float(x), 2) for x in m]}) + "\n")
    await pc.close()
    span = (frames["last"] - frames["first"]) if frames["n"] > 1 else 0.0
    summary.update({
        "frames": frames["n"],
        "first_frame_s": (frames["first"] - t_start) if frames["first"] else None,
        "span_s": round(span, 3),
        "fps": round((frames["n"] - 1) / span, 3) if span > 0 else None,
    })
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        s = asyncio.run(run(a.url, a.seconds, a.out))
        s["ok"] = s["frames"] > 0
    except Exception as e:  # noqa: BLE001 - one report line either way
        s = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
