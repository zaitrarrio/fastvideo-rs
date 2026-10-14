#!/usr/bin/env python3
"""Plays a WHEP URL (Cloudflare Stream) with aiortc and measures the audio it delivers: decoded Opus frames,
loudness per second (RMS dBFS), and how many seconds are not silent; video frames and size for reference.
GPU checks (docs/serve/e2e/h3-test-round-2026-10-14.md).

    python whep_audio.py --url <whep url> --seconds 40 --out audio.json
"""

import argparse
import asyncio
import json
import math
import time

import aiohttp
import numpy as np
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError


async def run(url, seconds):
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[RTCIceServer(urls="stun:stun.cloudflare.com:3478")]))
    pc.addTransceiver("video", direction="recvonly")
    pc.addTransceiver("audio", direction="recvonly")
    tracks = {}

    @pc.on("track")
    def on_track(track):
        tracks[track.kind] = track

    await pc.setLocalDescription(await pc.createOffer())
    t0 = time.time()
    async with aiohttp.ClientSession() as http:
        async with http.post(url, data=pc.localDescription.sdp, headers={"content-type": "application/sdp"}) as r:
            if r.status >= 300:
                return {"ok": False, "error": f"WHEP HTTP {r.status}"}
            answer = await r.text()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type="answer"))
    for _ in range(100):
        if "audio" in tracks and "video" in tracks:
            break
        await asyncio.sleep(0.1)
    res = {"ok": True, "tracks": sorted(tracks), "audio_frames": 0, "video_frames": 0, "video_size": None}
    per_sec = {}
    end = t0 + seconds

    async def audio():
        tr = tracks.get("audio")
        while tr and time.time() < end:
            try:
                f = await asyncio.wait_for(tr.recv(), timeout=2)
            except (asyncio.TimeoutError, MediaStreamError):
                continue
            a = f.to_ndarray().astype(np.float64) / 32768.0
            s = int(time.time() - t0)
            acc = per_sec.setdefault(s, [0.0, 0])
            acc[0] += float((a * a).sum())
            acc[1] += a.size
            res["audio_frames"] += 1
            res.setdefault("sample_rate", f.sample_rate)
            res.setdefault("layout", f.layout.name)

    async def video():
        tr = tracks.get("video")
        while tr and time.time() < end:
            try:
                f = await asyncio.wait_for(tr.recv(), timeout=2)
            except (asyncio.TimeoutError, MediaStreamError):
                continue
            res["video_frames"] += 1
            res["video_size"] = [f.width, f.height]

    await asyncio.gather(audio(), video())
    await pc.close()
    db = {s: (10 * math.log10(v[0] / v[1]) if v[0] > 0 else -120.0) for s, v in sorted(per_sec.items())}
    res["rms_dbfs_per_second"] = {str(k): round(v, 1) for k, v in db.items()}
    res["seconds_with_audio_above_-60dBFS"] = sum(1 for v in db.values() if v > -60)
    res["max_rms_dbfs"] = max(db.values()) if db else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = asyncio.run(run(a.url, a.seconds))
    json.dump(r, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k != "rms_dbfs_per_second"}))


if __name__ == "__main__":
    main()
