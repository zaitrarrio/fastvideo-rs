#!/usr/bin/env python3
"""The real `fal-client` (Python, tests/compat pin) against a pod: subscribe
(queue + status with logs + result) and run (sync).

    FAL_KEY=<key> FAL_QUEUE_RUN_HOST=<pod host> FAL_RUN_HOST=<pod host>/run \
        fal_client_check.py <app> [resolution]

Prints one JSON line.
"""

import json
import sys
import time

import fal_client

app = sys.argv[1]
res = sys.argv[2] if len(sys.argv) > 2 else "480P"
out = {"fal_client": getattr(fal_client, "__version__", "?"), "app": app}
logs = []
t0 = time.monotonic()
r = fal_client.subscribe(
    f"{app}/text-to-video",
    arguments={"prompt": "A paper lantern drifts over a night river", "resolution": res, "duration": 5, "seed": 11},
    with_logs=True,
    on_queue_update=lambda u: logs.append(type(u).__name__),
)
out["subscribe"] = {"wall_s": round(time.monotonic() - t0, 2), "updates": sorted(set(logs)),
                    "video": bool(r.get("video", {}).get("url")), "timings": r.get("timings")}
t0 = time.monotonic()
r = fal_client.run(f"{app}/text-to-video", arguments={"prompt": "A paper lantern drifts over a night river", "resolution": res, "duration": 5, "seed": 12})
out["run"] = {"wall_s": round(time.monotonic() - t0, 2), "video": bool(r.get("video", {}).get("url"))}
ok = out["subscribe"]["video"] and out["run"]["video"]
out["ok"] = ok
print(json.dumps(out))
sys.exit(0 if ok else 1)
