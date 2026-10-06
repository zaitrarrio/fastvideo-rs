import json, sys
for path in sys.argv[1:]:
    r = json.load(open(path))
    for c in r["checks"]:
        v = c["values"]
        if c["name"] in ("load", "longlive_weights"):
            print(c["name"], json.dumps(v)); continue
        if not c["name"].startswith("run/") or c["name"].count("/") != 1: 
            if not c["ok"]: print("FAILED CHECK", c["name"], json.dumps(v)[:300])
            continue
        w = v.get("windows", [])
        print(json.dumps({
            "run": c["name"][4:], "ok": c["ok"], "fps_wall": round(v["fps_steady_wall"], 2), "fps_engine": round(v["fps_steady_engine"], 2),
            "ttff": round(v["ttff_s"], 3), "block_s": v.get("block_s"), "kv_mib": v.get("kv_mib_end"), "mem_max": v.get("mem_used_mib_max"),
            "recaches": v.get("switch_recaches"), "switch": v.get("switch"), "graph": v.get("graph"),
            "win": [{k: (round(x[k], 3) if isinstance(x.get(k), float) else x.get(k)) for k in ("t0_s", "luma", "std", "mad", "seam_mad", "sharpness", "latent_hash", "recaches")} | {"fresh_mad": (x.get("fresh_decode") or {}).get("mad")} for x in w],
        }))
