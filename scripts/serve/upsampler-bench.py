#!/usr/bin/env python3
"""Benchmark of the prompt upsampler's local sidecar (docs/serve/upsampler-local.md), run ON a GPU pod next to a
running llama-server (deploy/runpod/fv-upsampler.sh). Stdlib only.

    python3 upsampler-bench.py --base http://127.0.0.1:18080/v1 --model qwen3-vl-4b-instruct \
        [--image-url URL] [--out results.json]

Cases (each prints one JSON line and is collected in --out):
  rewrite     a single-shot prompt rewrite (plain text answer)
  director    the owner's director request (system prompt + json_schema of the upsampler spec): a small world
              bible and three consecutive segments; every answer validated against the schema
  long        an H3 Max-style session: a ~40k-character premise, 12 prior segments and the entity registry, three
              consecutive segments; prefill tok/s, end-to-end latency, and the prompt cache (tokens evaluated vs
              cached per segment, from llama-server's `timings`)
  image       the opening segment with configure.image_url as an image_url content part (data: URL, and the http(s)
              URL when given), image encode included
Times are wall clock around each request; prefill / decode rates are llama-server's own `timings`.
"""
import argparse
import base64
import json
import random
import sys
import time
import urllib.request

SYSTEM_PROMPT = """You are the showrunner for a continuous, never-ending live video stream.
Each turn you write the NEXT segment of one unbroken take. You receive the world bible,
the registry of established entities, the recent segment history, and an optional new direction.

Rules:
- Continue directly from the last segment. No recaps, no time skips unless the direction demands one.
- If a direction is given, make it happen in this segment, staying consistent with established people and places.
- If no direction is given, advance the story by one natural beat.
- Refer to existing people, places, and props ONLY by their entity ids. Never describe their appearance.
- Introduce a new entity only if the story needs it; give it a short id and a concrete visual descriptor.
- "action" is present-tense, physical and visible: what moves, where, how. One continuous shot.
- "audio" describes ambience, foley and score. Spoken words go only in "dialogue", verbatim.
- Write concrete visual language. No meta commentary, no references to cameras "filming a show"."""

SCHEMA = {
    "name": "segment",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["beat_summary", "entity_ids", "new_entities", "action", "camera", "lighting", "audio", "dialogue"],
        "properties": {
            "beat_summary": {"type": "string"},
            "entity_ids": {"type": "array", "items": {"type": "string"}},
            "new_entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "kind", "name", "descriptor"],
                    "properties": {
                        "id": {"type": "string"},
                        "kind": {"type": "string", "enum": ["character", "location", "prop"]},
                        "name": {"type": "string"},
                        "descriptor": {"type": "string"},
                    },
                },
            },
            "action": {"type": "string"},
            "camera": {"type": "string"},
            "lighting": {"type": "string"},
            "audio": {"type": "string"},
            "dialogue": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["speaker_id", "line"],
                    "properties": {"speaker_id": {"type": "string"}, "line": {"type": "string"}},
                },
            },
        },
    },
}


def validate(v, s, path="$"):
    """The subset of JSON Schema the spec uses: type, required, additionalProperties false, properties, items, enum."""
    t = s.get("type")
    ok = {"object": dict, "array": list, "string": str}[t]
    if not isinstance(v, ok):
        return [f"{path}: not a {t}"]
    errs = []
    if "enum" in s and v not in s["enum"]:
        errs.append(f"{path}: {v!r} not in {s['enum']}")
    if t == "object":
        for k in s.get("required", []):
            if k not in v:
                errs.append(f"{path}.{k}: missing")
        if s.get("additionalProperties") is False:
            errs += [f"{path}.{k}: not allowed" for k in v if k not in s.get("properties", {})]
        for k, sub in s.get("properties", {}).items():
            if k in v:
                errs += validate(v[k], sub, f"{path}.{k}")
    if t == "array":
        for i, x in enumerate(v):
            errs += validate(x, s["items"], f"{path}[{i}]")
    return errs


def chat(base, model, messages, schema=None, max_tokens=700, temperature=0.8, timeout=300):
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": messages}
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": schema}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    wall = time.time() - t
    tm = out.get("timings", {})
    return out["choices"][0]["message"]["content"], wall, {
        "wall_s": round(wall, 3),
        "prompt_tokens": out.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": out.get("usage", {}).get("completion_tokens"),
        "evaluated": tm.get("prompt_n"),
        "cached": tm.get("cache_n"),
        "prefill_ms": tm.get("prompt_ms"),
        "prefill_tok_s": round(tm["prompt_per_second"], 1) if tm.get("prompt_per_second") else None,
        "decode_tok_s": round(tm["predicted_per_second"], 1) if tm.get("predicted_per_second") else None,
        "decode_ms": tm.get("predicted_ms"),
    }


def ttft(base, model, messages, max_tokens=200):
    """Time to first token (streamed) of a plain request."""
    body = {"model": model, "temperature": 0.7, "max_tokens": max_tokens, "messages": messages, "stream": True}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t = time.time()
    first = None
    n = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            delta = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                n += 1
                if first is None:
                    first = time.time() - t
    total = time.time() - t
    return {"ttft_s": round(first or total, 3), "total_s": round(total, 3), "chunks": n, "stream_tok_s": round((n - 1) / max(total - (first or 0), 1e-6), 1) if n > 1 else None}


ENTITIES = [
    {"id": "mira", "kind": "character", "name": "Mira", "descriptor": "a wiry woman in her thirties, cropped silver hair, oil-stained teal flight jacket"},
    {"id": "otto", "kind": "character", "name": "Otto", "descriptor": "a broad old man with a white beard, brass-rimmed goggles, leather apron"},
    {"id": "dock", "kind": "location", "name": "Skyport Dock 9", "descriptor": "a rusted floating airship dock above a sea of clouds, rope bridges, signal lanterns"},
    {"id": "kite", "kind": "prop", "name": "the Kite", "descriptor": "a small patched red airship with a canvas balloon and a copper gondola"},
]


def user_message(premise, registry, history, direction):
    ents = "\n".join(f'- {e["id"]} ({e["kind"]}, "{e["name"]}")' for e in registry)
    recent = "\n\n".join(f'[{s["index"]}] {s["summary"]}\n{s["prompt"]}' for s in history)
    return "\n\n".join([
        f"WORLD BIBLE:\n{premise}",
        f"ESTABLISHED ENTITIES:\n{ents or '(none)'}",
        f"RECENT SEGMENTS (oldest first):\n{recent or '(stream is starting)'}",
        f"NEW DIRECTION:\n{direction}" if direction else "NEW DIRECTION:\n(none, advance the story one beat)",
    ])


def assemble(out, registry):
    by = {e["id"]: e for e in registry}
    cast = [by[i] for i in out["entity_ids"] if i in by]
    loc = next((e for e in cast if e["kind"] == "location"), None)
    parts = [loc["descriptor"] if loc else ""] + [f'{e["name"]}: {e["descriptor"]}' for e in cast if e["kind"] != "location"]
    parts += [out["action"]] + [f'{by.get(d["speaker_id"], {}).get("name", "a voice")} says, "{d["line"]}"' for d in out["dialogue"]]
    parts += [f'Camera: {out["camera"]}', f'Lighting: {out["lighting"]}', f'Sound: {out["audio"]}']
    return " ".join(p for p in parts if p.strip())[:1800]


def session(base, model, premise, registry, history, directions, label, results, image_part=None):
    registry = [dict(e) for e in registry]
    history = list(history)
    for i, direction in enumerate(directions):
        user = user_message(premise, registry, history[-12:], direction)
        content = [image_part, {"type": "text", "text": user}] if (image_part and i == 0) else user
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        text, wall, m = chat(base, model, msgs, SCHEMA)
        try:
            out = json.loads(text)
            errs = validate(out, SCHEMA["schema"])
        except ValueError as e:
            out, errs = None, [f"not JSON: {e}"]
        for e in (out or {}).get("new_entities", []) if not errs else []:
            if all(x["id"] != e["id"] for x in registry):
                registry.append(e)
        if out and not errs:
            history.append({"index": len(history), "summary": out["beat_summary"], "prompt": assemble(out, registry)})
        row = {"case": label, "segment": i, "user_chars": len(user), "schema_ok": not errs, "errors": errs[:5], **m,
               "sample": (history[-1]["prompt"][:600] if out and not errs else text[:600])}
        print(json.dumps(row), flush=True)
        results.append(row)
    return history


def long_premise(n_chars=40000):
    rnd = random.Random(7)
    places = ["the Brass Market", "the cloud reefs", "the lighthouse of Vey", "the drowned observatory", "the copper canals", "the night ferry"]
    acts = ["smuggles", "repairs", "argues over", "steals", "charts", "rebuilds", "hides", "trades"]
    things = ["a weather engine", "a ledger of debts", "a map of the jet streams", "a crate of lantern glass", "the harbour master's seal", "an old distress beacon"]
    out = ["SERIES PREMISE. A long-running airship serial set above an endless sea of clouds."]
    i = 0
    while sum(len(x) + 1 for x in out) < n_chars:
        i += 1
        out.append(f"Episode note {i}: Mira {rnd.choice(acts)} {rnd.choice(things)} near {rnd.choice(places)} while Otto "
                   f"{rnd.choice(acts)} {rnd.choice(things)}; the tone stays warm, salt-bitten and hopeful, with long continuous shots, "
                   f"practical light from lanterns and the low sun, wind always moving cloth and rope.")
    return "\n".join(out)[:n_chars]


def history_of(n):
    return [{"index": k, "summary": f"Beat {k}: Mira and Otto work the dock as cargo swings in.",
             "prompt": f"Skyport Dock 9, rope bridges sway. Mira hauls a line while Otto checks a brass gauge, segment {k}. "
                       "Camera: slow lateral dolly. Lighting: low amber sun through cloud haze. Sound: wind, creaking rope, distant horns." * 2}
            for k in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18080/v1")
    ap.add_argument("--model", default="qwen3-vl-4b-instruct")
    ap.add_argument("--image-file", default="")
    ap.add_argument("--image-url", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--cases", default="rewrite,director,long,image")
    a = ap.parse_args()
    res = []
    cases = a.cases.split(",")
    if "rewrite" in cases:
        msgs = [{"role": "system", "content": "Rewrite the user's short video prompt into one vivid, concrete paragraph (60-90 words) for a text-to-video model: subject, action, setting, camera, lighting. No preamble."},
                {"role": "user", "content": "a fox running through snow at dawn"}]
        print(json.dumps({"case": "ttft", **ttft(a.base, a.model, msgs)}), flush=True)
        for k in range(2):
            text, wall, m = chat(a.base, a.model, msgs, max_tokens=200, temperature=0.7)
            row = {"case": "rewrite", "run": k, **m, "sample": text[:700]}
            print(json.dumps(row), flush=True)
            res.append(row)
    if "director" in cases:
        premise = "A warm, salt-bitten airship serial above an endless sea of clouds. Mira, a smuggler-pilot, and Otto, her old mechanic, keep the Kite flying out of Skyport Dock 9. Style: 35 mm film, long continuous takes, golden-hour light."
        session(a.base, a.model, premise, ENTITIES, [], ["Mira and Otto prepare the Kite for launch.", None, "A storm front appears on the horizon."], "director", res)
    if "long" in cases:
        premise = long_premise()
        session(a.base, a.model, premise, ENTITIES, history_of(12), [None, "Otto finds a stowaway in the cargo hold.", None], "long", res)
    if "image" in cases and a.image_file:
        raw = open(a.image_file, "rb").read()
        mime = "image/png" if raw[:4] == b"\x89PNG" else "image/jpeg"
        part = {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(raw).decode()}"}}
        premise = "A live stream that opens on the attached image: continue from exactly what it shows. Style: photoreal, natural light, one continuous take."
        session(a.base, a.model, premise, [], [], ["Open on the attached image and set the scene."], "image-data-url", res, image_part=part)
        if a.image_url:
            part = {"type": "image_url", "image_url": {"url": a.image_url}}
            try:
                session(a.base, a.model, premise, [], [], ["Open on the attached image and set the scene."], "image-http-url", res, image_part=part)
            except Exception as e:  # noqa: BLE001
                row = {"case": "image-http-url", "error": str(e)[:300]}
                print(json.dumps(row), flush=True)
                res.append(row)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
