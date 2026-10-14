"""h3-graphs-summary.py <results dir>: the h3-graphs-pod.sh results (docs/perf/h3-denoise-graphs.md) per cell and
request: denoise / total seconds, median step, peak MiB, the traced step, the graph captures, and the byte identity of
every checkpoint / prompt / canvas group across arms and requests (hashes.tsv)."""
import collections
import json
import os
import re
import statistics
import sys

root = sys.argv[1]
TIM = re.compile(r"^\[INFO\] h3/timings/(\S+) (\{.*\})$")
COLD = re.compile(r"^\[INFO\] h3/cold_generation (\{.*\})$")
GRAPHS = re.compile(r"h3 graphs: (\d+) steps: (\d+) replayed, (\d+) captured, (\d+) eager")
CAP = re.compile(r"h3 graphs: step (\d+) captured: (.*), graph pool high ([\d.]+) GiB")
TRACE = re.compile(r"^\[INFO\] h3/gpu_trace (\{.*\})$")


def cell_rows(d):
    rows = []
    # stderr.log on the pod; events.log (its h3 lines) once archived.
    err = os.path.join(d, "stderr.log")
    if not os.path.exists(err):
        err = os.path.join(d, "events.log")
    if not os.path.exists(err):
        return rows
    mode_queue = []
    caps = []
    traces = []
    for line in open(err, errors="replace"):
        line = line.rstrip("\n")
        m = GRAPHS.search(line)
        if m:
            mode_queue.append("replay" if m.group(2) != "0" and m.group(3) == "0" and m.group(4) == "0"
                              else "capture" if m.group(3) != "0" else "eager")
            continue
        m = CAP.search(line)
        if m:
            caps.append((int(m.group(1)), m.group(2), float(m.group(3))))
            continue
        m = TRACE.match(line)
        if m:
            traces.append(json.loads(m.group(1)))
            continue
        m = COLD.match(line)
        if m:
            mode_queue.append(None)  # cold run: consume its graph line if any
            continue
        m = TIM.match(line)
        if m:
            v = json.loads(m.group(2))
            v["name"] = m.group(1)
            rows.append(v)
    return rows, caps, traces


out = collections.OrderedDict()
for base in ("cells", "trace"):
    p = os.path.join(root, base)
    if not os.path.isdir(p):
        continue
    for cell in sorted(os.listdir(p)):
        r = cell_rows(os.path.join(p, cell))
        if not r:
            continue
        rows, caps, traces = r
        print(f"== {base}/{cell}")
        for v in rows:
            steps = v.get("step_s") or []
            tr = v.get("gpu_trace") or {}
            extra = ""
            if tr:
                extra = (f" trace: busy {tr.get('busy_s', 0):.3f} idle {tr.get('idle_s', 0) * 1e3:.1f}ms "
                         f"kernels {tr.get('kernel_count')} host_launches {tr.get('host_launches')} "
                         f"memcpy {json.dumps(tr.get('memcpy'))}")
            print(f"  {v['name']:28s} total {v['total_s']:7.2f} denoise {v['denoise_s']:7.3f} "
                  f"median step {statistics.median(steps) if steps else 0:.3f} peak {v.get('peak_mib')} MiB{extra}")
        for c in caps[:2]:
            print(f"  capture step {c[0]}: {c[1]} pool {c[2]:.2f} GiB")
        if caps:
            print(f"  max graph pool high {max(c[2] for c in caps):.2f} GiB over {len(caps)} captures")

h = os.path.join(root, "hashes.tsv")
if os.path.exists(h):
    groups = collections.defaultdict(list)
    for line in open(h):
        cell, clip, n, digest = line.rstrip("\n").split("\t")
        ck, arm = cell.split("-", 1)
        name = clip
        if name == "cold":
            continue
        key = (ck, re.sub(r"-[abc]$", "", name))
        groups[key].append((arm, name, n, digest))
    print("== byte identity (all arms and requests of a checkpoint / prompt / canvas)")
    for key, vals in sorted(groups.items()):
        digests = {d for _, _, _, d in vals}
        print(f"  {key}: {len(vals)} clips, {len(digests)} distinct, files {sorted({n for _, _, n, _ in vals})}"
              + ("" if len(digests) == 1 else f"  MISMATCH {vals}"))
