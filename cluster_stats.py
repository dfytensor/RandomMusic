# -*- coding: utf-8 -*-
"""Cluster corpus by tempo/brightness/energy to ground style clusters in real stats."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

feats = json.loads(Path(r"F:\yinyue\corpus\features.json").read_text(encoding="utf-8"))
rows = list(feats.values())
PITCH = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def cluster(r):
    bpm, cent, rms, dyn = r["bpm"], r["centroid_hz"], r["rms_db"], r["dyn_p95_p10_db"]
    if bpm < 75:
        return "C1-ballad"
    if bpm < 110:
        return "C2-midpop"
    if cent >= 2500 and dyn < 13:
        return "C4-dance"
    return "C3-rock"


def med(vals):
    a = np.array(vals)
    return dict(bpm=round(float(np.median(a)), 1), dur=round(float(np.median([r["duration_s"] for r in rows])), 0))


groups = {}
for r in rows:
    groups.setdefault(cluster(r), []).append(r)

out = {}
for name, rs in sorted(groups.items()):
    bpms = [r["bpm"] for r in rs]
    cents = [r["centroid_hz"] for r in rs]
    dyns = [r["dyn_p95_p10_db"] for r in rs]
    durs = [r["duration_s"] for r in rs]
    modes = Counter("minor" if "minor" in r["key"] else "major" for r in rs)
    keys = Counter(r["key"].split()[0] for r in rs).most_common(5)
    out[name] = dict(
        n=len(rs), share=round(len(rs) / len(rows) * 100),
        bpm_med=round(float(np.median(bpms)), 1),
        bpm_p25_p75=[round(float(np.percentile(bpms, 25))), round(float(np.percentile(bpms, 75)))],
        centroid_med=round(float(np.median(cents))),
        dyn_med=round(float(np.median(dyns)), 1),
        dur_med=round(float(np.median(durs))),
        major_pct=round(modes["major"] / len(rs) * 100),
        top_keys=[f"{k}({v})" for k, v in keys],
    )
print(json.dumps(out, ensure_ascii=False, indent=1))
Path(r"F:\yinyue\corpus\cluster_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
