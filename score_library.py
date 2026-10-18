# -*- coding: utf-8 -*-
"""全库 568 首 75分子量表评分：分布/最高分/96-100区间(110制)数量 -> eval/hit_library_score.json"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from hit_compare import subscore

lib = ROOT / "eval" / "library_wavs"
trs = ROOT / "eval" / "library_transcripts"

rows = []
for w in sorted(lib.glob("*.wav")):
    stem = w.stem
    tdir = trs / stem
    if not (tdir / "score.abc").exists():
        continue
    try:
        dur = (lambda i: i.frames / i.samplerate)(sf.info(str(w)))
        r = subscore(w, tdir, dur)
        r["name"] = stem[4:]
        rows.append(r)
    except Exception as e:
        print("FAIL", stem[:24], str(e)[:60], flush=True)
    if len(rows) % 100 == 0:
        print(f"[{len(rows)}]", flush=True)

vals = np.array([r["total"] for r in rows])
# 110 制的 96-100 区间换算到 75 制
zone_lo, zone_hi = 96 / 110 * 75, 100 / 110 * 75
in_zone = [r for r in rows if zone_lo <= r["total"] <= zone_hi]
above = [r for r in rows if r["total"] > zone_hi]
res = dict(
    n=len(rows),
    med=round(float(np.median(vals)), 1),
    p75=round(float(np.percentile(vals, 75)), 1),
    p90=round(float(np.percentile(vals, 90)), 1),
    p99=round(float(np.percentile(vals, 99)), 1),
    mx=round(float(vals.max()), 1),
    zone_96_100_count=len(in_zone),
    zone_96_100_songs=[f"{r['name']} ({r['total']})" for r in sorted(in_zone, key=lambda r: -r["total"])][:15],
    above_zone_count=len(above),
    above_zone_songs=[f"{r['name']} ({r['total']})" for r in sorted(above, key=lambda r: -r["total"])][:15],
    top10=[dict(name=r["name"][:30], total=r["total"], arc=r["arc"], fv=r["fv"],
                n_chorus=r["n_chorus"], cpos=r["cpos"])
           for r in sorted(rows, key=lambda r: -r["total"])[:10]],
)
(ROOT / "eval" / "hit_library_score.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=1,
               default=lambda o: float(o) if hasattr(o, "item") else str(o)), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "top10"}, ensure_ascii=False, indent=1))
print("TOP10:", json.dumps(res["top10"], ensure_ascii=False, indent=1))
