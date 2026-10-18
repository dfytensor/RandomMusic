# -*- coding: utf-8 -*-
"""听测集: 01_v2 同歌 11 seed 按 v3 副歌段分排序, 复制到 listen_test/rank01_听测"""
import json, shutil, sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
report = json.loads((ROOT/"eval"/"v3_validation.json").read_text(encoding='utf-8'))
norms = report['norms']
from score_v3 import measure, score_of

SRC = ROOT/"novel_songs"/"batch_gacha"/"01_v2"
OUT = ROOT/"listen_test"/"rank01_听测"
OUT.mkdir(parents=True, exist_ok=True)

cache_p = ROOT/"eval/seed_rank01_cache.json"
cache = json.loads(cache_p.read_text(encoding='utf-8')) if cache_p.exists() else {}

files = sorted(SRC.glob("seed*.flac"))
print(f"源文件 {len(files)} 个", flush=True)
rows = []
for f in files:
    m = cache.get(f.stem)
    if not m:
        for attempt in range(2):
            try:
                m = measure(f)
                if m: break
            except Exception as e:
                print(f"  {f.stem} 尝试{attempt+1} ERR {type(e).__name__}", flush=True)
        cache[f.stem] = m
        cache_p.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
    r = score_of(m) if m else None
    sc = r['score'] if r else None
    rows.append((sc, f, m))
    print(f"  {f.stem}: {sc} [{r['cat'] if r else '?'}] seg_t={m.get('seg_t') if m else '-'}", flush=True)

rows.sort(key=lambda x: -(x[0] if x[0] is not None else -999))
lines = ["# 01_v2《非我不可》11 seed 排名（v3 副歌段分）", "",
         "| 名次 | 分 | seed | 副歌段位置 | orn | gap | cent | dyn | tension | micro | BPM |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
print("\n最终排名:", flush=True)
for i, (sc, f, m) in enumerate(rows, 1):
    dst = OUT/f"rank{i:02d}_{f.stem}.flac"
    shutil.copy2(f, dst)
    print(f"  #{i:02d} {sc:6.1f}  {f.stem}", flush=True)
    if m:
        lines.append(f"| {i} | {sc:.1f} | {f.stem} | {m.get('seg_t')} | "
                     f"{m.get('orn'):.2f} | {m.get('gap'):.2f} | {m.get('cent'):.0f} | "
                     f"{m.get('dyn'):.1f} | {m.get('tension'):.1f} | {m.get('micro'):.1f} | {m.get('bpm'):.0f} |")
(OUT/"排名.md").write_text("\n".join(lines), encoding='utf-8')
vals = [r[0] for r in rows if r[0] is not None]
print(f"\n已复制到 {OUT}", flush=True)
print(f"分数: max={max(vals):.1f} min={min(vals):.1f} 中位={np.median(vals):.1f} 极差={max(vals)-min(vals):.1f}", flush=True)
