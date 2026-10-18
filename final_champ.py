# -*- coding: utf-8 -*-
"""终版: 冠军副歌段单窗分(与曲库同方法) + 跨窗离散度对比"""
import json, sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

report = json.loads((ROOT/"eval"/"v3_validation.json").read_text(encoding='utf-8'))
norms = report['norms']
test_scores = None
from score_v3 import measure, score_of

CHAMPS = [
    ("01 非我不可", r"novel_songs\batch_gacha\01_v2\seed58888.flac"),
    ("02 平凡的",   r"novel_songs\batch_gacha\02_wide\seed7778.flac"),
    ("03 白狐",     r"novel_songs\batch_gacha\03_v2\seed61001.flac"),
    ("04 龙珠劫",   r"novel_songs\batch_gacha\04\seed90406.flac"),
    ("05 阳光说的对", r"novel_songs\batch_gacha\05_wide\seed1111.flac"),
    ("06 第320次",  r"novel_songs\batch_gacha\06_wide\seed3690.flac"),
    ("07 烟花",     r"novel_songs\batch_gacha\07_adj\seed1357_r1.flac"),
    ("08 童话世界", r"novel_songs\batch_gacha\11\seed91104.flac"),
    ("09 青衫",     r"novel_songs\batch_gacha\09\seed90904.flac"),
    ("10 毒药",     r"novel_songs\batch_gacha\10\seed91007.flac"),
    ("11 尾行",     r"novel_songs\batch_gacha\11\seed91103.flac"),
    ("12 天涯",     r"novel_songs\batch_gacha\12\seed91202.flac"),
]

print("冠军副歌段评分(与曲库同方法):", flush=True)
rows = []
for name, rel in CHAMPS:
    p = ROOT/rel
    if not p.exists():
        print(f"  {name}: MISSING", flush=True); continue
    try:
        m = measure(p)
        r = score_of(m)
        sc = r['score'] if r else None
        rows.append((name, sc, r['cat'] if r else '?', m.get('seg_t') if m else None))
        print(f"  {name}: {sc} [{r['cat'] if r else '?'}] seg_t={m.get('seg_t') if m else '-'}", flush=True)
    except Exception as e:
        print(f"  {name}: ERR {type(e).__name__} {e}", flush=True)

vals = [r[1] for r in rows if r[1] is not None]
if vals:
    print(f"\n12冠军副歌段: 均值={np.mean(vals):.1f} 中位={np.median(vals):.1f} min={min(vals)} max={max(vals)}", flush=True)

# 曲库 test 对照
tm = json.loads((ROOT/"eval/v3_test_cache.json").read_text(encoding='utf-8'))
tscores = [score_of(m)['score'] for m in tm.values() if m and score_of(m)]
print(f"曲库test n={len(tscores)}: 中位={np.median(tscores):.1f} P25={np.percentile(tscores,25):.1f} P75={np.percentile(tscores,75):.1f}", flush=True)

# 跨窗离散度对比 (v33 caches)
def spreads(cache_p):
    c = json.loads(cache_p.read_text(encoding='utf-8'))
    out = []
    for k, d in c.items():
        vs = [v for v in d.values() if v is not None]
        if len(vs) >= 3: out.append(max(vs)-min(vs))
    return out
lib_sp = spreads(ROOT/"eval/v33_window_scores.json")
ai_sp = spreads(ROOT/"eval/v33_champions.json")
print(f"\n跨窗极差: 曲库 n={len(lib_sp)} 中位={np.median(lib_sp):.1f} | 冠军 n={len(ai_sp)} 中位={np.median(ai_sp):.1f}", flush=True)
