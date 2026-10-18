# -*- coding: utf-8 -*-
"""诊断: v3.5 相对 v3.3 式大幅下跌的 test 歌 — 新维哪维崩了? 是否分离伪影?"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
report = json.loads((ROOT/"eval/v3_validation.json").read_text(encoding='utf-8'))
norms = report['norms']
OLD6 = ["orn", "gap", "cent", "dyn", "tension", "micro"]
ART5 = ["vib_ratio", "vib_depth", "phase_R", "onset_rise_db", "long_smooth"]

def cat_of(bpm):
    if bpm < 100: return "C1"
    if bpm < 120: return "C2"
    if bpm < 135: return "C3"
    return "C4"

def dim_score(m, d):
    n = norms.get(cat_of(m['bpm']), {}).get(d); v = m.get(d)
    if not n or v is None: return None
    s = 100*(1 - abs(v-n['med'])/n['scale'])
    if v < n['p5']: s -= (n['p5']-v)/n['scale']*100
    if v > n['p95']: s -= (v-n['p95'])/n['scale']*100
    return s

def score_of(m, dims):
    parts = [dim_score(m, d) for d in dims]
    parts = [p for p in parts if p is not None]
    return float(np.mean(parts)) if parts else None

test_m = json.loads((ROOT/"eval/v3_test_cache.json").read_text(encoding='utf-8'))
art = json.loads((ROOT/"eval/v3_art_cache.json").read_text(encoding='utf-8'))

drops = []
for k, m in test_m.items():
    if not m: continue
    s6, s11 = score_of(m, OLD6), score_of(m, OLD6+ART5)
    if s6 is None or s11 is None: continue
    drops.append((s6-s11, k, s6, s11, m))
drops.sort(reverse=True)
big = [d for d in drops if d[0] > 25]
print(f"下跌>25分的歌: {len(big)}/{len(drops)} ({100*len(big)/len(drops):.1f}%)")
print(f"上涨>25分的歌: {len([d for d in drops if d[0] < -25])}")
print("\nTop15 下跌: 每维分(崩维标记*)")
for delta, k, s6, s11, m in drops[:15]:
    parts = []
    for d in OLD6+ART5:
        s = dim_score(m, d)
        mark = "*" if (s is not None and s < 0) else ""
        parts.append(f"{d[:4]}={('%.0f'%s) if s is not None else '--'}{mark}")
    print(f"  {k[:28]:28s} 6维={s6:5.1f} 11维={s11:5.1f} Δ={delta:5.1f} {' '.join(parts)}")

# 崩维统计: 各新维在 test 中触发 s<0 的数量
print("\n新维在 test 中触发负分(离谱越界)的数量:")
cnt = {d: 0 for d in ART5}
for k, m in test_m.items():
    if not m: continue
    for d in ART5:
        s = dim_score(m, d)
        if s is not None and s < 0: cnt[d] += 1
print("  ", cnt)
print("\n下跌Top5 的新维原始值 vs 人类 norm:")
for delta, k, s6, s11, m in drops[:5]:
    cat = cat_of(m['bpm'])
    parts = []
    for d in ART5:
        n = norms.get(cat, {}).get(d); v = m.get(d)
        if n and v is not None:
            z = (v-n['med'])/n['scale']
            parts.append(f"{d}={v:.3g}(z={z:+.1f})")
    print(f"  {k[:24]:24s} {' '.join(parts)}")
