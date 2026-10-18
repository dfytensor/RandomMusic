# -*- coding: utf-8 -*-
"""v3.5 sanity: ①test 集 v3.5 vs 6维分 Spearman ②冠军 6维/新维分拆 ③排名稳定性"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

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

def score_of(m, dims):
    if not m: return None
    cat = cat_of(m['bpm']); parts = []
    for d in dims:
        n = norms.get(cat, {}).get(d); v = m.get(d)
        if not n or v is None: continue
        s = 100*(1 - abs(v-n['med'])/n['scale'])
        if v < n['p5']: s -= (n['p5']-v)/n['scale']*100
        if v > n['p95']: s -= (v-n['p95'])/n['scale']*100
        parts.append(s)
    return float(np.mean(parts)) if parts else None

# ① test: v3.5(11) vs v3.3式(6)
test_m = json.loads((ROOT/"eval/v3_test_cache.json").read_text(encoding='utf-8'))
s11, s6 = {}, {}
for k, m in test_m.items():
    if not m: continue
    s11[k] = score_of(m, OLD6+ART5)
    s6[k] = score_of(m, OLD6)
ks = [k for k in s11 if s11[k] is not None and s6[k] is not None]
rho, p = spearmanr([s6[k] for k in ks], [s11[k] for k in ks])
print(f"① test n={len(ks)}  v3.3式(6维) vs v3.5(11维) Spearman ρ={rho:.3f} p={p:.2g}")
print(f"   6维中位={np.median([s6[k] for k in ks]):.1f}  11维中位={np.median([s11[k] for k in ks]):.1f}")
# 排名变动 top
r6 = {k: r for r, k in enumerate(sorted(ks, key=lambda x: -s6[x]))}
r11 = {k: r for r, k in enumerate(sorted(ks, key=lambda x: -s11[x]))}
flips = sorted(ks, key=lambda k: -abs(r6[k]-r11[k]))[:5]
for k in flips:
    print(f"   排名变动 {k[:24]:24s} 6维#{r6[k]:3d} -> 11维#{r11[k]:3d}")

# ② 人类 test 新维子分 vs 冠军新维子分
h_art = [s for k in ks if (s := score_of(test_m[k], ART5)) is not None]
print(f"\n② 人类test 新维子分: 中位={np.median(h_art):.1f} (n={len(h_art)})")
ch = json.loads((ROOT/"eval/v35_champions.json").read_text(encoding='utf-8'))
print("   冠军新维子分(按 beats tempo 定 cat):")
CHAMPS = {
    "01 非我不可": "seed58888", "02 平凡的": "seed7778", "03 白狐": "seed61001",
    "04 龙珠劫": "seed90406", "05 阳光说的对": "seed1111", "06 第320次": "seed3690",
    "07 烟花": "seed1357_r1", "08 童话世界": "seed91104", "09 青衫": "seed90904",
    "10 毒药": "seed91007", "11 尾行": "seed91103", "12 天涯": "seed91202",
}
rows = []
for name, seed in CHAMPS.items():
    art = ch.get(f"__art__{seed}")
    if not art: continue
    bj = ROOT/"album_voice"/"_stems"/seed/"beats.json"
    tempo = json.loads(bj.read_text(encoding="utf-8"))["tempo"] if bj.exists() else 120
    m = dict(art); m["bpm"] = tempo
    sa = score_of(m, ART5)
    rows.append((name, sa))
    print(f"   {name}: 新维子分={sa if sa is None else round(sa,1)} (tempo={tempo:.0f})")
vals = [r[1] for r in rows if r[1] is not None]
if vals:
    print(f"   冠军新维子分中位={np.median(vals):.1f} vs 人类test {np.median(h_art):.1f}")
else:
    print("   (冠军 __art__ 缓存缺失 — 重跑 score_champions_v35.py 后再看)")

# ③ 冠军排名 v3.3 vs v3.5
v33 = json.loads((ROOT/"eval/v33_champions.json").read_text(encoding='utf-8'))
old_med = {n: float(np.median([v for v in v33[n].values() if v is not None])) for n in CHAMPS if n in v33}
new_med = {n: float(np.median([v for v in ch[n].values() if v is not None])) for n in CHAMPS if n in ch}
common = sorted(set(old_med) & set(new_med))
rho2, p2 = spearmanr([old_med[n] for n in common], [new_med[n] for n in common])
print(f"\n③ 冠军排名 v3.3 vs v3.5 Spearman ρ={rho2:.3f} (n={len(common)})")
print("   排名对照 (v3.5 降序):")
for i, n in enumerate(sorted(new_med, key=lambda x: -new_med[x]), 1):
    rank33 = common.index(n) # noop
    order33 = sorted(old_med, key=lambda x: -old_med[x])
    print(f"   {i:2d}. {n:12s} v3.5={new_med[n]:5.1f} (v3.3排名{order33.index(n)+1:2d} 分{old_med[n]:5.1f})")
