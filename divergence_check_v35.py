# -*- coding: utf-8 -*-
"""背离修正验证: 原 v3 分与 onset_rise +0.43 正相关(奖赏字头猛)
检查 v3.5 分与 5 新维的 Spearman (12冠军 + 01/04 seed 池)"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).parent
report = json.loads((ROOT/"eval/v3_validation.json").read_text(encoding='utf-8'))
norms = report['norms']
DIMS = ["orn", "gap", "cent", "dyn", "tension", "micro",
        "vib_ratio", "vib_depth", "phase_R", "onset_rise_db", "long_smooth"]
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

# 样本: 12 冠军 + 01_v2 11 seed + 04_v3 8 seed
# 6 旧维来自混音 measure(), 5 新维来自 stem art_dims()
import librosa
from art_core import art_dims
from score_v3 import measure as full_measure

gacha = ROOT/"novel_songs"/"batch_gacha"
flac_by_seed = {p.stem: p for p in gacha.glob("**/seed*.flac")}

pools = []
for base in [ROOT/"album_voice"/"_stems", ROOT/"ai_stems"/"01_v2", ROOT/"ai_stems"/"04_v3"]:
    for d in sorted(base.iterdir()):
        if (d/"vocals.wav").exists() and (d/"beats.json").exists():
            pools.append(d)
seen, uniq = set(), []
for d in pools:
    if d.name in seen: continue
    seen.add(d.name); uniq.append(d)

rows, miss = [], []
for d in uniq:
    mix = flac_by_seed.get(d.name)
    if mix is None:
        miss.append(d.name); continue
    try:
        m = full_measure(mix, vocals_path=d/"vocals.wav", beats_path=d/"beats.json")
    except Exception as e:
        miss.append(f"{d.name}({type(e).__name__})"); continue
    rows.append((d.name, m))
print(f"n={len(rows)} (缺 mix: {miss})")

s35 = {n: score_of(m, DIMS) for n, m in rows}
# 原 v3 分: 6维-only (同 norms 的旧维部分, 即 v3.3 式)
s3 = {n: score_of(m, ["orn","gap","cent","dyn","tension","micro"]) for n, m in rows}

print(f"\n{'dim':14s} {'Spearman(v3分,维)':>18s} {'p':>8s}   {'Spearman(v3.5分,维)':>20s} {'p':>8s}")
for d in ART5:
    pairs3 = [(s3[n], m[d]) for n, m in rows if s3[n] is not None and m.get(d) is not None]
    pairs35 = [(s35[n], m[d]) for n, m in rows if s35[n] is not None and m.get(d) is not None]
    r3, p3 = spearmanr([x[0] for x in pairs3], [x[1] for x in pairs3])
    r35, p35 = spearmanr([x[0] for x in pairs35], [x[1] for x in pairs35])
    print(f"{d:14s} {r3:+18.3f} {p3:8.3f}   {r35:+20.3f} {p35:8.3f}")
print("\n期望: onset_rise 从 +0.43(奖赏字头猛) 变为 负/无相关(不再奖赏); vib/phase 不再盲区(应出现相关)")
