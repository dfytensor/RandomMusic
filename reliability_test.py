# -*- coding: utf-8 -*-
"""标准可靠性双测:
测1 重测信度: 同歌换段落重测, 分数应稳定 (ρ≥0.9 才可靠)
测2 共识金标: 20 首公认经典 vs 普通流行, 经典应显著更高 (Mann-Whitney p<0.05)
"""
import json, re, sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
sys.path.insert(0, str(ROOT))

report = json.loads((ROOT/"eval"/"v3_validation.json").read_text(encoding='utf-8'))
norms = report['norms']
DIMS = ["orn", "gap", "cent", "dyn", "tension", "micro"]

def cat_of(bpm):
    if bpm < 100: return "C1"
    if bpm < 120: return "C2"
    if bpm < 135: return "C3"
    return "C4"

def score_of(m):
    cat = cat_of(m['bpm']); parts = []
    for d in DIMS:
        n = norms.get(cat, {}).get(d)
        v = m.get(d)
        if not n or v is None: continue
        s = 100*(1 - abs(v-n['med'])/n['scale'])
        if v < n['p5']: s -= (n['p5']-v)/n['scale']*100
        if v > n['p95']: s -= (v-n['p95'])/n['scale']*100
        parts.append(s)
    return round(float(np.mean(parts)),1) if parts else None

# ===== 测1: 重测信度 (40 首, 换段落) =====
import librosa
from score_v3 import find_chorus_segment, measure as measure_orig

test_m = json.loads((ROOT/"eval/v3_test_cache.json").read_text(encoding='utf-8'))
wavs = {w.stem: w for w in (ROOT/"eval"/"library_wavs").glob("*.wav")}
keys = [k for k, v in test_m.items() if v][:40]

def measure_shift(path):
    """换段落: 主歌区(能量峰-50s 或 20s)"""
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    _, t0 = find_chorus_segment(str(path))
    alt = max(15, t0 - 55)  # 副歌前 55s ≈ 主歌
    seg = y[int(alt*sr):int((alt+30)*sr)]
    if len(seg) < SR*10:
        seg = y[:SR*30]
    # 直接复用 score_v3.measure 逻辑太深, 用简化: 调 segment
    import tempfile, soundfile as sf
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    sf.write(str(tmp), seg, SR)
    try:
        return measure_orig(tmp)
    finally:
        tmp.unlink(missing_ok=True)

pairs = []
for k in keys:
    w = wavs.get(k)
    if not w: continue
    try:
        s1 = score_of(test_m[k])
        m2 = measure_shift(w)
        s2 = score_of(m2) if m2 else None
        if s1 is not None and s2 is not None:
            pairs.append((s1, s2))
    except Exception as e:
        print(f"{k[:16]} ERR {type(e).__name__}", flush=True)

print(f"测1 重测信度 n={len(pairs)}", flush=True)
if len(pairs) >= 10:
    from scipy.stats import spearmanr
    rho, p = spearmanr([x[0] for x in pairs], [x[1] for x in pairs])
    print(f"  test-retest ρ={rho:.3f} p={p:.4f} {'PASS(≥0.9)' if rho>=0.9 else 'FAIL(<0.9)'}", flush=True)

# ===== 测2: 共识金标 (经典 vs 普通) =====
CLASSIC = ["真的爱你", "光辉岁月", "夜曲", "听妈妈的话", "安静", "我难过",
           "千年泪", "好想好想", "双截棍", "回到过去"]
MEDIOCRE = ["爱丫爱丫", "发呆", "樱花草", "光荣", "活埋", "会长大的幸福",
            "紫藤花", "美丽新世界", "一辈子的孤单", "猪之歌"]

def find_song(pat):
    for stem, w in wavs.items():
        if pat in stem: return stem
    return None

import subprocess
def measure_cached(stem):
    m = test_m.get(stem) or json.loads((ROOT/"eval/v3_gold_cache.json").read_text(encoding='utf-8')).get(stem) \
        or json.loads((ROOT/"eval/v3_train_cache.json").read_text(encoding='utf-8')).get(stem)
    return m

cl_scores, md_scores = [], []
for pat in CLASSIC:
    st = find_song(pat)
    if not st: print(f"  经典未找到: {pat}"); continue
    m = measure_cached(st)
    if m:
        s = score_of(m)
        if s is not None: cl_scores.append(s)
for pat in MEDIOCRE:
    st = find_song(pat)
    if not st: print(f"  普通未找到: {pat}"); continue
    m = measure_cached(st)
    if m:
        s = score_of(m)
        if s is not None: md_scores.append(s)

print(f"测2 共识金标: 经典 n={len(cl_scores)} 普通 n={len(md_scores)}", flush=True)
if cl_scores and md_scores:
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(cl_scores, md_scores, alternative='greater')
    print(f"  经典中位={np.median(cl_scores):.1f} 普通中位={np.median(md_scores):.1f} "
          f"U检验 p={p:.4f} {'PASS(p<0.05)' if p<0.05 else 'FAIL(p≥0.05)'}", flush=True)
    print(f"  经典: {sorted([round(x,1) for x in cl_scores])}", flush=True)
    print(f"  普通: {sorted([round(x,1) for x in md_scores])}", flush=True)
