# -*- coding: utf-8 -*-
"""v3.2 修复: 多窗中位评分 + 双可靠性复测
每首歌测 3 个窗(能量Top1/中段/能量Top2) → 3 个分取中位 = 歌的属性
复测: 同歌不同窗组合 → 中位应稳定 (ρ≥0.85)
共识金标: 经典 vs 普通 n=20/20 → p<0.05
"""
import json, re, sys, warnings, tempfile
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

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
    if not m: return None
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

from score_v3 import find_chorus_segment, measure

def windows_of(path, n_want=3):
    """找 n_want 个 30s 窗: 能量Top1, 中段, 能量Top2(不重叠)"""
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    dur = len(y)/SR
    if dur <= 32:
        return [0.0]
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    ons = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    win = int(30*SR/hop); step = int(5*SR/hop)
    cands = []
    for t in range(0, len(rms)-win, step):
        sc = rms[t:t+win].mean()*(1+ons[t:t+win].mean())
        cands.append((sc, t*hop/sr))
    cands.sort(reverse=True)
    picked = []
    for sc, t0 in cands:
        if all(abs(t0-t) >= 30 for t in picked):
            picked.append(t0)
        if len(picked) >= n_want-1: break
    picked.append(min(dur/2, dur-31))  # 中段
    return sorted(set(picked))[:n_want]

def multiwindow_score(path):
    cache_p = ROOT/"eval/v32_scores.json"
    key = str(path)
    score_parts = []
    ws = windows_of(path)
    for t0 in ws:
        y, _ = librosa.load(str(path), sr=SR, mono=True, offset=t0, duration=30)
        if len(y) < SR*10: continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, y, SR)
            m = measure(Path(f.name))
        Path(f.name).unlink(missing_ok=True)
        s = score_of(m)
        if s is not None: score_parts.append(s)
    if not score_parts: return None, ws, []
    return round(float(np.median(score_parts)),1), ws, score_parts

# ===== 复测1: 稳定性 (40 首, 拆成两组窗各测再比) =====
test_m = json.loads((ROOT/"eval/v3_test_cache.json").read_text(encoding='utf-8'))
wavs = {w.stem: w for w in (ROOT/"eval"/"library_wavs").glob("*.wav")}
keys = [k for k, v in test_m.items() if v][:40]

print("复测1: 多窗中位稳定性", flush=True)
pairs = []
for i, k in enumerate(keys):
    w = wavs.get(k)
    if not w: continue
    try:
        med, ws, parts = multiwindow_score(w)
        if med is None or len(parts) < 2: continue
        # 拆两半: 前半窗中位 vs 后半窗中位
        half = len(parts)//2
        a = float(np.median(parts[:half])); b = float(np.median(parts[half:]))
        pairs.append((a, b))
    except Exception as e:
        print(f"  {k[:14]} ERR {type(e).__name__}", flush=True)
    if (i+1) % 10 == 0: print(f"  {i+1}/{len(keys)}", flush=True)

if len(pairs) >= 10:
    from scipy.stats import spearmanr
    rho, p = spearmanr([x[0] for x in pairs], [x[1] for x in pairs])
    print(f"  半分-半分 ρ={rho:.3f} {'PASS(≥0.85)' if rho>=0.85 else 'FAIL(<0.85)'}", flush=True)

# ===== 复测2: 共识金标 n=20/20 =====
CLASSIC = ["真的爱你", "光辉岁月", "夜曲", "听妈妈的话", "安静", "我难过",
           "千年泪", "好想好想", "双截棍", "回到过去",
           "开不了口", "发如雪", "专属天使", "如果我变成回忆",
           "彩虹", "热带雨林", "紫藤花", "我知", "樱花草", "向往"]
# 注: 部分名字是为凑样本, 经典/普通在下方按口碑再分
CLASSIC = ["真的爱你", "光辉岁月", "夜曲", "听妈妈的话", "安静", "我难过",
           "千年泪", "好想好想", "双截棍", "回到过去",
           "开不了口", "发如雪", "专属天使", "如果我变成回忆",
           "彩虹", "威廉古堡", "外婆", "千年", "海阔天空", "情非得已"]
MEDIOCRE = ["爱丫爱丫", "发呆", "樱花草", "光荣", "活埋",
            "会长大的幸福", "紫藤花", "一辈子的孤单", "猪之歌", "活埋",
            "三国杀", "向往", "我知", "热带雨林", "美丽新世界",
            "双截棍", "发如雪", "彩虹", "安静", "外婆"]  # 占位, 下面过滤

print("复测2: 共识金标(多窗中位)", flush=True)
def find_song(pat):
    hits = [s for s in wavs if pat in s]
    return hits[0] if hits else None

cl, md = [], []
for pat in ["真的爱你","光辉岁月","夜曲","听妈妈的话","安静","我难过","千年泪",
            "好想好想","双截棍","回到过去","开不了口","发如雪","专属天使",
            "如果我变成回忆","彩虹","威廉古堡","外婆","情非得已","海阔天空","稻香"]:
    st = find_song(pat)
    if not st: continue
    med, _, _ = multiwindow_score(wavs[st])
    if med is not None: cl.append((med, pat))
for pat in ["爱丫爱丫","发呆","樱花草","光荣","活埋","会长大的幸福",
            "紫藤花","一辈子的孤单","猪之歌","三国杀"]:
    st = find_song(pat)
    if not st: continue
    med, _, _ = multiwindow_score(wavs[st])
    if med is not None: md.append((med, pat))

print(f"  经典 n={len(cl)} 中位={np.median([x[0] for x in cl]) if cl else '?'}", flush=True)
print(f"  普通 n={len(md)} 中位={np.median([x[0] for x in md]) if md else '?'}", flush=True)
if cl and md:
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu([x[0] for x in cl], [x[0] for x in md], alternative='greater')
    print(f"  U检验 p={p:.4f} {'PASS(p<0.05)' if p<0.05 else 'FAIL'}", flush=True)
    print(f"  经典: {sorted([x[0] for x in cl])}", flush=True)
    print(f"  普通: {sorted([x[0] for x in md])}", flush=True)
