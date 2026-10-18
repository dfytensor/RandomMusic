# -*- coding: utf-8 -*-
"""增量 5 窗稳定性复测（缓存已有窗分, 只补新窗）"""
import json, sys, warnings, tempfile
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
CACHE = ROOT/"eval"/"v32_window_scores.json"

def cat_of(bpm):
    if bpm < 100: return "C1"
    if bpm < 120: return "C2"
    if bpm < 135: return "C3"
    return "C4"

def score_of(m):
    if not m: return None
    cat = cat_of(m['bpm']); parts = []
    for d in DIMS:
        n = norms.get(cat, {}).get(d); v = m.get(d)
        if not n or v is None: continue
        s = 100*(1 - abs(v-n['med'])/n['scale'])
        if v < n['p5']: s -= (n['p5']-v)/n['scale']*100
        if v > n['p95']: s -= (v-n['p95'])/n['scale']*100
        parts.append(s)
    return round(float(np.mean(parts)),1) if parts else None

from score_v3 import measure

def ensure_windows(path, key, want_n=5):
    cache = json.loads(CACHE.read_text(encoding='utf-8')) if CACHE.exists() else {}
    entry = cache.get(key, {})
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    dur = len(y)/SR
    if dur <= 32:
        if not entry: entry["0.0"] = None
        # 单窗直接测
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    ons = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    win = int(30*SR/hop); step = int(5*SR/hop)
    cands = []
    for t in range(0, max(1, len(rms)-win), step):
        sc = rms[t:t+win].mean()*(1+ons[t:t+win].mean())
        cands.append((sc, round(t*hop/sr, 1)))
    cands.sort(reverse=True)
    # 取互不重叠前 want_n-1 + 中段
    picked = []
    for sc, t0 in cands:
        if all(abs(t0-t) >= 30 for t in picked):
            picked.append(t0)
        if len(picked) >= want_n-1: break
    mid = round(min(dur/2, dur-31), 1)
    if all(abs(mid-t) >= 30 for t in picked): picked.append(mid)
    for t0 in picked[:want_n]:
        sk = str(t0)
        if sk in entry and entry[sk] is not None: continue
        seg = y[int(t0*sr):int((t0+30)*sr)]
        if len(seg) < SR*10: entry[sk] = None; continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, seg, SR)
            m = measure(Path(f.name))
        Path(f.name).unlink(missing_ok=True)
        entry[sk] = score_of(m)
    cache[key] = entry
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
    return [v for v in entry.values() if v is not None]

test_m = json.loads((ROOT/"eval/v3_test_cache.json").read_text(encoding='utf-8'))
wavs = {w.stem: w for w in (ROOT/"eval"/"library_wavs").glob("*.wav")}
keys = [k for k, v in test_m.items() if v][:40]

print("补窗+复测 (40首)", flush=True)
pairs = []
for i, k in enumerate(keys):
    w = wavs.get(k)
    if not w: continue
    scs = ensure_windows(w, k, want_n=5)
    if len(scs) < 4: continue
    scs = sorted(scs)  # 伪随机顺序 → 拆 2-vs-3
    a = float(np.median(scs[:2])); b = float(np.median(scs[2:]))
    pairs.append((a, b))
    if (i+1) % 10 == 0: print(f"  {i+1}/{len(keys)} pairs={len(pairs)}", flush=True)

if len(pairs) >= 10:
    from scipy.stats import spearmanr
    rho, p = spearmanr([x[0] for x in pairs], [x[1] for x in pairs])
    print(f"5窗 2-vs-3 半分 ρ={rho:.3f} {'PASS(≥0.85)' if rho>=0.85 else 'FAIL'}", flush=True)

# 共识金标 5 窗复测
cl_names = ["真的爱你","光辉岁月","夜曲","听妈妈的话","安静","我难过","千年泪",
            "好想好想","双截棍","回到过去","开不了口","发如雪","专属天使",
            "如果我变成回忆","彩虹","威廉古堡","外婆","情非得已","海阔天空","稻香"]
md_names = ["爱丫爱丫","发呆","樱花草","光荣","活埋","会长大的幸福",
            "紫藤花","一辈子的孤单","猪之歌","三国杀"]
def find_song(pat):
    hits = [s for s in wavs if pat in s]
    return hits[0] if hits else None
cl, md = [], []
for grp, names in [(cl, cl_names), (md, md_names)]:
    for pat in names:
        st = find_song(pat)
        if not st: continue
        scs = ensure_windows(wavs[st], st, want_n=5)
        if scs: grp.append(float(np.median(scs)))
print(f"共识5窗: 经典 n={len(cl)} 中位={np.median(cl):.1f} | 普通 n={len(md)} 中位={np.median(md):.1f}", flush=True)
if cl and md:
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(cl, md, alternative='greater')
    print(f"U检验 p={p:.4f} {'PASS(p<0.05)' if p<0.05 else 'FAIL'}", flush=True)
