# -*- coding: utf-8 -*-
"""v3.3 终版验证: 固定平铺窗(不依赖能量排序) + 扩大共识样本
窗 = [0, 35, 70] 平铺 + 能量Top1(钳制) + 中段50 → 最多5窗(允许重叠)
评分 = 窗分中位; 共识金标(经典 vs 普通) 检验效度
"""
import json, sys, warnings, tempfile
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from scipy.stats import mannwhitneyu, spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
sys.path.insert(0, str(ROOT))
report = json.loads((ROOT/"eval"/"v3_validation.json").read_text(encoding='utf-8'))
norms = report['norms']
DIMS = ["orn", "gap", "cent", "dyn", "tension", "micro"]
CACHE = ROOT/"eval/v33_window_scores.json"

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

def window_scores(path, key, want=5):
    cache = json.loads(CACHE.read_text(encoding='utf-8')) if CACHE.exists() else {}
    entry = cache.get(key, {})
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    dur = len(y)/SR
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    ons = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    win = int(30*SR/hop); step = int(5*SR/hop)
    best_t, best_sc = 0.0, -1
    for t in range(0, max(1, len(rms)-win), step):
        sc = rms[t:t+win].mean()*(1+ons[t:t+win].mean())
        if sc > best_sc: best_sc, best_t = sc, t*hop/sr
    # 固定平铺: 0, 35, 70 (100s) + 能量峰(钳制0-70) + 中段(钳制0-70)
    tiles = [0.0, 35.0, 70.0] if dur >= 100 else [0.0, max(0, (dur-30)/2), max(0, dur-30)]
    tops = [round(min(70.0, max(0.0, best_t)), 1), round(min(70.0, max(0.0, dur/2-15)), 1)]
    order = [t for t in tiles + tops if t <= dur-25][:want]
    for t0 in order:
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

wavs = {w.stem: w for w in (ROOT/"eval"/"library_wavs").glob("*.wav")}

# 扩大共识: 经典(公认传世) vs 普通(明显较弱/边缘)
CLASSIC = ["真的爱你","光辉岁月","夜曲","听妈妈的话","安静","我难过","千年泪",
           "好想好想","双截棍","回到过去","开不了口","发如雪","彩虹","稻香",
           "威廉古堡","外婆","情非得已","海阔天空","专属天使","如果我变成回忆",
           "热带雨林","绿光","遇见","倔强","知足"]
MEDIOCRE = ["爱丫爱丫","发呆","樱花草","光荣","活埋","会长大的幸福",
            "紫藤花","一辈子的孤单","猪之歌","三国杀",
            "没有你在我有多难熬","向往","我知","真爱","会长大的幸福",
            "双侠","断桥残雪","灰色头像","玫瑰花的葬礼","城府"]

def find_song(pat):
    hits = [s for s in wavs if pat in s]
    return hits[0] if hits else None

cl, md = [], []
for grp, names in [(cl, CLASSIC), (md, MEDIOCRE)]:
    for pat in names:
        st = find_song(pat)
        if not st: print(f"  miss: {pat}", flush=True); continue
        scs = window_scores(wavs[st], st)
        if scs: grp.append((float(np.median(scs)), pat))
        else: print(f"  noscore: {pat}", flush=True)

print(f"\n共识样本: 经典 n={len(cl)} 普通 n={len(md)}", flush=True)
a = [x[0] for x in cl]; b = [x[0] for x in md]
print(f"经典中位={np.median(a):.1f} 普通中位={np.median(b):.1f}", flush=True)
if a and b:
    u, p = mannwhitneyu(a, b, alternative='greater')
    print(f"U检验 p={p:.4f} {'PASS(p<0.05)' if p<0.05 else 'FAIL'}", flush=True)
    print(f"经典: {sorted(round(x,1) for x in a)}", flush=True)
    print(f"普通: {sorted(round(x,1) for x in b)}", flush=True)

# 跨窗稳定性 (5窗拆2-3)
print("\n跨窗稳定性:", flush=True)
pairs = []
for k, d in json.loads(CACHE.read_text(encoding='utf-8')).items():
    scs = sorted(v for v in d.values() if v is not None)
    if len(scs) >= 4:
        pairs.append((float(np.median(scs[:2])), float(np.median(scs[2:]))))
if len(pairs) >= 10:
    rho, p = spearmanr([x[0] for x in pairs], [x[1] for x in pairs])
    print(f"  2v3 半分 n={len(pairs)} ρ={rho:.3f} (跨段内容不同, 参考值非门槛)", flush=True)
