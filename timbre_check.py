# -*- coding: utf-8 -*-
"""音色同质化验证: 11 seed (同歌) vs 11 首不同人类歌 的 MFCC 均值向量距离
用户听感: '所有AI都这个音色' → 预期: AI seed 间音色距离显著小于人类歌间
"""
import warnings
from pathlib import Path
import numpy as np
import librosa
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

def mfcc_mean(path, t0=0.0, dur=30.0):
    y, sr = librosa.load(str(path), sr=22050, mono=True, offset=t0, duration=dur)
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.concatenate([m.mean(1), m.std(1)])  # 26维音色签名

# AI: 11 seed (用各自副歌段 t)
import json
cache = json.loads((ROOT/"eval/seed_rank01_cache.json").read_text(encoding='utf-8'))
ai_vecs, names = [], []
for f in sorted((ROOT/"listen_test"/"rank01_听测").glob("rank*.flac")):
    seed = f.stem[7:]
    t0 = (cache.get(seed) or {}).get('seg_t', 60.0)
    try:
        ai_vecs.append(mfcc_mean(f, t0)); names.append(seed)
    except Exception as e:
        print(f"{seed} ERR {e}", flush=True)

# 人类: 随机 11 首曲库 (副歌区 78-108s, 同 vocal_expression 惯例)
lib = sorted((ROOT/"eval/library_wavs").glob("*.wav"))[:11]
hum_vecs = []
for w in lib:
    try:
        hum_vecs.append(mfcc_mean(w, 78.0))
    except Exception:
        try: hum_vecs.append(mfcc_mean(w, 0.0))
        except Exception as e: print(f"{w.stem} ERR", flush=True)

ai_D = squareform(pdist(np.array(ai_vecs)))[np.triu_indices(len(ai_vecs), 1)]
hum_D = squareform(pdist(np.array(hum_vecs)))[np.triu_indices(len(hum_vecs), 1)]
print(f"AI seed 间音色距离 (n={len(ai_D)} 对): 中位={np.median(ai_D):.1f}", flush=True)
print(f"人类歌间音色距离 (n={len(hum_D)} 对): 中位={np.median(hum_D):.1f}", flush=True)
print(f"比值: 人类/AI = {np.median(hum_D)/np.median(ai_D):.2f}×", flush=True)
from scipy.stats import mannwhitneyu
u, p = mannwhitneyu(ai_D, hum_D, alternative='less')
print(f"U检验 (AI距离<人类距离): p={p:.2e} {'→ 音色同质化实锤' if p<0.05 else '不支持'}", flush=True)
