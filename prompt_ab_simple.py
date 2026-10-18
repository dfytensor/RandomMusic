# -*- coding: utf-8 -*-
"""极简 Prompt A/B 评测"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
work = ROOT / "novel_songs" / "v2" / "prompt_ab"
tags = ["A_基线", "B_临界", "C_非线性", "D_呼吸", "E_爆发"]

results = {}
for tag in tags:
    f = work / f"{tag}.flac"
    if not f.exists(): continue
    y, sr = librosa.load(str(f), sr=SR, mono=True, duration=100)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms+1e-9)
    cent = float(librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean())
    zcr = float(np.mean(np.abs(np.diff(y)>0)))
    rms_db = float(librosa.amplitude_to_db(np.sqrt((y**2).mean())+1e-9))
    dyn = float(np.percentile(db,95)-np.percentile(db,10))
    results[tag] = dict(centroid=round(cent), zcr=round(zcr,4), rms_db=round(rms_db,1), dyn=round(dyn,1))

# 打印
keys = ["centroid", "zcr", "rms_db", "dyn"]
print("Prompt A/B 对比:", flush=True)
print(f"{'指标':12s}", end="")
for t in tags: print(f"  {t:8s}", end="")
print()

for k in keys:
    print(f"{k:12s}", end="")
    for t in tags:
        v = results.get(t,{}).get(k,"?")
        print(f"  {v:8}", end="")
    print()

# 保存
(ROOT/"eval"/"prompt_ab_results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
print("SAVED", flush=True)
