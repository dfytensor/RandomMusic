# -*- coding: utf-8 -*-
"""ACE-Step 抽卡择优：NetEase 评分 → 选最佳"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050

IDEAL = {"breath_dips": 2, "centroid_hz": 2050, "dyn_range_db": 12.5}


def quick_netease_score(y, sr=SR):
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    cent = float(librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean())
    w3 = int(3 * sr / 512)
    smooth = np.convolve(db, np.ones(12) / 12, mode="same")
    dips = sum(1 for i in range(w3, len(smooth) - w3)
               if smooth[i] == smooth[i-w3:i+w3].min() and smooth[i-w3:i+w3].max()-smooth[i]>3)
    s = 0.0
    s += max(0, 1 - abs(dips - 2) / 2) * 33
    s += max(0, 1 - abs(cent - 2050) / 1000) * 33
    rms_db = float(librosa.amplitude_to_db(np.sqrt((y**2).mean()) + 1e-9))
    s += max(0, 1 - abs(rms_db + 15) / 10) * 34
    return round(s, 1), dips, round(cent), round(rms_db, 1)


ace_dir = ROOT / "ace_step_output" / "gacha"
results = []
for f in sorted(ace_dir.glob("*.flac")):
    y, _ = librosa.load(str(f), sr=SR, mono=True, duration=100)
    s, dips, cent, rms = quick_netease_score(y, SR)
    results.append((f.name, s, dips, cent, rms))
    print(f"{f.name[:30]:32s} score={s} dips={dips} cent={cent} rms={rms}", flush=True)

results.sort(key=lambda r: -r[1])
print(f"\n最佳: {results[0][0][:30]} 分={results[0][1]}", flush=True)
