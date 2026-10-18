# -*- coding: utf-8 -*-
"""《蝴蝶泉边》1:18-1:32 逐帧解剖：F0形态/颤音/动态/频谱/节奏 -> 找情绪共鸣的声学构成"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050

# 目标段
T1, T2 = 78.0, 92.0  # 稍微放宽以覆盖 1:18-1:32

src = ROOT / "eval" / "library_wavs" / "lib_黄雅莉_-_蝴蝶泉边.wav"
y, _ = librosa.load(str(src), sr=SR, mono=True, offset=T1, duration=T2 - T1)

# ===== 1. F0 逐帧 =====
f0, vflag, vprob = librosa.pyin(y, fmin=65, fmax=700, sr=SR, frame_length=1024, hop_length=256)
ft = np.arange(len(f0)) * 256 / SR
voiced = vflag & ~np.isnan(f0)

# F0 轨迹采样点（每 0.5s）
print("=== F0 轨迹（每 0.5s）===")
for t in np.arange(0, T2 - T1, 0.5):
    idx = int(t * SR / 256)
    if idx < len(f0) and not np.isnan(f0[idx]):
        note = 69 + 12 * np.log2(f0[idx] / 440)
        print(f"  {t:5.1f}s  {f0[idx]:6.1f}Hz  MIDI {note:.1f}")

# ===== 2. 颤音分析（每个浊音段） =====
cents = 1200 * np.log2(f0 / 55.0)
runs, start = [], None
win = int(0.1 * SR / 256)
for i, v in enumerate(voiced):
    if v and start is None:
        start = i
    elif not v and start is not None:
        if ft[i - 1] - ft[start] >= 0.3:
            runs.append((start, i))
        start = None
if start is not None:
    runs.append((start, len(voiced)))

print(f"\n=== 浊音段 {len(runs)} 个 ===")
for a, b in runs:
    dur = ft[b - 1] - ft[a]
    if dur < 0.2:
        continue
    seg_c = cents[a:b] - np.nanmedian(cents[a:b])
    slow = np.convolve(seg_c, np.ones(win) / win, mode="same")
    fast = seg_c - slow
    spec = np.abs(np.fft.rfft(np.nan_to_num(fast)))
    freqs = np.fft.rfftfreq(len(fast), d=256 / SR)
    band = (freqs >= 3) & (freqs <= 12)
    if band.sum() == 0:
        continue
    pk_f = freqs[band][np.argmax(spec[band])]
    pk_e = spec[band].max()
    band_e = spec[band].sum() + 1e-9
    reg = pk_e / band_e
    ext = np.nanstd(fast)
    drift = np.nanstd(slow)
    print(f"  {ft[a]:5.1f}-{ft[b-1]:5.1f}s ({dur:.2f}s) 颤音={pk_f:.1f}Hz 幅={ext:.0f}c 规整={reg:.2f} 漂移={drift:.0f}c")

# ===== 3. 动态包络（0.25s 窗） =====
rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=256)[0]
db = librosa.amplitude_to_db(rms + 1e-9)
tf = np.arange(len(db)) * 256 / SR

print("\n=== 动态包络（每 0.5s dB）===")
for t in np.arange(0, T2 - T1, 0.5):
    idx = int(t * SR / 256)
    if idx < len(db):
        bar = "█" * max(0, int((db[idx] + 60) / 2))
        print(f"  {t:5.1f}s {db[idx]:6.1f}dB {bar}")

# ===== 4. 频谱通量（变化率） =====
flux = librosa.onset.onset_strength(y=y, sr=SR, hop_length=256)
print(f"\n频谱通量: 均值={flux.mean():.2f} std={flux.std():.2f} 峰值数={len(librosa.onset.onset_detect(y=y, sr=SR, hop_length=256, delta=0.05))}")

# ===== 5. 峰值因子 =====
crest = 20 * np.log10(np.abs(y).max() / (np.sqrt((y ** 2).mean()) + 1e-9))
print(f"峰值因子: {crest:.1f}dB")

# ===== 6. 立体声宽度（如果有） =====
data, sr0 = sf.read(str(src), always_2d=True)
mid = data.mean(axis=1)
side = (data[:, 0] - data[:, 1]) / 2
width = float(np.sqrt((side ** 2).mean()) / (np.sqrt((mid ** 2).mean()) + 1e-9))
print(f"立体声宽度: {width:.3f}")
