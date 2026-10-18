# -*- coding: utf-8 -*-
"""v3.5 articulation 核心: 全曲人声 stem 上测 5 新维
算法逐字取自 articulation2~5 已验证版 (全曲口径, 对齐人类30 vs AI冠军12 基准)
  vib_ratio/vib_depth: 0.8-2.5s run, f0 cents 去趋势, 自相关 4-9Hz (art4)
  phase_R: run onset 对伴奏拍相位集中度 (art2/3), 拍点来自 beats.json
  onset_rise_db: onset 前导中位 -> 后3帧 dB 跃升, cap 60 (art5)
  long_smooth: >=0.8s run 的 |diff dB| 中位 (art5)
"""
import json
import warnings
from pathlib import Path

import numpy as np
import librosa

warnings.filterwarnings("ignore")
SR = 22050
HOP = 512
FRAME = 1024
GAP_RUN = int(0.15 * SR / HOP)  # 150ms 桥接辅音闭塞
ART_DIMS = ["vib_ratio", "vib_depth", "phase_R", "onset_rise_db", "long_smooth"]


def extract_runs(voiced, gap_tol=GAP_RUN):
    runs, in_run, start, gap = [], False, 0, 0
    for i, val in enumerate(voiced):
        if val:
            if not in_run:
                start = i; in_run = True
            gap = 0
        else:
            if in_run:
                gap += 1
                if gap > gap_tol:
                    runs.append((start, i - gap)); in_run = False
    if in_run:
        runs.append((start, len(voiced) - 1))
    return runs


def art_dims(vocals_path, beats_path=None):
    """全曲人声 stem -> 5 新维; beats 缺失时 phase_R=None; runs<8 时全 None"""
    hop_s = HOP / SR
    y, _ = librosa.load(str(vocals_path), sr=SR, mono=True)
    f0, vf, _ = librosa.pyin(y, fmin=65, fmax=700, sr=SR, frame_length=FRAME, hop_length=HOP)
    voiced = vf & ~np.isnan(f0)
    runs = extract_runs(voiced)
    out = {d: None for d in ART_DIMS}
    if len(runs) < 8:
        return out
    run_dur = [(e - s + 1) * hop_s for s, e in runs]

    # --- vib ---
    vib_rates, vib_depths = [], []
    for (s, e), d in zip(runs, run_dur):
        if not (0.8 <= d <= 2.5):
            continue
        seg = f0[s:e + 1]
        valid = ~np.isnan(seg)
        if valid.sum() < 40:
            continue
        seg = seg[valid]
        cents = 1200 * np.log2(seg / np.median(seg))
        t = np.arange(len(cents)) * hop_s
        try:
            coef = np.polyfit(t, cents, 1)
            resid = cents - np.polyval(coef, t)
        except Exception:
            continue
        if resid.std() < 3:
            continue
        ac = np.correlate(resid, resid, mode="full")[len(resid) - 1:]
        ac = ac / (ac[0] + 1e-9)
        lags = np.arange(len(ac)) * hop_s
        mask = (lags >= 1 / 9) & (lags <= 1 / 4)
        if mask.sum() < 2:
            continue
        peak_lag = lags[mask][np.argmax(ac[mask])]
        rate = 1.0 / peak_lag if peak_lag > 0 else None
        if rate and 3.5 <= rate <= 9.5:
            vib_rates.append(float(rate))
            vib_depths.append(float(resid.std()))
    out["vib_ratio"] = float(len(vib_rates) / max(len(runs), 1))
    out["vib_depth"] = float(np.median(vib_depths)) if vib_depths else None

    # --- phase_R (拍点来自 accomp beat_track 的 beats.json) ---
    if beats_path and Path(beats_path).exists():
        try:
            btd = json.loads(Path(beats_path).read_text(encoding="utf-8"))
            beat_times = np.array(btd["beat_times"])
            if len(beat_times) >= 16:
                beat_dur = float(np.median(np.diff(beat_times)))
                onset_times = np.array([s * hop_s for s, e in runs])
                phase = []
                for ot in onset_times:
                    idx = np.searchsorted(beat_times, ot)
                    if 0 < idx < len(beat_times):
                        nb = beat_times[idx - 1] if abs(ot - beat_times[idx - 1]) < abs(beat_times[idx] - ot) else beat_times[idx]
                        phase.append(((ot - nb) / beat_dur) % 1.0)
                phase = np.array(phase)
                if len(phase) > 4:
                    out["phase_R"] = float(np.sqrt(np.mean(np.cos(2 * np.pi * phase)) ** 2 +
                                                   np.mean(np.sin(2 * np.pi * phase)) ** 2))
        except Exception:
            pass

    # --- onset_rise_db / long_smooth ---
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    smooth_absdiff = []
    for (s, e), d in zip(runs, run_dur):
        if d < 0.8:
            continue
        seg_db = 20 * np.log10(rms[s:e + 1] + 1e-9)
        if len(seg_db) >= 6:
            smooth_absdiff.append(float(np.median(np.abs(np.diff(seg_db)))))
    onset_rise = []
    for i in range(1, len(runs)):
        prev_end = runs[i - 1][1]
        s, e = runs[i]
        pre = rms[max(0, prev_end + 1):s + 1]
        if len(pre) < 3:
            continue
        rise = (20 * np.log10(rms[s + 1:e + 1][:3].mean() + 1e-9)
                - 20 * np.log10(np.median(pre) + 1e-9))
        if rise > 0:
            onset_rise.append(float(min(rise, 60)))
    out["onset_rise_db"] = float(np.median(onset_rise)) if onset_rise else None
    out["long_smooth"] = float(np.median(smooth_absdiff)) if smooth_absdiff else None
    return out
