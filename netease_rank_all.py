# -*- coding: utf-8 -*-
"""NetEase 评分 × 涌现优化：
用 42 首 NetEase 验证指标（呼吸/质心/动态）作为涌现信号，
对 12 首 AI 曲 + 9 首 ACE 曲做全面评分排名，
找出哪些歌在"温暖音色+大动态+少呼吸"三轴上最接近真曲爆款。
"""
import json, math, sys, warnings
from pathlib import Path
import numpy as np
import librosa
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050

# ===== 真曲基准（NetEase 42 首标定） =====
IDEAL = {
    "breath_dips":   {"v": 2.0,  "tol": 1.5, "w": 1.0, "dir": "低好", "note": "唱满一点好"},
    "centroid_hz":   {"v": 2050, "tol": 500, "w": 1.0, "dir": "暗好", "note": "温暖音色好"},
    "dyn_range_db":  {"v": 12.5, "tol": 3.0, "w": 1.0, "dir": "大好", "note": "有起伏好"},
    "glide_ratio":   {"v": 0.68, "tol": 0.15, "w": 0.5, "dir": "高好", "note": "滑音多好"},
    "onset_iv_cv":   {"v": 0.48, "tol": 0.10, "w": 0.5, "dir": "适中", "note": "律动感"},
    "zcr":           {"v": 0.085,"tol": 0.01, "w": 0.3, "dir": "低好", "note": "音色暖"},
    "rms_db":        {"v": -15.1, "tol": 2.0, "w": 0.3, "dir": "适中", "note": "不吵"},
    "long_drift":    {"v": 68.5, "tol": 35.0, "w": 0.3, "dir": "高好", "note": "有人味"},
    "low_attack_ms": {"v": 67.4, "tol": 20.0, "w": 0.3, "dir": "高好", "note": "圆润"},
}


def compute_netease_metrics(flac_path):
    y, sr = librosa.load(str(flac_path), sr=SR, mono=True, duration=100)

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    cent = float(librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean())
    dyn = float(np.percentile(db, 95) - np.percentile(db, 10))
    rms_db = float(librosa.amplitude_to_db(np.sqrt((y**2).mean()) + 1e-9))
    zcr = float(np.mean(np.abs(np.diff(y) > 0)))

    w3 = int(3 * sr / 512)
    smooth = np.convolve(db, np.ones(12) / 12, mode="same")
    dips = 0
    for i in range(w3, len(smooth) - w3):
        window = smooth[i - w3:i + w3]
        if smooth[i] == window.min() and (window.max() - window.min()) > 3:
            if i == w3 or smooth[i] <= smooth[i - w3 // 2]:
                dips += 1

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    glide = round(1.0 / (onset_env.std() + 0.1), 3)

    ons = librosa.onset.onset_detect(y=y, sr=sr, hop_length=256, units="time", delta=0.06)
    iv = np.diff(ons)
    iv_cv = round(float(iv.std() / iv.mean()), 3) if len(iv) > 5 else 1.0

    sos = butter(4, 150 / (sr / 2), btype="low", output="sos")
    low = sosfilt(sos, y)
    lr = librosa.feature.rms(y=low, frame_length=1024, hop_length=256)[0]
    thr = lr.mean() + lr.std()
    atks, rise, st = [], False, 0
    for i, v in enumerate(lr):
        if not rise and v > thr: rise, st = True, i
        elif rise and v < thr:
            if i > st: atks.append((i - st) * 256 / sr * 1000)
            rise = False
    latk = round(float(np.median(atks)), 1) if len(atks) >= 3 else None

    hi_energy = S = None
    S_spec = np.abs(librosa.stft(y))**2
    fr = librosa.fft_frequencies(sr=sr)
    hi_mask = fr >= 2000
    hi_e = S_spec[hi_mask].sum(axis=0)
    hi_var = float(np.var(hi_e / (hi_e.sum() + 1e-9))) * 10000

    return dict(breath_dips=dips, centroid_hz=round(cent), dyn_range_db=round(dyn, 1),
                glide_ratio=glide, onset_iv_cv=iv_cv, zcr=round(zcr, 4), rms_db=rms_db,
                long_drift=round(hi_var, 1), low_attack_ms=latk)


def netease_score(metrics):
    """0-100 分，越高越接近真曲爆款分布"""
    total, max_total, details = 0.0, 0.0, []
    for key, cfg in IDEAL.items():
        x = metrics.get(key)
        if x is None:
            continue
        dev = abs(x - cfg["v"]) / cfg["tol"]
        s = max(0.0, 1 - dev * 0.5) * 100  # 容差=2×tol 也可得分（减半）
        total += cfg["w"] * s
        max_total += cfg["w"] * 100
        details.append(dict(metric=key, value=x, ideal=cfg["v"], score=round(s, 1)))
    return round(total / max(max_total, 1), 1), details


# ===== 评价全部 =====
song_dirs = [
    ("YuE2", ROOT / "novel_songs" / "final"),
    ("ACE", ROOT / "ace_step_output"),
    ("ACE-gacha", ROOT / "ace_step_output" / "gacha"),
    ("YuE2-prosody", ROOT / "novel_songs" / "v2" / "yuegongri_prosody"),
    ("YuE2-v2", ROOT / "novel_songs" / "v2" / "novel_01_欠天地一笔"),
]

all_scores = []
for source, sdir in song_dirs:
    if not sdir.exists(): continue
    for f in sorted(sdir.glob("*.flac")):
        try:
            m = compute_netease_metrics(f)
            s, det = netease_score(m)
            all_scores.append(dict(file=f.name, source=source, score=s,
                                   metrics=m, details=det))
        except Exception as e:
            print(f"SKIP {f.name[:24]}: {str(e)[:50]}", flush=True)

# ===== 曲库真曲基线 =====
lib_wav = ROOT / "eval" / "library_wavs"
corpus_scores = []
for w in sorted(lib_wav.glob("*.wav"))[:50]:
    try:
        m = compute_netease_metrics(w)
        s, _ = netease_score(m)
        corpus_scores.append(s)
    except: pass

import numpy as _np
print(f"\n真曲基线 (n={len(corpus_scores)}): 中位 {_np.median(corpus_scores):.1f} · "
      f"P25 {_np.percentile(corpus_scores,25):.1f} · P75 {_np.percentile(corpus_scores,75):.1f}",
      flush=True)

# ===== 生成曲排名 =====
print(f"\n{'='*60}")
print(f"{'排名':>4s} {'来源':8s} {'曲目':24s} {'NetEase分':>8s}")
print(f"{'-'*60}")
for i, r in enumerate(sorted(all_scores, key=lambda x: -x["score"]), 1):
    src = "YuE2" if "novel_songs" in str(r.get("file","")) else "ACE"
    print(f"  {i:2d}. {r['file'][:28]:28s} {r['score']:5.1f}  [{src}]", flush=True)

(ROOT / "eval" / "netease_final_ranking.json").write_text(
    json.dumps(all_scores, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print("SAVED", flush=True)
