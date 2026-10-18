# -*- coding: utf-8 -*-
"""用 NetEase 评分体系评价 ACE-Step vs YuE2 生成的歌"""
import json, math, sys, warnings
from pathlib import Path
import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050

# ===== NetEase 验证过的 9 指标（42 首真曲标定） =====
VALIDATED = [
    ("breath_dips",   -1, 2.0,   1.0,  1.0),
    ("centroid_hz",   -1, 2050,  400,  1.0),
    ("dyn_range_db",  +1, 12.5,  3.0,  1.0),
    ("glide_ratio",   +1, 0.68,  0.15, 0.5),
    ("onset_iv_cv",   +1, 0.48,  0.10, 0.5),
    ("zcr",           -1, 0.085, 0.01, 0.3),
    ("rms_db",        -1, -15.1, 2.0,  0.3),
    ("long_drift",    +1, 68.5,  35.0, 0.3),
    ("low_attack_ms", +1, 67.4,  20.0, 0.3),
]


def compute_netease_metrics(flac_path):
    y, sr = librosa.load(str(flac_path), sr=SR, mono=True, duration=100)
    
    # 1) 呼吸间隙
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    w3 = int(3 * sr / 512)
    smooth = np.convolve(db, np.ones(12)/12, mode="same")
    dips = 0
    for i in range(w3, len(smooth)-w3):
        window = smooth[i-w3:i+w3]
        if smooth[i] == window.min() and (window.max()-window.min()) > 3:
            if i == w3 or smooth[i] <= smooth[i-w3//2]:
                dips += 1
    
    # 2) 频谱质心
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid = float(cent.mean())
    
    # 3) 动态范围
    dyn_range = float(np.percentile(db, 95) - np.percentile(db, 10))
    
    # 4) 滑音比率（近似）
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    glide = round(1.0 / (onset_env.std() + 0.1), 3)
    
    # 5) 起音间隔 CV
    ons = librosa.onset.onset_detect(y=y, sr=sr, hop_length=256, units="time", delta=0.06)
    iv = np.diff(ons)
    iv_cv = round(float(iv.std()/iv.mean()), 3) if len(iv) > 5 else 1.0
    
    # 6) 过零率
    zcr = round(float(np.mean(np.abs(np.diff(y) > 0))), 4)
    
    # 7) 响度
    rms_val = round(float(librosa.amplitude_to_db(np.sqrt((y**2).mean())+1e-9)), 1)
    
    # 8) 长音漂移（近似）
    hi_mask = freqs >= 2000
    hi_energy = S[hi_mask].sum(axis=0)
    hi_var = float(np.var(hi_energy / (hi_energy.sum() + 1e-9)))
    long_drift = round(hi_var * 10000, 1)
    
    # 9) 低频冲击速度
    sos = butter(4, 150/(sr/2), btype="low", output="sos")
    low = sosfilt(sos, y)
    lr = librosa.feature.rms(y=low, frame_length=1024, hop_length=256)[0]
    thr = lr.mean() + lr.std()
    atks, rise, st = [], False, 0
    for i, v in enumerate(lr):
        if not rise and v > thr: rise, st = True, i
        elif rise and v < thr:
            if i > st: atks.append((i-st)*256/sr*1000)
            rise = False
    low_atk = round(float(np.median(atks)), 1) if len(atks) >= 3 else None
    
    return dict(breath_dips=dips, centroid_hz=round(centroid), dyn_range_db=round(dyn_range,1),
                glide_ratio=glide, onset_iv_cv=iv_cv, zcr=zcr, rms_db=rms_val,
                long_drift=long_drift, low_attack_ms=low_atk)


def score_netease(metrics):
    total, max_total, details = 0.0, 0.0, []
    for key, direction, med, iqr, weight, note in VALIDATED:
        x = metrics.get(key)
        if x is None:
            continue
        dev = abs(x - med) / (iqr + 1e-9)
        s = max(0.0, 1 - dev) * 100
        total += weight * s
        max_total += weight * 100
        details.append(dict(metric=key, value=x, ideal=med, deviation=round(dev,2), score=round(s,1)))
    final = round(total / max(max_total, 1), 1)
    return final, details


# ===== 评价曲库真曲（基线） =====
lib_wav = ROOT / "eval" / "library_wavs"
lib_tr = ROOT / "eval" / "library_transcripts"
corpus_scores = []
corpus_names = []
for w in sorted(lib_wav.glob("*.wav"))[:50]:
    tdir = lib_tr / w.stem
    rows = parse_lab(tdir, "structure.lab") if False else None
    ch = None
    for line in (tdir/"structure.lab").read_text(encoding="utf-8").splitlines() if (tdir/"structure.lab").exists() else []:
        q = line.split("\t")
        if len(q) >= 3 and q[2].strip().lower() == "chorus":
            ch = (float(q[0]), min(float(q[1]), ch[0]+30) if ch else float(q[0])+30)
            break
    if not ch: continue
    try:
        y, _ = librosa.load(str(w), sr=SR, mono=True, offset=ch[0], duration=min(ch[1]-ch[0], 30))
        m = compute_netease_metrics(y, SR)
        s, _ = score_netease(m)
        corpus_scores.append(s)
        corpus_names.append(w.stem)
    except: pass

import numpy as np
print(f"真曲基线 (n={len(corpus_scores)}): 中位 {np.median(corpus_scores):.1f} · "
      f"P25 {np.percentile(corpus_scores,25):.1f} · P75 {np.percentile(corpus_scores,75):.1f} · "
      f"最大 {max(corpus_scores):.1f}", flush=True)

# ===== 评价生成曲 =====
ace_dir = ROOT / "ace_step_output"
yue_dir = ROOT / "novel_songs" / "final"

pairs = []
for ace_f in sorted(ace_dir.glob("*.flac")):
    no = ace_f.stem.split("_")[1]
    for yue_f in sorted(yue_dir.glob("*.flac")):
        if yue_f.name.startswith(no + "_") and "_hum" not in yue_f.name:
            pairs.append((no, yue_f, ace_f))
            break

print(f"\n=== 逐曲对比 ===", flush=True)
for no, yue_f, ace_f in pairs:
    y_m = compute_netease_metrics(yue_f, SR)
    a_m = compute_netease_metrics(ace_f, SR)
    y_s, _ = score_netease(y_m)
    a_s, _ = score_netease(a_m)
    winner = "YuE2" if y_s > a_s else ("ACE" if a_s > y_s else "平手")
    print(f"  {no} {title if False else no}: YuE2={y_s:.1f} vs ACE={a_s:.1f} → {winner}", flush=True)
    # 打印细节
    for k, direction, med, iqr, w, note in VALIDATED:
        yv = y_m.get(k)
        av = a_m.get(k)
        if yv is not None and av is not None:
            better = "YuE2" if abs(yv-med)<abs(av-med) else "ACE"
            print(f"    {k}: YuE2={yv} ACE={av} (真曲{med}) → {better}")

# ===== ACE-Step 歌 =====
print("\n=== ACE-Step 生成曲评分 ===", flush=True)
for ace_f in sorted(ace_dir.glob("*.flac")):
    y, _ = librosa.load(str(ace_f), sr=SR, mono=True, duration=100)
    m = compute_netease_metrics(y, SR)
    s, _ = score_netease(m)
    print(f"  {ace_f.stem}: {s}", flush=True)
