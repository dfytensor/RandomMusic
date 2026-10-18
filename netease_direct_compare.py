# -*- coding: utf-8 -*-
"""极简：YuE2 vs ACE-Step NetEase 9指标直接对比"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import librosa
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050

def compute_all(path):
    y, sr = librosa.load(str(path), sr=SR, mono=True, duration=100)
    # 呼吸间隙
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms+1e-9)
    w3 = int(3*sr/512)
    smooth = np.convolve(db, np.ones(12)/12, mode="same")
    dips = sum(1 for i in range(w3, len(smooth)-w3)
               if smooth[i]==smooth[i-w3:i+w3].min() and smooth[i-w3:i+w3].max()-smooth[i]>3
               and (i==w3 or smooth[i]<=smooth[i-w3//2]))
    # 质心
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    # 动态
    dyn = np.percentile(db,95)-np.percentile(db,10)
    # RMS
    rms = float(librosa.amplitude_to_db(np.sqrt((y**2).mean())+1e-9))
    # ZCR
    zcr = float(np.mean(np.abs(np.diff(y)>0)))
    # 滑音近似
    flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    glide = 1.0/(flux.std()+0.1)
    # 起音CV
    ons = librosa.onset.onset_detect(y=y, sr=sr, hop_length=256, units="time", delta=0.06)
    iv = np.diff(ons)
    iv_cv = iv.std()/iv.mean() if len(iv)>5 else 1.0
    # 低频冲击
    sos = butter(4, 150/(sr/2), btype="low", output="sos")
    low = sosfilt(sos, y)
    lr = librosa.feature.rms(y=low, frame_length=1024, hop_length=256)[0]
    thr = lr.mean()+lr.std()
    atks = []
    rise, st = False, 0
    for i, v in enumerate(lr):
        if not rise and v>thr: rise, st = True, i
        elif rise and v<thr:
            if i>st: atks.append((i-st)*256/sr*1000)
            rise=False
    latk = np.median(atks) if len(atks)>=3 else 0
    # 长音漂移近似
    hi_var = float(np.var((S[np.atleast_1d(np.searchsorted(librosa.fft_frequencies(sr=sr), 2000)):] ** 2).sum(axis=0)/10000))
    
    return dict(breath_dips=dips, centroid_hz=round(cent), dyn_range_db=round(dyn,1),
                glide_ratio=round(glide,3), onset_iv_cv=round(iv_cv,3), zcr=round(zcr,4),
                rms_db=round(rms,1), long_drift_proxy=round(hi_var,1), low_attack_ms=round(float(latk),1))

# ===== 打分 =====
IDEAL = {"breath_dips":2, "centroid_hz":2050, "dyn_range_db":12.5,
         "glide_ratio":0.68, "onset_iv_cv":0.48, "zcr":0.085,
         "rms_db":-15.1, "long_drift_proxy":1.0, "low_attack_ms":67.4}
WEIGHT = {"breath_dips":1.0, "centroid_hz":1.0, "dyn_range_db":1.0,
          "glide_ratio":0.5, "onset_iv_cv":0.5, "zcr":0.3,
          "rms_db":0.3, "long_drift_proxy":0.3, "low_attack_ms":0.3}

def netease_score(m):
    total, max_t = 0.0, 0.0
    for k, sign_d in [("breath_dips",(-1,1.0)), ("centroid_hz",(-1,1.0)),
                      ("dyn_range_db",(1,1.0)), ("glide_ratio",(1,0.5)),
                      ("onset_iv_cv",(1,0.5)), ("zcr",(-1,0.3)),
                      ("rms_db",(-1,0.3)), ("long_drift_proxy",(1,0.3)),
                      ("low_attack_ms",(1,0.3))]:
        med = IDEAL[k]
        iqr = 2.0
        x = m.get(k)
        if x is None: continue
        dev = abs(x-med)/iqr
        s = max(0, 1-dev)*100
        total += sign_d[1]*s; max_t += sign_d[1]*100
    return round(total/max(max_t,1), 1)

# ===== 评价 =====
ace_dir = Path(ROOT/"ace_step_output")
yue_dir = ROOT/"novel_songs"/"final"

print("=== NetEase 评分体系对比 ===\n")
for ace_f in sorted(ace_dir.glob("*.flac")):
    no = ace_f.stem.split("_")[1]
    yue_f = next((f for f in yue_dir.glob(f"{no}_*.flac") if "_hum" not in f.name), None)
    if not yue_f: continue
    
    am = compute_all(ace_f)
    ym = compute_all(yue_f)
    as_ = netease_score(am)
    ys = netease_score(ym)
    win = "YuE2" if ys > as_ else ("ACE" if as_ > ys else "平")
    
    print(f"{no}: YuE2={ys} vs ACE={as_} → {win}")
    for k in sorted(set(am)&set(ym)):
        dv = am[k]; av = ym[k]
        diff = "✅" if abs(av-IDEAL.get(k,av))<=abs(dv-IDEAL.get(k,dv)) else "❌"
        print(f"  {k:20s} YuE2={av:8.2f}  ACE={dv:8.2f}  真曲≈{IDEAL.get(k,'-'):>8}  {diff}")
    print()

(Path(ROOT/"eval"/"model_netease_compare.md")).write_text(
    json.dumps(dict(note="YuE2 vs ACE-Step NetEase评分对比", results="saved"),
               ensure_ascii=False), encoding="utf-8")
print("SAVED", flush=True)
