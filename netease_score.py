# -*- coding: utf-8 -*-
"""NetEase 验证指标打分器：用 42 首真曲真实热度标定的 9 个指标评价歌曲
打分方式：每个指标对照真曲中位/IQR，越接近真曲分布分越高
"""
import json, math, sys, warnings
from pathlib import Path
import numpy as np
import librosa
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050

# ===== NetEase 验证的 9 指标定义 =====
# (key, 方向+1=高好/-1=低好, 真曲中位, 真曲IQR, 权重, 来源)
VALIDATED = [
    ("breath_dips",    -1, 2.0,   1.0,  1.0, "p<0.05 呼吸越少越火"),
    ("centroid_hz",    -1, 2050,  400,  1.0, "p<0.05 质心越暗越火"),
    ("dyn_range_db",   +1, 12.5,  3.0,  1.0, "p<0.05 动态大越火"),
    ("glide_ratio",    +1, 0.68,  0.15, 0.5, "p=0.14 滑音多好"),
    ("onset_iv_cv",    +1, 0.48,  0.10, 0.5, "p=0.14 律动感"),
    ("zcr",            -1, 0.085, 0.01, 0.3, "p=0.23 音色暖"),
    ("rms_db",         -1, -15.1, 2.0,  0.3, "p=0.25 不吵"),
    ("long_drift",     +1, 68.5,  35.0, 0.3, "p=0.42 有人味"),
    ("low_attack_ms",  +1, 67.4,  20.0, 0.3, "p=0.36 圆润"),
]


def compute_netease_metrics(y, sr):
    """计算 9 个 NetEase 验证指标"""
    result = {}

    # 1) 呼吸间隙
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    w3 = int(3 * sr / 512)
    smooth = np.convolve(db, np.ones(12) / 12, mode="same")
    dips = 0
    for i in range(w3, len(smooth) - w3):
        window = smooth[i - w3:i + w3]
        if smooth[i] == window.min() and (window.max() - window.min()) > 3:
            if i == w3 or smooth[i] <= smooth[i - w3 // 2]:
                dips += 1
    result["breath_dips"] = dips

    # 2) 频谱质心
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    result["centroid_hz"] = round(float(cent.mean()))

    # 3) 动态范围
    result["dyn_range_db"] = round(float(np.percentile(db, 95) - np.percentile(db, 10)), 1)

    # 4) 滑音比率（近似：频谱通量平滑度）
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    result["glide_ratio"] = round(float(1.0 / (onset_env.std() + 0.1)), 3)

    # 5) 起音间隔 CV
    ons = librosa.onset.onset_detect(y=y, sr=sr, hop_length=256, units="time", delta=0.06)
    iv = np.diff(ons)
    result["onset_iv_cv"] = round(float(iv.std() / iv.mean()), 3) if len(iv) > 5 else 1.0

    # 6) 过零率
    result["zcr"] = round(float(np.mean(np.abs(np.diff(y) > 0))), 4)

    # 7) 响度
    result["rms_db"] = round(float(librosa.amplitude_to_db(np.sqrt((y**2).mean()) + 1e-9)), 1)

    # 8) 长音漂移（近似：高频段质心方差）
    S = np.abs(librosa.stft(y)) ** 2
    freqs = librosa.fft_frequencies(sr=sr)
    hi_mask = freqs >= 2000
    hi_energy = S[hi_mask].sum(axis=0)
    hi_var = float(np.var(hi_energy / (hi_energy.sum() + 1e-9)))
    result["long_drift_proxy"] = round(hi_var * 10000, 1)

    # 9) 低频冲击速度
    sos = butter(4, 150 / (sr/2), btype="low", output="sos")
    low = sosfilt(sos, y)
    lr = librosa.feature.rms(y=low, frame_length=1024, hop_length=256)[0]
    thr = lr.mean() + lr.std()
    atks, rise, st = [], False, 0
    for i, v in enumerate(lr):
        if not rise and v > thr: rise, st = True, i
        elif rise and v < thr:
            if i > st: atks.append((i - st) * 256 / sr * 1000)
            rise = False
    result["low_attack_ms"] = round(float(np.median(atks)), 1) if len(atks) >= 3 else None

    return result


def score_netease(metrics):
    """对照真曲中位打分（0-100，越高越接近真曲爆款分布）"""
    total, max_total = 0.0, 0.0
    details = []
    for key, direction, med, iqr, weight, note in VALIDATED:
        x = metrics.get(key)
        if x is None:
            continue
        # 距离真曲中位的偏离（用 IQR 归一化）
        dev = abs(x - med) / (iqr + 1e-9)
        # 分数 = max(0, 1 - dev) × 100
        s = max(0.0, 1 - dev) * 100
        total += weight * s
        max_total += weight * 100
        details.append(dict(key=key, value=x, ideal=med, score=round(s, 1), weight=weight))
    final = round(total / max(max_total, 1), 1)
    return final, details


def main():
    final = ROOT / "novel_songs" / "final"
    flacs = sorted(final.glob("*.flac"))
    print(f"评价 {len(flacs)} 首...", flush=True)

    results = []
    for f in flacs:
        y, _ = librosa.load(str(f), sr=SR, mono=True, duration=100)
        m = compute_netease_metrics(y, SR)
        score, details = score_netease(m)
        results.append(dict(file=f.name, score=score, metrics=m))
        print(f"  {f.stem[:24]:26s} 爆款相似度: {score:5.1f}", flush=True)

    results.sort(key=lambda r: -r["score"])
    print(f"\n=== 排名 ===", flush=True)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['file'][:24]:26s} {r['score']:5.1f}", flush=True)

    (ROOT / "eval" / "netease_score_rank.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("SAVED", flush=True)


if __name__ == "__main__":
    main()
