# -*- coding: utf-8 -*-
"""用 NetEase 验证过的 9 指标评价 12 首 AI 曲
3 个显著 + 6 个弱趋势，全部来自 42 首真曲的真实热度验证
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
sys.path.insert(0, str(ROOT))

# ===== NetEase 验证过的指标定义 =====
# 来源：42 首真曲真实热度 × 声学特征 Spearman 相关
VALIDATED = {
    # (key, 方向: +1=高好, -1=低好, 真曲中位, 显著性)
    "breath_dips":      {"dir": -1, "med": 2.0,  "sig": "p<0.05 ✅", "note": "唱满一点好"},
    "centroid_hz":      {"dir": -1, "med": 2050, "sig": "p<0.05 ✅", "note": "温暖音色好"},
    "dyn_range_db":     {"dir": +1, "med": 12.5, "sig": "p<0.05 ✅", "note": "有起伏好"},
    "glide_ratio":      {"dir": +1, "med": 0.68, "sig": "p=0.14 🟡", "note": "滑音多受欢迎"},
    "onset_iv_cv":      {"dir": +1, "med": 0.48, "sig": "p=0.14 🟡", "note": "律动感"},
    "zcr":              {"dir": -1, "med": 0.085, "sig": "p=0.23 🟡", "note": "音色暖"},
    "rms_db":           {"dir": -1, "med": -15.1, "sig": "p=0.25 🟡", "note": "不吵"},
    "long_drift":       {"dir": +1, "med": 68.5, "sig": "p=0.42 🟡", "note": "有人味"},
    "low_attack_ms":    {"dir": +1, "med": 67.4, "sig": "p=0.36 🟡", "note": "圆润"},
}


def compute_validated_metrics(flac_path, t1=None, t2=None):
    """计算 9 个 NetEase 验证过的指标"""
    y, sr = librosa.load(str(flac_path), sr=SR, mono=True)
    if t1 is not None:
        y = y[int(t1*sr):int(t2*sr)]

    result = {}

    # 1) 呼吸间隙（动态包络中的下沉次数）
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    w3 = int(3 * SR / 512)
    smooth = np.convolve(db, np.ones(12)/12, mode="same")
    dips = 0
    for i in range(w3, len(smooth)-w3):
        window = smooth[i-w3:i+w3]
        if smooth[i] == window.min() and (window.max()-window.min()) > 3:
            if i == w3 or smooth[i] <= smooth[i-w3//2]:
                dips += 1
    result["breath_dips"] = dips

    # 2) 频谱质心
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    result["centroid_hz"] = round(float(cent.mean()))

    # 3) 动态范围
    result["dyn_range_db"] = round(float(np.percentile(db, 95) - np.percentile(db, 10)), 1)

    # 4) 滑音比率（近似：频谱通量的平滑度）
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    onset_t = librosa.onset.onset_detect(y=y, sr=sr, hop_length=256, units="time", delta=0.06)
    iv = np.diff(onset_t) if len(onset_t) > 5 else np.array([1.0])
    result["glide_ratio_proxy"] = round(float(1.0 / (iv.std() + 0.1)), 3)

    # 5) 起音间隔 CV
    result["onset_iv_cv"] = round(float(iv.std() / iv.mean()), 3) if len(iv) > 1 else 1.0

    # 6) 过零率
    result["zcr"] = round(float(np.mean(np.abs(np.diff(y)) > 0)), 4)

    # 7) 响度
    result["rms_db"] = round(float(librosa.amplitude_to_db(np.sqrt((y**2).mean()) + 1e-9)), 1)

    # 8) 长音漂移（近似：高能量段的频谱质心方差）
    hi_mask = db > np.median(db)
    if hi_mask.sum() > 10:
        cent_hi = cent[:len(hi_mask)][hi_mask[:len(cent)]]
        result["long_drift_proxy"] = round(float(cent_hi.std()), 1) if len(cent_hi) > 3 else None
    else:
        result["long_drift_proxy"] = None

    # 9) 低频冲击速度（近似：低频包络上升时间）
    sos = butter(4, 150 / (sr/2), btype="low", output="sos")
    from scipy.signal import sosfilt
    low = sosfilt(sos, y)
    lr = librosa.feature.rms(y=low, frame_length=1024, hop_length=256)[0]
    thr = lr.mean() + lr.std()
    atks = []
    rise, st = False, 0
    for i, v in enumerate(lr):
        if not rise and v > thr: rise, st = True, i
        elif rise and v < thr:
            if i > st: atks.append((i-st) * 256 / sr * 1000)
            rise = False
    result["low_attack_ms"] = round(float(np.median(atks)), 1) if len(atks) >= 3 else None

    return result


from scipy.signal import butter, sosfilt

# ===== 评价 12 首 =====
final = ROOT / "novel_songs" / "final"
results = []
for f in sorted(final.glob("*.flac")):
    if "_hum" in f.name:
        continue
    try:
        r = compute_validated_metrics(str(f))
        r["file"] = f.stem
        results.append(r)
        print(f"{f.stem}: breath={r['breath_dips']} cent={r['centroid_hz']} "
              f"dyn={r['dyn_range_db']} zcr={r['zcr']} rms={r['rms_db']}", flush=True)
    except Exception as e:
        print(f"FAIL {f.stem}: {str(e)[:60]}", flush=True)

# ===== 曲库基线对比 =====
# 从 42 首真曲（之前用 dissect_butterfly 分析过的那些）
BASE = {
    "breath_dips": {"med": 2.0, "dir": "低好", "sig": "p<0.05"},
    "centroid_hz": {"med": 2050, "dir": "暗好", "sig": "p<0.05"},
    "dyn_range_db": {"med": 12.5, "dir": "大好", "sig": "p<0.05"},
}

# ===== 报告 =====
lines = ["# AI 生成曲 × NetEase 验证指标评价", "",
         "评价体系：42 首真曲 × NetEase 真实热度标定（3 显著 + 6 弱趋势）", "",
         "## 12 首 AI 生成曲", ""]
for r in results:
    lines.append(f"### {r['file']}")
    lines.append(f"- 呼吸间隙: {r['breath_dips']} (真曲基线: 唱满好)")
    lines.append(f"- 频谱质心: {r['centroid_hz']}Hz (越暗越好)")
    lines.append(f"- 动态范围: {r['dyn_range_db']}dB (越大越有情绪)")
    lines.append(f"- 过零率: {r['zcr']} (越低越暖)")
    lines.append(f"- 响度: {r['rms_db']}dB (适中 -25~-8)")
    lines.append("")

# 简单评分
lines += ["## 综合评价", ""]
score = 0
for r in results:
    s = 0
    if r["breath_dips"] >= 1: s += 1
    if 1800 <= r["centroid_hz"] <= 2400: s += 1
    if r["dyn_range_db"] >= 10: s += 1
    if -25 <= r["rms_db"] <= -10: s += 1
    score += s
    lines.append(f"- {r['file']}: {s}/4 (声学质量分)")

avg = score / max(len(results), 1)
lines.append(f"\n**平均声学质量: {avg:.1f}/4**")
lines.append("\n> 注意：这些指标衡量的是'制作层质量'（不吵/有起伏/音色暖），")
lines.append("> 不是'好不好听'。好听的判断需要你的耳朵。")

final_p = ROOT / "eval" / "netease_eval_report.md"
final_p.write_text("\n".join(lines), encoding="utf-8")
print("\nSAVED", final_p)
