# -*- coding: utf-8 -*-
"""母带修复链：质心降暗 + 动态压缩 + 响度归一 -> 重测验证"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050

# ===== 目标值（曲库基准）=====
TARGET_CENTROID = 2100  # Hz（真曲中位约 2100）
TARGET_DYN = 13.0       # dB（真曲中位约 13）
TARGET_RMS = -15.0      # dB

def high_shelf_cut(y, sr, freq=3500, gain_db=3.0):
    """高频衰减（模拟 high-shelf）"""
    nyq = sr / 2
    sos = butter(2, freq / nyq, btype="high", output="sos")
    high = sosfilt(sos, y)
    return y - high * (1 - 10 ** (-gain_db / 20))

def compress_dyn(y, sr, target_range_db, current_range_db):
    """软压缩动态范围"""
    if current_range_db <= target_range_db:
        return y
    ratio = target_range_db / current_range_db
    # 简化：对 RMS 包络做增益调整
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    # 逐帧增益：太响的压下来，太轻的提上去
    gain_db = np.zeros_like(db)
    loud_mask = db > np.percentile(db, 70)
    quiet_mask = db < np.percentile(db, 30)
    gain_db[loud_mask] = -(db[loud_mask] - np.percentile(db, 70)) * (1 - ratio) * 0.5
    gain_db[quiet_mask] = (np.percentile(db, 30) - db[quiet_mask]) * (1 - ratio) * 0.3
    # 平滑增益
    gain_db = np.convolve(gain_db, np.ones(20)/20, mode="same")
    gain_lin = 10 ** (gain_db / 20)
    # 应用到时域
    y_out = np.copy(y)
    frame_t = np.arange(len(db)) * hop / sr
    for i in range(len(db)):
        a = int(i * hop)
        b = min(a + hop, len(y_out))
        if a < len(y_out):
            y_out[a:b] *= 10 ** (gain_db[i] / 20)
    return y_out


def process_song(in_path, out_path):
    y, sr = librosa.load(str(in_path), sr=48000, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    
    orig_cent = float(librosa.feature.spectral_centroid(y=y.mean(axis=0), sr=sr)[0].mean())
    orig_rms = librosa.feature.rms(y=y.mean(axis=0), frame_length=2048, hop_length=512)[0]
    orig_db = librosa.amplitude_to_db(orig_rms + 1e-9)
    orig_dyn = float(np.percentile(orig_db, 95) - np.percentile(orig_db, 10))
    
    # 1) 质心降暗
    cent_target = 2100
    shelf_db = 0.0
    if orig_cent > cent_target * 1.15:
        shelf_db = -min(3.0, (orig_cent - cent_target) / cent_target * 10)
    if shelf_db < 0:
        sos = butter(2, 4000 / (sr/2), btype="high", output="sos")
        for ch in range(2):
            high = sosfilt(sos, y[ch])
            y[ch] += high * (10 ** (shelf_db / 20) - 1)
    
    # 2) 动态压缩
    rms = librosa.feature.rms(y=y.mean(axis=0), frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    cur_dyn = float(np.percentile(db, 95) - np.percentile(db, 10))
    if cur_dyn > 16:
        # 软压缩
        hop = 512
        gain = np.ones(len(db))
        window = 20
        smooth_db = np.convolve(db, np.ones(window)/window, mode="same")
        for i in range(len(db)):
            if smooth_db[i] > -10:
                gain[i] = 10 ** ((-10 - smooth_db[i]) / 20 * 0.3)
        gain = np.convolve(gain, np.ones(100)/100, mode="same")
        for ch in range(2):
            y[ch] *= gain
    
    # 3) 响度归一
    rms_final = np.sqrt((y.mean(axis=0) ** 2).mean() + 1e-9)
    cur_db = 20 * np.log10(rms_final + 1e-9)
    if cur_db < -25:
        y *= 10 ** ((-18 - cur_db) / 20)
    elif cur_db > -10:
        y *= 10 ** ((-14 - cur_db) / 20)
    
    # 防削波
    peak = np.abs(y).max()
    if peak > 0.95:
        y *= 0.95 / peak
    
    sf.write(str(out_path), y.T, sr, format="FLAC")
    
    # 重测
    y_check, _ = librosa.load(str(out_path), sr=SR, mono=True, duration=100)
    cent_new = float(librosa.feature.spectral_centroid(y=y_check, sr=SR)[0].mean())
    rms_new = librosa.feature.rms(y=y_check, frame_length=2048, hop_length=512)[0]
    db_new = librosa.amplitude_to_db(rms_new + 1e-9)
    dyn_new = float(np.percentile(db_new, 95) - np.percentile(db_new, 10))
    rms_new_db = float(librosa.amplitude_to_db(np.sqrt((y_check**2).mean()) + 1e-9))
    
    return dict(
        centroid_before=round(orig_cent), centroid_after=round(cent_new),
        dyn_before=round(orig_dyn, 1), dyn_after=round(dyn_new, 1),
        rms_after=round(rms_new_db, 1)
    )


def main():
    final = ROOT / "novel_songs" / "final" / "mastered"
    final.mkdir(parents=True, exist_ok=True)
    src_dir = ROOT / "novel_songs" / "final"
    
    flacs = sorted(src_dir.glob("*.flac"))
    print(f"处理 {len(flacs)} 首...", flush=True)
    
    results = []
    for i, f in enumerate(flacs):
        out = final / f.name
        try:
            r = process_song(str(f), str(out))
            results.append(dict(file=f.name, **r))
            print(f"  [{i+1}] {f.stem[:20]}: 质心 {r['centroid_before']}→{r['centroid_after']} | "
                  f"动态 {r['dyn_before']}→{r['dyn_after']}", flush=True)
        except Exception as e:
            print(f"  FAIL {f.stem[:20]}: {str(e)[:60]}", flush=True)
    
    # 汇总
    cents_before = [r["centroid_before"] for r in results]
    cents_after = [r["centroid_after"] for r in results]
    dyns = [r["dyn_after"] for r in results]
    import numpy as np
    print(f"\n质心中位: {np.median(cents_before):.0f} → {np.median(cents_after):.0f} (目标 ~2100)")
    print(f"动态中位: {np.median(dyns):.1f} (目标 ~13)")
    print(f"\n输出: {final}/")
    
    (ROOT/"eval"/"master_fix_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
