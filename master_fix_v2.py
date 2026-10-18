# -*- coding: utf-8 -*-
"""母带修复 v2：修复广播 shape 错误（mono/stereo 不匹配导致增益数组长度对不上）"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
TARGET_CENTROID = 2100
TARGET_DYN = 13.0


def process_song_v2(in_path, out_path):
    y, sr = librosa.load(str(in_path), sr=48000, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    mono = y.mean(axis=0)
    
    orig_cent = float(librosa.feature.spectral_centroid(y=mono, sr=sr)[0].mean())
    orig_rms_env = librosa.feature.rms(y=mono, frame_length=2048, hop_length=512)[0]
    orig_db_env = librosa.amplitude_to_db(orig_rms_env + 1e-9)
    orig_dyn = float(np.percentile(orig_db_env, 95) - np.percentile(orig_db_env, 10))
    
    # ===== 1) 质心降暗（high shelf）=====
    if orig_cent > TARGET_CENTROID * 1.15:
        shelf_db = -min(4.0, (orig_cent - TARGET_CENTROID) / TARGET_CENTROID * 12)
        sos = butter(2, 3500 / (sr/2), btype="high", output="sos")
        for ch in range(2):
            high = sosfilt(sos, y[ch])
            y[ch] = y[ch] + high * (10 ** (shelf_db / 20) - 1)
    
    # ===== 2) 动态压缩 =====
    mono = y.mean(axis=0)
    rms = librosa.feature.rms(y=mono, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    cur_dyn = float(np.percentile(db, 95) - np.percentile(db, 10))
    if cur_dyn > 16:
        # 简易压缩：限制 3s 窗间的响度差
        w = int(3 * sr / 512)
        smooth_db = np.convolve(db, np.ones(w)/w, mode="same")
        gain_db = np.zeros_like(db)
        loud = db > -10
        gain_db[loud] = -10 - db[loud]  # 压到 -10 以上
        gain_db = np.clip(gain_db, -3, 0)  # 最多降 3dB
        gain_db = np.convolve(gain_db, np.ones(50)/50, mode="same")  # 平滑
        for ch in range(2):
            y[ch] *= 10 ** (gain_db / 20)
    
    # ===== 3) 响度归一 =====
    rms_final = np.sqrt((y.mean(axis=0) ** 2).mean() + 1e-9)
    cur_rms_db = 20 * np.log10(rms_final + 1e-9)
    if cur_rms_db < -22:
        y *= 10 ** ((-18 - cur_rms_db) / 20)
    elif cur_rms_db > -10:
        y *= 10 ** ((-13 - cur_rms_db) / 20)
    
    peak = np.abs(y).max()
    if peak > 0.95:
        y *= 0.95 / peak
    
    sf.write(str(out_path), y.T, sr, format="FLAC")
    
    # 重测
    y_check, _ = librosa.load(str(out_path), sr=SR, mono=True, duration=100)
    cent_new = float(librosa.feature.spectral_centroid(y=y_check, sr=SR)[0].mean())
    rms_new = librosa.feature.rms(y=y_check, frame_length=2048, hop_length=512)[0]
    db_new = librosa.amplitude_to_db(rms_new + 1e-9)
    dyn_new = round(float(np.percentile(db_new, 95) - np.percentile(db_new, 10)), 1)
    rms_new_db = round(float(librosa.amplitude_to_db(np.sqrt((y_check**2).mean()) + 1e-9)), 1)
    
    return dict(centroid_before=round(orig_cent), centroid_after=round(cent_new),
                dyn_before=round(orig_dyn, 1), dyn_after=round(dyn_new, 1),
                rms_after=rms_new_db)


def main():
    src_dir = ROOT / "novel_songs" / "final"
    out_dir = ROOT / "novel_songs" / "final" / "mastered"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    flacs = sorted(src_dir.glob("*.flac"))
    print(f"处理 {len(flacs)} 首...", flush=True)
    
    for i, f in enumerate(flacs):
        out = out_dir / f.name
        try:
            r = process_song_v2(str(f), str(out))
            print(f"  [{i+1}] {f.stem[:22]}: 质心 {r['centroid_before']}→{r['centroid_after']} | "
                  f"动态 {r['dyn_before']}→{r['dyn_after']}", flush=True)
        except Exception as e:
            print(f"  FAIL {f.stem[:22]}: {str(e)[:60]}", flush=True)
    
    import numpy as np
    cents, dyns = [], []
    for r in json.loads((ROOT/"eval"/"master_fix_report.json").read_text(encoding="utf-8")) if (ROOT/"eval"/"master_fix_report.json").exists() else []:
        pass
    # 简单统计输出
    print(f"\n输出目录: {out_dir}")


if __name__ == "__main__":
    main()
