# -*- coding: utf-8 -*-
"""articulation5: 长音质量收尾
矛盾: AI 长音内 RMS 动态更大(34.9 vs 28.9dB) 但用户听感'长音死'
假设: 人类=平滑渐强/渐弱(受控), AI=逐帧抖动(失控) -> 测轨迹平滑度/方向性
补充: onset attack 锐度 (字头清晰度)
"""
import json, warnings
from pathlib import Path
import numpy as np
import librosa
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
HOP = 512
FRAME = 1024
GAP_RUN = int(0.15 * SR / HOP)
CACHE = ROOT / "eval" / "articulation5_cache.json"

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

def analyze(voc_path):
    y, _ = librosa.load(str(voc_path), sr=SR, mono=True)
    f0, vf, _ = librosa.pyin(y, fmin=65, fmax=700, sr=SR, frame_length=FRAME, hop_length=HOP)
    voiced = vf & ~np.isnan(f0)
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    runs = extract_runs(voiced)
    if len(runs) < 8:
        return None
    hop_s = HOP / SR

    smooth_absdiff, dir_ratio, attack_db = [], [], []
    for (s, e) in runs:
        d = (e - s + 1) * hop_s
        if d < 0.8:  # 只看长音
            continue
        seg_db = 20 * np.log10(rms[s:e+1] + 1e-9)
        if len(seg_db) < 6:
            continue
        # 逐帧绝对差 (dB/frame): 越小越平滑
        ad = np.abs(np.diff(seg_db))
        smooth_absdiff.append(float(np.median(ad)))
        # 方向性: 线性斜率符号占比 vs 抖动占比
        x = np.arange(len(seg_db))
        slope = np.polyfit(x, seg_db, 1)[0] * len(seg_db)  # 总趋势幅度
        resid = seg_db - np.polyval(np.polyfit(x, seg_db, 1), x)
        total_range = float(seg_db.max() - seg_db.min())
        trend_frac = abs(slope) / (total_range + 1e-9)  # 趋势占总变化比例
        dir_ratio.append(float(min(trend_frac, 1.0)))
        # jitter (去趋势残差) 占比
    # attack: onset 前 100ms 到后 50ms 的能量上升 (取有前导静音的 run)
    onset_rise = []
    for i in range(1, len(runs)):
        prev_end = runs[i-1][1]
        s, e = runs[i]
        pre = rms[max(0, prev_end+1):s+1]  # 前导段
        if len(pre) < 3:
            continue
        onset_db = 20*np.log10(rms[s+1:e+1][:3].mean() + 1e-9)  # onset 后短窗
        presil_db = 20*np.log10(np.median(pre) + 1e-9)           # 前导中位
        rise = onset_db - presil_db
        if rise > 0:
            onset_rise.append(float(min(rise, 60)))
    return {
        "long_smooth_db_per_frame": float(np.median(smooth_absdiff)) if smooth_absdiff else None,
        "long_trend_frac": float(np.median(dir_ratio)) if dir_ratio else None,
        "onset_rise_db": float(np.median(onset_rise)) if onset_rise else None,
        "n_long": len(smooth_absdiff),
    }

def collect(tag, voc_paths):
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    out = []
    for voc in voc_paths:
        key = str(voc.relative_to(ROOT)).replace("\\", "/")
        if key in cache and cache[key]:
            out.append((key, cache[key])); continue
        try:
            m = analyze(voc)
        except Exception as e:
            m = {"error": f"{type(e).__name__}: {e}"}
        cache[key] = m
        out.append((key, m))
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return out

def main():
    human = sorted((ROOT/"human_stems").glob("*/vocals.wav"))
    champ = sorted((ROOT/"album_voice"/"_stems").glob("*/vocals.wav"))
    seed01 = sorted((ROOT/"ai_stems"/"01_v2").glob("*/vocals.wav"))
    seed04 = sorted((ROOT/"ai_stems"/"04_v3").glob("*/vocals.wav"))
    print(f"human={len(human)} champ={len(champ)} 01v2={len(seed01)} 04v3={len(seed04)}", flush=True)

    hr = collect("human", human)
    cr = collect("champ", champ)
    s1 = collect("01v2", seed01)
    s4 = collect("04v3", seed04)

    for tag, res in [("人类", hr), ("AI冠军", cr), ("AI-01v2", s1), ("AI-04v3", s4)]:
        valid = [m for _, m in res if m and "error" not in m and m.get("long_smooth_db_per_frame") is not None]
        if not valid:
            print(f"\n{tag}: no valid"); continue
        print(f"\n{tag} (n={len(valid)})")
        for k in ["long_smooth_db_per_frame", "long_trend_frac", "onset_rise_db"]:
            vals = [m[k] for m in valid if m.get(k) is not None]
            if vals:
                print(f"  {k:28s} {np.median(vals):.4f}")

    print("\n=== 人类 vs AI冠军 Mann-Whitney ===", flush=True)
    hv = [m for _, m in hr if m and "error" not in m and m.get("long_smooth_db_per_frame") is not None]
    cv = [m for _, m in cr if m and "error" not in m and m.get("long_smooth_db_per_frame") is not None]
    tests = {}
    for k in ["long_smooth_db_per_frame", "long_trend_frac", "onset_rise_db"]:
        a = [m[k] for m in hv if m.get(k) is not None]
        b = [m[k] for m in cv if m.get(k) is not None]
        if len(a) >= 3 and len(b) >= 3:
            U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            tests[k] = {"human_med": float(np.median(a)), "ai_med": float(np.median(b)), "p": float(p)}
            mark = "*" if p < 0.05 else ("~" if p < 0.1 else " ")
            print(f"{mark} {k:28s} 人类={np.median(a):.4f} AI={np.median(b):.4f} p={p:.4f}")

    (ROOT/"eval"/"articulation5_summary.json").write_text(
        json.dumps({"human": {k: np.median([m[k] for m in [x for _,x in hr] if m.get(k) is not None]) for k in ["long_smooth_db_per_frame","long_trend_frac","onset_rise_db"]},
                    "ai_champ": {k: np.median([m[k] for m in [x for _,x in cr] if m.get(k) is not None]) for k in ["long_smooth_db_per_frame","long_trend_frac","onset_rise_db"]},
                    "mannwhitney": tests}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSAVED eval/articulation5_summary.json", flush=True)

if __name__ == "__main__":
    main()
