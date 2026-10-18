# -*- coding: utf-8 -*-
"""时值分布分析: voicing-run 持续时间 = '每个字词占的拍'
人类30 vs AI冠军12 vs AI多seed19
核心假设: AI 无长音、时值均匀 -> 长音比例低、CV低、句尾留白少
"""
import json, sys, warnings, hashlib
from pathlib import Path
import numpy as np
import librosa
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
HOP = 512
FRAME = 1024
GAP_TOL = int(0.15 * SR / HOP)  # 150ms 内的清音段桥接为同一 run (辅音闭塞)
CACHE = ROOT / "eval" / "articulation_cache.json"

def runs_from_path(path):
    """返回 (voiced_flag数组, f0数组, sr, hop)"""
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    f0, vf, _ = librosa.pyin(y, fmin=65, fmax=700, sr=SR, frame_length=FRAME, hop_length=HOP)
    return vf & ~np.isnan(f0), f0, len(y)/SR

def extract_runs(voiced):
    """voiced(bool array) -> runs[(start_idx, end_idx)] 桥接 GAP_TOL 内空隙"""
    if len(voiced) == 0:
        return []
    v = voiced.copy()
    runs = []
    in_run = False
    start = 0
    gap = 0
    for i, val in enumerate(v):
        if val:
            if not in_run:
                start = i
                in_run = True
            gap = 0
        else:
            if in_run:
                gap += 1
                if gap > GAP_TOL:
                    runs.append((start, i - gap))
                    in_run = False
    if in_run:
        runs.append((start, len(v) - 1))
    return runs

def metrics_from_runs(runs, hop, total_s, f0=None):
    if not runs:
        return None
    durs = np.array([(e - s + 1) * hop / SR for s, e in runs])  # seconds
    gaps = np.array([(runs[i+1][0] - runs[i][1] - 1) * hop / SR for i in range(len(runs)-1)])
    n = len(durs)
    dur_total = durs.sum()
    # 长音: >0.75s / >1.0s / >1.5s
    long_ratio_075 = float((durs > 0.75).mean())
    long_ratio_100 = float((durs > 1.00).mean())
    long_ratio_150 = float((durs > 1.50).mean())
    # 时值多样性: 分箱熵
    bins = np.array([0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5, 2.5, 10.0])
    hist, _ = np.histogram(durs, bins=bins)
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())
    # 句尾/乐句停顿: >300ms / >600ms 的 gap 占比
    if len(gaps):
        pause_ratio_300 = float((gaps > 0.3).mean())
        pause_ratio_600 = float((gaps > 0.6).mean())
        mean_gap = float(gaps.mean())
    else:
        pause_ratio_300 = pause_ratio_600 = mean_gap = 0.0
    # 音节率 (run/秒) — 用 run 覆盖的时长做分母更公平
    coverage = dur_total / total_s if total_s else 0
    # f0 变化: run 内 f0 range 中位数 (滑音/转音丰富度)
    f0_med_range = None
    if f0 is not None:
        ranges = []
        for s, e in runs:
            seg = f0[s:e+1]
            seg = seg[~np.isnan(seg)]
            if len(seg) > 3:
                ranges.append(float(seg.max() - seg.min()))
        if ranges:
            f0_med_range = float(np.median(ranges))
    return {
        "n_runs": n,
        "total_s": round(total_s, 1),
        "syllable_rate": round(n / total_s, 3) if total_s else None,
        "coverage": round(coverage, 3),
        "dur_median": round(float(np.median(durs)), 3),
        "dur_mean": round(float(durs.mean()), 3),
        "dur_cv": round(float(durs.std() / durs.mean()), 3),
        "dur_entropy": round(entropy, 3),
        "long_ratio_075": round(long_ratio_075, 3),
        "long_ratio_100": round(long_ratio_100, 3),
        "long_ratio_150": round(long_ratio_150, 3),
        "max_dur": round(float(durs.max()), 3),
        "pause_ratio_300": round(pause_ratio_300, 3),
        "pause_ratio_600": round(pause_ratio_600, 3),
        "mean_gap": round(mean_gap, 3),
        "f0_med_range_cents": round(f0_med_range, 1) if f0_med_range else None,
    }

def collect_group(name, paths):
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    out = []
    for p in paths:
        key = str(p.relative_to(ROOT)).replace("\\", "/")
        if key in cache:
            out.append((key, cache[key]))
            continue
        try:
            voiced, f0, total_s = runs_from_path(p)
            runs = extract_runs(voiced)
            m = metrics_from_runs(runs, HOP, total_s, f0=f0)
        except Exception as e:
            m = {"error": f"{type(e).__name__}: {e}"}
        cache[key] = m
        out.append((key, m))
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return out

def summarize(tag, results):
    valid = [m for _, m in results if m and "error" not in m]
    if not valid:
        print(f"{tag}: no valid"); return None
    keys = ["syllable_rate", "dur_median", "dur_cv", "dur_entropy",
            "long_ratio_075", "long_ratio_100", "long_ratio_150", "max_dur",
            "pause_ratio_300", "pause_ratio_600", "coverage", "f0_med_range_cents"]
    med = {k: float(np.median([m[k] for m in valid if m.get(k) is not None])) for k in keys}
    print(f"\n{tag} (n={len(valid)})")
    for k in keys:
        if med[k] is not None:
            print(f"  {k:22s} {med[k]:.3f}")
    return med

def main():
    # 收集路径
    human = sorted((ROOT/"human_stems").glob("*/vocals.wav"))
    champ = sorted((ROOT/"album_voice"/"_stems").glob("*/vocals.wav"))
    seed01 = sorted((ROOT/"ai_stems"/"01_v2").glob("*/vocals.wav"))
    seed04 = sorted((ROOT/"ai_stems"/"04_v3").glob("*/vocals.wav"))
    print(f"human={len(human)} champ={len(champ)} seed01={len(seed01)} seed04={len(seed04)}", flush=True)

    print("=== 人类 ===", flush=True)
    hr = collect_group("human", human)
    print("=== AI冠军 ===", flush=True)
    cr = collect_group("champ", champ)
    print("=== AI seed01 ===", flush=True)
    s1 = collect_group("01_v2", seed01)
    print("=== AI seed04 ===", flush=True)
    s4 = collect_group("04_v3", seed04)

    m_h = summarize("人类", hr)
    m_c = summarize("AI冠军", cr)
    m_1 = summarize("AI-01v2", s1)
    m_4 = summarize("AI-04v3", s4)

    # 统计检验: 人类 vs AI冠军 (Mann-Whitney)
    print("\n=== 人类 vs AI冠军 Mann-Whitney U ===", flush=True)
    hv = [m for _, m in hr if m and "error" not in m]
    cv = [m for _, m in cr if m and "error" not in m]
    keys = ["syllable_rate", "dur_median", "dur_cv", "dur_entropy",
            "long_ratio_075", "long_ratio_100", "long_ratio_150", "max_dur",
            "pause_ratio_300", "pause_ratio_600", "coverage"]
    sig = {}
    for k in keys:
        a = [m[k] for m in hv if m.get(k) is not None]
        b = [m[k] for m in cv if m.get(k) is not None]
        if len(a) >= 3 and len(b) >= 3:
            U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            sig[k] = {"human_med": float(np.median(a)), "ai_med": float(np.median(b)),
                      "U": int(U), "p": float(p)}
            mark = "*" if p < 0.05 else " "
            print(f"{mark} {k:22s} 人类={np.median(a):.3f} AI={np.median(b):.3f} p={p:.4f}")

    # 多 seed 内部一致性 (01_v2 11个): 看是否 seed 不敏感
    print("\n=== AI-01v2 内部一致 (n=11) ===", flush=True)
    v1 = [m for _, m in s1 if m and "error" not in m]
    if v1:
        for k in ["syllable_rate", "long_ratio_100", "dur_cv"]:
            vals = [m[k] for m in v1 if m.get(k) is not None]
            if vals:
                print(f"  {k:22s} med={np.median(vals):.3f} std={np.std(vals):.3f} range=[{min(vals):.3f},{max(vals):.3f}]")

    # 保存汇总
    summary = {"human": m_h, "ai_champ": m_c, "ai_01v2": m_1, "ai_04v3": m_4, "mannwhitney_vs_champ": sig}
    (ROOT/"eval"/"articulation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSAVED eval/articulation_summary.json", flush=True)

if __name__ == "__main__":
    main()
