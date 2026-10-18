# -*- coding: utf-8 -*-
"""articulation3: 直击用户原话
1) 每个字词占用的节拍是否接近 -> IOI 量化分布 (字间隔是不是全贴 1/4、1/8 拍)
2) 没小节 -> 乐句长度是否落 4/8 小节整数倍
3) rubato 呼吸感 -> 人声局部速度波动 (vocal IOI 滑窗 CV)
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
GAP_PHRASE = int(1.0 * SR / HOP)
CACHE = ROOT / "eval" / "articulation3_cache.json"

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

def analyze(voc_path, acc_path):
    y, _ = librosa.load(str(voc_path), sr=SR, mono=True)
    f0, vf, _ = librosa.pyin(y, fmin=65, fmax=700, sr=SR, frame_length=FRAME, hop_length=HOP)
    voiced = vf & ~np.isnan(f0)
    runs = extract_runs(voiced)
    if len(runs) < 8 or not acc_path or not Path(acc_path).exists():
        return None
    hop_s = HOP / SR
    ya, _ = librosa.load(str(acc_path), sr=SR, mono=True)
    tempo, beats = librosa.beat.beat_track(y=ya, sr=SR, hop_length=HOP)
    tempo_val = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
    if len(beat_times) < 16:
        return None
    beat_dur = float(np.median(np.diff(beat_times)))
    onset_times = np.array([s * hop_s for s, e in runs])

    # ---- 1) IOI 量化: IOI/beat 距最近 0.25 拍的误差 ----
    ioi = np.diff(onset_times)
    ioi = ioi[(ioi > 0.12) & (ioi < 4.0)]  # 过滤静音断句
    ioi_beats = ioi / beat_dur
    # 距最近 0.25 拍的残差 (拍为单位)
    grid = np.round(ioi_beats / 0.25) * 0.25
    resid = np.abs(ioi_beats - grid)
    quant_err = float(resid.mean())          # 越小越贴网格
    quant_err_std = float(resid.std())
    # 分布熵 (标准化到 [0,1]): 10 个 0.25拍箱
    hist, _ = np.histogram(ioi_beats, bins=np.arange(0.25, 4.26, 0.25))
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())  # max ~ log2(16)=4
    entropy_norm = float(entropy / np.log2(min(16, len(hist)))) if len(hist) > 1 else 0

    # ---- 2) 小节感: 乐句时长 / 小节时长 -> 距最近整数小节误差 ----
    bar_dur = beat_dur * 4
    phrases, cur = [], [0]
    for i in range(1, len(runs)):
        if runs[i][0] - runs[i-1][1] - 1 > GAP_PHRASE:
            phrases.append(cur); cur = [i]
        else:
            cur.append(i)
    phrases.append(cur)
    phrases = [p for p in phrases if len(p) >= 3]
    bar_errs = []
    for p in phrases:
        t0 = runs[p[0]][0] * hop_s
        t1 = runs[p[-1]][1] * hop_s
        dur_bars = (t1 - t0) / bar_dur
        if dur_bars < 1: continue
        err = abs(dur_bars - round(dur_bars))
        # 距最近的 2/4 小节倍数 (更严格的流行句法)
        err_sem = min(abs(dur_bars - k) for k in [2, 3, 4, 6, 8, 12, 16]) if dur_bars <= 16 else abs(dur_bars - round(dur_bars))
        bar_errs.append({"err_any": float(err), "err_semantic": float(err_sem), "dur_bars": float(dur_bars)})
    if not bar_errs:
        return None
    bar_err_med = float(np.median([b["err_any"] for b in bar_errs]))
    bar_err_sem_med = float(np.median([b["err_semantic"] for b in bar_errs]))

    # ---- 3) Rubato: 人声局部 IOI 速度波动 (滑窗 6 个 IOI, 步进 3) ----
    local_rates = []
    for i in range(0, len(ioi) - 5, 3):
        w = ioi[i:i+6]
        local_rates.append(w.mean())
    if len(local_rates) >= 3:
        local_rates = np.array(local_rates)
        rubato_cv = float(local_rates.std() / local_rates.mean())
        rubato_range = float((local_rates.max() - local_rates.min()) / local_rates.mean())
    else:
        rubato_cv = rubato_range = None

    # ---- 4) onset 贴拍相位 (相对伴奏 beat) 重算 ----
    phase = []
    for ot in onset_times:
        idx = np.searchsorted(beat_times, ot)
        if 0 < idx < len(beat_times):
            nearest = beat_times[idx-1] if abs(ot-beat_times[idx-1]) < abs(beat_times[idx]-ot) else beat_times[idx]
            phase.append(((ot - nearest) / beat_dur) % 1.0)
    phase = np.array(phase)
    R = float(np.sqrt(np.mean(np.cos(2*np.pi*phase))**2 + np.mean(np.sin(2*np.pi*phase))**2)) if len(phase) > 4 else None

    return {
        "tempo": round(tempo_val, 1), "beat_dur": round(beat_dur, 3),
        "n_ioi": len(ioi),
        "ioi_quant_err": round(quant_err, 4),
        "ioi_quant_err_std": round(quant_err_std, 4),
        "ioi_entropy_norm": round(entropy_norm, 3),
        "n_phrases": len(phrases),
        "bar_err_med": round(bar_err_med, 3),
        "bar_err_sem_med": round(bar_err_sem_med, 3),
        "rubato_cv": round(rubato_cv, 4) if rubato_cv is not None else None,
        "rubato_range": round(rubato_range, 4) if rubato_range is not None else None,
        "phase_R": round(R, 3) if R is not None else None,
    }

def collect(tag, voc_paths):
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    out = []
    for voc in voc_paths:
        key = str(voc.relative_to(ROOT)).replace("\\", "/")
        if key in cache and cache[key]:
            out.append((key, cache[key])); continue
        acc = voc.parent / "accomp.wav"
        try:
            m = analyze(voc, acc)
        except Exception as e:
            m = {"error": f"{type(e).__name__}: {e}"}
        cache[key] = m
        out.append((key, m))
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return out

def summarize(tag, results):
    valid = [m for _, m in results if m and "error" not in m and "ioi_quant_err" in m]
    if not valid:
        print(f"{tag}: no valid"); return None
    keys = ["tempo", "ioi_quant_err", "ioi_quant_err_std", "ioi_entropy_norm",
            "bar_err_med", "bar_err_sem_med", "rubato_cv", "rubato_range", "phase_R"]
    med = {}
    print(f"\n{tag} (n={len(valid)})")
    for k in keys:
        vals = [m[k] for m in valid if m.get(k) is not None]
        if vals:
            med[k] = float(np.median(vals))
            print(f"  {k:22s} {med[k]:.4f}")
    return med

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

    mh = summarize("人类", hr)
    mc = summarize("AI冠军", cr)
    m1 = summarize("AI-01v2", s1)
    m4 = summarize("AI-04v3", s4)

    print("\n=== 人类 vs AI冠军 Mann-Whitney ===", flush=True)
    hv = [m for _, m in hr if m and "error" not in m and "ioi_quant_err" in m]
    cv = [m for _, m in cr if m and "error" not in m and "ioi_quant_err" in m]
    tests = {}
    for k in ["ioi_quant_err", "ioi_entropy_norm", "bar_err_med", "bar_err_sem_med",
              "rubato_cv", "rubato_range", "phase_R"]:
        a = [m[k] for m in hv if m.get(k) is not None]
        b = [m[k] for m in cv if m.get(k) is not None]
        if len(a) >= 3 and len(b) >= 3:
            U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            tests[k] = {"human_med": float(np.median(a)), "ai_med": float(np.median(b)), "p": float(p)}
            mark = "*" if p < 0.05 else " "
            print(f"{mark} {k:22s} 人类={np.median(a):.4f} AI={np.median(b):.4f} p={p:.4f}")

    summary = {"human": mh, "ai_champ": mc, "ai_01v2": m1, "ai_04v3": m4, "mannwhitney": tests}
    (ROOT/"eval"/"articulation3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSAVED eval/articulation3_summary.json", flush=True)

if __name__ == "__main__":
    main()
