# -*- coding: utf-8 -*-
"""articulation4: 真因候选 (用户感知'情绪过得快')
A. 乐句内音符均匀度: 同一乐句内时值 CV — 人类长短相间, AI 均匀机枪
B. 颤音: 0.8-2s 持续音的 f0 调制 (深度/速率) — AI 直音=死, 或过量=机器
C. 乐句收尾: 末尾 f0 是否下滑/portamento (人唱收句下落, AI 直切)
D. 段落建置: 全曲分8窗能量剖面, 副歌对比度 (verse-chorus 落差)
E. 方向性 rubato: 乐句内速度趋势 (人唱句末渐慢, AI 无方向)
"""
import json, warnings
from pathlib import Path
import numpy as np
import librosa
from scipy import stats, signal as sps

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
HOP = 512
FRAME = 1024
GAP_RUN = int(0.15 * SR / HOP)
GAP_PHRASE = int(1.0 * SR / HOP)
CACHE = ROOT / "eval" / "articulation4_cache.json"

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
    total_s = len(y) / SR
    f0, vf, _ = librosa.pyin(y, fmin=65, fmax=700, sr=SR, frame_length=FRAME, hop_length=HOP)
    voiced = vf & ~np.isnan(f0)
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    runs = extract_runs(voiced)
    if len(runs) < 8:
        return None
    hop_s = HOP / SR
    run_dur = np.array([(e - s + 1) * hop_s for s, e in runs])

    # ---- A. 乐句分段 + 乐句内时值 CV ----
    phrases, cur = [], [0]
    for i in range(1, len(runs)):
        if runs[i][0] - runs[i-1][1] - 1 > GAP_PHRASE:
            phrases.append(cur); cur = [i]
        else:
            cur.append(i)
    phrases.append(cur)
    phrases = [p for p in phrases if len(p) >= 4]  # 至少4音才有CV意义

    phr_cv, phr_arc_r2, phr_end_f0slope, phr_ioi_trend = [], [], [], []
    for p in phrases:
        d = run_dur[p]
        if len(d) >= 4 and d.mean() > 0:
            phr_cv.append(float(d.std() / d.mean()))
        # B/收尾: f0 末 300ms 斜率 (半音/秒) — 收句下落=负
        s0, e0 = runs[p[0]][0], runs[p[-1]][1]
        seg = f0[s0:e0+1]
        valid = ~np.isnan(seg)
        if valid.sum() > 20:
            t = np.arange(len(seg)) * hop_s
            f0m = librosa.hz_to_midi(np.where(valid, seg, np.nan))
            # 末300ms
            tail = f0m[-int(0.3/hop_s):]
            tail = tail[~np.isnan(tail)]
            if len(tail) >= 4:
                tt = np.arange(len(tail)) * hop_s
                slope = float(np.polyfit(tt, tail - tail[0], 1)[0])  # 半音/秒
                phr_end_f0slope.append(slope)
        # 能量拱形: 乐句内 RMS 中位序列 -> 二次拟合 R² (拱=开口向下)
        seg_rms = rms[s0:e0+1]
        # 按 run 取中位
        run_rms = []
        for i in p:
            rs, re_ = runs[i]
            run_rms.append(float(np.median(rms[rs:re_+1])))
        if len(run_rms) >= 4:
            rr = 20*np.log10(np.array(run_rms)+1e-9)
            x = np.arange(len(rr))
            # 线性 vs 二次
            coef1 = np.polyfit(x, rr, 1); coef2 = np.polyfit(x, rr, 2)
            pred1 = np.polyval(coef1, x); pred2 = np.polyval(coef2, x)
            ss_res1 = ((rr-pred1)**2).sum(); ss_res2 = ((rr-pred2)**2).sum()
            ss_tot = ((rr-rr.mean())**2).sum()
            r2_lin = 1 - ss_res1/ss_tot if ss_tot > 0 else 0
            r2_quad = 1 - ss_res2/ss_tot if ss_tot > 0 else 0
            phr_arc_r2.append(float(max(r2_quad, 0)))
        # 方向性 rubato: 乐句内相邻 run IOI 是否递增 (句末变慢)
        if len(p) >= 4:
            ioi = np.diff([runs[i][0] for i in p]) * hop_s
            if len(ioi) >= 3 and ioi.std() > 0:
                slope_ioi = float(np.polyfit(np.arange(len(ioi)), ioi, 1)[0])
                phr_ioi_trend.append(slope_ioi)

    # ---- C. 颤音: 0.8-2.5s 的 run, f0 去趋势后 4-8Hz 调制深度 ----
    vib_rates, vib_depths = [], []
    for (s, e), d in zip(runs, run_dur):
        if not (0.8 <= d <= 2.5): continue
        seg = f0[s:e+1]
        valid = ~np.isnan(seg)
        if valid.sum() < 40: continue
        seg = seg[valid]
        if len(seg) < 40: continue
        cents = 1200 * np.log2(seg / np.median(seg))
        # 去线性趋势
        t = np.arange(len(cents)) * hop_s
        try:
            coef = np.polyfit(t, cents, 1)
            resid = cents - np.polyval(coef, t)
        except Exception:
            continue
        if resid.std() < 3:  # 几乎无波动, 跳过
            continue
        # 自相关找主频 (4-9 Hz)
        ac = np.correlate(resid, resid, mode="full")[len(resid)-1:]
        ac = ac / (ac[0] + 1e-9)
        lags = np.arange(len(ac)) * hop_s
        mask = (lags >= 1/9) & (lags <= 1/4)
        if mask.sum() < 2: continue
        peak_lag = lags[mask][np.argmax(ac[mask])]
        rate = 1.0 / peak_lag if peak_lag > 0 else None
        if rate and 3.5 <= rate <= 9.5:
            vib_rates.append(float(rate))
            vib_depths.append(float(resid.std()))

    # ---- D. 段落建置: 8 窗能量/人声覆盖剖面 + 副歌对比度 ----
    n_win = 8
    win_len = len(y) // n_win
    win_rms, win_cov, win_cent = [], [], []
    for i in range(n_win):
        seg = y[i*win_len:(i+1)*win_len]
        seg_rms = rms[i*int(len(seg)/len(rms)) if len(rms) >= n_win else i:][:0]
        wr = float(np.sqrt((seg**2).mean()))
        win_rms.append(wr)
    # 用帧级 rms 更准
    frames_per_win = len(rms) // n_win
    win_rms = [float(np.median(rms[i*frames_per_win:(i+1)*frames_per_win])) for i in range(n_win)]
    win_cov = [float(voiced[i*frames_per_win:(i+1)*frames_per_win].mean()) for i in range(n_win)]
    peak_win = int(np.argmax(win_rms))
    contrast = float((max(win_rms) - min(win_rms)) / (np.median(win_rms) + 1e-9))
    # 峰窗位置: 是否在后半段 (5-8) — 人类流行曲副歌靠后, 若峰在窗1=开头直给
    peak_pos_frac = peak_win / (n_win - 1)

    return {
        "total_s": round(total_s, 1),
        "n_phrases": len(phrases),
        "phrase_cv_med": float(np.median(phr_cv)) if phr_cv else None,
        "phrase_arc_r2_med": float(np.median(phr_arc_r2)) if phr_arc_r2 else None,
        "phrase_end_f0slope_med": float(np.median(phr_end_f0slope)) if phr_end_f0slope else None,
        "phrase_end_f0_down_ratio": float(np.mean(np.array(phr_end_f0slope) < -1.0)) if phr_end_f0slope else None,
        "phr_ioi_trend_med": float(np.median(phr_ioi_trend)) if phr_ioi_trend else None,
        "phr_ioi_slowdown_ratio": float(np.mean(np.array(phr_ioi_trend) > 0)) if phr_ioi_trend else None,
        "vib_rate_med": float(np.median(vib_rates)) if vib_rates else None,
        "vib_depth_med": float(np.median(vib_depths)) if vib_depths else None,
        "vib_present_ratio": len(vib_rates) / max(len(runs), 1),
        "win_rms": [round(x, 4) for x in win_rms],
        "win_cov": [round(x, 3) for x in win_cov],
        "section_contrast": round(contrast, 3),
        "peak_win_pos": round(peak_pos_frac, 3),
        "phrases_per_min": round(len(phrases) / (total_s/60), 2),
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

def summarize(tag, results):
    valid = [m for _, m in results if m and "error" not in m and m.get("phrase_cv_med") is not None]
    if not valid:
        print(f"{tag}: no valid"); return None
    keys = ["phrases_per_min", "phrase_cv_med", "phrase_arc_r2_med",
            "phrase_end_f0slope_med", "phrase_end_f0_down_ratio",
            "phr_ioi_trend_med", "phr_ioi_slowdown_ratio",
            "vib_rate_med", "vib_depth_med", "vib_present_ratio",
            "section_contrast", "peak_win_pos"]
    med = {}
    print(f"\n{tag} (n={len(valid)})")
    for k in keys:
        vals = [m[k] for m in valid if m.get(k) is not None]
        if vals:
            med[k] = float(np.median(vals))
            print(f"  {k:28s} {med[k]:.4f}")
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
    hv = [m for _, m in hr if m and "error" not in m and m.get("phrase_cv_med") is not None]
    cv = [m for _, m in cr if m and "error" not in m and m.get("phrase_cv_med") is not None]
    tests = {}
    for k in ["phrases_per_min", "phrase_cv_med", "phrase_arc_r2_med",
              "phrase_end_f0slope_med", "phrase_end_f0_down_ratio",
              "phr_ioi_trend_med", "phr_ioi_slowdown_ratio",
              "vib_rate_med", "vib_depth_med", "vib_present_ratio",
              "section_contrast", "peak_win_pos"]:
        a = [m[k] for m in hv if m.get(k) is not None]
        b = [m[k] for m in cv if m.get(k) is not None]
        if len(a) >= 3 and len(b) >= 3:
            U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            tests[k] = {"human_med": float(np.median(a)), "ai_med": float(np.median(b)), "p": float(p)}
            mark = "*" if p < 0.05 else ("~" if p < 0.1 else " ")
            print(f"{mark} {k:28s} 人类={np.median(a):.4f} AI={np.median(b):.4f} p={p:.4f}")

    summary = {"human": mh, "ai_champ": mc, "ai_01v2": m1, "ai_04v3": m4, "mannwhitney": tests}
    (ROOT/"eval"/"articulation4_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSAVED eval/articulation4_summary.json", flush=True)

if __name__ == "__main__":
    main()
