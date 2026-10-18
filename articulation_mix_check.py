# -*- coding: utf-8 -*-
"""阶段0快验: 5 articulation 新维在【混音副歌30s窗】上是否保持人类vs AI方向
对照: ① stem 同窗(排除窗口混淆)  ② 已发表的 stem 全曲值(参考基准)
产出: eval/articulation_mix_cache.json + eval/articulation_mix_summary.json
判定: 方向一致 且 p_mix<0.05(或与参考同向且 p<0.1) 的维数 >=4/5 -> 混音方案可行
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
GAP_RUN = int(0.15 * SR / HOP)  # 150ms 桥接
CACHE = ROOT / "eval" / "articulation_mix_cache.json"

# 已发表 stem 全曲参考 (eval/articulation{2,3,4,5}_summary.json)
REF = {
    "vib_ratio":       dict(h=0.3458, ai=0.2152, p=0.0208, dir="h>"),
    "vib_depth":       dict(h=145.5,  ai=105.5,  p=0.0269, dir="h>"),
    "phase_R":         dict(h=0.2215, ai=0.3155, p=0.0621, dir="h<"),
    "onset_rise_db":   dict(h=10.147, ai=16.259, p=0.0006, dir="h<"),
    "long_smooth":     dict(h=0.8157, ai=0.5974, p=3.1e-5, dir="h>"),
}
DIMS = ["vib_ratio", "vib_depth", "phase_R", "onset_rise_db", "long_smooth"]


def find_chorus_segment(path, seg_len=30):
    """与 score_v3.measure 同实现 (能量+起音密度峰 30s 窗); 不 import score_v3 以免触发全管线"""
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    if len(y) < seg_len * SR:
        return y, 0.0
    hop = HOP
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    ons = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    n = len(rms)
    win = int(seg_len * SR / hop)
    step = int(5 * SR / hop)
    best_score, best_t = -1, 0
    for t in range(0, n - win, step):
        sc = rms[t:t + win].mean() * (1 + ons[t:t + win].mean())
        if sc > best_score:
            best_score, best_t = sc, t
    t0 = best_t * hop / sr
    return y[int(t0 * sr):int((t0 + seg_len) * sr)], round(t0, 1)


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


def dims5(y):
    """在 30s 窗 y 上算 5 新维 + long_trend_frac; 算法逐字取自 articulation2~5 已验证版"""
    if len(y) < SR * 10:
        return None
    hop_s = HOP / SR
    f0, vf, _ = librosa.pyin(y, fmin=65, fmax=700, sr=SR, frame_length=FRAME, hop_length=HOP)
    voiced = vf & ~np.isnan(f0)
    runs = extract_runs(voiced)
    out = {"n_runs": len(runs)}
    if len(runs) < 6:
        return None
    run_dur = [(e - s + 1) * hop_s for s, e in runs]

    # --- vib: 0.8-2.5s run, f0 cents 去线性趋势, 自相关 4-9Hz 主频 (articulation4) ---
    vib_rates, vib_depths = [], []
    for (s, e), d in zip(runs, run_dur):
        if not (0.8 <= d <= 2.5):
            continue
        seg = f0[s:e + 1]
        valid = ~np.isnan(seg)
        if valid.sum() < 40:
            continue
        seg = seg[valid]
        cents = 1200 * np.log2(seg / np.median(seg))
        t = np.arange(len(cents)) * hop_s
        try:
            coef = np.polyfit(t, cents, 1)
            resid = cents - np.polyval(coef, t)
        except Exception:
            continue
        if resid.std() < 3:
            continue
        ac = np.correlate(resid, resid, mode="full")[len(resid) - 1:]
        ac = ac / (ac[0] + 1e-9)
        lags = np.arange(len(ac)) * hop_s
        mask = (lags >= 1 / 9) & (lags <= 1 / 4)
        if mask.sum() < 2:
            continue
        peak_lag = lags[mask][np.argmax(ac[mask])]
        rate = 1.0 / peak_lag if peak_lag > 0 else None
        if rate and 3.5 <= rate <= 9.5:
            vib_rates.append(float(rate))
            vib_depths.append(float(resid.std()))
    out["vib_ratio"] = float(len(vib_rates) / max(len(runs), 1))
    out["vib_depth"] = float(np.median(vib_depths)) if vib_depths else None

    # --- phase_R: run onset 对窗内 beat 相位集中度 (articulation2/3) ---
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP)
        beat_times = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
        onset_times = np.array([s * hop_s for s, e in runs])
        if len(beat_times) >= 16 and len(onset_times) >= 5:
            beat_dur = float(np.median(np.diff(beat_times)))
            phase = []
            for ot in onset_times:
                idx = np.searchsorted(beat_times, ot)
                if 0 < idx < len(beat_times):
                    nb = beat_times[idx - 1] if abs(ot - beat_times[idx - 1]) < abs(beat_times[idx] - ot) else beat_times[idx]
                    phase.append(((ot - nb) / beat_dur) % 1.0)
            phase = np.array(phase)
            if len(phase) > 4:
                R = float(np.sqrt(np.mean(np.cos(2 * np.pi * phase)) ** 2 +
                                  np.mean(np.sin(2 * np.pi * phase)) ** 2))
                out["phase_R"] = R
            else:
                out["phase_R"] = None
        else:
            out["phase_R"] = None
    except Exception:
        out["phase_R"] = None

    # --- onset_rise_db / long_smooth / long_trend_frac (articulation5, frame=1024) ---
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    onset_rise, smooth_absdiff, dir_ratio = [], [], []
    for (s, e) in runs:
        d = (e - s + 1) * hop_s
        if d >= 0.8:
            seg_db = 20 * np.log10(rms[s:e + 1] + 1e-9)
            if len(seg_db) >= 6:
                smooth_absdiff.append(float(np.median(np.abs(np.diff(seg_db)))))
                x = np.arange(len(seg_db))
                sl = np.polyfit(x, seg_db, 1)
                pred = np.polyval(sl, x)
                slope_total = sl[0] * len(seg_db)
                total_range = float(seg_db.max() - seg_db.min())
                dir_ratio.append(float(min(abs(slope_total) / (total_range + 1e-9), 1.0)))
    for i in range(1, len(runs)):
        prev_end = runs[i - 1][1]
        s, e = runs[i]
        pre = rms[max(0, prev_end + 1):s + 1]
        if len(pre) < 3:
            continue
        onset_db = 20 * np.log10(rms[s + 1:e + 1][:3].mean() + 1e-9)
        presil_db = 20 * np.log10(np.median(pre) + 1e-9)
        rise = onset_db - presil_db
        if rise > 0:
            onset_rise.append(float(min(rise, 60)))
    out["onset_rise_db"] = float(np.median(onset_rise)) if onset_rise else None
    out["long_smooth"] = float(np.median(smooth_absdiff)) if smooth_absdiff else None
    out["long_trend_frac"] = float(np.median(dir_ratio)) if dir_ratio else None
    return out


def stem_cut(stem_path, t0, dur=30):
    """按 mix 同一 t0 截 stem 同窗"""
    ys, _ = librosa.load(str(stem_path), sr=SR, mono=True)
    a, b = int(t0 * SR), int((t0 + dur) * SR)
    seg = ys[a:b]
    if len(seg) < SR * 10:
        return None
    return seg


def build_groups():
    groups = {}
    # 人类 30: mix=library wav, stem=human_stems
    hum = []
    for d in sorted((ROOT / "human_stems").iterdir()):
        if not (d / "vocals.wav").exists():
            continue
        mix = ROOT / "eval" / "library_wavs" / f"{d.name}.wav"
        if mix.exists():
            hum.append((mix, d / "vocals.wav"))
    groups["human"] = hum
    # AI 冠军 12
    CHAMPS = [
        r"novel_songs\batch_gacha\01_v2\seed58888.flac",
        r"novel_songs\batch_gacha\02_wide\seed7778.flac",
        r"novel_songs\batch_gacha\03_v2\seed61001.flac",
        r"novel_songs\batch_gacha\04\seed90406.flac",
        r"novel_songs\batch_gacha\05_wide\seed1111.flac",
        r"novel_songs\batch_gacha\06_wide\seed3690.flac",
        r"novel_songs\batch_gacha\07_adj\seed1357_r1.flac",
        r"novel_songs\batch_gacha\11\seed91104.flac",
        r"novel_songs\batch_gacha\09\seed90904.flac",
        r"novel_songs\batch_gacha\10\seed91007.flac",
        r"novel_songs\batch_gacha\11\seed91103.flac",
        r"novel_songs\batch_gacha\12\seed91202.flac",
    ]
    ch = []
    for rel in CHAMPS:
        p = ROOT / rel
        st = ROOT / "album_voice" / "_stems" / p.stem / "vocals.wav"
        if p.exists() and st.exists():
            ch.append((p, st))
    groups["champ"] = ch
    # AI 多 seed: 01_v2 (11) / 04_v3 (8)
    for gname, sub, stmdir in [("seed01", "01_v2", "01_v2"), ("seed04", "04", "04_v3")]:
        lst = []
        for d in sorted((ROOT / "ai_stems" / stmdir).iterdir()):
            if not (d / "vocals.wav").exists():
                continue
            mix = ROOT / "novel_songs" / "batch_gacha" / sub / f"{d.name}.flac"
            if mix.exists():
                lst.append((mix, d / "vocals.wav"))
        groups[gname] = lst
    return groups


def main():
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    groups = build_groups()
    for g, lst in groups.items():
        print(f"{g}: n={len(lst)}", flush=True)

    # 测量 (断点续跑)
    for g, lst in groups.items():
        for mix, stem in lst:
            key = str(mix.relative_to(ROOT)).replace("\\", "/")
            if key in cache and cache[key].get("mix") is not None:
                continue
            try:
                y30, t0 = find_chorus_segment(mix)
                mix_d = dims5(y30)
                st_d = None
                if stem.exists():
                    ys = stem_cut(stem, t0)
                    if ys is not None:
                        st_d = dims5(ys)
                cache[key] = {"group": g, "t0": t0, "mix": mix_d, "stem": st_d}
                CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            except Exception as e:
                print(f"  ERR {key}: {type(e).__name__}: {e}", flush=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    def vals(group, variant, dim):
        out = []
        for k, v in cache.items():
            if v.get("group") != group:
                continue
            d = v.get(variant)
            if d and d.get(dim) is not None:
                out.append(d[dim])
        return out

    summary = {"groups": {g: sum(1 for v in cache.values() if v.get("group") == g) for g in groups},
               "mw_mix": {}, "mw_stem_win": {}, "agree_mix_vs_stem": {}, "verdict": {}}
    print("\n=== Mann-Whitney 人类 vs AI冠军 (混音30s窗) ===", flush=True)
    print(f"{'dim':16s} {'mix人类':>10s} {'mixAI':>10s} {'p_mix':>9s} | "
          f"{'stem窗人类':>10s} {'stem窗AI':>10s} {'p_stem':>9s} | {'ρ(mix,stem)':>11s} 判定", flush=True)
    hold = 0
    for dim in DIMS:
        row = {}
        # 混音窗: human vs champ
        hv = vals("human", "mix", dim)
        cv = vals("champ", "mix", dim)
        if len(hv) >= 3 and len(cv) >= 3:
            _, p_mix = stats.mannwhitneyu(hv, cv, alternative="two-sided")
            hmed, cmed = float(np.median(hv)), float(np.median(cv))
        else:
            p_mix, hmed, cmed = 1.0, float("nan"), float("nan")
        # stem 同窗: human vs champ
        hs = vals("human", "stem", dim)
        cs = vals("champ", "stem", dim)
        if len(hs) >= 3 and len(cs) >= 3:
            _, p_stem = stats.mannwhitneyu(hs, cs, alternative="two-sided")
            hsmed, csmed = float(np.median(hs)), float(np.median(cs))
        else:
            p_stem, hsmed, csmed = 1.0, float("nan"), float("nan")
        # 配对一致: 全部 pair 上 mix vs stem 同窗 Spearman
        xs, ys_ = [], []
        for k, v in cache.items():
            if v.get("mix") and v.get("stem") and v["mix"].get(dim) is not None and v["stem"].get(dim) is not None:
                xs.append(v["mix"][dim]); ys_.append(v["stem"][dim])
        rho = float(stats.spearmanr(xs, ys_)[0]) if len(xs) >= 8 else None
        # 判定: 混音方向与参考一致 且 p_mix<0.05 -> 保持; 同向且 p_mix<0.1 -> 部分
        ref = REF[dim]
        dir_ok = (hmed > cmed and ref["dir"] == "h>") or (hmed < cmed and ref["dir"] == "h<")
        if dir_ok and p_mix < 0.05:
            verdict = "保持"
            hold += 1
        elif dir_ok and p_mix < 0.1:
            verdict = "~部分"
        elif dir_ok:
            verdict = "同向ns"
        else:
            verdict = "反向"
        summary["mw_mix"][dim] = dict(human=hmed, ai=cmed, p=float(p_mix), dir_ok=dir_ok)
        summary["mw_stem_win"][dim] = dict(human=hsmed, ai=csmed, p=float(p_stem))
        summary["agree_mix_vs_stem"][dim] = dict(rho=rho, n=len(xs))
        summary["verdict"][dim] = verdict
        print(f"{dim:16s} {hmed:10.3f} {cmed:10.3f} {p_mix:9.4f} | {hsmed:10.3f} {csmed:10.3f} "
              f"{p_stem:9.4f} | {(rho if rho is not None else float('nan')):11.3f} {verdict}", flush=True)

    summary["hold"] = hold
    summary["decision"] = "混音方案可行 (>=4/5保持)" if hold >= 4 else (
        "需补看同向ns维数后定" if hold >= 3 else "混音方案不成立 -> 走 stem 方案")
    print(f"\n保持 {hold}/5 -> {summary['decision']}", flush=True)
    print(f"参考(stem全曲): " + "; ".join(
        f"{d}: {REF[d]['h']} vs {REF[d]['ai']} p={REF[d]['p']}" for d in DIMS), flush=True)

    (ROOT / "eval" / "articulation_mix_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SAVED eval/articulation_mix_summary.json", flush=True)


if __name__ == "__main__":
    main()
