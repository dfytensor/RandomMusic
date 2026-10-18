# -*- coding: utf-8 -*-
"""articulation 深挖: 长音质量 + 乐句结构 + 旋律运动 + 节拍量化
用户感知"情绪刚起来就过去了" ≠ 字太短(实测AI长音更多), 真因候选:
  A. 长音是死的: 无渐强渐弱、无滑音、无颤音调制 -> 长而不表情
  B. 乐句不建置: 乐句内动态/音域平, 无 crescendo 到顶点
  C. 旋律不动: 音高事件少, 同音保持, 旋律间隔单调
  D. 节拍量化死板: 音节 onset 全贴网格, 无 rubato/切分
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
GAP_RUN = int(0.15 * SR / HOP)    # run 内桥接 150ms
GAP_PHRASE = int(1.0 * SR / HOP)  # 乐句分隔: 1.0s
CACHE = ROOT / "eval" / "articulation2_cache.json"

def load_stems(voc_path):
    y, _ = librosa.load(str(voc_path), sr=SR, mono=True)
    f0, vf, _ = librosa.pyin(y, fmin=65, fmax=700, sr=SR, frame_length=FRAME, hop_length=HOP)
    voiced = vf & ~np.isnan(f0)
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    return y, f0, voiced, rms

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

def analyze_song(voc_path, acc_path=None):
    y, f0, voiced, rms = load_stems(voc_path)
    total_s = len(y) / SR
    runs = extract_runs(voiced)
    if len(runs) < 5:
        return None
    hop_s = HOP / SR
    # run 中位 f0 (Hz -> MIDI)
    run_f0, run_rms, run_dur = [], [], []
    for s, e in runs:
        seg_f0 = f0[s:e+1]; seg_f0 = seg_f0[~np.isnan(seg_f0)]
        seg_rms = rms[s:e+1]
        if len(seg_f0) < 3: continue
        run_f0.append(float(np.median(seg_f0)))
        run_rms.append(float(np.median(seg_rms)))
        run_dur.append((e - s + 1) * hop_s)
    if len(run_f0) < 5:
        return None
    run_f0 = np.array(run_f0); run_rms = np.array(run_rms); run_dur = np.array(run_dur)
    run_midi = librosa.hz_to_midi(run_f0)

    # --- C. 旋律运动 ---
    intervals = np.abs(np.diff(run_midi))  # 半音
    static_ratio = float((intervals < 0.5).mean())  # 同音重复占比
    interval_cv = float(intervals.std() / intervals.mean()) if intervals.mean() > 0 else 0

    # --- 乐句分隔 (gap > 1s) ---
    phrases, cur = [], [0]
    for i in range(1, len(runs)):
        gap_frames = runs[i][0] - runs[i-1][1] - 1
        if gap_frames > GAP_PHRASE:
            phrases.append(cur); cur = [i]
        else:
            cur.append(i)
    phrases.append(cur)
    phrases = [p for p in phrases if len(p) >= 3]  # 至少3个run

    phrase_stats = []
    for p in phrases:
        seg_midi = run_midi[p]
        seg_rms = run_rms[p]
        seg_db = 20 * np.log10(seg_rms + 1e-9)
        n = len(p)
        dur = sum(run_dur[i] for i in p)
        mel_range = float(seg_midi.max() - seg_midi.min())  # 半音
        dyn_range = float(seg_db.max() - seg_db.min())      # dB
        # 乐句内音高斜率 (线性回归斜率: 半音/run序号)
        x = np.arange(n)
        slope_mel = float(np.polyfit(x, seg_midi, 1)[0]) if n >= 3 else 0
        slope_dyn = float(np.polyfit(x, seg_db, 1)[0]) if n >= 3 else 0
        # 顶点位置 (能量最高 run 的相对位置)
        peak_pos = float(np.argmax(seg_db) / max(n-1, 1))
        phrase_stats.append({
            "n": n, "dur": dur, "mel_range": mel_range, "dyn_range": dyn_range,
            "slope_mel": slope_mel, "slope_dyn": slope_dyn, "peak_pos": peak_pos,
        })

    # --- A. 长音质量 (run >= 0.8s) ---
    long_expr = []
    for (s, e), d in zip(runs, run_dur):
        if d < 0.8: continue
        seg_rms = rms[s:e+1]
        seg_db = 20 * np.log10(seg_rms + 1e-9)
        dyn_in = float(seg_db.max() - seg_db.min())
        slope_in = float(np.polyfit(np.arange(len(seg_db)), seg_db, 1)[0]) if len(seg_db) > 5 else 0
        seg_f0 = f0[s:e+1]; valid = ~np.isnan(seg_f0)
        f0_range_cents = None
        if valid.sum() > 5:
            f0v = seg_f0[valid]
            f0_range_cents = float(1200 * np.log2(f0v.max() / f0v.min()))
        long_expr.append({"dyn_in_db": dyn_in, "slope_in_db": slope_in, "f0_range_cents": f0_range_cents})

    # --- D. IOI 量化 (需要 accomp 节拍) ---
    ioi_quant = None
    if acc_path and Path(acc_path).exists():
        try:
            ya, _ = librosa.load(str(acc_path), sr=SR, mono=True)
            tempo, beats = librosa.beat.beat_track(y=ya, sr=SR, hop_length=HOP)
            tempo_val = float(tempo) if np.isscalar(tempo) or np.size(tempo)==1 else float(np.atleast_1d(tempo)[0])
            onset_times = np.array([s * hop_s for s, e in runs])
            if len(beats) > 8 and len(onset_times) > 5:
                beat_times = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
                beat_dur = np.median(np.diff(beat_times))
                # onset 相对最近 beat 的 phase (0~1)
                phase = []
                for ot in onset_times:
                    idx = np.searchsorted(beat_times, ot)
                    if 0 < idx < len(beat_times):
                        nearest = beat_times[idx-1] if abs(ot - beat_times[idx-1]) < abs(beat_times[idx] - ot) else beat_times[idx]
                        phase.append(((ot - nearest) / beat_dur) % 1.0)
                phase = np.array(phase)
                # Rayleigh test: 是否聚集在 0 (贴网格)
                R = np.sqrt((np.cos(2*np.pi*phase).mean())**2 + (np.sin(2*np.pi*phase).mean())**2)
                z = len(phase) * R**2
                p_ray = np.exp(-z)  # 近似 p 值
                ioi_quant = {
                    "tempo": round(tempo_val, 1),
                    "phase_concentration_R": round(float(R), 3),
                    "phase_rayleigh_p": round(float(p_ray), 4),
                    "phase_entropy": round(float(-(np.histogram(phase, bins=10, range=(0,1), density=True)[0]*np.histogram(phase, bins=10, range=(0,1), density=True)[0]).sum() * 0.1), 3),
                }
        except Exception as ex:
            ioi_quant = {"error": str(ex)}

    return {
        "total_s": round(total_s, 1),
        "n_runs": len(runs),
        "static_ratio": round(static_ratio, 3),
        "interval_cv": round(interval_cv, 3),
        "median_interval_semi": round(float(np.median(intervals)), 2),
        "n_phrases": len(phrases),
        "phrase_stats": phrase_stats,
        "long_expr": long_expr,
        "ioi": ioi_quant,
    }

def collect(tag, voc_dir_list):
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    out = []
    for voc in voc_dir_list:
        key = str(voc.relative_to(ROOT)).replace("\\", "/")
        if key in cache:
            out.append((key, cache[key])); continue
        acc = voc.parent / "accomp.wav"
        try:
            m = analyze_song(voc, acc_path=acc if acc.exists() else None)
        except Exception as e:
            m = {"error": f"{type(e).__name__}: {e}"}
        cache[key] = m
        out.append((key, m))
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return out

def agg_phrase_stats(members):
    """聚合(已解包的)歌曲列表的所有乐句"""
    all_ps = []
    for m in members:
        if m and "phrase_stats" in m:
            all_ps.extend(m["phrase_stats"])
    if not all_ps:
        return None
    return {
        "n_phrases": len(all_ps),
        "mel_range_med": float(np.median([p["mel_range"] for p in all_ps])),
        "dyn_range_med": float(np.median([p["dyn_range"] for p in all_ps])),
        "slope_mel_abs_med": float(np.median([abs(p["slope_mel"]) for p in all_ps])),
        "slope_dyn_abs_med": float(np.median([abs(p["slope_dyn"]) for p in all_ps])),
        "peak_pos_med": float(np.median([p["peak_pos"] for p in all_ps])),
        "n_runs_per_phrase": float(np.median([p["n"] for p in all_ps])),
        "dur_per_phrase": float(np.median([p["dur"] for p in all_ps])),
    }

def agg_long_expr(members):
    all_le = []
    for m in members:
        if m and "long_expr" in m:
            all_le.extend(m["long_expr"])
    if not all_le:
        return None
    return {
        "n_long": len(all_le),
        "dyn_in_db_med": float(np.median([x["dyn_in_db"] for x in all_le])),
        "slope_in_abs_med": float(np.median([abs(x["slope_in_db"]) for x in all_le])),
        "f0_range_cents_med": float(np.median([x["f0_range_cents"] for x in all_le if x["f0_range_cents"] is not None])) if any(x["f0_range_cents"] for x in all_le) else None,
    }

def summarize(tag, results):
    valid = [m for _, m in results if m and "error" not in m and "phrase_stats" in m]
    if not valid:
        print(f"{tag}: no valid"); return None
    static = float(np.median([m["static_ratio"] for m in valid]))
    icv = float(np.median([m["interval_cv"] for m in valid]))
    ii = float(np.median([m["median_interval_semi"] for m in valid]))
    ps = agg_phrase_stats(valid)
    le = agg_long_expr(valid)
    ioi_list = [m["ioi"] for m in valid if m.get("ioi") and "phase_concentration_R" in m.get("ioi", {})]
    ioi_R = float(np.median([x["phase_concentration_R"] for x in ioi_list])) if ioi_list else None
    ioi_p = float(np.median([x["phase_rayleigh_p"] for x in ioi_list])) if ioi_list else None
    ioi_tempo = float(np.median([x["tempo"] for x in ioi_list])) if ioi_list else None
    print(f"\n{tag} (n={len(valid)})")
    print(f"  静音比(同音重复)     {static:.3f}")
    print(f"  旋律间隔CV           {icv:.3f}")
    print(f"  旋律间隔中位(半音)   {ii:.2f}")
    if ps:
        print(f"  乐句数中位           {ps['n_phrases']/len(valid):.1f}")
        print(f"  乐句内音域(半音)     {ps['mel_range_med']:.1f}")
        print(f"  乐句内动态(dB)       {ps['dyn_range_med']:.1f}")
        print(f"  乐句音高斜率绝对值   {ps['slope_mel_abs_med']:.3f} 半音/run")
        print(f"  乐句动态斜率绝对值   {ps['slope_dyn_abs_med']:.3f} dB/run")
        print(f"  能量顶点位置         {ps['peak_pos_med']:.2f} (0=句首 1=句尾)")
        print(f"  每句 run 数          {ps['n_runs_per_phrase']:.1f}")
        print(f"  每句时长(s)          {ps['dur_per_phrase']:.1f}")
    if le:
        print(f"  长音(>0.8s)内动态    {le['dyn_in_db_med']:.1f} dB")
        print(f"  长音内斜率绝对值     {le['slope_in_abs_med']:.3f} dB/frame")
        if le['f0_range_cents_med']:
            print(f"  长音内f0摆动(音分)   {le['f0_range_cents_med']:.0f}")
    if ioi_R is not None:
        print(f"  节拍相位集中度R      {ioi_R:.3f} (1=全贴拍 0=均匀)")
        print(f"  Rayleigh p           {ioi_p:.4f} (<0.05=显著贴拍)")
        print(f"  伴奏BPM中位          {ioi_tempo:.1f}")
    return {"static_ratio": static, "interval_cv": icv, "median_interval_semi": ii,
            "phrase": ps, "long_expr": le,
            "ioi_R": ioi_R, "ioi_p": ioi_p, "ioi_tempo": ioi_tempo}

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

    # 关键指标 Mann-Whitney: 人类 vs AI冠军
    print("\n=== 人类 vs AI冠军 关键检验 ===", flush=True)
    hv = [m for _, m in hr if m and "error" not in m and "phrase_stats" in m]
    cv = [m for _, m in cr if m and "error" not in m and "phrase_stats" in m]

    tests = {}
    # 静音比
    a = [m["static_ratio"] for m in hv]; b = [m["static_ratio"] for m in cv]
    U, p = stats.mannwhitneyu(a, b); tests["static_ratio"] = {"p": float(p), "human": float(np.median(a)), "ai": float(np.median(b))}
    print(f"  static_ratio       人类={np.median(a):.3f} AI={np.median(b):.3f} p={p:.4f}")
    # 乐句内音域
    a = [m["phrase_stats"] and np.median([x["mel_range"] for x in m["phrase_stats"]]) for m in hv]
    b = [m["phrase_stats"] and np.median([x["mel_range"] for x in m["phrase_stats"]]) for m in cv]
    U, p = stats.mannwhitneyu(a, b); tests["phrase_mel_range"] = {"p": float(p), "human": float(np.median(a)), "ai": float(np.median(b))}
    print(f"  phrase_mel_range   人类={np.median(a):.1f} AI={np.median(b):.1f} p={p:.4f}")
    # 乐句内动态
    a = [np.median([x["dyn_range"] for x in m["phrase_stats"]]) for m in hv]
    b = [np.median([x["dyn_range"] for x in m["phrase_stats"]]) for m in cv]
    U, p = stats.mannwhitneyu(a, b); tests["phrase_dyn_range"] = {"p": float(p), "human": float(np.median(a)), "ai": float(np.median(b))}
    print(f"  phrase_dyn_range   人类={np.median(a):.1f} AI={np.median(b):.1f} p={p:.4f}")
    # 长音内动态
    a = [np.median([x["dyn_in_db"] for x in m["long_expr"]]) for m in hv if m.get("long_expr")]
    b = [np.median([x["dyn_in_db"] for x in m["long_expr"]]) for m in cv if m.get("long_expr")]
    U, p = stats.mannwhitneyu(a, b); tests["long_note_dyn_in"] = {"p": float(p), "human": float(np.median(a)), "ai": float(np.median(b))}
    print(f"  long_note_dyn_in   人类={np.median(a):.1f} AI={np.median(b):.1f} p={p:.4f}")
    # 节拍相位集中度
    a = [m["ioi"]["phase_concentration_R"] for m in hv if m.get("ioi") and "phase_concentration_R" in m.get("ioi", {})]
    b = [m["ioi"]["phase_concentration_R"] for m in cv if m.get("ioi") and "phase_concentration_R" in m.get("ioi", {})]
    if len(a) >= 3 and len(b) >= 3:
        U, p = stats.mannwhitneyu(a, b); tests["ioi_phase_R"] = {"p": float(p), "human": float(np.median(a)), "ai": float(np.median(b))}
        print(f"  ioi_phase_R        人类={np.median(a):.3f} AI={np.median(b):.3f} p={p:.4f}")

    summary = {"human": mh, "ai_champ": mc, "ai_01v2": m1, "ai_04v3": m4, "mannwhitney": tests}
    (ROOT/"eval"/"articulation2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSAVED eval/articulation2_summary.json", flush=True)

if __name__ == "__main__":
    main()
