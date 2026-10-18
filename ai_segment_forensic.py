# -*- coding: utf-8 -*-
"""《平账人》46-78s AI 味取证：副歌段 vs 曲库同段落副歌分布。
指纹维度：颤音/滑音/音内稳定度/时长机械度/宏观动态/循环周期性
"""
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 24000
HOP = 512


def read_mono_24k(path, t1=0.0, t2=None):
    data, sr = sf.read(str(path), always_2d=True)
    data = data.mean(axis=1)
    if sr != SR:
        import librosa
        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=SR)
    a = int(t1 * SR)
    b = int(t2 * SR) if t2 else len(data)
    return data[a:b].astype(np.float32)


def parse_lab(tdir, name):
    rows = []
    p = Path(tdir) / name
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        q = line.split("\t")
        if len(q) >= 3:
            rows.append((float(q[0]), float(q[1]), q[2].strip()))
    return rows


def first_span(rows, name, limit):
    for t1, t2, n in rows:
        if n.lower() == name and t1 < limit - 7:
            return t1, min(t2, limit)
    return None


def f0_metrics(y):
    f0, vflag, vprob = librosa.pyin(y, fmin=65, fmax=700, sr=SR, frame_length=1024, hop_length=HOP)
    frame_t = np.arange(len(f0)) * HOP / SR
    cents = 1200 * np.log2(f0 / 55.0)
    voiced = vflag & ~np.isnan(f0)
    vr = float(voiced.mean())
    win = max(3, int(round(0.1 * SR / HOP)))  # ~100ms
    runs, start = [], None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if frame_t[i - 1] - frame_t[start] >= 0.4:
                runs.append((start, i))
            start = None
    if start is not None and frame_t[-1] - frame_t[start] >= 0.4:
        runs.append((start, len(voiced)))

    vib_rate, vib_extent, vib_reg, intra_stab, glide_frac = [], [], [], [], []
    win = max(3, int(round(0.1 * SR / HOP)))  # ~100ms
    for a, b in runs:
        seg = cents[a:b]
        seg = seg - np.nanmedian(seg)
        slow = np.convolve(seg, np.ones(win) / win, mode="same")
        fast = seg - slow
        spec = np.abs(np.fft.rfft(np.nan_to_num(fast)))
        freqs = np.fft.rfftfreq(len(fast), d=HOP / SR)
        band = (freqs >= 3.5) & (freqs <= 9)
        if band.sum() == 0 or np.nanstd(seg) < 20:
            continue
        pk = freqs[band][np.argmax(spec[band])]
        band_e = spec[band].sum() + 1e-9
        vib_rate.append(float(pk))
        vib_extent.append(float(np.nanstd(fast)))
        vib_reg.append(float(spec[band].max() / band_e))
        intra_stab.append(float(np.nanstd(slow)))
        # 滑音占比：帧间斜率大且非颤音的帧
        slope = np.abs(np.diff(slow)) / (HOP / SR)  # cents/s
        gl = slope > 150
        glide_frac.append(float(gl.mean()))

    def med(x):
        return round(float(np.median(x)), 3) if x else None

    return dict(voiced_ratio=round(vr, 3),
                vibrato_rate_hz=med(vib_rate),
                vibrato_extent_cents=med(vib_extent),
                vibrato_regularity=med(vib_reg),
                intra_note_drift_cents=med(intra_stab),
                glide_ratio=med(glide_frac),
                n_vocal_runs=len(runs))


def dynamics_metrics(y):
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    db_s = np.convolve(db, np.ones(24) / 24, mode="same")  # ~0.5s
    flux = librosa.onset.onset_strength(y=y, sr=SR, hop_length=512)
    return dict(macro_dyn_std_db=round(float(db_s.std()), 2),
                dyn_range_db=round(float(db_s.max() - db_s.min()), 2),
                spectral_flux_std=round(float(flux.std()), 3))


def loopiness(y, n_mels=48):
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=n_mels, hop_length=1024)
    lm = np.log(m + 1e-6)
    lm = (lm - lm.mean(axis=1, keepdims=True)) / (lm.std(axis=1, keepdims=True) + 1e-9)
    fr = lm.shape[1]
    lags = range(int(3 * SR / 1024), min(int(12 * SR / 1024), fr // 2))
    best, best_lag = -1, 0
    corr_curve = []
    for lag in lags:
        c = float(np.corrcoef(lm[:, :fr - lag].ravel(), lm[:, lag:].ravel())[0, 1])
        corr_curve.append(c)
        if c > best:
            best, best_lag = c, lag
    cc = np.array(corr_curve)
    loopiness = round(float(best - cc.mean()), 3)
    return dict(loop_peak_corr=round(float(best), 3),
                loop_lag_s=round(best_lag * 1024 / SR, 1),
                loopiness=loopiness)


def symbolic_dur(path, t1, t2):
    sys.path.insert(0, str(ROOT))
    from abc_compare import parse_abc
    meta, vocal, secs = parse_abc(str(path))
    notes = [(o, m, d) for o, m, d in vocal if t1 <= o < t2]
    if len(notes) < 8:
        return None
    durs = np.array([d for _, _, d in notes])
    return dict(note_dur_cv=round(float(durs.std() / durs.mean()), 3),
                notes=len(notes),
                med_note_s=round(float(np.median(durs)), 3))


def analyze_segment(path, tdir, t1, t2, want_symbolic=True):
    y = read_mono_24k(path, t1, t2)
    r = {}
    r.update(f0_metrics(y))
    r.update(dynamics_metrics(y))
    r.update(loopiness(y))
    if want_symbolic and tdir:
        abc = Path(tdir) / "score.abc"
        if abc.exists():
            sym = symbolic_dur(abc, t1, t2)
            if sym:
                r.update(sym)
    return r


def main():
    cache_p = ROOT / "eval" / "ai_segment_cache.json"
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}

    seg_flac = ROOT / "novel_songs" / "final" / "02_平账人.flac"
    seg_tdir = ROOT / "eval" / "novel_transcripts" / "novel_02_平账人"

    target = analyze_segment(seg_flac, seg_tdir, 46.0, 78.0)
    own_verse = analyze_segment(seg_flac, seg_tdir, 21.5, 40.7, want_symbolic=True)
    print("target+own done", flush=True)

    corpus_ch, corpus_v = [], []
    manifest = json.loads((ROOT / "corpus" / "sample_manifest.json").read_text(encoding="utf-8"))
    done = 0
    for m in manifest:
        stem = Path(m["wav"]).stem
        tdir = ROOT / "corpus" / "transcriptions" / stem
        rows = parse_lab(tdir, "structure.lab")
        if not rows:
            continue
        ch = first_span(rows, "chorus", 99)
        vv = first_span(rows, "verse", 99)
        if ch and ch[1] - ch[0] >= 8:
            key = f"ch::{stem}"
            if key not in cache:
                cache[key] = analyze_segment(m["wav"], tdir, ch[0], min(ch[1], ch[0] + 25),
                                             want_symbolic=True)
            corpus_ch.append(cache[key])
            done += 1
            print(f"chorus {done}: {stem[:24]}", flush=True)
        if vv and vv[1] - vv[0] >= 8:
            key = f"vv::{stem}"
            if key not in cache:
                cache[key] = analyze_segment(m["wav"], tdir, vv[0], min(vv[1], vv[0] + 25),
                                             want_symbolic=False)
            corpus_v.append(cache[key])
        if done >= 24:
            break
    cache_p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    def show(title, rows):
        keys = [k for k in target if isinstance(target.get(k), (int, float))]
        print(f"\n## {title} (n={len(rows)})")
        print(f"{'指标':26s} {'目标段':>10s} {'P25':>8s} {'中位':>8s} {'P75':>8s} {'判读':>6s}")
        for k in keys:
            vals = np.array([r[k] for r in rows if r.get(k) is not None], float)
            if len(vals) < 5 or target.get(k) is None:
                continue
            p25, med, p75 = np.percentile(vals, [25, 50, 75])
            tv = target[k]
            inside = p25 <= tv <= p75
            mark = "OK" if inside else "❗偏离"
            print(f"{k:26s} {tv:10.3f} {p25:8.3f} {med:8.3f} {p75:8.3f} {mark:>6s}")

    show("目标段《平账人》46-78s（预副歌+第一副歌）", corpus_ch)
    print(f"\n自身对照-主歌段(21.5-40.7s): {json.dumps(own_verse, ensure_ascii=False)}")
    (ROOT / "eval" / "ai_segment_02.json").write_text(
        json.dumps(dict(target=target, own_verse=own_verse,
                        corpus_chorus=corpus_ch, corpus_verse=corpus_v),
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("SAVED eval/ai_segment_02.json")


if __name__ == "__main__":
    main()
