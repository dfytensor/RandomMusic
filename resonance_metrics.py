# -*- coding: utf-8 -*-
"""共鸣四指标：颤音变化度/长音漂移/呼吸间隙/级进铺垫比
全库采集+标定 -> eval/resonance_metrics_report.md
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
sys.path.insert(0, str(ROOT))
from ai_segment_forensic import parse_lab, first_span


def resonance_metrics(y):
    """副歌段音频 -> 四个共鸣指纹"""
    f0, vflag, vprob = librosa.pyin(y, fmin=65, fmax=700, sr=SR,
                                    frame_length=1024, hop_length=256)
    ft = np.arange(len(f0)) * 256 / SR
    voiced = vflag & ~np.isnan(f0)
    cents = 1200 * np.log2(f0 / 55.0)
    win = max(3, int(round(0.1 * SR / 256)))

    # 收集浊音段
    runs, start = [], None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if ft[i - 1] - ft[start] >= 0.3:
                runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(voiced)))

    vib_rates, vib_exts, drifts = [], [], []
    for a, b in runs:
        seg_c = cents[a:b] - np.nanmedian(cents[a:b])
        if np.nanstd(seg_c) < 15:
            continue
        slow = np.convolve(seg_c, np.ones(win) / win, mode="same")
        fast = seg_c - slow
        spec = np.abs(np.fft.rfft(np.nan_to_num(fast)))
        freqs = np.fft.rfftfreq(len(fast), d=256 / SR)
        band = (freqs >= 3) & (freqs <= 12)
        if band.sum() == 0:
            continue
        pk = freqs[band][np.argmax(spec[band])]
        vib_rates.append(pk)
        vib_exts.append(float(np.nanstd(fast)))
        drifts.append(float(np.nanstd(slow)))

    # 指纹 1: 颤音速率变化度 (CV)
    vib_cv = round(float(np.std(vib_rates) / np.mean(vib_rates)), 3) if len(vib_rates) >= 3 else None

    # 指纹 2: 长音漂移 (≥0.5s 段的最大慢漂移)
    long_drifts = []
    for a, b in runs:
        if b - a >= int(0.5 * SR / 256):
            seg_c = cents[a:b] - np.nanmedian(cents[a:b])
            slow = np.convolve(seg_c, np.ones(win * 2) / (win * 2), mode="same")
            long_drifts.append(float(np.nanstd(slow)))
    long_drift = round(max(long_drifts), 1) if long_drifts else None

    # 指纹 3: 呼吸间隙（动态包络中 >3dB 的下沉次数）
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    w3 = int(3 * SR / 512)
    smooth = np.convolve(db, np.ones(12) / 12, mode="same")
    dips = 0
    for i in range(w3, len(smooth) - w3):
        window = smooth[i - w3:i + w3]
        if smooth[i] == window.min() and (window.max() - window.min()) > 3:
            if i == w3 or smooth[i] <= smooth[i - w3 // 2]:
                dips += 1

    # 指纹 4: 级进铺垫比（首个≥5度跳前的连续级进步数 / 总步数）
    pitches = []
    for a, b in runs:
        seg_c = cents[a:b]
        for i in range(len(seg_c)):
            if not np.isnan(seg_c[i]):
                pitches.append(seg_c[i])
    leaps_prep = None
    if len(pitches) > 10:
        intervals = np.abs(np.diff(pitches))
        steps = [x for x in intervals if x < 0.5]  # 去除静音跳变
        leaps_before = 0
        for i, iv in enumerate(intervals):
            if iv >= 5:
                leaps_before = i
                break
        leaps_prep = round(leaps_before / max(len(intervals), 1), 3) if intervals.any() else None

    return dict(vibrato_cv=vib_cv, long_drift=long_drift,
                breath_dips=dips, leap_prep=leaps_prep)


def parse_lab_safe(tdir):
    rows = []
    p = Path(tdir) / "structure.lab"
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        q = line.split("\t")
        if len(q) >= 3:
            rows.append((float(q[0]), float(q[1]), q[2].strip().lower()))
    return rows


def first_chorus(rows, limit=99):
    for t1, t2, n in rows:
        if n == "chorus" and t1 < limit - 10:
            return t1, min(t2, limit)
    return None


def main():
    # 曲库采集
    lib_wav = ROOT / "eval" / "library_wavs"
    lib_tr = ROOT / "eval" / "library_transcripts"
    cache_p = ROOT / "eval" / "resonance_cache.json"
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}

    import time
    t0 = time.time()
    corpus = []
    wavs = sorted(lib_wav.glob("*.wav"))
    for i, w in enumerate(wavs):
        if w.stem in cache:
            corpus.append(cache[w.stem])
            continue
        rows = parse_lab_safe(lib_tr / w.stem)
        ch = first_chorus(rows, 99)
        if not ch or ch[1] - ch[0] < 8:
            cache[w.stem] = {"skip": True}
            continue
        try:
            y, _ = librosa.load(str(w), sr=SR, mono=True)
            t2 = min(ch[1], ch[0] + 30)
            seg = y[int(ch[0] * SR):int(t2 * SR)]
            r = resonance_metrics(seg)
            cache[w.stem] = r
            corpus.append(r)
        except Exception:
            cache[w.stem] = {"skip": True}
        if (i + 1) % 30 == 0:
            cache_p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            print(f"[{i+1}/{len(wavs)}] el={time.time()-t0:.0f}s", flush=True)
    cache_p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"corpus done: {len([c for c in corpus if c.get('vibrato_cv') is not None])}", flush=True)

    # 生成 12 首
    from novel_songs_manifest import SONGS
    from hit_score import TRANSCRIPTS
    novel = []
    for s in SONGS:
        flac = ROOT / "novel_songs" / "final" / f"{s['no']}_{s['title']}.flac"
        tdir = TRANSCRIPTS[s["no"]]
        rows = parse_lab_safe(tdir)
        ch = first_chorus(rows, 300)
        if not ch or ch[1] - ch[0] < 8:
            continue
        try:
            y, _ = librosa.load(str(flac), sr=SR, mono=True)
            t2 = min(ch[1], ch[0] + 30)
            seg = y[int(ch[0] * SR):int(t2 * SR)]
            r = resonance_metrics(seg)
            r["name"] = f"{s['no']} {s['title']}"
            novel.append(r)
            print(f"novel {no if False else s['no']}: vib_cv={r['vibrato_cv']} drift={r['long_drift']} dips={r['breath_dips']} prep={r['leap_prep']}", flush=True)
        except Exception as e:
            print(f"novel FAIL {s['no']}: {str(e)[:60]}", flush=True)

    # ===== 标定 =====
    def d_of(key):
        cv = np.array([r[key] for r in corpus if r.get(key) is not None], float)
        nv = np.array([r[key] for r in novel if r.get(key) is not None], float)
        if len(cv) < 10 or len(nv) < 3:
            return None, None, None
        d = (nv.mean() - cv.mean()) / (np.sqrt(((len(nv)-1)*nv.var()+(len(cv)-1)*cv.var())/(len(nv)+len(cv)-2)) + 1e-9)
        return d, float(np.median(cv)), float(np.median(nv))

    defs = [
        ("vibrato_cv", "颤音速率变化度", "越高越活"),
        ("long_drift", "长音漂移(音分)", "越高越人味"),
        ("breath_dips", "呼吸间隙数", "越多越自然"),
        ("leap_prep", "级进铺垫比", "越高越有爆发准备"),
    ]
    lines = ["# 共鸣四指标标定（真曲 n≥30 vs 生成 n=12）", "",
             "| 指标 | 真曲中位 | 真曲IQR | 生成中位 | d | 级别 |", "|---|---|---|---|---|---|"]
    grades = {}
    for key, cname, meaning in defs:
        cv = [r[key] for r in corpus if r.get(key) is not None]
        nv = [r[key] for r in novel if r.get(key) is not None]
        if not cv or not nv:
            continue
        p25, med, p75 = np.percentile(cv, [25, 50, 75])
        d, _, _ = d_of(key) or (None, None, None)
        if d is None:
            d = 0
        g = "A 可靠" if abs(d) >= 0.8 else ("B 方向性" if abs(d) >= 0.5 else "C 仅趋势")
        grades[key] = g
        lines.append(f"| {cname} | {np.median(cv):.2f} | [{p25:.2f}-{p75:.2f}] | {np.median(nv):.2f} | {d:+.2f} | {g} |")
    lines += ["", "## 逐曲（生成 12 首）", ""]
    for r in novel:
        lines.append(f"### {r.get('name', '?')}")
        for key, cname, _ in defs:
            v = r.get(key)
            if v is not None:
                lines.append(f"- {cname}: {v}")
    (ROOT / "eval" / "resonance_metrics_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
