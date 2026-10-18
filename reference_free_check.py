# -*- coding: utf-8 -*-
"""无参考质量自检（map §7）。用法:
    python reference_free_check.py song1.flac [song2.flac ...] --expected 260 --out report.md
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

warnings.filterwarnings("ignore")
SR = 22050
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def key_of(chroma_mean):
    scores = []
    for rot in range(12):
        c = np.roll(chroma_mean, -rot)
        scores.append((np.corrcoef(c, MAJOR)[0, 1], f"{PITCH[rot]} major"))
        scores.append((np.corrcoef(c, MINOR)[0, 1], f"{PITCH[rot]} minor"))
    return max(scores, key=lambda x: x[0])[1]


def check(path, expected_dur):
    data, sr = sf.read(str(path), always_2d=True)
    mono = data.mean(axis=1)
    clip_ratio = float((np.abs(data) >= 0.99).mean())
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    dur = len(y) / SR
    w = len(y) // 3
    res = dict(file=Path(path).name, duration_s=round(dur, 1), clip_ratio=round(clip_ratio, 6))
    bpms, keys = [], []
    for i in range(3):
        seg = y[i * w:(i + 1) * w]
        tempo, _ = librosa.beat.beat_track(y=seg, sr=SR, units="time")
        bpms.append(float(np.atleast_1d(tempo)[0]))
        chroma = librosa.feature.chroma_stft(y=seg, sr=SR).mean(axis=1)
        keys.append(key_of(chroma / (chroma.max() + 1e-9)))
    res["bpm_windows"] = [float(round(b, 1)) for b in bpms]
    # 倍频归一：把每个窗的 BPM 折到中位数同一八度再比稳定性（拍频检测常见×2/÷2 抖动）
    med_bpm = float(np.median(bpms))
    normed = []
    for b in bpms:
        while b < med_bpm / 1.5:
            b *= 2
        while b > med_bpm * 1.5:
            b /= 2
        normed.append(b)
    res["bpm_stability_pct"] = float(round((max(normed) - min(normed)) / med_bpm * 100, 1))
    res["key_windows"] = keys

    def norm_key(k):
        name, mode = k.split()
        return (PITCH.index(name) + (3 if mode == "minor" else 0)) % 12, mode

    def scale_pcs(k):
        # 大/小调七声音阶 pcs
        name, mode = k.split()
        i = PITCH.index(name)
        steps = [2, 2, 1, 2, 2, 2, 1] if mode == "major" else [2, 1, 2, 2, 1, 2, 2]
        return {(i + sum(steps[:j])) % 12 for j in range(7)}

    from collections import Counter
    scales = [scale_pcs(k) for k in keys]
    overlaps = []
    for a in range(len(scales)):
        for b in range(a + 1, len(scales)):
            overlaps.append(len(scales[a] & scales[b]) / 7)
    res["key_stable"] = bool(overlaps) and (sum(overlaps) / len(overlaps)) >= 5 / 7
    top, cnt = Counter(norm_key(k) for k in keys).most_common(1)[0]
    res["key_majority"] = f"{PITCH[top[0]]} {top[1]} ({cnt}/{len(keys)})"

    S = np.abs(librosa.stft(y[: SR * 90]))
    freqs = librosa.fft_frequencies(sr=SR)
    vocal_band = S[(freqs >= 300) & (freqs <= 3400)].sum()
    res["vocal_band_share"] = round(float(vocal_band / (S.sum() + 1e-9)), 3)
    res["rms_db"] = round(float(librosa.amplitude_to_db(np.sqrt((y ** 2).mean()) + 1e-9)), 1)

    checks = {
        "bpm_stable_≤8%": res["bpm_stability_pct"] <= 8,
        "key_stable": bool(res["key_stable"]),
        "vocal_present_≥0.35": res["vocal_band_share"] >= 0.35,
        "rms_in_-25~-8": -25 <= res["rms_db"] <= -8,
        "no_clipping": clip_ratio < 0.0001,
    }
    if expected_dur:
        ok = abs(dur - expected_dur) / expected_dur <= 0.20
        checks[f"duration_±20%_of_{expected_dur}s"] = bool(ok)
    res["checks"] = checks
    res["pass"] = sum(checks.values())
    res["total"] = len(checks)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--expected", type=float, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = [check(f, a.expected) for f in a.files]
    best = max(rows, key=lambda r: r["pass"] / r["total"])
    lines = ["# 无参考质量自检", "",
             f"- 最佳候选: **{best['file']}** ({best['pass']}/{best['total']} 项通过)", "",
             "| 检查项 | " + " | ".join(r["file"] for r in rows) + " |",
             "|---|" + "---|" * len(rows)]
    all_checks = sorted({k for r in rows for k in r["checks"]})
    for k in all_checks:
        lines.append(f"| {k} | " + " | ".join(("✅" if r["checks"].get(k) else "❌") for r in rows) + " |")
    lines += ["", "## 明细", "", "```json",
              json.dumps([{k: v for k, v in r.items() if k != "checks"} for r in rows],
                         ensure_ascii=False, indent=1,
                         default=lambda o: float(o) if hasattr(o, "item") else str(o)), "```"]
    report = "\n".join(lines)
    print(report)
    if a.out:
        Path(a.out).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
