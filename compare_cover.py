# -*- coding: utf-8 -*-
"""Compare original vs cover: audio features + transcription melody similarity."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import librosa

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from abc_compare import parse_abc, to_grid, dtw_similarity, transpose_min_mad, pitch_hist, hist_cosine

ROOT = Path(__file__).parent
SR = 22050
WINDOW = 90.0
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def audio_feats(path):
    import soundfile as sf
    info = sf.info(str(path))
    dur = round(float(info.frames) / info.samplerate, 1)
    y, _ = librosa.load(str(path), sr=SR, mono=True, duration=WINDOW)
    tempo, _ = librosa.beat.beat_track(y=y, sr=SR, units="time")
    bpm = round(float(np.atleast_1d(tempo)[0]), 1)
    chroma = librosa.feature.chroma_stft(y=y, sr=SR).mean(axis=1)
    scores = []
    for rot in range(12):
        c = np.roll(chroma, -rot)
        scores.append((np.corrcoef(c, MAJOR)[0, 1], f"{PITCH[rot]} major"))
        scores.append((np.corrcoef(c, MINOR)[0, 1], f"{PITCH[rot]} minor"))
    key = max(scores, key=lambda x: x[0])[1]
    rms = librosa.feature.rms(y=y)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=SR)[0]
    return dict(duration_s=dur, bpm=bpm, key=key,
                rms_db=round(float(librosa.amplitude_to_db(rms.mean() + 1e-9)), 1),
                centroid_hz=round(float(cent.mean())))


def circle_dist(k1, k2):
    def idx(k):
        name, mode = k.split()
        i = PITCH.index(name)
        i = i + 3 if mode == "minor" else i
        return i % 12
    d = abs(idx(k1) - idx(k2))
    return min(d, 12 - d)


def main(orig_path, cover_path, orig_abc, cover_abc, out_md):
    fo = audio_feats(orig_path)
    fc = audio_feats(cover_path)

    mo, vo, so = parse_abc(orig_abc)
    mc, vc, sc = parse_abc(cover_abc)
    total_o = max(mo["total_sec"], vo[-1][0] + vo[-1][2] if vo else 1)
    total_c = max(mc["total_sec"], vc[-1][0] + vc[-1][2] if vc else 1)
    go = to_grid(vo, total_o)
    gc = to_grid(vc, total_c)
    best = transpose_min_mad(go, gc)
    hist_sim = hist_cosine(pitch_hist(vo), pitch_hist(vc))

    # section boundary alignment: pair occurrences by order
    from collections import defaultdict
    def sec_list(secs, total):
        d = defaultdict(list)
        for n, p in secs:
            d[n].append(round(p / total, 3))
        return d
    smo, smc = sec_list(so, total_o), sec_list(sc, total_c)
    sect_rows = []
    seen = set()
    for name, _ in so:
        if name in seen:
            continue
        seen.add(name)
        if name not in smo:
            continue
        po_list, pc_list = smo[name], smc.get(name, [])
        for k, po in enumerate(po_list):
            pc = pc_list[k] if k < len(pc_list) else None
            err = None if pc is None else round(abs(po - pc) * max(total_o, 1), 1)
            label = name if len(po_list) == 1 else f"{name}#{k+1}"
            sect_rows.append((label, po, pc, err))

    # bpm comparison with octave-normalization
    def bpm_norm(b, ref):
        while b < ref / 1.5:
            b *= 2
        while b > ref * 1.5:
            b /= 2
        return b
    bpm_c_n = bpm_norm(fc["bpm"], fo["bpm"])
    bpm_err = round(abs(bpm_c_n - fo["bpm"]) / fo["bpm"] * 100, 1)

    lines = [
        "# 还原度对比报告：《江南》原曲 vs YuE2 Cover", "",
        f"- 原曲: `{orig_path}`", f"- 生成: `{cover_path}`", "",
        "## 1. 音频特征（librosa，前 90s 窗口）", "",
        "| 指标 | 原曲 | 生成 | 差异 |",
        "|---|---|---|---|",
        f"| 时长 (s) | {fo['duration_s']} | {fc['duration_s']} | {abs(fo['duration_s']-fc['duration_s'])/fo['duration_s']*100:.1f}% |",
        f"| BPM | {fo['bpm']} | {fc['bpm']} (归一后 {bpm_c_n}) | {bpm_err}% |",
        f"| 调性 | {fo['key']} | {fc['key']} | 五度圈距离 {circle_dist(fo['key'], fc['key'])} 步 |",
        f"| RMS (dB) | {fo['rms_db']} | {fc['rms_db']} | - |",
        f"| 频谱质心 (Hz) | {fo['centroid_hz']} | {fc['centroid_hz']} | - |", "",
        "## 2. 旋律还原（双方均 SheetSage2 转谱，Vocal 声部）", "",
        f"- 转谱时长: 原曲 {total_o:.0f}s / 生成 {total_c:.0f}s",
        f"- 最优移调: {best['transpose']} 半音（音准层面已对齐）",
        f"- **对齐帧平均音高误差: {best['mean_abs_semitones']} 半音**",
        f"- **音符帧相似度: {best['note_frame_similarity']*100:.1f}%**",
        f"- 双方都为音符的帧内音准一致率: {best['pitch_acc_on_both_notes']*100:.1f}%",
        f"- 音高直方图余弦相似度: {hist_sim}", "",
        "## 3. 结构对齐（段落起点，占总时长比例）", "",
        "| 段落 | 原曲位置 | 生成位置 | 误差 (s) |",
        "|---|---|---|---|",
    ]
    for name, po, pc, err in sect_rows:
        lines.append(f"| {name} | {po} | {pc if pc is not None else '缺失'} | {err if err is not None else '-'} |")

    verdict = []
    if best["note_frame_similarity"] >= 0.8:
        verdict.append("旋律还原: 达标 (≥0.8)")
    else:
        verdict.append(f"旋律还原: 未达标 ({best['note_frame_similarity']})")
    if bpm_err <= 6:
        verdict.append("速度: 达标 (≤6%)")
    else:
        verdict.append(f"速度: 偏差 {bpm_err}%")
    if abs(fo['duration_s'] - fc['duration_s']) / fo['duration_s'] <= 0.15:
        verdict.append("时长: 达标 (≤15%)")
    cd = circle_dist(fo['key'], fc['key'])
    verdict.append(f"调性: {'达标' if cd <= 1 else '偏差'} (五度圈 {cd} 步)")
    lines += ["", "## 4. 验收结论", "", *[f"- {v}" for v in verdict]]
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(
        r"D:\音乐\林俊杰 - 江南.mp3",
        ROOT / "covers" / "jiangnan" / "cover.flac",
        ROOT / "covers" / "jiangnan" / "score_full" / "score.abc",
        ROOT / "covers" / "jiangnan" / "cover_score" / "score.abc",
        ROOT / "covers" / "jiangnan" / "compare_report.md",
    )
