# -*- coding: utf-8 -*-
"""节奏唱拍指纹：副歌段人声起音相对节拍网格的偏离分布
真歌手：系统性推/拉（球状偏移30-80ms）；AI：死扣网格或均匀散布
-> eval/groove_report.md
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import librosa
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS
from ai_segment_forensic import parse_lab, first_span
from hit_score import TRANSCRIPTS

SR = 22050


def groove_metrics(y_seg, sr=SR):
    tempo, beats = librosa.beat.beat_track(y=y_seg, sr=sr, units="time", hop_length=512)
    beat_t = np.atleast_1d(beats)
    if len(beat_t) < 8:
        return None
    period = 60.0 / float(np.atleast_1d(tempo)[0])
    ons = librosa.onset.onset_detect(y=y_seg, sr=sr, hop_length=256, units="time",
                                     delta=0.06, backtrack=False)
    if len(ons) < 10:
        return None
    phases = np.array([((o - beat_t[0]) % period) / period for o in ons])
    ang = phases * 2 * np.pi
    R = float(abs(np.exp(1j * ang).mean()))
    return dict(beat_R=round(R, 3), period_ms=round(period * 1000))


def groove(path, t1, t2):
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    seg = y[int(t1 * SR):int(t2 * SR)]
    # 人声频带 300-3400Hz
    sos = butter(4, [300 / (SR / 2), 3400 / (SR / 2)], btype="band", output="sos")
    voc = sosfilt(sos, seg)
    # 节拍网格
    tempo, beats = librosa.beat.beat_track(y=seg, sr=SR, units="time", hop_length=512)
    beat_t = np.atleast_1d(beats)
    if len(beat_t) < 8:
        return None
    period = 60.0 / float(np.atleast_1d(tempo)[0])
    # 人声起音
    ons = librosa.onset.onset_detect(y=voc, sr=SR, hop_length=256, units="time",
                                     delta=0.06, backtrack=False)
    if len(ons) < 10:
        return None
    # 节拍相位圆统计：集中度 R（0=散，1=锁死）+ 推拉方向
    period = period if period else 0.5
    phases = np.array([((o - beat_t[0]) % period) / period for o in ons])
    ang = phases * 2 * np.pi
    R = float(np.abs(np.exp(1j * ang).mean()))
    mean_dir = float(np.angle(np.exp(1j * ang).mean()) / (2 * np.pi) % 1)
    return dict(n_onsets=len(ons),
                beat_R=round(R, 3),
                phase_dir=round(mean_dir, 3),
                period_ms=round(period * 1000))


def main():
    final = ROOT / "novel_songs" / "final"
    novel = {}
    for s in SONGS:
        no = s["no"]
        flac = final / f"{no}_{s['title']}.flac"
        tdir = TRANSCRIPTS[no]
        rows = parse_lab(tdir, "structure.lab")
        ch = first_span(rows, "chorus", 1e9)
        if not ch:
            continue
        r = groove(flac, ch[0], min(ch[1], ch[0] + 30))
        if r:
            novel[f"{no}_{s['title']}"] = r
            print(no, s["title"], json.dumps(r), flush=True)

    corpus = []
    manifest = json.loads((ROOT / "corpus" / "sample_manifest.json").read_text(encoding="utf-8"))
    for m in manifest:
        stem = Path(m["wav"]).stem
        rows = parse_lab(ROOT / "corpus" / "transcriptions" / stem, "structure.lab")
        ch = first_span(rows, "chorus", 99)
        if ch and ch[1] - ch[0] >= 8:
            r = groove(m["wav"], ch[0], min(ch[1], ch[0] + 30))
            if r:
                corpus.append(r)
    print(f"corpus n={len(corpus)}")

    keys = ["beat_R", "phase_dir", "period_ms"]
    lines = ["# 节奏唱拍指纹（副歌段）", "",
             "| 指标 | 曲库P25 | 中位 | P75 | " + " | ".join(list(novel)[:4]) + " |", ""]
    table = ["| 指标 | 曲库中位(IQR) | " + " | ".join(list(novel)) + " |",
             "|---|" + "---|" * (len(novel) + 1)]
    for k in keys:
        cv = np.array([r[k] for r in corpus if r.get(k) is not None], float)
        row = f"| {k} | {np.median(cv):.1f} ({np.percentile(cv,25):.0f}-{np.percentile(cv,75):.0f}) |"
        for name, r in novel.items():
            row += f" {r.get(k)} |"
        table.append(row)
    lines += table
    (ROOT / "eval" / "groove_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(table))


if __name__ == "__main__":
    main()
