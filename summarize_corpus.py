# -*- coding: utf-8 -*-
"""Summarize corpus features -> music production map."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

feats = json.loads(Path(r"F:\yinyue\corpus\features.json").read_text(encoding="utf-8"))
rows = list(feats.values())
PITCH_ORDER = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def key_root(key):
    return PITCH_ORDER.index(key.split()[0]) + (0 if "major" in key else 0)


def quartiles(vals):
    a = np.array(vals)
    return dict(min=round(a.min(), 1), p25=round(np.percentile(a, 25), 1),
                med=round(np.percentile(a, 50), 1), p75=round(np.percentile(a, 75), 1),
                max=round(a.max(), 1))


def mode_share(vals):
    c = Counter(vals)
    return [(k, v, f"{v/len(vals)*100:.0f}%") for k, v in c.most_common(12)]


bpms = [r["bpm"] for r in rows]
durs = [r["duration_s"] for r in rows]
cents = [r["centroid_hz"] for r in rows]
dyns = [r["dyn_p95_p10_db"] for r in rows]
rmss = [r["rms_db"] for r in rows]
keys = mode_share([r["key"] for r in rows])
modes = Counter(("minor" if "minor" in r["key"] else "major") for r in rows)

# tempo classes
slow = sum(1 for b in bpms if b < 75)
mid = sum(1 for b in bpms if 75 <= b < 110)
fast = sum(1 for b in bpms if b >= 110)

# key histogram (circle of fifths distance from C)
key_hist = Counter(r["key"] for r in rows)

report = f"""# 语料分析报告（{len(rows)} 首，D:\\音乐）
## 时长
{json.dumps(quartiles(durs), ensure_ascii=False)}
## BPM
分布: <75: {slow} ({slow/len(rows)*100:.0f}%) | 75-110: {mid} ({mid/len(rows)*100:.0f}%) | >=110: {fast} ({fast/len(rows)*100:.0f}%)
{json.dumps(quartiles(bpms), ensure_ascii=False)}
## 调性 Top12
{chr(10).join(f'- {k}: {v} ({s})' for k, v, s in keys)}
大小调: major={modes['major']} ({modes['major']/len(rows)*100:.0f}%) / minor={modes['minor']} ({modes['minor']/len(rows)*100:.0f}%)
## 频谱质心 (Hz)
{json.dumps(quartiles(cents), ensure_ascii=False)}
## 动态范围 p95-p10 (dB)
{json.dumps(quartiles(dyns), ensure_ascii=False)}
## 平均响度 RMS (dB)
{json.dumps(quartiles(rmss), ensure_ascii=False)}
"""
Path(r"F:\yinyue\corpus\analysis_summary.md").write_text(report, encoding="utf-8")
print(report)

jn = feats.get("林俊杰 - 江南.mp3")
print("江南 features:", json.dumps(jn, ensure_ascii=False))
