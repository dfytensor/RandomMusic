# -*- coding: utf-8 -*-
"""Prompt A/B 评测：5 个变体 × 9 指标 → 找出哪些 prompt 词真正改变输出"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
sys.path.insert(0, str(ROOT))
from netease_score import compute_netease_metrics, score_netease

work = ROOT / "novel_songs" / "v2" / "prompt_ab"
tags = ["A_基线", "B_临界", "C_非线性", "D_呼吸", "E_爆发"]

all_metrics = {}
for tag in tags:
    f = work / f"{tag}.flac"
    if not f.exists(): continue
    y, _ = librosa.load(str(f), sr=SR, mono=True, duration=100)
    m = compute_netease_metrics(str(f), SR)
    all_metrics[tag] = m

# 打印对比表
print("=== Prompt A/B 对比 ===\n")
metric_keys = ["breath_dips", "centroid_hz", "dyn_range_db", "onset_iv_cv", "zcr", "rms_db"]
metric_names = ["呼吸间隙", "质心Hz", "动态dB", "起音CV", "过零率", "RMS"]

print(f"{'指标':12s}", end="")
for tag in tags:
    print(f"  {tag:8s}", end="")
print()

for mk, mn in zip(metric_keys, metric_names):
    print(f"{mn:12s}", end="")
    for tag in tags:
        m = all_metrics.get(tag, {})
        print(f"  {m.get(mk,'?'):8}", end="")
    print()

# 差异分析
print("\n=== 差异分析 ===")
base = all_metrics.get("A_基线", {})
for tag in tags[1:]:
    m = all_metrics.get(tag, {})
    diffs = []
    for mk, mn in zip(metric_keys, metric_names):
        bv = base.get(mk, 0)
        mv = m.get(mk, 0)
        if bv and mv and bv != mv:
            diff = mv - bv
            pct = round(diff / abs(bv) * 100, 1) if bv else 0
            diffs.append(f"{mn}:{mv}({pct:+.0f}%)")
    print(f"{tag}: {', '.join(diffs)}")
