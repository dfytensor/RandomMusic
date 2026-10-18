# -*- coding: utf-8 -*-
"""按 (BPM, 质心, 动态) KMeans 4 类 → 每类 4 维统计 → 类别目标表
输出 eval/category_targets.md + category_targets.json
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
lib = json.loads((ROOT/"eval"/"lib_metric_cache.json").read_text(encoding="utf-8"))
dims = json.loads((ROOT/"eval"/"library_dims_cache.json").read_text(encoding="utf-8"))

rows = []
for name, d in dims.items():
    if not d or d.get("orn") is None or d.get("gap") is None:
        continue
    lm = lib.get(name, {})
    period = lm.get("period_ms")
    bpm = round(60000/period, 1) if period else None
    rows.append(dict(name=name, bpm=bpm, orn=d["orn"], gap=d["gap"],
                     cent=d["cent"], dyn=d["dyn"]))

X = []
for r in rows:
    X.append([r["bpm"] or 117.5, r["cent"], r["dyn"]])
X = np.array(X)
Xn = (X - X.mean(0)) / (X.std(0) + 1e-9)

# KMeans 4 类
rng = np.random.RandomState(42)
k = 4
centroids = Xn[rng.choice(len(Xn), k, replace=False)]
for _ in range(50):
    dist = ((Xn[:, None, :] - centroids[None]) ** 2).sum(-1)
    lab = dist.argmin(1)
    for j in range(k):
        if (lab == j).any():
            centroids[j] = Xn[lab == j].mean(0)
dist = ((Xn[:, None, :] - centroids[None]) ** 2).sum(-1)
lab = dist.argmin(1)

# 类命名：按 BPM 中位排序 C1(慢)→C4(快)
order = sorted(range(k), key=lambda j: np.median(X[lab == j, 0]))
remap = {j: f"C{i+1}" for i, j in enumerate(order)}

def p25_75(v):
    return f"{np.percentile(v,25):.2f}~{np.percentile(v,75):.2f}"

lines = ["# 曲库 4 维分类统计（571 首全量 → 4 类）", "",
         "| 类 | n | BPM中位 | 质心中位 | 动态中位 | orn/s 中位(P25~75) | gapCV 中位(P25~75) | 抽卡目标 orn | gapCV | cent | dyn |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
targets = {}
for j in range(k):
    cname = remap[j]
    sel = [r for r, l in zip(rows, lab) if l == j]
    orn = np.array([r["orn"] for r in sel]); gap = np.array([r["gap"] for r in sel])
    cent = np.array([r["cent"] for r in sel]); dyn = np.array([r["dyn"] for r in sel])
    bpm = np.array([r["bpm"] or 117.5 for r in sel])
    t = dict(orn=round(float(np.median(orn)), 2), gap=round(float(np.median(gap)), 2),
             cent=round(float(np.median(cent))), dyn=round(float(np.median(dyn)), 1))
    targets[cname] = dict(n=len(sel), bpm_med=round(float(np.median(bpm)),1),
                          bpm_p25_75=p25_75(bpm), targets=t,
                          orn_p25_75=p25_75(orn), gap_p25_75=p25_75(gap),
                          cent_p25_75=f"{np.percentile(cent,25):.0f}~{np.percentile(cent,75):.0f}",
                          dyn_p25_75=f"{np.percentile(dyn,25):.1f}~{np.percentile(dyn,75):.1f}")
    lines.append(f"| {cname} | {len(sel)} | {np.median(bpm):.0f} | {np.median(cent):.0f} | {np.median(dyn):.1f} "
                 f"| {np.median(orn):.2f} ({p25_75(orn)}) | {np.median(gap):.2f} ({p25_75(gap)}) "
                 f"| **{t['orn']}** | **{t['gap']}** | **{t['cent']}** | **{t['dyn']}** |")

lines += ["", "## 用法", "",
          "生成前先按歌的情绪/BPM 定类（C1 慢 → C4 快），",
          "抽卡目标不再用全局中位 (0.09/1.38/2135/14.6)，",
          "而是用该类的 median 作为四维 target 评分择优。"]
(ROOT/"eval"/"category_targets.md").write_text("\n".join(lines), encoding="utf-8")
(ROOT/"eval"/"category_targets.json").write_text(
    json.dumps(targets, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n".join(lines))
