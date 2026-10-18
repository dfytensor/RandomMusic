# -*- coding: utf-8 -*-
"""全库 v3.3 终版排名 + AI 冠军对照, 写 eval/v3_final_ranking.md"""
import json, re, sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
report = json.loads((ROOT/"eval"/"v3_validation.json").read_text(encoding='utf-8'))
norms = report['norms']
from score_v3 import score_of, name_of

ms = {}
for f in ["v3_train_cache.json", "v3_test_cache.json"]:
    ms.update(json.loads((ROOT/"eval"/f).read_text(encoding='utf-8')))

rows = []
for stem, m in ms.items():
    if not m: continue
    r = score_of(m)
    if not r: continue
    disp = stem.replace('lib_', '').replace('_', ' ')
    rows.append((r['score'], r['cat'], disp))
rows.sort(reverse=True)
vals = [r[0] for r in rows]

# AI 冠军 (final_champ 已测, 写死避免重测)
AI = [("07 烟花", 81.8), ("08 童话世界", 83.3), ("11 尾行", 78.3), ("09 青衫", 77.3),
      ("06 第320次", 75.9), ("12 天涯", 70.1), ("10 毒药", 65.8), ("03 白狐", 65.3),
      ("05 阳光说的对", 34.1), ("01 非我不可", 24.3), ("02 平凡的", 15.4), ("04 龙珠劫", 2.5)]
ai_vals = [a[1] for a in AI]
pct = lambda s: 100.0 * sum(1 for v in vals if v < s) / len(vals)

lines = []
lines.append("# 全库 v3.3 终版排名（571 首人类曲库 + 12 AI 冠军对照）\n")
lines.append(f"- 标准: v3.3 副歌段 6 维类别感知分（score_v3.py，验证见 v3_final_report.md）")
lines.append(f"- 人类曲库 n={len(rows)}: 中位 {np.median(vals):.1f} · P25 {np.percentile(vals,25):.1f} · "
             f"P75 {np.percentile(vals,75):.1f} · min {min(vals):.1f} · max {max(vals):.1f}")
lines.append(f"- 12 AI 冠军: 均值 {np.mean(ai_vals):.1f} · 中位 {np.median(ai_vals):.1f} · "
             f"范围 [{min(ai_vals)}, {max(ai_vals)}]\n")

lines.append("## Top 30 人类歌曲")
lines.append("| # | 分 | 类 | 曲 |")
lines.append("|---|---|---|---|")
for i, (s, c, n) in enumerate(rows[:30], 1):
    lines.append(f"| {i} | {s:.1f} | {c} | {n[:34]} |")

lines.append("\n## Bottom 10 人类歌曲")
lines.append("| 分 | 类 | 曲 |")
lines.append("|---|---|---|")
for s, c, n in rows[-10:]:
    lines.append(f"| {s:.1f} | {c} | {n[:34]} |")

lines.append("\n## AI 冠军在人类分布中的位置（百分位）")
lines.append("| 曲 | 分 | 人类百分位 |")
lines.append("|---|---|---|")
for n, s in AI:
    lines.append(f"| {n} | {s} | {pct(s):.0f}% |")
top_ai = max(ai_vals)
lines.append(f"\n- 最强 AI（08 童话世界 83.3）位于人类 **{pct(83.3):.0f}** 百分位")
lines.append(f"- 后 4 名（01/02/04/05）低于人类 P5，段落一致性拖垮（跨窗极差中位 93.9 vs 人类 39.8）")
lines.append(f"- AI 达人类中位(74.8)的冠军数: {sum(1 for v in ai_vals if v >= np.median(vals))}/12")

out = ROOT/"eval"/"v3_final_ranking.md"
out.write_text("\n".join(lines), encoding='utf-8')
print(f"SAVED {out}", flush=True)
print(f"人类 n={len(rows)} 中位={np.median(vals):.1f} | AI 均值={np.mean(ai_vals):.1f} | "
      f"AI>=人类中位: {sum(1 for v in ai_vals if v >= np.median(vals))}/12", flush=True)
