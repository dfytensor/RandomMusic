# -*- coding: utf-8 -*-
"""全库标定：真曲(568) vs 生成(12) 逐指纹效应量 -> 更新分级 + AI味阈值
-> eval/metric_library_validation.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import librosa

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from ai_flavor_check import METRICS, ai_score, load_reference
from ai_segment_forensic import parse_lab, first_span
from groove_fingerprint import groove_metrics  # noqa
import ai_segment_forensic as asf

cache = json.loads((ROOT / "eval" / "lib_metric_cache.json").read_text(encoding="utf-8"))
corpus = [v for v in cache.values() if not v.get("skip") and v.get("beat_R") is not None
          and v.get("voiced_ratio") is not None]
print(f"corpus valid: {len(corpus)}")

# 生成 12 首（final 版）：beat_R + 指纹
from novel_songs_manifest import SONGS
from hit_score import TRANSCRIPTS
final = ROOT / "novel_songs" / "final"
novel = []
for s in SONGS:
    no = s["no"]
    flac = final / f"{no}_{s['title']}.flac"
    tdir = TRANSCRIPTS[no]
    rows = parse_lab(tdir, "structure.lab")
    ch = first_span(rows, "chorus", 1e9)
    if not ch:
        continue
    y, _ = librosa.load(str(flac), sr=22050, mono=True)
    t2 = min(ch[1], ch[0] + 30, len(y) / 22050)
    seg = y[int(ch[0] * 22050):int(t2 * 22050)]
    g = __import__("groove_fingerprint").groove_metrics(seg)
    from ai_segment_forensic import f0_metrics, dynamics_metrics, loopiness
    f = f0_metrics(seg)
    f.update(dynamics_metrics(seg))
    f.update(loopiness(seg))
    if g:
        f["beat_R"] = g["beat_R"]
        f["period_ms"] = g["period_ms"]
    f["name"] = f"{no} {s['title']}"
    novel.append(f)

rank = {r["no"]: r["score"] for r in json.loads(
    (ROOT / "eval" / "ai_flavor_rank.json").read_text(encoding="utf-8"))}


def d_of(key, sign):
    cv = np.array([r[key] for r in corpus if r.get(key) is not None], float)
    nv = np.array([r[key] for r in novel if r.get(key) is not None], float)
    if len(cv) < 10 or len(nv) < 5:
        return None, None, None
    d = (nv.mean() - cv.mean()) / (np.sqrt(((len(nv)-1)*nv.var()+(len(cv)-1)*cv.var())/(len(nv)+len(cv)-2)) + 1e-9)
    return float(d), float(np.median(cv)), float(np.median(nv))


KEYS = [k for k, _, _ in METRICS] + ["beat_R"]
NAMES = {"voiced_ratio": "人声占比", "n_vocal_runs": "乐句数", "notes": "音符数",
         "vibrato_rate_hz": "颤音速率", "vibrato_extent_cents": "颤音幅度",
         "intra_note_drift_cents": "音内漂移", "macro_dyn_std_db": "宏观动态",
         "dyn_range_db": "动态范围", "loop_lag_s": "循环周期", "beat_R": "唱拍锁死度"}
NAMES["beat_R"] = "唱拍锁死度"

lines = ["# 全库标定结果（真曲 n=568 vs 生成 n=12）", "",
         "| 指标 | 真曲中位 | 生成中位 | Cohen's d | 新级别 |", "|---|---|---|---|---|"]
grades = {}
for k in KEYS:
    sign = next((s for kk, s, _ in METRICS if kk == k), +1)
    d, cm, nm = d_of(k, sign)
    if d is None:
        continue
    g = "A 可靠" if abs(d) >= 0.8 else ("B 方向性" if abs(d) >= 0.5 else "C 仅趋势")
    grades[k] = g
    lines.append(f"| {NAMES.get(k, k)} | {cm:.2f} | {nm:.2f} | {d:+.2f} | {g} |")

# AI味总分重算（真曲分布 n=568 的指纹 vs 生成12）
ref_metrics = corpus
def score_with_lib(metrics):
    total = 0.0
    for key, sign, w in METRICS:
        cv = np.array([r[key] for r in ref_metrics if r.get(key) is not None], float)
        x = metrics.get(key)
        if len(cv) < 10 or x is None:
            continue
        med, iqr = float(np.median(cv)), float(np.percentile(cv, 75) - np.percentile(cv, 25)) + 1e-9
        total += w * min(max(0.0, sign * (x - med) / (0.7413 * iqr)), 6.0)
    return round(total, 2)

ai_scores = []
for r in novel:
    s = score_with_lib(r)
    ai_scores.append(s)
cs_scores = [score_with_lib(r) for r in corpus if r.get("voiced_ratio") is not None]
thr = float(np.percentile(cs_scores, 90))
over = sum(1 for x in ai_scores if x > thr)
lines += ["", "## 大样本标定的 AI 味分（全库指纹分布）", "",
          f"- 真曲：中位 {np.median(cs_scores):.2f} · P90 **{thr:.2f}** · 最大 {max(cs_scores):.2f}",
          f"- 生成 12 首：中位 {np.median(ai_scores):.2f} · 超阈值 {over}/12",
          "", "### 生成曲逐首（AI味分 / 唱拍锁死度）", ""]
for r, s in zip(novel, ai_scores):
    lines.append(f"- {r['name']}: AI味 {s} · beat_R {r.get('beat_R')}")
(ROOT / "eval" / "metric_library_validation.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
