# -*- coding: utf-8 -*-
"""关联: articulation 指标 × v3评分 (全 AI 样本) — 新维度是否已被 v3 覆盖/有增量"""
import json, warnings
from pathlib import Path
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

rescan = json.loads((ROOT/"eval"/"rescan_v3_cache.json").read_text(encoding="utf-8"))
# stem -> v3 score
score_by_stem = {}
for rel, v in rescan.items():
    if v is None: continue
    stem = Path(rel).stem
    score_by_stem[stem] = v["score"]  # 同 seed 名在不同池可能是同歌不同渲染, 用 stem 匹配

def load(tag):
    p = ROOT/"eval"/f"{tag}"
    if not p.exists(): return {}
    return json.loads(p.read_text(encoding="utf-8"))

art = {}
for tag in ["articulation2_cache.json", "articulation3_cache.json",
            "articulation4_cache.json", "articulation5_cache.json"]:
    d = load(tag)
    for k, v in d.items():
        if not v: continue
        # key 形如 .../<seed>/vocals.wav 或 .../<池>/<seed>.flac -> 用 seed 目录名
        parts = k.split("/")
        stem = parts[-2] if parts[-1].startswith("vocals") else Path(k).stem
        art.setdefault(stem, {}).update(v or {})

# 配对
rows = []
for stem, m in art.items():
    if stem in score_by_stem and any(k in m for k in ["vib_present_ratio", "phase_R", "onset_rise_db"]):
        rows.append((score_by_stem[stem], m))
print(f"paired n={len(rows)}", flush=True)

metrics = ["vib_present_ratio", "vib_depth_med", "phase_R", "onset_rise_db",
           "long_smooth_db_per_frame", "long_trend_frac", "phrase_cv_med",
           "ioi_entropy_norm", "rubato_cv", "dur_cv", "long_ratio_100"]
print(f"{'metric':28s} {'Spearman ρ':>10s} {'p':>8s}")
out = {}
for k in metrics:
    pairs = [(sc, m[k]) for sc, m in rows if m.get(k) is not None]
    if len(pairs) >= 8:
        a, b = zip(*pairs)
        rho, p = stats.spearmanr(a, b)
        out[k] = {"rho": float(rho), "p": float(p), "n": len(pairs)}
        mark = "*" if p < 0.05 else " "
        print(f"{mark} {k:27s} {rho:+.3f} {p:8.4f}  n={len(pairs)}")

(ROOT/"eval"/"articulation_v3_corr.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("SAVED eval/articulation_v3_corr.json")
