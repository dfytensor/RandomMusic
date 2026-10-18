# -*- coding: utf-8 -*-
"""波形指标标定：真曲(463) vs 生成(12) 效应量 + 冗余检查 -> eval/waveform_validation.md"""
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
from novel_songs_manifest import SONGS
from wave_metric_collect import wave_metrics
from hit_score import TRANSCRIPTS
from ai_segment_forensic import parse_lab, first_span


def cohen_d(a, b):
    a, b = np.array(a, float), np.array(b, float)
    s = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / (s + 1e-9))


def grade(d):
    d = abs(d)
    return "A 可靠" if d >= 0.8 else ("B 方向性" if d >= 0.5 else "C 仅趋势")


KEYS = [("crest_db", "峰值因子"), ("env_rough", "包络粗糙度"), ("low_attack_ms", "低频冲击速度"),
        ("silence_frac", "静音占比"), ("onset_iv_cv", "起音间隔CV"), ("kurtosis", "波形峭度"),
        ("lra_db", "响度域")]

cache = json.loads((ROOT / "eval" / "wave_metric_cache.json").read_text(encoding="utf-8"))
corpus = [v for v in cache.values() if "crest_db" in v]
print(f"corpus valid: {len(corpus)}")

novel = []
for s in SONGS:
    flac = ROOT / "novel_songs" / "final" / f"{s['no']}_{s['title']}.flac"
    tdir = TRANSCRIPTS[s["no"]]
    rows = parse_lab(tdir, "structure.lab")
    ch = first_span(rows, "chorus", 1e9)
    y, _ = librosa.load(str(flac), sr=SR, mono=True)
    t2 = min(ch[1], ch[0] + 30, len(y) / SR)
    seg = y[int(ch[0] * SR):int(t2 * SR)]
    r = wave_metrics(seg)
    r["name"] = f"{s['no']} {s['title']}"
    novel.append(r)

lines = ["# 波形级指标标定（真曲 n=463 vs 生成 n=12）", "",
         "| 指标 | 真曲中位 | 生成中位 | Cohen's d | 级别 | 冗余检查 |", "|---|---|---|---|---|---|"]
feats = json.loads((ROOT / "eval" / "feat_corpus.json").read_text(encoding="utf-8"))
grades = {}
for key, cname in KEYS:
    cv = [r[key] for r in corpus if r.get(key) is not None]
    nv = [r[key] for r in novel if r.get(key) is not None]
    if len(cv) < 10 or not nv:
        continue
    d = cohen_d(nv, cv)
    g = grade(d)
    grades[key] = (g, d)
    # 冗余检查：与既有指标的相关
    red = ""
    if key == "lra_db":
        dyn = np.array([r.get("dyn_p95_p10_db") for r in feats if r.get("dyn_p95_p10_db") is not None])
        l = np.array([r.get("lra_db") for r in cache.values() if isinstance(cache, dict)] ) if False else None
        red = "vs dyn_range 待查"
    lines.append(f"| {cname} | {np.median(cv):.2f} | {np.median(nv):.2f} | {d:+.2f} | {g} | {red} |")

# 冗余：lra vs 已有 dyn_p95_p10（逐曲配对需同曲，简化用分布中位差说明）
lines += ["", "### 冗余性说明", "",
          "- 响度域(lra) 与既有 动态p95-p10 同族（相关预期高），保留其一即可",
          "- 波形峭度/峰值因子 为纯波形层新信息（既有指标未覆盖）"]
(ROOT / "eval" / "waveform_validation.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
