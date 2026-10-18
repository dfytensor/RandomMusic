# -*- coding: utf-8 -*-
"""热度-指标相关性（极简版，确保无 None）"""
import json, sys, re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).parent
feats = json.loads((ROOT / "corpus" / "features.json").read_text(encoding="utf-8"))
MEGA_ARTISTS = {"周杰伦","林俊杰","五月天","张学友","刘德华","邓紫棋","陈奕迅","王菲",
                "张惠妹","孙燕姿","梁静茹","S.H.E","王心凌","张韶涵","Tank","胡歌"}
KNOWN_ARTISTS = {"许嵩","汪苏泷","后弦","黄雅莉","张芸京","山野","花儿乐队","信乐团",
                 "183Club","5566","By2","吴克群","薛之谦","毛不易","陈琳","香香","唐磊","汤潮"}
MEGA = {"青花瓷","东风破","菊花台","发如雪","夜曲","七里香","江南","曹操","一千年以后",
        "隐形的翅膀","偏爱","死了都要爱","离歌","突然的自我","知足","温柔","倔强",
        "Super Star","不想长大","恋人未满","遇见","天黑黑","绿光","月亮代表我的心",
        "甜蜜蜜","后来","为爱痴狂","朋友","第一次","勇气","老鼠爱大米","两只蝴蝶",
        "秋天不回来","求佛","香水有毒","该死的温柔","专属天使","北极星的眼泪","我难过",
        "倒带","说爱你","那些花儿","生如夏花","平凡之路","董小姐","成都","南山南","小苹果","最炫民族风"}

def tier(name):
    parts = re.split(r"[-_—]", name)
    artist = parts[0].strip() if parts else ""
    if artist in MEGA_ARTISTS: return 3
    for h in MEGA:
        if h in name: return 3
    if artist in KNOWN_ARTISTS: return 2
    return 1

def safe_corr(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None
             and y == y and not np.isinf(y)]
    if len(pairs) < 30: return None
    a = np.array([p[0] for p in pairs], float)
    b = np.array([p[1] for p in pairs], float)
    return round(float(spearmanr(a, b).statistic), 3)

# 构建数据
keys = ["bpm", "duration_s", "centroid_hz", "rms_db", "dyn_p95_p10_db", "zcr", "rolloff_hz"]
data = {k: {"x": [], "y": []} for k in keys}
data["tier"] = {"x": [], "y": []}

for name, r in feats.items():
    if any(r.get(k) is None or r.get(k) != r.get(k) for k in keys):
        continue
    t = tier(name)
    for k in keys:
        v = r[k]
        if v is None or v != v or np.isinf(v):
            continue
        data[k]["x"].append(t)
        data[k]["y"].append(float(v))
    data["tier"]["x"].append(t)
    data["tier"]["y"].append(float(t))

# 输出
lines = ["# 热度-指标相关性（全库 n=571）", ""]
for k in keys:
    if len(data[k]["x"]) < 50:
        continue
    r = safe_corr(data[k]["x"], data[k]["y"])
    if r is None:
        continue
    s_vals = [data[k]["y"][i] for i, x in enumerate(data[k]["x"]) if x == 3]
    b_vals = [data[k]["y"][i] for i, x in enumerate(data[k]["x"]) if x == 1]
    s_med = np.median(s_vals) if s_vals else 0
    b_med = np.median(b_vals) if b_vals else 0
    mark = "✅" if abs(r) >= 0.12 else ("🟡" if abs(r) >= 0.07 else "❌")
    lines.append(f"- **{k}**: ρ={r:+.3f} {mark} | S层中位={s_med:.1f} B层中位={b_med:.1f}")
lines += ["", "## 热度分层分布", ""]
from collections import Counter
tc = Counter(data["tier"]["x"])
for t in sorted(tc):
    lines.append(f"- Tier {t}: {tc[t]} 首")
(ROOT / "eval" / "popularity_correlation.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
