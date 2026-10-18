# -*- coding: utf-8 -*-
"""从曲库提炼'中国风'子集的音频基准（BPM/调式/亮度），为仙侠专辑重定规格。"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

feats = json.loads(Path(r"F:\yinyue\corpus\features.json").read_text(encoding="utf-8"))

# 已知中国风/仙侠意象曲名关键词（2000-2010 华语中国风浪潮）
KEYS = ["江南", "曹操", "东风破", "发如雪", "青花瓷", "菊花台", "千里之外", "西厢",
        "红尘", "江湖", "水墨", "月光", "仙人", "剑", "飞天", "长安", "烟花",
        "朱砂", "燕", "城府", "断桥", "清明", "三国", "武林", "英雄", "精忠",
        "新贵妃", "逐梦", "牵丝戏", "典狱司", "泸沽寻梦"]

hits = [r for name, r in feats.items() if any(k in name for k in KEYS)]
print(f"中国风子集: {len(hits)} 首")
for r in hits:
    print("  -", r["file"][:30], r["bpm"], r["key"])

bpms = [r["bpm"] for r in hits]
modes = Counter("minor" if "minor" in r["key"] else "major" for r in hits)
cents = [r["centroid_hz"] for r in hits]
keys = Counter(r["key"].split()[0] for r in hits).most_common(6)


def q(vals, p):
    return round(float(np.percentile([v for v in vals if v is not None], p)), 1)


profile = dict(
    n=len(hits),
    bpm_p25=q(bpms, 25), bpm_med=q(bpms, 50), bpm_p75=q(bpms, 75),
    minor_pct=round(modes["minor"] / len(hits) * 100),
    centroid_med=q(cents, 50),
    top_keys=keys,
)
print(json.dumps(profile, ensure_ascii=False, indent=1))
Path(r"F:\yinyue\corpus\guofeng_profile.json").write_text(
    json.dumps(dict(profile, songs=[r["file"] for r in hits]), ensure_ascii=False, indent=1),
    encoding="utf-8")
