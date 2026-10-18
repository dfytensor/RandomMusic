# -*- coding: utf-8 -*-
"""对照: 同一共识组在 v3.3(6维) 缓存 vs v3.5(11维) 缓存下的 U 检验 + 每维方向"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).parent

CLASSIC = ["真的爱你","光辉岁月","夜曲","听妈妈的话","安静","我难过","千年泪",
           "好想好想","双截棍","回到过去","开不了口","发如雪","彩虹","稻香",
           "威廉古堡","外婆","情非得已","海阔天空","专属天使","如果我变成回忆",
           "热带雨林","绿光","遇见","倔强","知足"]
MEDIOCRE = ["爱丫爱丫","发呆","樱花草","光荣","活埋","会长大的幸福",
            "紫藤花","一辈子的孤单","猪之歌","三国杀",
            "没有你在我有多难熬","向往","我知","真爱","会长大的幸福",
            "双侠","断桥残雪","灰色头像","玫瑰花的葬礼","城府"]

def match(pats, keys):
    out = []
    for p in pats:
        hits = [k for k in keys if p in k]
        if hits: out.append((hits[0], p))
    return out

v33 = json.loads((ROOT/"eval/v33_window_scores.json").read_text(encoding="utf-8"))
v35 = json.loads((ROOT/"eval/v35_window_scores.json").read_text(encoding="utf-8"))

for tag, cache in [("v3.3(6维)", v33), ("v3.5(11维)", v35)]:
    cl, md = [], []
    for grp, names in [(cl, CLASSIC), (md, MEDIOCRE)]:
        for st, p in match(names, cache.keys()):
            scs = [v for v in cache[st].values() if v is not None]
            if scs: grp.append(float(np.median(scs)))
    u, p = mannwhitneyu(cl, md, alternative="greater")
    print(f"{tag}: 经典n={len(cl)} med={np.median(cl):.1f} 普通n={len(md)} med={np.median(md):.1f} p={p:.4f}")

# 每维方向: 全库 train 条目里, 经典 vs 普通 的维中位 (来自 art_cache+train_cache)
train = json.loads((ROOT/"eval/v3_train_cache.json").read_text(encoding="utf-8"))
art = json.loads((ROOT/"eval/v3_art_cache.json").read_text(encoding="utf-8"))
import re
def norm_name(s):
    s = s.replace("lib_", ""); s = re.sub(r"[_]+", " ", s)
    return re.sub(r"\s+", "", s).lower()
keys_norm = {norm_name(k): k for k in train.keys()}
DIMS = ["orn","gap","cent","dyn","tension","micro","vib_ratio","vib_depth","phase_R","onset_rise_db","long_smooth"]
def vals(names):
    out = {d: [] for d in DIMS}
    for p in names:
        hits = [k for k in keys_norm if p.lower().replace(" ","") in k]
        if not hits: continue
        k = keys_norm[hits[0]]
        m = dict(train[k]); m.update(art.get(k, {}))
        for d in DIMS:
            if m.get(d) is not None: out[d].append(m[d])
    return out
cl_v, md_v = vals(CLASSIC), vals(MEDIOCRE)
print("\n每维 (经典中位 vs 普通中位, MW p):")
for d in DIMS:
    a, b = cl_v[d], md_v[d]
    if len(a) >= 3 and len(b) >= 3:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        print(f"  {d:14s} 经典={np.median(a):8.3f} n={len(a):2d}  普通={np.median(b):8.3f} n={len(b):2d}  p={p:.4f}")
