# -*- coding: utf-8 -*-
"""好听预测器 v0：用你的耳朵当标签，从现有特征找"好听"的声学关联
-> eval/taste_analysis.md
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent

# 你的打分（1-5 分，5=好听）
# ← 这里就是你做的每一轮人耳终审
RATINGS = {
    "08_月供日": 4,        # 你说"可以"
    "06_第320代记账人": 4, # 你说"可以"
    "01_欠天地一笔": 3,    # 你说"不行"后重做了，暂定 3
    "02_平账人": 2,        # "一般"
    "03_白户": 2,
    "04_剁手魔": 2,
    "05_心疼说成省钱": 2,
    "07_规矩": 2,
    "09_断供": 2,
    "10_糕与药包": 2,
    "11_烂尾劫": 2,
    "12_新契": 2,
}

# 对比参考：曲库中你收藏的歌（你听着好听才会收进 D:\音乐）
# 这些歌你没打分，但你收了 = 你觉得好听 = 分数 4-5
# 而你没收的同期歌曲 = 你觉得一般 = 分数 1-2
# ← 这就是天然的标签来源！

# 从曲库中你**没有**收集的歌（同艺人同时代）做负样本
# ← 这个需要联网查，暂时先跳过

# ===== 加载我们已有的所有特征 =====
features = {}
for r in json.loads((ROOT / "eval" / "feat_novel.json").read_text(encoding="utf-8")):
    features[r["file"]] = r

for r in json.loads((ROOT / "eval" / "feat_corpus.json").read_text(encoding="utf-8")):
    features[r["file"]] = r

# 加载旋律指标
mel_cache = json.loads((ROOT / "eval" / "wave_metric_cache.json").read_text(encoding="utf-8"))
fp_cache = json.loads((ROOT / "eval" / "lib_metric_cache.json").read_text(encoding="utf-8"))

# ===== 对已打分的歌，输出全部特征 =====
lines = ["# 好听预测器 v0（基于你的耳朵标签）", ""]
lines += ["## 你已打分的歌", ""]
for name, rating in RATINGS.items():
    stem = name + ".flac"
    feat_key = next((k for k in features if name in k or k in name), None)
    if feat_key:
        f = features[feat_key]
        lines.append(f"### {name}（你的评分: {rating}/5）")
        for k in sorted(f.keys()):
            if k not in ("file",):
                lines.append(f"- {k}: {f[k]}")
        lines.append("")

# ===== 找区分特征 =====
# 你打了 4 分的（好听）vs 2 分的（一般）→ 差异就是"好听"的声学特征
good_keys = [k for k, v in RATINGS.items() if v >= 4]
bad_keys = [k for k, v in RATINGS.items() if v <= 2]

def cohen_d(a, b):
    a, b = np.array(a, float), np.array(b, float)
    s = np.sqrt(((len(a)-1)*a.var() + (len(b)-1)*b.var()) / (len(a)+len(b)-2))
    return float((a.mean() - b.mean()) / (s + 1e-9))

# 收集好/坏歌曲的所有共同特征
def find_feat(name_pattern):
    for k, v in features.items():
        if name_pattern[:8] in k:
            return v
    return None

if good_keys and bad_keys:
    good_feats = [find_feat(k) for k in good_keys if find_feat(k)]
    bad_feats = [find_feat(k) for k in bad_keys if find_feat(k)]
    good_feats = [f for f in good_feats if f]
    bad_feats = [f for f in bad_feats if f]

    if good_feats and bad_feats:
        all_keys = set()
        for f in good_feats + bad_feats:
            all_keys.update(k for k in f.keys() if k != "file")
        lines += ["## 好听 vs 一般的声学差异（n_good={} n_bad={}）".format(
            len(good_feats), len(bad_feats)), "",
                  "| 指标 | 好听组均值 | 一般组均值 | d | 方向 |", "|---|---|---|---|---|"]
        for k in sorted(all_keys):
            if k == "file" or k in ("key", "key_majority", "mode"):
                continue
            ga = [f[k] for f in good_feats if f.get(k) is not None and isinstance(f.get(k), (int, float))]
            ba = [f[k] for f in bad_feats if f.get(k) is not None and isinstance(f.get(k), (int, float))]
            if len(ga) < 1 or len(ba) < 1:
                continue
            d = cohen_d(ga, ba)
            direction = "好听组更高" if d > 0 else "一般组更高"
            strength = "**强**" if abs(d) >= 0.8 else ("中" if abs(d) >= 0.5 else "弱")
            lines.append(f"| {k} | {np.mean(ga):.2f} | {np.mean(ba):.2f} | {d:+.2f} | {direction} ({strength}) |")

lines += ["", "## 下一步", ""]
lines += [
    "1. **扩大打分样本**：给曲库中的歌也打分（好听/不好听），特别是 S 级和 B 级各 20 首",
    "2. **加入 MERT 嵌入**：用 MERT 向量做回归，比手工特征更全面",
    "3. **训练好听预测器**：用打分数据训练回归模型，预测新生成曲的好听概率",
    "4. **闭环**：预测器打分 → 低分的自动换 seed 重生成 → 高分的交付",
]

Path(ROOT / "eval" / "taste_analysis.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
