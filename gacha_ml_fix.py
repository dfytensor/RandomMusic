# -*- coding: utf-8 -*-
"""修复版：正确的 key 匹配 + 全部缓存合并 + RF 分类"""
import json, sys, warnings
from pathlib import Path
import numpy as np
from collections import Counter

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# 加载所有缓存
fp_cache = json.loads((ROOT/"eval"/"lib_metric_cache.json").read_text(encoding="utf-8"))
wv_cache = json.loads((ROOT/"eval"/"wave_metric_cache.json").read_text(encoding="utf-8"))
res_cache = json.loads((ROOT/"eval"/"resonance_cache.json").read_text(encoding="utf-8"))
ms_list = json.loads((ROOT/"corpus"/"melody_stats.json").read_text(encoding="utf-8"))["per_song"]

# ms_list 的 file 字段就是 transcript 目录名
ms_map = {r["file"]: r for r in ms_list if r.get("file")}

# 找交集：三个缓存都有的 stem
fp_keys = set(k for k in fp_cache if not fp_cache[k].get("skip") and fp_cache[k].get("voiced_ratio") is not None)
wv_keys = set(k for k in wv_cache if "crest_db" in wv_cache[k])
res_keys = set(k for k in res_cache if res_cache[k].get("vibrato_cv") is not None)
ms_keys = set(k for k in ms_map)

common = fp_keys & wv_keys & res_keys & ms_keys
print(f"交集样本: {len(common)}", flush=True)

# 构建特征矩阵
all_keys_set = set()
samples = {}
for k in common:
    row = {}
    # 指纹
    for kk, vv in fp_cache[k].items():
        if isinstance(vv, (int, float)):
            row["fp_" + kk] = float(vv)
    # 波形
    for kk, vv in wv_cache[k].items():
        if isinstance(vv, (int, float)):
            row["wf_" + kk] = float(vv)
    # 共鸣
    for kk, vv in res_cache[k].items():
        if isinstance(vv, (int, float)):
            row["res_" + kk] = float(vv)
    # 旋律
    ms = ms_map.get(k, {})
    for kk in ("density","pitch_range","leap_ratio","motif_repeat","lift_mean","rest_ratio"):
        v = ms.get(kk)
        if v is not None:
            row["mel_" + kk] = float(v)
    samples[k] = row
    all_keys_set.update(row.keys())

fn_all = sorted(all_keys_set)
X = np.array([[samples[k].get(fn, 0) for fn in fn_all] for k in common])
y = np.array([tier(stem) for stem in common])

from collections import Counter
tc = Counter(y)
print(f"样本: {X.shape[0]}  特征: {X.shape[1]}  分布: {dict(sorted(tc.items()))}")

def tier(stem):
    # 从 features.json 获取
    pass

# 直接从 file stem 匹配 features.json 的热度
feats = json.loads((ROOT/"corpus"/"features.json").read_text(encoding="utf-8"))
import re
MEGA_A = {"周杰伦","林俊杰","五月天","张学友","邓紫棋","陈奕迅","王菲","张惠妹",
          "孙燕姿","梁静茹","S.H.E","王心凌","张韶涵","胡歌"}
KNOWN_A = {"许嵩","汪苏泷","后弦","黄雅莉","张芸京","山野","花儿乐队","信乐团",
           "183Club","5566","By2","吴克群","薛之谦","毛不易","陈琳","香香","唐磊","汤潮"}
MEGA_S = {"青花瓷","东风破","菊花台","发如雪","江南","曹操","隐形的翅膀","偏爱",
          "死了都要爱","突然的自我","温柔","知足","Super Star","不想长大","遇见",
          "天黑黑","绿光"}

def tier(stem):
    clean = stem.replace("lib_","",1).replace("_"," ")
    # 在 feats 里找
    for name in feats:
        if stem.replace("_"," ")[:8] in name[:12] or name.replace(".mp3","").replace(" ","_")[:12] == stem.replace("_"," ")[:12]:
            pass
    # 简单方式：用关键字
    for a in MEGA_A:
        if a in stem: return 3
    for h in MEGA_S:
        if h.replace(" ","_") in stem or h in stem: return 3
    for a in KNOWN_A:
        if a in stem: return 2
    return 1

y = np.array([tier(k) for k in common])
tc = Counter(y)
print(f"热度分布: {dict(sorted(tc.items()))}")

# ===== ML =====
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

scaler = StandardScaler()
X_s = scaler.fit_transform(X)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "RF-500": RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, class_weight="balanced"),
    "ET-500": ExtraTreesClassifier(n_estimators=500, max_depth=15, random_state=42, class_weight="balanced"),
    "GB-300": GradientBoostingClassifier(n_estimators=300, max_depth=8, random_state=42, subsample=0.8),
}

results = []
for name, model in models.items():
    pred = cross_val_predict(model, X_s, y, cv=skf)
    acc = float(np.mean(pred == y))
    f1m = float(f1_score(y, pred, average="macro"))
    results.append((name, acc, f1m))
    print(f"{name}: acc={acc*100:.1f}% f1={f1m:.3f}", flush=True)

best = max(results, key=lambda x: x[1])
print(f"\n最佳: {best[0]} acc={best[1]*100:.1f}%", flush=True)
