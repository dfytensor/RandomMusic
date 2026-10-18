# -*- coding: utf-8 -*-
"""修复版 v2：只要求三个缓存交集，melody 缺失补 0"""
import json, sys, warnings, re
from pathlib import Path
import numpy as np
from collections import Counter

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

fp_cache = json.loads((ROOT/"eval"/"lib_metric_cache.json").read_text(encoding="utf-8"))
wv_cache = json.loads((ROOT/"eval"/"wave_metric_cache.json").read_text(encoding="utf-8"))
res_cache = json.loads((ROOT/"eval"/"resonance_cache.json").read_text(encoding="utf-8"))
ms_list = json.loads((ROOT/"corpus"/"melody_stats.json").read_text(encoding="utf-8"))["per_song"]
ms_map = {r["file"]: r for r in ms_list if r.get("file")}

# 交集
fp_ok = set(k for k, v in fp_cache.items() if not v.get("skip") and v.get("voiced_ratio") is not None)
wv_ok = set(k for k, v in wv_cache.items() if "crest_db" in v)
rs_ok = set(k for k, v in res_cache.items() if v.get("vibrato_cv") is not None)
common = sorted(fp_ok & wv_ok & rs_ok)
print(f"交集: {len(common)}", flush=True)

# ===== 构建特征 =====
MEGA_A = {"周杰伦","林俊杰","五月天","张学友","邓紫棋","陈奕迅","王菲","张惠妹",
          "孙燕姿","梁静茹","S.H.E","王心凌","张韶涵","胡歌"}
KNOWN_A = {"许嵩","汪苏泷","后弦","黄雅莉","张芸京","山野","花儿乐队","信乐团",
           "183Club","5566","By2","吴克群","薛之谦","毛不易","陈琳","香香","唐磊","汤潮"}
MEGA_S = {"青花瓷","东风破","菊花台","发如雪","江南","曹操","隐形的翅膀","偏爱",
          "死了都要爱","突然的自我","温柔","知足","Super Star","不想长大","遇见","天黑黑","绿光"}

def tier_of_stem(stem):
    clean = stem.replace("lib_","",1).replace("_"," ")
    for a in MEGA_A:
        if a in clean: return 3
    for h in MEGA_S:
        if h.replace(" ","_") in stem or h in clean: return 3
    for a in KNOWN_A:
        if a in clean: return 2
    return 1

rows, labels, keys_used = [], [], []
for k in common:
    fp = fp_cache[k]; wv = wv_cache[k]; rs = res_cache[k]
    ms = ms_map.get(k, {})
    
    row = {}
    for src, prefix in [(fp, "fp_"), (wv, "wf_"), (rs, "res_")]:
        for kk, vv in src.items():
            if isinstance(vv, (int, float)):
                row[prefix + kk] = float(vv)
    for kk in ("density","pitch_range","leap_ratio","motif_repeat","lift_mean","rest_ratio"):
        v = ms.get(kk)
        row["mel_" + kk] = float(v) if v is not None else 0.0
    
    rows.append(row)
    labels.append(tier_of_stem(k))
    keys_used.append(k)

fn_all = sorted(set(k for r in rows for k in r))
X = np.array([[r.get(k, 0.0) for k in fn_all] for r in rows])
y = np.array(labels)
tc = Counter(y)
print(f"特征: {X.shape[1]}  分布: {dict(sorted(tc.items()))}", flush=True)

# ===== ML =====
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score

scaler = StandardScaler()
X_s = scaler.fit_transform(X)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "RF": RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, class_weight="balanced"),
    "ET": ExtraTreesClassifier(n_estimators=500, max_depth=15, random_state=42, class_weight="balanced"),
    "GB": GradientBoostingClassifier(n_estimators=300, max_depth=8, random_state=42, subsample=0.8),
    "Voting": VotingClassifier([
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight="balanced")),
        ("et", ExtraTreesClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight="balanced")),
        ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)),
    ], voting="soft"),
}

best_acc, best_name, best_model = 0, "", None
for name, model in models.items():
    pred = cross_val_predict(model, X_s, y, cv=skf)
    acc = float(np.mean(pred == y))
    f1m = float(f1_score(y, pred, average="macro"))
    print(f"{name}: acc={acc*100:.1f}% f1={f1m:.3f}", flush=True)
    if acc > best_acc:
        best_acc = acc
        best_name = name
        best_model = model

print(f"\n最佳: {best_name} {best_acc*100:.1f}%", flush=True)

# 特征重要性
best_model.fit(X_s, y)
imp = best_model.feature_importances_
idx = np.argsort(imp)[::-1]
print("\nTop 20:")
for i in range(min(20, len(idx))):
    fn = fn_all[idx[i]] if idx[i] < len(fn_all) else "?"
    print(f"  {i+1}. {fn}: {imp[idx[i]]:.4f}")
