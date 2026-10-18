# -*- coding: utf-8 -*-
"""继续提升：XGBoost + LightGBM + ExtraTrees调参 + Stacking"""
import json, sys, warnings
from pathlib import Path
import numpy as np
from collections import Counter

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# 加载数据（同 gacha_ml_v2）
fp_cache = json.loads((ROOT/"eval"/"lib_metric_cache.json").read_text(encoding="utf-8"))
wv_cache = json.loads((ROOT/"eval"/"wave_metric_cache.json").read_text(encoding="utf-8"))
res_cache = json.loads((ROOT/"eval"/"resonance_cache.json").read_text(encoding="utf-8"))
ms_list = json.loads((ROOT/"corpus"/"melody_stats.json").read_text(encoding="utf-8"))["per_song"]
ms_map = {r["file"]: r for r in ms_list if r.get("file")}

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

fp_ok = set(k for k,v in fp_cache.items() if not v.get("skip") and v.get("voiced_ratio") is not None)
wv_ok = set(k for k,v in wv_cache.items() if "crest_db" in v)
rs_ok = set(k for k,v in res_cache.items() if v.get("vibrato_cv") is not None)
common = sorted(fp_ok & wv_ok & rs_ok)

rows, labels = [], []
for k in common:
    fp = fp_cache[k]; wv = wv_cache[k]; rs = res_cache[k]
    ms = ms_map.get(k, {})
    row = {}
    for src, prefix in [(fp,"fp_"),(wv,"wf_"),(rs,"res_")]:
        for kk,vv in src.items():
            if isinstance(vv,(int,float)):
                row[prefix+kk] = float(vv)
    for kk in ("density","pitch_range","leap_ratio","motif_repeat","lift_mean","rest_ratio"):
        v = ms.get(kk)
        row["mel_"+kk] = float(v) if v is not None else 0.0
    rows.append(row)
    labels.append(tier_of_stem(k))

fn_all = sorted(set(k for r in rows for k in r))
X = np.array([[r.get(k,0.0) for k in fn_all] for r in rows])
y = np.array(labels)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

from sklearn.model_selection import StratifiedKFold, cross_val_predict
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ===== 多模型 =====
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

results = []

def try_model(name, model, X_input):
    pred = cross_val_predict(model, X_input, y, cv=skf)
    acc = float(np.mean(pred == y))
    f1m = float(f1_score(y, pred, average="macro"))
    results.append((name, acc, f1m, model))
    print(f"{name}: acc={acc*100:.1f}% f1={f1m:.3f}", flush=True)
    return acc

# ExtraTrees 调参
for depth in (None, 25, 20):
    for n_est in (500, 800):
        try_model(f"ET(d={depth},n={n_est})",
                  ExtraTreesClassifier(n_estimators=n_est, max_depth=depth,
                                       random_state=42, class_weight="balanced"), X_s)

# RF 调参
try_model("RF(n=500,d=15)", RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, class_weight="balanced"), X_s)

# XGBoost
try:
    from xgboost import XGBClassifier
    try_model("XGB", XGBClassifier(n_estimators=300, max_depth=8, random_state=42,
                                    use_label_encoder=False, eval_metric="mlogloss"), X_s)
except ImportError:
    print("XGBoost not available", flush=True)

# LightGBM
try:
    from lightgbm import LGBMClassifier
    try_model("LGBM", LGBMClassifier(n_estimators=300, max_depth=8, random_state=42,
                                      verbose=-1), X_s)
except ImportError:
    print("LightGBM not available", flush=True)

# ===== Stacking =====
print("\n=== Stacking ===", flush=True)
from sklearn.ensemble import StackingClassifier

estimators = [
    ("et", ExtraTreesClassifier(n_estimators=500, max_depth=None, random_state=42, class_weight="balanced")),
    ("rf", RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, class_weight="balanced")),
    ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)),
]
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(C=1.0, max_iter=1000),
    cv=3, n_jobs=-1
)
try_model("Stacking", stack, X_s)

# ===== 排序 =====
results.sort(key=lambda r: -r[1])
print("\n=== 排名 ===", flush=True)
for name, acc, f1m, _ in results:
    print(f"  {acc*100:.1f}%  f1={f1m:.3f}  {name}", flush=True)

best = results[0]
print(f"\n最终最佳: {best[0]} {best[1]*100:.1f}%", flush=True)
