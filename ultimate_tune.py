# -*- coding: utf-8 -*-
"""终极提升：SVM调参 + MERT层选择 + 手工+MERT拼接 + 模型融合"""
import json, sys, warnings, os
from pathlib import Path
import numpy as np
from collections import Counter

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ===== 加载 MERT 全库嵌入 =====
z = np.load(str(ROOT/"eval"/"mert_all_embeddings.npz"), allow_pickle=True)
mert_names = [str(n) for n in z["names"]]
mert_embs = z["embs"]

# ===== 热度 =====
MEGA_A = {"周杰伦","林俊杰","五月天","张学友","邓紫棋","陈奕迅","王菲","张惠妹",
          "孙燕姿","梁静茹","S.H.E","王心凌","张韶涵","胡歌"}
KNOWN_A = {"许嵩","汪苏泷","后弦","黄雅莉","张芸京","山野","花儿乐队","信乐团",
           "183Club","5566","By2","吴克群","薛之谦","毛不易","陈琳","香香","唐磊","汤潮"}
MEGA_S = {"青花瓷","东风破","菊花台","发如雪","江南","曹操","隐形的翅膀","偏爱",
          "死了都要爱","突然的自我","温柔","知足","Super Star","不想长大","遇见","天黑黑","绿光"}

def tier(stem):
    clean = stem.replace("lib_","",1).replace("_"," ")
    for a in MEGA_A:
        if a in clean: return 3
    for h in MEGA_S:
        if h in clean or h.replace(" ","_") in clean: return 3
    for a in KNOWN_A:
        if a in clean: return 2
    return 1

y = np.array([tier(n) for n in mert_names])
tc = Counter(y)
print(f"样本: {len(y)}  分布: {dict(sorted(tc.items()))}", flush=True)

# ===== 加载手工特征（35维） =====
feats = json.loads((ROOT/"corpus"/"features.json").read_text(encoding="utf-8"))
fp_cache = json.loads((ROOT/"eval"/"lib_metric_cache.json").read_text(encoding="utf-8"))
wv_cache = json.loads((ROOT/"eval"/"wave_metric_cache.json").read_text(encoding="utf-8"))
res_cache = json.loads((ROOT/"eval"/"resonance_cache.json").read_text(encoding="utf-8"))
ms_list = json.loads((ROOT/"corpus"/"melody_stats.json").read_text(encoding="utf-8"))["per_song"]
ms_map = {r["file"]: r for r in ms_list if r.get("file")}

manual_keys = set()
manual_data = {}
for name, r in feats.items():
    stem = "lib_" + name.replace(".mp3","").replace(".flac","").replace(" ","_")[:60]
    if stem not in mert_names_set if (mert_names_set := set(mert_names)) else False:
        continue
    row = {}
    for k, v in r.items():
        if isinstance(v, (int, float)) and v == v:
            row[k] = float(v)
    fp = fp_cache.get(stem, {})
    if not fp.get("skip"):
        for k, v in fp.items():
            if isinstance(v, (int, float)):
                row["fp_"+k] = float(v)
    wv = wv_cache.get(stem, {})
    for k, v in wv.items():
        if isinstance(v, (int, float)):
            row["wf_"+k] = float(v)
    rs = res_cache.get(stem, {})
    for k, v in rs.items():
        if isinstance(v, (int, float)):
            row["res_"+k] = float(v)
    ms = ms_map.get(stem.replace("lib_","",1), {})
    for k in ("density","pitch_range","leap_ratio","motif_repeat","lift_mean","rest_ratio"):
        v = ms.get(k)
        if v is not None:
            row["mel_"+k] = float(v)
    manual_data[stem] = row
    manual_keys.update(row.keys())

fn_manual = sorted(manual_keys)
X_manual = np.array([[manual_data[n].get(k, 0.0) for k in fn_manual] for n in mert_names])

# MERT
X_mert = mert_embs

print(f"手工: {X_manual.shape}  MERT: {X_mert.shape}", flush=True)

# ===== 拼接 =====
X_combined = np.hstack([X_manual, X_mert])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_comb_s = scaler.fit_transform(X_combined)
X_mert_s = scaler.fit_transform(X_mert)
X_man_s = scaler.fit_transform(X_manual)

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate(name, model, X_input):
    pred = cross_val_predict(model, X_input, y, cv=skf)
    acc = float(np.mean(pred == y))
    f1m = float(f1_score(y, pred, average="macro"))
    return acc, f1m

# ===== 实验 =====
print("\n=== 1. 纯 MERT 1024 维 ===", flush=True)
for name, model in [
    ("SVM(C=10)", SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
    ("SVM(C=50)", SVC(kernel="rbf", C=50, gamma="scale", random_state=42)),
    ("SVM(C=100)", SVC(kernel="rbf", C=100, gamma="scale", random_state=42)),
    ("ET(500)", ExtraTreesClassifier(n_estimators=500, random_state=42, class_weight="balanced")),
    ("RF(500)", RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced")),
]:
    acc, f1m = evaluate(name, model, X_mert_s)
    print(f"  {name}: {acc*100:.1f}% f1={f1m:.3f}", flush=True)

print("\n=== 2. 手工 35 维 ===", flush=True)
for name, model in [
    ("SVM(C=10)", SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
    ("ET(500)", ExtraTreesClassifier(n_estimators=500, random_state=42, class_weight="balanced")),
]:
    acc, f1m = evaluate(name, model, X_man_s)
    print(f"  {name}: {acc*100:.1f}% f1={f1m:.3f}", flush=True)

print("\n=== 3. 拼接 (35+1024=1059维) ===", flush=True)
for name, model in [
    ("SVM(C=10)", SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
    ("SVM(C=50)", SVC(kernel="rbf", C=50, gamma="scale", random_state=42)),
    ("ET(500)", ExtraTreesClassifier(n_estimators=500, random_state=42, class_weight="balanced")),
    ("RF(500)", RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced")),
]:
    acc, f1m = evaluate(name, model, X_comb_s)
    print(f"  {name}: {acc*100:.1f}% f1={f1m:.3f}", flush=True)

# ===== 4. PCA 降维后再拼 =====
from sklearn.decomposition import PCA
print("\n=== 4. PCA 降维 MERT ===", flush=True)
for n_comp in (64, 128, 256):
    pca = PCA(n_components=n_comp, random_state=42)
    mert_pca = pca.fit_transform(X_mert_s)
    X_pca = np.hstack([X_man_s, mert_pca])
    for name, model in [
        ("SVM(C=50)", SVC(kernel="rbf", C=50, gamma="scale", random_state=42)),
        ("ET(500)", ExtraTreesClassifier(n_estimators=500, random_state=42, class_weight="balanced")),
    ]:
        acc, f1m = evaluate(name, model, X_pca)
        print(f"  PCA{n_comp}+{name}: {acc*100:.1f}% f1={f1m:.3f}", flush=True)

# ===== 5. 混合：MERT PCA + 手工 + 比值特征 =====
print("\n=== 5. 最优组合 ===", flush=True)
best_results = []

for n_comp in (64, 128, 256):
    pca = PCA(n_components=n_comp, random_state=42)
    mert_p = pca.fit_transform(X_mert_s)
    X_full = np.hstack([X_man_s, mert_p])
    
    for mname, model in [
        ("SVM(C=50)", SVC(kernel="rbf", C=50, gamma="scale", random_state=42)),
        ("SVM(C=100)", SVC(kernel="rbf", C=100, gamma="scale", random_state=42)),
        ("ET(500)", ExtraTreesClassifier(n_estimators=500, random_state=42, class_weight="balanced")),
        ("RF(500)", RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced")),
    ]:
        acc, f1m = evaluate(name, model, X_full)
        best_results.append((f"PCA{n_comp}+{mname}", acc, f1m))
        print(f"  PCA{n_comp}+{mname}: {acc*100:.1f}% f1={f1m:.3f}", flush=True)

best_results.sort(key=lambda x: -x[1])
print(f"\n🏆 最佳: {best_results[0][0]} {best_results[0][1]*100:.1f}% f1={best_results[0][2]:.3f}", flush=True)
