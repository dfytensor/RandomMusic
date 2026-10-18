# -*- coding: utf-8 -*-
"""MERT 1024 维嵌入 × 热度分类（XGBoost/MLP/RF 对比）"""
import json, sys, warnings
from pathlib import Path
import numpy as np
from collections import Counter

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

# ===== 热度 =====
MEGA_A = {"周杰伦","林俊杰","五月天","张学友","邓紫棋","陈奕迅","王菲","张惠妹",
          "孙燕姿","梁静茹","S.H.E","王心凌","张韶涵","胡歌"}
KNOWN_A = {"许嵩","汪苏泷","后弦","黄雅莉","张芸京","山野","花儿乐队","信乐团",
           "183Club","5566","By2","吴克群","薛之谦","毛不易","陈琳","香香","唐磊","汤潮"}
MEGA_S = {"青花瓷","东风破","菊花台","发如雪","江南","曹操","隐形的翅膀","偏爱",
          "死了都要爱","突然的自我","温柔","知足","Super Star","不想长大","遇见",
          "天黑黑","绿光"}

def tier(stem):
    clean = stem.replace("lib_", "", 1).replace("_", " ")
    for a in MEGA_A:
        if a in clean: return 3
    for h in MEGA_S:
        if h in clean or h.replace(" ", "_") in clean: return 3
    for a in KNOWN_A:
        if a in clean: return 2
    return 1

# ===== 加载 =====
z = np.load(str(ROOT/"eval"/"mert_all_embeddings.npz"), allow_pickle=True)
names = [str(n) for n in z["names"]]
embs = z["embs"]

X = embs
y = np.array([tier(n) for n in names])
tc = Counter(y)
print(f"样本: {X.shape[0]}  维度: {X.shape[1]}  分布: {dict(sorted(tc.items()))}", flush=True)

# ===== 标准化 =====
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# ===== 模型 =====
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "RF": RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42, class_weight="balanced"),
    "ET": ExtraTreesClassifier(n_estimators=500, max_depth=None, random_state=42, class_weight="balanced"),
    "GB": GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42),
    "LR": LogisticRegression(C=1.0, max_iter=2000, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=42,
                          early_stopping=True, learning_rate="adaptive"),
    "SVM": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
}

lines = ["# MERT 1024维嵌入 × 热度分类（全库 571 首）", "",
         f"样本: {X.shape[0]} · 维度: {X.shape[1]}", "",
         "| 模型 | Accuracy | F1-macro |", "|---|---|---|"]

best_acc, best_name, best_pred = 0, "", None
for name, model in models.items():
    try:
        pred = cross_val_predict(model, X_s, y, cv=skf)
        acc = float(np.mean(pred == y))
        f1m = float(f1_score(y, pred, average="macro"))
        print(f"{name}: acc={acc*100:.1f}% f1={f1m:.3f}", flush=True)
        lines.append(f"| {name} | {acc*100:.1f}% | {f1m:.3f} |")
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_pred = pred
    except Exception as e:
        print(f"{name} FAIL: {str(e)[:80]}", flush=True)

# ===== 对比手工特征基线 =====
lines += ["", "## 对比", "",
          "- 手工特征基线（34维 RF）: 55.7%"]
if best_acc > 0.55:
    lines.append(f"- **MERT 1024维 {best_name}: {best_acc*100:.1f}% → 提升 +{(best_acc-0.557)*100:.1f}%** ✅")
else:
    lines.append(f"- MERT 1024维 {best_name}: {best_acc*100:.1f}% → 未超过手工特征基线")

lines += ["", "## 逐类报告", "", "```", classification_report(y, best_pred), "```"]

# ===== 保存 =====
(ROOT/"eval"/"mert_classification_report.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines), flush=True)
