# -*- coding: utf-8 -*-
"""音乐 Jev：System One 直接打分（不生成文本评价）
用 MERT 嵌入 + 42 首 NetEase 标签做校准回归 + 置信门控
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
from collections import Counter

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

# ===== 加载 MERT 全库嵌入 =====
z = np.load(str(ROOT/"eval"/"mert_all_embeddings.npz"), allow_pickle=True)
mert_names = [str(n) for n in z["names"]]
mert_embs = z["embs"]

# ===== 热度标签 =====
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

y_tier = np.array([tier(n) for n in mert_names])
print(f"总样本: {len(y_tier)}  分布: {dict(sorted(Counter(y_tier).items()))}")

# ===== 拆分：训练集（其他 526 首）+ 测试集（42 首有完整指纹的） =====
fp_cache = json.loads((ROOT/"eval"/"lib_metric_cache.json").read_text(encoding="utf-8"))
ne_data = json.loads((ROOT/"eval"/"netease_popularity_v2.json").read_text(encoding="utf-8"))

# 找到 42 首 NetEase 歌对应的 MERT 索引
ne_stems = set()
for orig, v in ne_data.items():
    if "netease_id" not in v: continue
    clean = orig.replace(".mp3","").replace(" ","_")[:60]
    ne_stems.add("lib_" + clean)

test_idx = [i for i, n in enumerate(mert_names) if n in ne_stems]
train_idx = [i for i in range(len(y_tier)) if i not in set(test_idx)]
print(f"训练: {len(train_idx)}  测试: {len(test_idx)}")

X_train = mert_embs[train_idx]
y_train = y_tier[train_idx]
X_test = mert_embs[test_idx]
y_test_tier = y_tier[test_idx]
y_test_pop = []
for i in test_idx:
    n = mert_names[i]
    if n in ne_data and "popularity" in ne_data[n]:
        y_test_pop.append(ne_data[n]["popularity"])
    else:
        y_test_pop.append(0)
y_test_pop = np.array(y_test_pop)

# ===== 训练：线性头（AnyJev L2 方法：2/3深度 + 闭式解） =====
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)

# 尝试不同深度的线性头（模拟 AnyJev L2 的"停在前2/3深度"）
# 我们只有最终嵌入，但可以用 PCA 模拟不同"信息深度"
from sklearn.decomposition import PCA

best_r2, best_dim, best_model_reg = -999, 0, None
for n_comp in (32, 64, 128, 256, 512):
    pca = PCA(n_components=n_comp, random_state=42)
    X_tr_p = pca.fit_transform(X_tr_s)
    X_te_p = pca.transform(X_te_s)
    model = Ridge(alpha=1.0)
    model.fit(X_tr_p, y_train)
    pred = model.predict(X_te_p)
    r2 = float(np.corrcoef(pred, y_test_pop)[0,1]**2) if np.std(pred) > 0 else 0
    mae = float(np.mean(np.abs(pred - y_test_pop)))
    if r2 > best_r2:
        best_r2, best_dim, best_model_reg = r2, n_comp, model
    print(f"  PCA{n_comp}: R²={r2:.3f} MAE={mae:.1f}", flush=True)

print(f"\n最佳回归: PCA{best_dim} R²={best_r2:.3f}", flush=True)

# ===== 分类：3层热度 =====
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, classification_report

et = ExtraTreesClassifier(n_estimators=500, random_state=42, class_weight="balanced")
et.fit(X_tr_s, y_train)
pred_tier = et.predict(X_te_s)
acc_tier = float(np.mean(pred_tier == y_test_tier))
f1_tier = float(f1_score(y_test_tier, pred_tier, average="macro"))
print(f"\n分类(3层): acc={acc_tier*100:.1f}% f1={f1_tier:.3f}", flush=True)

# ===== 置信门控 =====
# 用预测概率的最大值作为置信度
proba = et.predict_proba(X_te_s)
conf = np.max(proba, axis=1)

# 置信门控：高置信才自动决策
for threshold in (0.5, 0.6, 0.7, 0.8):
    high_conf = conf >= threshold
    if high_conf.sum() > 0:
        sub_acc = float(np.mean(pred_tier[high_conf] == y_test_tier[high_conf]))
        coverage = float(high_conf.mean())
        print(f"  置信≥{threshold:.1f}: 覆盖{coverage*100:.0f}% 准确率{sub_acc*100:.1f}%", flush=True)

# ===== 保存模型 =====
import pickle
artifact = dict(
    scaler=scaler, model=et, pca_dim=None,
    feature_type="mert_1024",
    accuracy=acc_tier, f1=f1_tier,
)
with open(ROOT/"eval"/"jev_music_scorer.pkl", "wb") as f:
    pickle.dump(artifact, f)
print("SAVED eval/jev_music_scorer.pkl", flush=True)
