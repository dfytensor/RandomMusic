# -*- coding: utf-8 -*-
"""1) 保存最优模型  2) 8分类全特征重跑  3) 报告"""
import json, pickle, re, sys, warnings
from pathlib import Path
import numpy as np
from collections import Counter

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ===== 1. 保存最优模型 =====
z = np.load(str(ROOT/"eval"/"mert_all_embeddings.npz"), allow_pickle=True)
mert_names = [str(n) for n in z["names"]]
mert_embs = z["embs"]

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

# 手工特征
feats = json.loads((ROOT/"corpus"/"features.json").read_text(encoding="utf-8"))
fp_cache = json.loads((ROOT/"eval"/"lib_metric_cache.json").read_text(encoding="utf-8"))
wv_cache = json.loads((ROOT/"eval"/"wave_metric_cache.json").read_text(encoding="utf-8"))
res_cache = json.loads((ROOT/"eval"/"resonance_cache.json").read_text(encoding="utf-8"))
ms_list = json.loads((ROOT/"corpus"/"melody_stats.json").read_text(encoding="utf-8"))["per_song"]
ms_map = {r["file"]: r for r in ms_list if r.get("file")}

fn_manual_keys = set()
manual_rows = []
for stem in mert_names:
    row = {}
    fp = fp_cache.get(stem, {})
    if not fp.get("skip"):
        for k, v in fp.items():
            if isinstance(v, (int, float)): row["fp_"+k] = float(v)
    wv = wv_cache.get(stem, {})
    for k, v in wv.items():
        if isinstance(v, (int, float)): row["wf_"+k] = float(v)
    rs = res_cache.get(stem, {})
    for k, v in rs.items():
        if isinstance(v, (int, float)): row["res_"+k] = float(v)
    ms = ms_map.get(stem.replace("lib_","",1), {})
    for k in ("density","pitch_range","leap_ratio","motif_repeat","lift_mean","rest_ratio"):
        v = ms.get(k)
        if v is not None: row["mel_"+k] = float(v)
    manual_rows.append(row)
    fn_manual_keys.update(row.keys())

fn_manual = sorted(fn_manual_keys)
X_manual = np.array([[r.get(k, 0.0) for k in fn_manual] for r in manual_rows])

# MERT
X_mert = mert_embs

from sklearn.preprocessing import StandardScaler, label_binarize
scaler = StandardScaler()
X_man_s = scaler.fit_transform(X_manual)
X_mert_s = scaler.fit_transform(X_mert)

# PCA64
from sklearn.decomposition import PCA
pca = PCA(n_components=64, random_state=42)
mert_pca = pca.fit_transform(X_mert_s)
X_comb = np.hstack([X_man_s, mert_pca])

# 保存最优模型配置
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced")
model.fit(X_comb, y)

model_artifact = {
    "model": model,
    "pca": pca,
    "scaler_manual": scaler,
    "feature_names_manual": fn_manual,
    "config": "RF(500) + PCA64(MERT) + 手工35维 拼接 = 66.2%"
}
with open(ROOT/"eval"/"best_model.pkl", "wb") as f:
    pickle.dump(model_artifact, f)
print("✅ 最优模型已保存: eval/best_model.pkl", flush=True)

# ===== 2. 8 分类 =====
print("\n=== 8 分类（全特征）===")

# 细分层标签
TOP_TIER = {"周杰伦", "林俊杰", "五月天", "张学友"}
KNOWN_ARTISTS = {"许嵩","汪苏泷","后弦","黄雅莉","张芸京","山野","花儿乐队","信乐团",
                 "183Club","5566","By2","吴克群","薛之谦","毛不易","陈琳","香香","唐磊","汤潮"}
OST_KEYWORDS = ["电视剧","电影","主题曲","插曲","片头","片尾"]
NET_SONGS = ["老鼠爱大米","两只蝴蝶","秋天不回来","求佛","香水有毒","该死的温柔"]

def fine_tier(name):
    artist = re.split(r"[-_—]", name)[0].strip() if re.search(r"[-_—]", name) else ""
    song = re.split(r"[-_—]", name)[-1].strip()
    is_ost = any(k in name for k in OST_KEYWORDS)
    is_mega_artist = artist in MEGA_A
    is_top = artist in TOP_TIER
    is_known = artist in KNOWN_ARTISTS
    is_mega_song = any(h in name for h in MEGA_S)
    is_hit = any(h in name for h in {"雨蝶","红色高跟鞋","三国恋","六月的雨","逍遥叹",
        "城府","断桥残雪","清明雨上","有何不可","三国杀","小星星","会呼吸的痛"})
    
    if is_top and is_mega_song: return "S1"
    if is_mega_artist and is_mega_song: return "S2"
    if is_mega_artist: return "S3"
    if is_ost and is_mega_artist: return "A1"
    if is_hit or is_known: return "A2"
    if is_mega_artist: return "B1"
    if is_known: return "B2"
    if is_ost: return "B4"
    if any(k in name for k in NET_SONGS): return "B3"
    return "B5"

y_8 = np.array([fine_tier(n.replace("lib_","",1).replace("_"," ")) for n in mert_names])
tc8 = Counter(y_8)
print(f"8类分布: {dict(sorted(tc8.items()))}")

# 8 分类 ML
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score

rf8 = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, class_weight="balanced")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pred8 = cross_val_predict(rf8, X_comb, y_8, cv=skf)
acc8 = float(np.mean(pred8 == y_8))
f18 = float(f1_score(y_8, pred8, average="macro"))
baseline8 = max(tc8.values()) / len(y_8)
print(f"8类准确率: {acc8*100:.1f}% (最大类基线 {baseline8*100:.1f}%) f1={f18:.3f}")

# 3 分类对照
rf3 = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced")
pred3 = cross_val_predict(rf3, X_comb, y, cv=skf)
acc3 = float(np.mean(pred3 == y))

print(f"3类准确率: {float(np.mean(pred3 == y))*100:.1f}%")
print(f"8类 vs 3类: {'8类更好' if acc8 > acc3 else '3类更好'}")
