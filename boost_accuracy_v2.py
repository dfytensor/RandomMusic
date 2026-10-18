# -*- coding: utf-8 -*-
"""35→80维特征合并+XGBoost分类（修复版，极简无外部依赖）"""
import json, re, sys, warnings
from pathlib import Path
import numpy as np
from collections import Counter

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050

MEGA_A = {"周杰伦","林俊杰","五月天","张学友","邓紫棋","陈奕迅","王菲","张惠妹",
          "孙燕姿","梁静茹","S.H.E","王心凌","张韶涵","胡歌"}
KNOWN_A = {"许嵩","汪苏泷","后弦","黄雅莉","张芸京","山野","花儿乐队","信乐团",
           "183Club","5566","By2","吴克群","薛之谦","毛不易","陈琳","香香","唐磊","汤潮"}
MEGA_S = {"青花瓷","东风破","菊花台","发如雪","江南","曹操","隐形的翅膀","偏爱",
          "死了都要爱","突然的自我","温柔","知足","Super Star","不想长大","遇见",
          "天黑黑","绿光"}

def tier(name):
    clean = name.replace(" ", "_")
    for a in MEGA_A:
        if a in clean: return 3
    for h in MEGA_S:
        if h in clean: return 3
    for a in KNOWN_A:
        if a in clean: return 2
    return 1

# ===== 加载所有缓存 =====
feats = json.loads((ROOT/"corpus"/"features.json").read_text(encoding="utf-8"))
fp_cache = json.loads((ROOT/"eval"/"lib_metric_cache.json").read_text(encoding="utf-8"))
wv_cache = json.loads((ROOT/"eval"/"wave_metric_cache.json").read_text(encoding="utf-8"))
res_cache = json.loads((ROOT/"eval"/"resonance_cache.json").read_text(encoding="utf-8"))
ms_list = json.loads((ROOT/"corpus"/"melody_stats.json").read_text(encoding="utf-8"))["per_song"]

ms_map = {}
for r in ms_list:
    if r.get("file"):
        ms_map[r["file"]] = r

# 构建 wav_stem -> feature_key 映射
stem_map = {}
for name in feats:
    clean = name.replace(".mp3", "").replace(".flac", "").replace(" ", "_")
    stem_map["lib_" + clean[:60]] = name

# ===== 合并特征 =====
X_rows, y_rows, all_keys = [], [], set()
raw_data = []

for stem, fp in fp_cache.items():
    if fp.get("skip"): continue
    wv = wv_cache.get(stem, {})
    if "crest_db" not in wv: continue
    rs = res_cache.get(stem, {})
    
    orig_name = stem_map.get(stem)
    if not orig_name:
        continue
    r_basic = feats.get(orig_name, {})
    ms_key = stem.replace("lib_", "", 1)
    ms = ms_map.get(ms_key, {})
    if not ms: continue
    
    row = {}
    # 基础
    for k in ("bpm","duration_s","centroid_hz","rms_db","dyn_p95_p10_db","zcr","rolloff_hz","hpr","key_conf"):
        v = r_basic.get(k)
        row[k] = float(v) if v is not None and v == v else 0.0
    # 指纹
    for k in ("voiced_ratio","n_vocal_runs","vibrato_rate_hz","vibrato_extent_cents",
              "intra_note_drift_cents","glide_ratio","macro_dyn_std_db","dyn_range_db",
              "spectral_flux_std","loop_lag_s","loop_peak_corr","loopiness"):
        v = fp.get(k)
        row["fp_" + k] = float(v) if v is not None and v == v else 0.0
    # 波形
    for k in ("crest_db","env_rough","low_attack_ms","silence_frac","onset_iv_cv","kurtosis","lra_db"):
        v = wv.get(k)
        row["wf_" + k] = float(v) if v is not None and v == v else 0.0
    # 共鸣
    for k in ("vibrato_cv","long_drift","breath_dips","leap_prep"):
        v = rs.get(k)
        row["res_" + k] = float(v) if v is not None and v == v else 0.0
    # 旋律
    for k in ("density","pitch_range","leap_ratio","motif_repeat","lift_mean","rest_ratio"):
        v = ms.get(k)
        row["mel_" + k] = float(v) if v is not None and v == v else 0.0
    
    raw_data.append((stem, row, tier(orig_name)))

# 构建特征名列表
all_keys = set()
for _, row, _ in raw_data:
    all_keys.update(row.keys())
fn_all = sorted(all_keys)

# 构建矩阵
for stem, row, t in raw_data:
    vec = [row.get(k, 0.0) for k in fn_all]
    X_rows.append(vec)
    y_rows.append(t)

X = np.array(X_rows, dtype=np.float64)
y = np.array(y_rows)
tc = Counter(y)
print(f"样本: {X.shape[0]}  特征: {X.shape[1]}  分布: {dict(sorted(tc.items()))}", flush=True)

# ===== 交互特征 =====
n_base = len(fn_all)
new_feats = []
for i in range(min(n_base, 20)):
    for j in range(i+1, min(n_base, 20)):
        if i != j:
            new_feats.append(X[:, i] * X[:, j])
if new_feats:
    X_interact = np.array(new_feats).T
    # 限制数量
    X_interact = X_interact[:, :30]
    X = np.hstack([X, X_interact])
    fn_all += [f"interact_{i}" for i in range(X_interact.shape[1])]
print(f"加交互后: {X.shape[1]} 维", flush=True)

# ===== 标准化 =====
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# ===== 多模型 =====
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "RF": RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, class_weight="balanced"),
    "ET": ExtraTreesClassifier(n_estimators=500, max_depth=15, random_state=42, class_weight="balanced"),
    "GB": GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42, subsample=0.8),
    "Voting": VotingClassifier([
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight="balanced")),
        ("et", ExtraTreesClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight="balanced")),
        ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)),
    ], voting="soft"),
}

best_acc, best_name, best_pred = 0, "", None
lines = [f"# 80+维特征热度分类（提升版）", "",
         f"样本: {X.shape[0]} · 特征: {X.shape[1]} · 类别: 3", "",
         "| 模型 | Accuracy | F1-macro |", "|---|---|---|"]

for name, model in models.items():
    try:
        pred = cross_val_predict(model, X_s, y, cv=skf)
        acc = float(np.mean(pred == y))
        f1m = float(f1_score(y, pred, average="macro"))
        lines.append(f"| {name} | {acc*100:.1f}% | {f1m:.3f} |")
        print(f"{name}: acc={acc*100:.1f}% f1={f1m:.3f}", flush=True)
        if acc > best_acc:
            best_acc, best_name, best_pred = acc, name, pred
    except Exception as e:
        print(f"{name} FAIL: {str(e)[:60]}", flush=True)

lines += ["", f"最佳: **{best_name} {best_acc*100:.1f}%** (基线 {max(Counter(y).values())/len(y)*100:.0f}%)"]
(ROOT/"eval"/"boost_result.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines), flush=True)
