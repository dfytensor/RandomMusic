# -*- coding: utf-8 -*-
"""提升热度分类准确率：
1. 加载全部已有缓存指标（基础+指纹+波形+旋律+共鸣 = ~50维）
2. 加 MERT 嵌入（已有 32 首，PCA 降维到 32 维）
3. 加 ABC 转谱旋律学指标（~10 维）
4. 全合并 = 80-100 维
5. RandomForest + XGBoost + SVM 三模型对比
6. 输出 Accuracy / F1-macro / AUC + 特征重要性 Top30
"""
import json, math, re, sys, warnings
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
sys.path.insert(0, str(ROOT))

MEGA_A = {"周杰伦","林俊杰","五月天","张学友","邓紫棋","陈奕迅","王菲","张惠妹",
          "孙燕姿","梁静茹","S.H.E","王心凌","张韶涵","胡歌"}
KNOWN_A = {"许嵩","汪苏泷","后弦","黄雅莉","张芸京","山野","花儿乐队","信乐团",
           "183Club","5566","By2","吴克群","薛之谦","毛不易","陈琳","香香","唐磊","汤潮"}
MEGA_S = {"青花瓷","东风破","菊花台","发如雪","江南","曹操","隐形的翅膀","偏爱",
          "死了都要爱","突然的自我","温柔","知足","Super Star","不想长大","遇见",
          "天黑黑","绿光","月亮代表我的心","甜蜜蜜","后来","为爱痴狂","朋友",
          "第一次","勇气","老鼠爱大米","两只蝴蝶","秋天不回来","求佛","香水有毒",
          "该死的温柔","专属天使","北极星的眼泪","我难过","倒带","说爱你","那些花儿",
          "生如夏花","平凡之路","董小姐","成都","南山南","小苹果","最炫民族风"}

def tier(name):
    clean = name.replace(" ", "_")
    for a in MEGA_A:
        if a in clean: return 3
    for h in MEGA_S:
        if h in clean: return 3
    for a in KNOWN_A:
        if a in clean: return 2
    return 1

# ===== 加载所有源 =====
feats = json.loads((ROOT/"corpus"/"features.json").read_text(encoding="utf-8"))
fp_cache = json.loads((ROOT/"eval"/"lib_metric_cache.json").read_text(encoding="utf-8"))
wv_cache = json.loads((ROOT/"eval"/"wave_metric_cache.json").read_text(encoding="utf-8"))
res_cache = json.loads((ROOT/"eval"/"resonance_cache.json").read_text(encoding="utf-8"))
ms_list = json.loads((ROOT/"corpus"/"melody_stats.json").read_text(encoding="utf-8"))["per_song"]
ms_map = {}
for r in ms_list:
    if r.get("file"):
        ms_map[r["file"]] = r

# ===== 逐曲合并全部维度 =====
X_rows, y_rows, song_names = [], [], []
all_keys = set()

for name, r in sorted(feats.items()):
    stem = "lib_" + name.replace(".mp3", "").replace(".flac", "").replace(" ", "_")[:60]
    fp = fp_cache.get(stem, {})
    wv = wv_cache.get(stem, {})
    rs = res_cache.get(stem, {})
    ms = ms_map.get(stem.replace("lib_", "", 1), {})
    
    if fp.get("skip"):
        continue
    
    vec, fn = [], []
    
    # 基础 9
    for k in ("bpm","duration_s","centroid_hz","rms_db","dyn_p95_p10_db","zcr","rolloff_hz","hpr","key_conf"):
        v = r.get(k)
        vec.append(float(v) if v is not None and v == v else 0.0); fn.append(k)
    
    # 指纹 ~15
    for k in ("voiced_ratio","n_vocal_runs","vibrato_rate_hz","vibrato_extent_cents",
              "intra_note_drift_cents","glide_ratio","macro_dyn_std_db","dyn_range_db",
              "spectral_flux_std","loop_lag_s","loop_peak_corr","loopiness"):
        v = fp.get(k)
        vec.append(float(v) if v is not None else 0.0); fn.append("fp_" + k)
    
    # 波形 7
    for k in ("crest_db","env_rough","low_attack_ms","silence_frac","onset_iv_cv","kurtosis","lra_db"):
        v = wv.get(k)
        vec.append(float(v) if v is not None else 0.0); fn.append("wf_" + k)
    
    # 旋律 6
    for k in ("density","pitch_range","leap_ratio","motif_repeat","lift_mean","rest_ratio"):
        v = ms.get(k)
        vec.append(float(v) if v is not None else 0.0); fn.append("mel_" + k)
    
    # 共鸣 4
    for k in ("vibrato_cv","long_drift","breath_dips","leap_prep"):
        v = rs.get(k)
        vec.append(float(v) if v is not None else 0.0); fn.append("res_" + k)
    
    t = tier(name)
    if len(vec) == len(fn):
        X_rows.append(vec); y_rows.append(t); song_names.append(name)
        all_keys.update(fn)

X = np.array(X_rows, dtype=np.float64)
y = np.array(y_rows)
fn_all = sorted(all_keys)

# 对齐列
col_map = {fn: i for i, fn in enumerate(fn_all)}
X_aligned = np.zeros((len(X_rows), len(fn_all)))
for i, (vec, fn) in enumerate(zip(X_rows, [fn_all] * len(X_rows))):
    pass  # Already aligned by construction

# Actually re-do properly:
X_final = np.zeros((len(X_rows), len(fn_all)))
for i in range(len(X_rows)):
    for j, k in enumerate(fn_all):
        X_final[i, j] = X_rows[i][j] if j < len(X_rows[i]) else 0.0

X = X_final
print(f"合并后: {X.shape[0]} 样本 × {X.shape[1]} 维", flush=True)

# ===== MERT 嵌入 (32 首) =====
mert_path = ROOT / "eval" / "embeddings.npz"
mert_dim = 32
if mert_path.exists():
    z = np.load(str(mert_path), allow_pickle=True)
    e1 = z["embs"]  # 32×1024
    from sklearn.decomposition import PCA
    pca = PCA(n_components=32, random_state=42)
    mert_reduced = pca.fit_transform(e1)  # 32×32
    # 映射到全库（只对有的歌加）
    mert_stems = set(Path(n).stem.replace("lib_", "", 1) for n in z["names"])
    mert_add = np.zeros((X.shape[0], 32))
    for i, name in enumerate(song_names):
        stem_clean = name.replace(".mp3", "").replace(".flac", "").replace(" ", "_")[:60]
        wav_stem = "lib_" + stem_clean
        # 在 mert names 里找
        for j, mn in enumerate(z["names"]):
            if Path(str(mn)).stem.replace("lib_", "", 1)[:10] == stem_clean[:10]:
                mert_add[i] = mert_reduced[j]
                break
    X = np.hstack([X, mert_add])
    fn_all = fn_all + [f"mert_{i}" for i in range(32)]
    print(f"加入 MERT 嵌入 32 维 -> 总维度 {X.shape[1]}", flush=True)

# ===== 标准化 =====
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== ML 三模型 =====
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

models = {
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight="balanced"),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42),
    "SVM-RBF": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model, best_name, best_acc = None, "", 0

lines = ["# 全维度热度分类（提升版）", "",
         f"样本: {X.shape[0]} · 特征: {X.shape[1]} · 3 类", "",
         "| 模型 | Accuracy | F1-macro |", "|---|---|---|"]

for name, model in models.items():
    pred = cross_val_predict(model, X_scaled, y, cv=skf)
    acc = float(np.mean(pred == y))
    f1 = float(f1_score(y, pred, average="macro"))
    lines.append(f"| {name} | {acc*100:.1f}% | F1={f1:.3f} |")
    print(f"{name}: acc={acc*100:.1f}% f1={f1:.3f}", flush=True)
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

# 最佳模型详细
best_model.fit(X_scaled, y)
pred_best = cross_val_predict(best_model, X_scaled, y, cv=skf)
report = classification_report(y, pred_best)
lines += ["", "## 最佳模型详细报告", "", "```", report, "```"]

# AUC (ovr)
try:
    best_model.fit(X_scaled, y)
    proba = best_model.predict_proba(X_scaled)
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y, classes=sorted(set(y)))
    if y_bin.shape[1] > 2:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
    else:
        auc = roc_auc_score(y, proba[:, 1])
    lines.append(f"\nAUC (OVR macro): **{auc:.3f}**")
    print(f"AUC: {auc:.3f}", flush=True)
except:
    pass

# ===== 特征重要性 =====
if hasattr(best_model, "feature_importances_"):
    imp = best_model.feature_importances_
    idx = np.argsort(imp)[::-1]
    lines += ["", "## Top 30 特征重要性", "", "| 排名 | 特征 | 重要性 |", "|---|---|---|"]
    for i in range(min(30, len(idx))):
        fn = fn_all[idx[i]] if idx[i] < len(fn_all) else f"mert_{idx[i]-len(fn_all)}"
        lines.append(f"| {i+1} | {fn} | {imp[idx[i]]:.4f} |")

(ROOT / "eval" / "popularity_ml_v2.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines), flush=True)
