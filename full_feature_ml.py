# -*- coding: utf-8 -*-
"""全特征 ML：合并基础+指纹+波形+旋律 = 全维度热度分类"""
import json, math, re, sys, warnings
from pathlib import Path
import numpy as np
from collections import Counter

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
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

def lib_stem(name):
    name_clean = name.replace(".mp3", "").replace(".flac", "")
    return "lib_" + name_clean.replace(" ", "_")[:60]

# ===== 加载所有源 =====
feats = json.loads((ROOT/"corpus"/"features.json").read_text(encoding="utf-8"))
fp_cache = json.loads((ROOT/"eval"/"lib_metric_cache.json").read_text(encoding="utf-8"))
wv_cache = json.loads((ROOT/"eval"/"wave_metric_cache.json").read_text(encoding="utf-8"))
ms_list = json.loads((ROOT/"corpus"/"melody_stats.json").read_text(encoding="utf-8"))["per_song"]

# 旋律统计按 stem 索引
ms_map = {}
for r in ms_list:
    if r.get("file"):
        ms_map[r["file"]] = r

# ===== 逐曲合并 =====
X, y, names, all_fn = [], [], [], None
skipped = 0
for name, r in sorted(feats.items()):
    stem = lib_stem(name)
    fp = fp_cache.get(stem, {})
    wv = wv_cache.get(stem, {})
    if fp.get("skip") or "crest_db" not in wv:
        skipped += 1
        continue
    
    ms_key = stem.replace("lib_", "", 1)
    ms = ms_map.get(ms_key, {})
    
    vec, fn = [], []
    # 基础 9
    for k in ("bpm","duration_s","centroid_hz","rms_db","dyn_p95_p10_db","zcr","rolloff_hz","hpr","key_conf"):
        v = r.get(k)
        if v is not None and v == v:
            vec.append(float(v)); fn.append(k)
        else:
            vec.append(0.0); fn.append(k)
    # 指纹
    for k in ("voiced_ratio","n_vocal_runs","vibrato_rate_hz","vibrato_extent_cents",
              "intra_note_drift_cents","glide_ratio","macro_dyn_std_db","dyn_range_db",
              "spectral_flux_std","loop_lag_s","loop_peak_corr","loopiness"):
        v = fp.get(k)
        vec.append(float(v) if v is not None else 0.0); fn.append("fp_" + k)
    # 波形
    for k in ("crest_db","env_rough","low_attack_ms","silence_frac","onset_iv_cv","kurtosis","lra_db"):
        v = wv.get(k)
        vec.append(float(v) if v is not None else 0.0); fn.append("wf_" + k)
    # 旋律
    for k in ("density","pitch_range","leap_ratio","motif_repeat","lift_mean","rest_ratio"):
        v = ms.get(k)
        vec.append(float(v) if v is not None else 0.0); fn.append("mel_" + k)
    
    if not all_fn:
        all_fn = fn
    if len(vec) == len(all_fn):
        X.append(vec); y.append(tier(name)); names.append(name)
    else:
        skipped += 1

X = np.array(X); y = np.array(y)
from collections import Counter
tc = Counter(y)
print(f"样本: {X.shape[0]}  特征: {X.shape[1]}  分布: {dict(sorted(tc.items()))}", flush=True)

# ===== ML: RandomForest 分类 =====
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
pred = cross_val_predict(rf, X, y, cv=5)
acc3 = float(np.mean(pred == y))
baseline3 = max(Counter(y).values()) / len(y)
print(f"3类准确率: {acc3*100:.1f}% (基线 {baseline3*100:.1f}%)", flush=True)

rf.fit(X, y)
imp = rf.feature_importances_
idx = np.argsort(imp)[::-1]

# ===== 输出 Top30 =====
lines = [f"# 全特征 ML（{X.shape[1]} 维 × {X.shape[0]} 首）", "",
         f"3类准确率: **{acc3*100:.1f}%**（基线 {baseline3*100:.0f}%）",
         f"跳过: {skipped}", "",
         "| 排名 | 特征 | 重要性 |", "|---|---|---|"]
for i in range(min(30, len(idx))):
    fn = all_fn[idx[i]] if idx[i] < len(all_fn) else "?"
    lines.append(f"| {i+1} | {fn} | {imp[idx[i]]:.4f} |")

# 逐特征热度ρ
from scipy.stats import spearmanr
lines += ["", "## 逐特征热度ρ", ""]
rho_list = []
for i in range(X.shape[1]):
    if len(set(X[:, i])) >= 2:
        r, _ = spearmanr(X[:, i], y)
        rho_list.append((all_fn[i], round(float(r), 3)))
    else:
        rho_list.append((all_fn[i], 0.0))
rho_list.sort(key=lambda t: -abs(t[1]))
for k, r in rho_list[:15]:
    lines.append(f"- {k}: ρ={r:+.3f}")

lines += ["", "## Top25 重要性明细", ""]
for i in range(min(25, len(idx))):
    fn = all_fn[idx[i]]
    lines.append(f"- {fn}: {imp[idx[i]]:.4f}")

(ROOT/"eval"/"full_feature_ml.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:16]), flush=True)
print("...", flush=True)
print("Top10 特征:")
for i in range(min(10, len(idx))):
    fn = all_fn[idx[i]] if idx[i] < len(all_fn) else "?"
    print(f"  {i+1}. {fn}: {imp[idx[i]]:.4f}")
print("SAVED", flush=True)
