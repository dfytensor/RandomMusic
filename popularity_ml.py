# -*- coding: utf-8 -*-
"""全指标 × 热度 ML 分析：RandomForest 特征重要性 + 留一验证
用全部 571 首 × 所有缓存指标，预测热度层级，看哪些特征真的能预测爆款
-> eval/popularity_ml_analysis.md
"""
import json, re, sys, warnings
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

# ===== 热度标签 =====
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

def tier(stem):
    for a in MEGA_A:
        if a in stem: return 3
    for h in MEGA_S:
        if h in stem: return 3
    for a in KNOWN_A:
        if a in stem: return 2
    return 1

# ===== 合并全部特征 =====
feats = json.loads((ROOT/"corpus"/"features.json").read_text(encoding="utf-8"))
fp_cache = json.loads((ROOT/"eval"/"lib_metric_cache.json").read_text(encoding="utf-8"))
wv_cache = json.loads((ROOT/"eval"/"wave_metric_cache.json").read_text(encoding="utf-8"))
ms_data = json.loads((ROOT/"corpus"/"melody_stats.json").read_text(encoding="utf-8"))["per_song"]

X, y, names, feature_names = [], [], [], []
feat_names = list(next(iter(feats.values())).keys())
feat_names = [k for k in feat_names if k not in ("file",) and 
              isinstance(next(iter(feats.values()))[k], (int, float))]

for name, r in feats.items():
    t = tier(name.replace(" ", "_"))
    vec = []
    vec_names = []
    for k in feat_names:
        v = r.get(k)
        if v is not None and v == v:
            vec.append(float(v))
            vec_names.append(k)
    
    # 挂载指纹/波形缓存
    wav_stem = "lib_" + name.replace(" ", "_")[:60]
    fp = fp_cache.get(wav_stem, {})
    wv = wv_cache.get(wav_stem, {})
    for src, prefix in [(fp, "fp_"), (wv, "wf_")]:
        for k, v in src.items():
            if isinstance(v, (int, float)) and v == v and k not in ("skip",):
                vec.append(float(v))
                vec_names.append(prefix + k)
    
    # 旋律指标
    abc = ROOT / "eval" / "library_transcripts" / wav_stem / "score.abc"
    if abc.exists():
        try:
            sys.path.insert(0, str(ROOT))
            from melody_stats import analyze_song
            m = analyze_song(str(abc))
            for k in ("density", "pitch_range", "leap_ratio", "motif_repeat", "lift_mean", "rest_ratio"):
                if m.get(k) is not None:
                    vec.append(float(m[k]))
                    vec_names.append("mel_" + k)
        except:
            pass
    
    # 对齐维度（用第一个向量的长度做标准）
    if not feature_names:
        feature_names = vec_names
    if len(vec) == len(feature_names):
        X.append(vec)
        y.append(t)
        names.append(name)

X = np.array(X)
y = np.array(y)
print(f"样本: {X.shape[0]}  特征: {X.shape[1]}  热度分布: {np.bincount(y)[1:]}", flush=True)

# ===== RandomForest 特征重要性 =====
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score

# 分类（S/A/B 三层）
rf_c = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_c.fit(X, y)
importances = rf_c.feature_importances_
idx = np.argsort(importances)[::-1]

# 准确率
from sklearn.model_selection import cross_val_predict
pred = cross_val_predict(RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
                         X, y, cv=5)
acc = float(np.mean(pred == y))

# ===== 输出 =====
lines = ["# 全指标 × 热度 ML 分析（RandomForest + 交叉验证）", "",
         f"样本: {X.shape[0]} 首 × {X.shape[1]} 维特征",
         f"分类准确率 (5-fold CV): **{acc*100:.1f}%**（随机基线 ~{100/max(3, len(set(y))):.0f}%）",
         f"热度分布: S={sum(y==3)} A={sum(y==2)} B={sum(y==1)}", "",
         "## RandomForest 特征重要性 Top20", "",
         "| 排名 | 特征 | 重要性 |", "|---|---|---|"]
for i in range(min(20, len(idx))):
    fn = feature_names[idx[i]] if idx[i] < len(feature_names) else f"feature_{idx[i]}"
    lines.append(f"| {i+1} | {fn} | {importances[idx[i]]:.4f} |")

# ===== 回归（连续热度 1-3） =====
rf_r = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_r.fit(X, y.astype(float))
reg_importances = rf_r.feature_importances_
reg_idx = np.argsort(reg_importances)[::-1]
lines += ["", "## 回归特征重要性 Top15", "",
          "| 排名 | 特征 | 重要性 |", "|---|---|---|"]
for i in range(min(15, len(reg_idx))):
    fn = feature_names[reg_idx[i]] if reg_idx[i] < len(feature_names) else f"feature_{reg_idx[i]}"
    lines.append(f"| {i+1} | {fn} | {reg_importances[reg_idx[i]]:.4f} |")

# ===== 解读 =====
top5 = [(feature_names[i], importances[i]) for i in idx[:5]]
lines += ["", "## 解读", ""]
lines.append("1. **分类准确率**：如果显著高于随机基线，说明声学特征确实编码了部分热度信息")
lines.append("2. **特征重要性**：排名靠前的特征就是真正预测爆款的声学维度")
lines.append("3. **注意**：高重要性 ≠ 因果关系——可能反映的是时代/制作水平的混淆变量")
lines.append("")
lines.append("**真正结论**：如果准确率 <60%，说明声学特征只能编码热度的很小一部分，")
lines.append("大部分热度来自旋律写作、歌词、演唱、时代——这些是声学指标测不到的。")

(ROOT / "eval" / "popularity_ml_analysis.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[-20:]))
