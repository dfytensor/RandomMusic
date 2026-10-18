# -*- coding: utf-8 -*-
"""金标终局验证: 全特征扫描 × 清洗后 n=32 NetEase 流行度
1) 单特征 Spearman 全扫 (lib_metric 49维 + v3 6维 + emotion)
2) Top 特征组合 (秩平均 / LOOCV 岭回归) → ρ
目标 ρ≥0.4; 达不到则以历史 ML 失败一并定论
"""
import json, re, sys, warnings
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

gold = json.loads((ROOT/"eval"/"netease_popularity_v2.json").read_text(encoding='utf-8'))

def norm(s):
    s = s.replace('.mp3', '').replace('.wav', '').replace('lib_', '')
    s = re.sub(r'[_]+', ' ', s)
    return re.sub(r'\s+', '', s).lower()

def clean_title(k):
    s = re.sub(r'\.mp3$', '', k)
    s = re.sub(r'[《\[（(].*?[》\])）)]', '', s)
    s = re.sub(r'电影.*|电视剧.*', '', s)
    return re.split(r'\s*-\s*', s)[-1].strip()

gold_valid = {k: v for k, v in gold.items() if v.get('popularity') and v.get('netease_id')}
def suspect(k):
    exp = clean_title(k); got = gold_valid[k]['netease_name']
    if exp[:4] and exp[:4] in got: return False
    if got[:4] and got[:4] in exp: return False
    return True
clean = {k: float(v['popularity']) for k, v in gold_valid.items() if not suspect(k)}
print(f"清洗金标 n={len(clean)}", flush=True)

# ===== 特征集 =====
feats = {}   # stem -> {feat: val}
def add(src):
    if isinstance(src, list):
        src = {d['file']: d for d in src if isinstance(d, dict) and 'file' in d}
    if not isinstance(src, dict): return
    for stem, d in src.items():
        if not isinstance(d, dict): continue
        feats.setdefault(norm(stem), {}).update(
            {k: v for k, v in d.items() if isinstance(v, (int, float)) and np.isfinite(v)})

libm = json.loads((ROOT/"eval"/"lib_metric_cache.json").read_text(encoding='utf-8'))
add(libm)
for f in ["v3_gold_cache.json", "v3_test_cache.json", "v3_train_cache.json",
          "library_dims_cache.json", "wave_metric_cache.json",
          "library_emotion_scores.json"]:
    p = ROOT/"eval"/f
    if p.exists(): add(json.loads(p.read_text(encoding='utf-8')))

# 对齐
rows = []
for k, pop in clean.items():
    fv = feats.get(norm(k))
    if fv: rows.append((fv, pop, k))
print(f"特征可对齐 n={len(rows)}", flush=True)

# ===== 单特征扫描 (覆盖 ≥85%, 缺失中位补齐) =====
fset = {}
for fv, pop, k in rows:
    for f, v in fv.items():
        if isinstance(v, (int, float)) and np.isfinite(v):
            fset.setdefault(f, []).append(v)
common = [f for f, v in fset.items() if len(v) >= 0.85*len(rows) and len(set(v)) > 3]
print(f"可用特征数 {len(common)}", flush=True)
for f in common:
    have = {}
    for fv, pop, k in rows:
        v = fv.get(f)
        if isinstance(v, (int, float)) and np.isfinite(v):
            have[k] = float(v)
    med = float(np.median(list(have.values())))
    for fv, pop, k in rows:
        if k not in have: fv[f] = med

results = []
y = [r[1] for r in rows]
for f in common:
    x = [r[0][f] for r in rows]
    rho, p = spearmanr(x, y)
    results.append((rho, p, f))
results.sort(key=lambda t: -abs(t[0]))
print("\n单特征 Top10 |ρ|:", flush=True)
for rho, p, f in results[:10]:
    print(f"  {f:24s} rho={rho:+.3f} p={p:.3f}", flush=True)

# ===== 组合: 秩平均 top-k (k=2..5, 全组合) =====
from itertools import combinations
top_feats = [f for _, _, f in results[:8] if True]
def rankdata(x):
    order = np.argsort(np.argsort(x))
    return order.astype(float)
y_r = rankdata(y)
best_combos = []
for k in range(2, 6):
    for combo in combinations(top_feats, k):
        # 方向: 按单特征 rho 符号对齐
        signs = []
        rr = dict((f, r) for r, _, f in results)
        signs = [np.sign(rr[f]) if rr[f] != 0 else 1 for f in combo]
        m = np.mean([signs[i]*rankdata([r[0][f] for r in rows]) for i, f in enumerate(combo)], axis=0)
        rho, p = spearmanr(m, y)
        best_combos.append((rho, p, combo))
best_combos.sort(key=lambda t: -t[0])
print("\n秩平均组合 Top5:", flush=True)
for rho, p, combo in best_combos[:5]:
    print(f"  rho={rho:+.3f} p={p:.4f} {combo}", flush=True)

# ===== LOOCV 岭回归 (top8 特征, 方向对齐后标准化) =====
from numpy.linalg import solve
rr = dict((f, r) for r, _, f in results)
X = np.column_stack([rankdata([r[0][f] for r in rows]) for f in top_feats])
# 方向修正
signs = np.array([1 if rr[f] >= 0 else -1 for f in top_feats])
X = X * signs
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
yv = np.array(y, dtype=float)
preds = np.zeros(len(yv))
lam = 1.0
for i in range(len(yv)):
    tr = np.arange(len(yv)) != i
    Xt, yt = X[tr], yv[tr]
    w = solve(Xt.T @ Xt + lam*np.eye(X.shape[1]), Xt.T @ yt)
    preds[i] = X[i] @ w
rho_cv, p_cv = spearmanr(preds, yv)
print(f"\nLOOCV 岭回归(top{len(top_feats)}特征): rho={rho_cv:+.3f} p={p_cv:.4f}", flush=True)

rho0 = max(r[0] for r in results)
print(f"\n判定: 最佳单特征 rho={rho0:+.3f}, 最佳组合 rho={max(b[0] for b in best_combos):+.3f}, "
      f"LOOCV rho={rho_cv:+.3f} → {'PASS(≥0.4)' if max(rho0, max(b[0] for b in best_combos), rho_cv) >= 0.4 else 'FAIL(<0.4) 金标不可达定论'}", flush=True)
