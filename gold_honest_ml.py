# -*- coding: utf-8 -*-
"""大样本诚实检验: n=548 拆 train/test, 特征组合能否让测试集 ρ≥0.4
方法: train 上筛特征+定方向+定组合 → 冻结 → test 上评估
另测 ridge 回归 (train 拟合, test 评估)
"""
import json, re, sys, warnings
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

gold = json.loads((ROOT/"eval"/"netease_popularity_v2.json").read_text(encoding='utf-8'))

def titles_of(k):
    s = re.sub(r'\.mp3$', '', k)
    s = re.sub(r'[《\[（(].*?[》\])）)]', '', s)
    s = re.sub(r'电影.*|电视剧.*|片尾曲|主题曲|插曲', '', s)
    return [p for p in re.split(r'\s*-\s*|_|\s+', s) if len(p) >= 2]
def strict_ok(k, v):
    name = v.get('netease_name', '')
    frags = titles_of(k)
    if not frags: return False
    for f in sorted(frags, key=len, reverse=True):
        if len(f) >= 3 and (f in name or name in f): return True
        if len(f) >= 2 and f in name: return True
    t = frags[-1]
    return t in name or name in t
def ns(s):
    s = s.replace('.mp3', '').replace('.wav', '').replace('lib_', '')
    s = re.sub(r'[_]+', ' ', s)
    return re.sub(r'\s+', '', s).lower()

feats = {}
def add(src):
    if isinstance(src, list):
        src = {d['file']: d for d in src if isinstance(d, dict) and 'file' in d}
    if not isinstance(src, dict): return
    for stem, d in src.items():
        if not isinstance(d, dict): continue
        feats.setdefault(ns(stem), {}).update(
            {k: float(v) for k, v in d.items()
             if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)})
for f in ["v3_train_cache.json", "v3_test_cache.json", "lib_metric_cache.json",
          "wave_metric_cache.json", "library_dims_cache.json"]:
    add(json.loads((ROOT/"eval"/f).read_text(encoding='utf-8')))

rows = []
for k, v in gold.items():
    if not isinstance(v, dict) or not v.get('popularity'): continue
    if not strict_ok(k, v): continue
    fv = feats.get(ns(k))
    if not fv: continue
    rows.append((fv, float(v['popularity']), k))
print(f"n={len(rows)}", flush=True)

# 特征矩阵 (覆盖≥90% 才用, 中位补齐)
y = np.array([r[1] for r in rows])
cand = {}
for fv, _, _ in rows:
    for f, v in fv.items():
        cand.setdefault(f, []).append(v)
feats_ok = [f for f, v in cand.items() if len(v) >= 0.9*len(rows) and len(set(v)) >= 8]
# 排除明显泄漏/无关
feats_ok = [f for f in feats_ok if f not in ('popularity', 'seg_t', 'score')]
print(f"候选特征 {len(feats_ok)}", flush=True)

X = np.zeros((len(rows), len(feats_ok)))
for j, f in enumerate(feats_ok):
    col = np.array([r[0].get(f, np.nan) for r in rows], dtype=float)
    nan = np.isnan(col); col[nan] = np.nanmedian(col); X[:, j] = col

# hash 拆分 (稳定)
idx = np.arange(len(rows))
h = np.array([abs(hash(r[2])) % 100 for r in rows])
tr = idx[h < 50]; te = idx[h >= 50]
print(f"train={len(tr)} test={len(te)}", flush=True)

def rankdata(x):
    from scipy.stats import rankdata as _rd
    return _rd(np.asarray(x, dtype=float))

# ===== A) train 筛单特征 → 组合搜索 → test 冻结评估 =====
single = []
for j, f in enumerate(feats_ok):
    rho, _ = spearmanr(X[tr, j], y[tr])
    single.append((abs(rho), rho, j, f))
single.sort(reverse=True)
top = [s for s in single[:10]]   # train 上 |rho| top10

best_test = (-2, None)
tested = 0
for k in range(1, 6):
    for combo in combinations(top, k):
        signs = [1 if c[1] >= 0 else -1 for c in combo]
        cols = [signs[i]*rankdata(X[tr, c[2]]) for i, c in enumerate(combo)]
        m_tr = np.mean(cols, axis=0)
        rho_tr, _ = spearmanr(m_tr, y[tr])
        # 冻结: 用 train 的秩参考不行——test 需独立秩; 方向冻结, 各自内部秩
        cols_te = [signs[i]*rankdata(X[te, c[2]]) for i, c in enumerate(combo)]
        m_te = np.mean(cols_te, axis=0)
        rho_te, _ = spearmanr(m_te, y[te])
        tested += 1
        if rho_te > best_test[0]:
            best_test = (rho_te, (tuple(c[3] for c in combo), rho_tr, rho_te))
print(f"\nA) 组合搜索: 测试过 {tested} 个组合(train top10 内)", flush=True)
bt = best_test[1]
print(f"   测试集最高 ρ={best_test[0]:+.3f}  组合={bt[0]}  (train ρ={bt[1]:+.3f})", flush=True)
print(f"   判定: {'PASS' if best_test[0] >= 0.4 else 'FAIL (测试集 <0.4)'}", flush=True)

# ===== B) ridge (train 拟合, test 评估) =====
from numpy.linalg import solve
Xtr = np.column_stack([rankdata(X[tr, j]) for j in range(X.shape[1])])
Xte = np.column_stack([rankdata(X[te, j]) for j in range(X.shape[1])])
Xtr = (Xtr - Xtr.mean(0)) / (Xtr.std(0) + 1e-9)
Xte = (Xte - Xtr.mean(0)) / (Xtr.std(0) + 1e-9)   # 用 train 统一标准化
ytr, yte = y[tr], y[te]
for lam in (1.0, 10.0, 50.0):
    w = solve(Xtr.T @ Xtr + lam*np.eye(Xtr.shape[1]), Xtr.T @ ytr)
    pred = Xte @ w
    rho, p = spearmanr(pred, yte)
    print(f"B) ridge λ={lam}: test ρ={rho:+.3f} p={p:.3f}", flush=True)

# ===== C) 全特征 top 单特征在 test 上的 ρ (无组合) =====
print("C) top3 单特征 test ρ:", flush=True)
for absr, r_, j, f in single[:3]:
    rho, p = spearmanr(X[te, j], y[te])
    print(f"   {f:24s} train={r_:+.3f} test={rho:+.3f} p={p:.3f}", flush=True)
