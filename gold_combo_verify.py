# -*- coding: utf-8 -*-
"""组合分双验证:
A) 共识效度: 组合分能否区分 经典 vs 普通 (独立于流行度的第二效标)
B) 选择偏差: bootstrap OOB — 每次重抽选最优组合, 袋外评估真实 ρ
"""
import json, re, sys, warnings
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from itertools import combinations

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
gv = {k: v for k, v in gold.items() if v.get('popularity') and v.get('netease_id')}
def susp(k):
    e = clean_title(k); g = gv[k]['netease_name']
    if e[:4] and e[:4] in g: return False
    if g[:4] and g[:4] in e: return False
    return True
clean = {k: float(v['popularity']) for k, v in gv.items() if not susp(k)}

feats = {}
def add(src):
    if isinstance(src, list):
        src = {d['file']: d for d in src if isinstance(d, dict) and 'file' in d}
    if not isinstance(src, dict): return
    for stem, d in src.items():
        if not isinstance(d, dict): continue
        feats.setdefault(norm(stem), {}).update(
            {k: v for k, v in d.items() if isinstance(v, (int, float)) and np.isfinite(v)})
for f in ["v3_gold_cache.json", "v3_test_cache.json", "v3_train_cache.json", "wave_metric_cache.json"]:
    add(json.loads((ROOT/"eval"/f).read_text(encoding='utf-8')))

rows = [(feats[norm(k)], pop, k) for k, pop in clean.items() if norm(k) in feats]
print(f"金标 n={len(rows)}", flush=True)

COMBO = {"cent": -1, "dyn": +1, "bpm": -1, "gap": +1}
def rankdata(x):
    return np.argsort(np.argsort(np.asarray(x, dtype=float))).astype(float)

def combo_score(fvs):
    """4特征秩平均(方向对齐), 返回越大越流行的分"""
    cols = []
    for f, s in COMBO.items():
        v = [fv.get(f, np.nan) for fv in fvs]
        v = np.array(v, dtype=float)
        nan = np.isnan(v); v[nan] = np.nanmedian(v)
        r = rankdata(v)
        cols.append(s*r)
    return np.mean(cols, axis=0)

X = [r[0] for r in rows]; y = np.array([r[1] for r in rows])
sc = combo_score(X)
rho, p = spearmanr(sc, y)
print(f"A) 冻结组合 rho={rho:+.3f} p={p:.4f}", flush=True)

# ===== A) 共识效度 =====
CL = ["真的爱你","光辉岁月","夜曲","听妈妈的话","安静","我难过","千年泪","好想好想",
      "双截棍","回到过去","开不了口","发如雪","彩虹","稻香","威廉古堡","外婆",
      "情非得已","海阔天空","专属天使","如果我变成回忆","热带雨林","绿光","遇见","倔强","知足"]
MD = ["爱丫爱丫","发呆","樱花草","光荣","活埋","会长大的幸福","紫藤花",
      "一辈子的孤单","猪之歌","三国杀"]
feats_all = feats
def fvs_of(names):
    out = []
    for pat in names:
        hit = [k for k in feats_all if pat in k]
        if hit: out.append(feats_all[hit[0]])
    return out
# 共识组直接用冻结组合分(与金标同一函数, 全库特征)
all_keys = list(feats_all)
def find(pat):
    h = [k for k in all_keys if pat in k]
    return feats_all[h[0]] if h else None
cl_f = [f for f in (find(p) for p in CL) if f]
md_f = [f for f in (find(p) for p in MD) if f]
cl_s = combo_score(cl_f); md_s = combo_score(md_f)
u, pu = mannwhitneyu(cl_s, md_s, alternative='greater')
print(f"A2) 共识: 经典 n={len(cl_s)} 中位={np.median(cl_s):.1f} | 普通 n={len(md_s)} 中位={np.median(md_s):.1f} "
      f"U p={pu:.4f} {'PASS' if pu<0.05 else 'FAIL'}", flush=True)

# ===== B) Bootstrap OOB (整个选组合流程) =====
rng = np.random.default_rng(42)
Xm = [dict(fv) for fv in X]
n = len(rows)
oob_pools = []
feats_names = sorted(set().union(*[set(fv) for fv in Xm]) & {"cent","dyn","bpm","gap","orn","tension","micro","loopiness","beat_R","dyn_range_db","voiced_ratio","loop_peak_corr"})
B = 300
oob_spearman = []
combo_win = {}
for b in range(B):
    idx = rng.choice(n, n, replace=True)
    oob = np.setdiff1d(np.arange(n), idx)
    if len(oob) < 8: continue
    yb = y[idx]
    best_r, best_c = -2, None
    for k in (2, 3, 4, 5):
        for c in combinations(feats_names, k):
            # 方向按袋内单特征 rho
            signs = []
            ok = True
            for f in c:
                xv = np.array([Xm[i].get(f, np.nan) for i in idx], dtype=float)
                nan = np.isnan(xv); xv[nan] = np.nanmedian(xv)
                if len(set(xv)) < 3: ok = False; break
                rr, _ = spearmanr(xv, yb)
                signs.append(1 if rr >= 0 else -1)
            if not ok: continue
            cols = []
            for f, s in zip(c, signs):
                xv = np.array([Xm[i].get(f, np.nan) for i in idx], dtype=float)
                nan = np.isnan(xv); xv[nan] = np.nanmedian(xv)
                r_all = rankdata(np.concatenate([xv, [Xm[j].get(f, np.nan) for j in oob]]))
                cols.append(s*r_all[:len(idx)])
                # 袋外用同一秩空间
            m_in = np.mean(cols, axis=0)
            rr_in, _ = spearmanr(m_in, yb)
            if rr_in > best_r: best_r, best_c = rr_in, (c, signs)
    if best_c is None: continue
    c, signs = best_c
    cols_out = []
    for f, s in zip(c, signs):
        xv = np.array([Xm[i].get(f, np.nan) for i in list(idx)+list(oob)], dtype=float)
        nan = np.isnan(xv); xv[nan] = np.nanmedian(xv)
        r_all = rankdata(xv)
        cols_out.append(s*r_all[len(idx):])
    m_out = np.mean(cols_out, axis=0)
    rr_oob, _ = spearmanr(m_out, y[oob])
    oob_spearman.append(rr_oob)
    combo_win[c] = combo_win.get(c, 0) + 1
    if (b+1) % 50 == 0: print(f"  boot {b+1}/{B}", flush=True)

oob = np.array(oob_spearman)
print(f"\nB) Bootstrap OOB: n={len(oob)} 中位rho={np.median(oob):+.3f} 均值={np.mean(oob):+.3f} "
      f"P(rho>=0.4)={np.mean(oob>=0.4):.2f}", flush=True)
top_wins = sorted(combo_win.items(), key=lambda x: -x[1])[:3]
print(f"  最常胜组合: {top_wins}", flush=True)
