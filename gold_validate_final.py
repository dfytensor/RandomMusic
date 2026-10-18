# -*- coding: utf-8 -*-
"""金标终验: n≈550 上 v3.3 副歌段分 vs NetEase popularity (严格清洗)"""
import json, re, sys, warnings
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
report = json.loads((ROOT/"eval"/"v3_validation.json").read_text(encoding='utf-8'))
norms = report['norms']
from score_v3 import score_of

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
    tail = frags[-1]
    return tail in name or name in tail

def norm_stem(s):
    s = s.replace('.mp3', '').replace('.wav', '').replace('lib_', '')
    s = re.sub(r'[_]+', ' ', s)
    return re.sub(r'\s+', '', s).lower()

# 曲库测量 (train+test 缓存)
ms = {}
for f in ["v3_train_cache.json", "v3_test_cache.json"]:
    ms.update(json.loads((ROOT/"eval"/f).read_text(encoding='utf-8')))
ms_idx = {norm_stem(k): v for k, v in ms.items() if v}
print(f"曲库测量 n={len(ms_idx)}", flush=True)

rows = []
for k, v in gold.items():
    if not isinstance(v, dict) or not v.get('popularity'): continue
    if not strict_ok(k, v): continue
    m = ms_idx.get(norm_stem(k))
    if not m: continue
    r = score_of(m)
    if not r: continue
    rows.append((r['score'], float(v['popularity']), k))
print(f"验证对齐 n={len(rows)} (strict_ok & 有pop & 有测量)", flush=True)

y = [r[0] for r in rows]; p = [r[1] for r in rows]
rho, pv = spearmanr(y, p)
dist = "饱和" if p.count(100.0) > len(p)*0.3 else "正常"
print(f"\n主结果: Spearman ρ={rho:+.3f} p={pv:.2e}  {'PASS(ρ≥0.4)' if rho >= 0.4 else 'FAIL(ρ<0.4)'}", flush=True)
print(f"pop 分布: =100的 {p.count(100.0)}/{len(p)} ({dist}), min={min(p):.0f} med={np.median(p):.0f}", flush=True)

# 分层: pop<100 子集 (非饱和) 上的相关
sub = [(s, pp) for s, pp, _ in rows if pp < 100]
if len(sub) >= 30:
    rho2, pv2 = spearmanr([x[0] for x in sub], [x[1] for x in sub])
    print(f"非饱和子集 n={len(sub)}: ρ={rho2:+.3f} p={pv2:.2e}", flush=True)

# pop=100 vs <100 两组分数差 (点二列)
a = [s for s, pp, _ in rows if pp == 100]
b = [s for s, pp, _ in rows if pp < 100]
from scipy.stats import mannwhitneyu
u, pu = mannwhitneyu(a, b, alternative='greater')
print(f"满分组 n={len(a)} 中位分={np.median(a):.1f} | 非满分组 n={len(b)} 中位分={np.median(b):.1f} | U p={pu:.4f}", flush=True)

# 高流行(top25%) vs 低流行(bottom25%)
thr_hi = np.percentile(p, 75); thr_lo = np.percentile(p, 25)
hi = [s for s, pp, _ in rows if pp >= thr_hi]
lo = [s for s, pp, _ in rows if pp <= thr_lo]
u2, pu2 = mannwhitneyu(hi, lo, alternative='greater')
print(f"流行上四分位 n={len(hi)} 分中位={np.median(hi):.1f} | 下四分位 n={len(lo)} 分中位={np.median(lo):.1f} | U p={pu2:.4f}", flush=True)
