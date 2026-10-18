# -*- coding: utf-8 -*-
"""金标清洗 + 重测（从 gold 缓存直接关联）"""
import json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(r'F:\yinyue')
gold = json.loads((ROOT/"eval"/"netease_popularity_v2.json").read_text(encoding='utf-8'))
gold_m = json.loads((ROOT/"eval"/"v3_gold_cache.json").read_text(encoding='utf-8'))
report = json.loads((ROOT/"eval"/"v3_validation.json").read_text(encoding='utf-8'))
norms = report['norms']
DIMS = ["orn", "gap", "cent", "dyn", "tension", "micro"]

def norm(s):
    s = s.replace('.mp3', '').replace('.wav', '').replace('lib_', '')
    s = re.sub(r'[_]+', ' ', s)
    return re.sub(r'\s+', '', s).lower()

def clean_title(k):
    s = re.sub(r'\.mp3$', '', k)
    s = re.sub(r'[《\[（(].*?[》\])）)]', '', s)
    s = re.sub(r'电影.*|电视剧.*', '', s)
    return re.split(r'\s*-\s*', s)[-1].strip()

# 金标 key 索引
gold_valid = {k: v for k, v in gold.items() if v.get('popularity') and v.get('netease_id')}
gidx = {norm(k): k for k in gold_valid}

# 清洗
def suspect(k):
    exp = clean_title(k)
    got = gold_valid[k]['netease_name']
    if exp[:4] and exp[:4] in got: return False
    if got[:4] and got[:4] in exp: return False
    return True
sus = {k for k in gold_valid if suspect(k)}
print(f'清洗 {len(sus)}/{len(gold_valid)} 错误匹配')

def cat_of(bpm):
    if bpm < 100: return "C1"
    if bpm < 120: return "C2"
    if bpm < 135: return "C3"
    return "C4"

def score_of(m):
    cat = cat_of(m['bpm']); parts = []
    for d in DIMS:
        n = norms.get(cat, {}).get(d)
        v = m.get(d)
        if not n or v is None: continue
        s = 100*(1 - abs(v-n['med'])/n['scale'])
        if v < n['p5']: s -= (n['p5']-v)/n['scale']*100
        if v > n['p95']: s -= (v-n['p95'])/n['scale']*100
        parts.append(s)
    return round(float(np.mean(parts)),1) if parts else None

rows = []
for stem, m in gold_m.items():
    if not m: continue
    k = gidx.get(norm(stem))
    if not k or k in sus: continue
    sc = score_of(m)
    if sc is None: continue
    rows.append((sc, float(gold_valid[k]['popularity']), k[:34]))

print(f'清洗后 n={len(rows)}')
if len(rows) >= 8:
    from scipy.stats import spearmanr
    rho, p = spearmanr([x[0] for x in rows], [x[1] for x in rows])
    verdict = 'PASS' if rho >= 0.35 else 'FAIL'
    print(f'清洗后 Spearman ρ={rho:.3f} p={p:.4f} {verdict} (目标≥0.35)')
    for s, pop, name in sorted(rows, key=lambda x: -x[1])[:6]:
        print(f'  流行{pop:5.1f} 分{s:5.1f} {name}')
    print('  ...')
    for s, pop, name in sorted(rows, key=lambda x: x[1])[:6]:
        print(f'  流行{pop:5.1f} 分{s:5.1f} {name}')
