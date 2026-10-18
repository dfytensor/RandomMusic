# -*- coding: utf-8 -*-
"""金标清洗 + 重测相关性（测量已缓存，只重跑统计）"""
import json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(r'F:\yinyue')
gold = json.loads((ROOT/"eval"/"netease_popularity_v2.json").read_text(encoding='utf-8'))
report = json.loads((ROOT/"eval"/"v3_validation.json").read_text(encoding='utf-8'))

def clean_title(k):
    s = re.sub(r'\.mp3$', '', k)
    # 去掉影视剧标注
    s = re.sub(r'[《\[（(].*?[》\])）)]', '', s)
    s = re.sub(r'电影.*|电视剧.*', '', s)
    parts = re.split(r'\s*-\s*', s)
    return parts[-1].strip()

suspect = set()
for k, v in gold.items():
    if not v.get('netease_id'): continue
    exp = clean_title(k)
    got = v['netease_name']
    # 严格: got 必须含 exp 前4字 或 exp 含 got 前4字, 否则清洗
    if not (exp[:4] and exp[:4] in got) and not (got[:4] and got[:4] in exp):
        suspect.add(k)
print(f'清洗掉 {len(suspect)} 首错误匹配')

det = [r for r in report['gold_detail']]
# gold_detail 里 song 是 stem，需要映射回 mp3 key
gold_by_norm = {}
for k in gold:
    s = re.sub(r'\.mp3$', '', k); s = re.sub(r'[_]+', ' ', s)
    gold_by_norm[re.sub(r'\s+', '', s).lower()] = k

clean_rows = []
for r in det:
    stem = r['song']
    # stem 形如 lib_Beyond_-_真的爱你 (截断20字)
    for nk, orig in gold_by_norm.items():
        if nk.startswith(re.sub(r'lib_', '', re.sub(r'[_\s]+', '', stem)).lower()[:8]) or \
           re.sub(r'lib_', '', re.sub(r'[_\s]+', '', stem)).lower()[:8] in nk:
            if orig in suspect: break
            pop = gold[orig].get('popularity')
            if pop:
                clean_rows.append((r['score'], float(pop), orig[:30]))
            break

print(f'清洗后金标 n={len(clean_rows)}')
if len(clean_rows) >= 10:
    from scipy.stats import spearmanr
    rho, p = spearmanr([x[0] for x in clean_rows], [x[1] for x in clean_rows])
    print(f'清洗后 Spearman ρ={rho:.3f} p={p:.4f} {"PASS" if rho>=0.35 else "FAIL"}')
    # 按流行度排序看
    for s, pop, name in sorted(clean_rows, key=lambda x: -x[1])[:5]:
        print(f'  高流行 {pop} -> 分 {s:.1f} {name}')
    for s, pop, name in sorted(clean_rows, key=lambda x: x[1])[:5]:
        print(f'  低流行 {pop} -> 分 {s:.1f} {name}')
