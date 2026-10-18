# -*- coding: utf-8 -*-
"""换嗓前后 v3 分对比: 12 冠军 vs album_voice/*_seedvc.flac"""
import json, sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
report = json.loads((ROOT/"eval"/"v3_validation.json").read_text(encoding='utf-8'))
from score_v3 import measure, score_of

CH = [("01", r"novel_songs\batch_gacha\01_v2\seed58888.flac"),
      ("02", r"novel_songs\batch_gacha\02_wide\seed7778.flac"),
      ("03", r"novel_songs\batch_gacha\03_v2\seed61001.flac"),
      ("04", r"novel_songs\batch_gacha\04\seed90406.flac"),
      ("05", r"novel_songs\batch_gacha\05_wide\seed1111.flac"),
      ("06", r"novel_songs\batch_gacha\06_wide\seed3690.flac"),
      ("07", r"novel_songs\batch_gacha\07_adj\seed1357_r1.flac"),
      ("08", r"novel_songs\batch_gacha\11\seed91104.flac"),
      ("09", r"novel_songs\batch_gacha\09\seed90904.flac"),
      ("10", r"novel_songs\batch_gacha\10\seed91007.flac"),
      ("11", r"novel_songs\batch_gacha\11\seed91103.flac"),
      ("12", r"novel_songs\batch_gacha\12\seed91202.flac")]

cache_p = ROOT/"eval"/"v3_seedvc_cache.json"
cache = json.loads(cache_p.read_text(encoding='utf-8')) if cache_p.exists() else {}

def score_file(p, key):
    if key in cache and cache[key] is not None:
        return cache[key], score_of(cache[key])
    for _ in range(2):
        try:
            m = measure(p)
            if m:
                cache[key] = m
                cache_p.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
                return m, score_of(m)
        except Exception as e:
            print(f"  retry {key}: {type(e).__name__}", flush=True)
    cache[key] = None
    cache_p.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
    return None, None

print("| # | 原分 | 换嗓后 | Δ | 原cent | 新cent | 类别 |", flush=True)
orig_scores, new_scores = [], []
for no, rel in CH:
    p0 = ROOT/rel
    p1 = ROOT/"album_voice"/f"{no}_seedvc.flac"
    if not p1.exists():
        print(f"| {no} | - | 缺文件 | | | | |"); continue
    m0, s0 = score_file(p0, f"orig_{no}")
    m1, s1 = score_file(p1, f"seedvc_{no}")
    o, n = (s0['score'] if s0 else None), (s1['score'] if s1 else None)
    if o is not None and n is not None:
        orig_scores.append(o); new_scores.append(n)
        c0, c1 = s0.get('cat'), s1.get('cat')
        print(f"| {no} | {o:.1f} | {n:.1f} | {n-o:+.1f} | "
              f"{m0.get('cent'):.0f} | {m1.get('cent'):.0f} | "
              f"{'类别变 '+c0+'→'+c1 if c0!=c1 else c0} |", flush=True)
    else:
        print(f"| {no} | {o} | {n} | ? | | | |", flush=True)

if orig_scores:
    a, b = np.array(orig_scores), np.array(new_scores)
    from scipy.stats import wilcoxon
    try:
        _, p = wilcoxon(a, b)
    except Exception:
        p = float('nan')
    print(f"\n原: 中位={np.median(a):.1f} 均值={a.mean():.1f}", flush=True)
    print(f"新: 中位={np.median(b):.1f} 均值={b.mean():.1f}", flush=True)
    print(f"配对Wilcoxon p={p:.4f} | 平均Δ={np.mean(b-a):+.1f} | 相关ρ={np.corrcoef(a,b)[0,1]:+.3f}", flush=True)
