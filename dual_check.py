# -*- coding: utf-8 -*-
"""双任务: (A) 11 seed 演唱表现力层交叉验证 (B) 03_v2 6 seed 第二盲测集"""
import json, shutil, sys, warnings
from pathlib import Path
import numpy as np
import librosa
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from vocal_expression import vocal_expression_metrics
from score_v3 import measure, score_of

# ===== (A) 01_v2 11 seed 演唱层 =====
rank_cache = json.loads((ROOT/"eval/seed_rank01_cache.json").read_text(encoding='utf-8'))
v3_rows = json.loads((ROOT/"eval/seed_rank01_v3scores.json").read_text(encoding='utf-8')) \
    if (ROOT/"eval/seed_rank01_v3scores.json").exists() else None

# 从 listen_test 目录反查 v3 排名
files = sorted((ROOT/"listen_test"/"rank01_听测").glob("rank*.flac"))
ve = {}
for f in files:
    rank = int(f.name[4:6])
    stem = f.stem[7:]  # strip rankXX_
    m = rank_cache.get(stem)
    t0 = m.get('seg_t', 60.0) if m else 60.0
    try:
        y, sr = librosa.load(str(f), sr=22050, mono=True, offset=t0, duration=30)
        r = vocal_expression_metrics(y, sr)
        ve[rank] = r
        print(f"rank{rank:02d} {stem}: pitch_dev={r.get('pitch_dev')} gap_cv={r.get('gap_cv')} "
              f"orn={r.get('orn_density')} slide={r.get('slide_frac')} dyn={r.get('phrase_dyn')}", flush=True)
    except Exception as e:
        print(f"rank{rank:02d} ERR {e}", flush=True)

pairs = [(rk, ve[rk].get('pitch_dev')) for rk in ve if ve[rk].get('pitch_dev') is not None]
if len(pairs) >= 6:
    rho, p = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
    # rank 小=好; pitch_dev 人耳感知为"跑调感" = 越低越好 → 期望正相关 (rank大 pitch_dev大)
    print(f"\n(A) v3排名 vs pitch_dev: ρ={rho:+.3f} p={p:.4f} (期望为正: 越差的排名跑调感越强)", flush=True)
for key in ('gap_cv', 'orn_density', 'slide_frac', 'phrase_dyn'):
    pr = [(rk, ve[rk].get(key)) for rk in ve if ve[rk].get(key) is not None]
    if len(pr) >= 6:
        rho, p = spearmanr([a for a, _ in pr], [b for _, b in pr])
        print(f"    vs {key}: ρ={rho:+.3f} p={p:.4f}", flush=True)

# ===== (B) 03_v2 6 seed 第二盲测集 =====
report = json.loads((ROOT/"eval"/"v3_validation.json").read_text(encoding='utf-8'))
norms = report['norms']
SRC = ROOT/"novel_songs"/"batch_gacha"/"03_v2"
OUT = ROOT/"listen_test"/"rank03_听测"
OUT.mkdir(parents=True, exist_ok=True)
cache_p = ROOT/"eval/seed_rank03_cache.json"
cache = json.loads(cache_p.read_text(encoding='utf-8')) if cache_p.exists() else {}
rows = []
for f in sorted(SRC.glob("seed*.flac")):
    m = cache.get(f.stem)
    if not m:
        for _ in range(2):
            try:
                m = measure(f)
                if m: break
            except Exception as e:
                print(f"  {f.stem} retry ERR {type(e).__name__}", flush=True)
        cache[f.stem] = m
        cache_p.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
    sc = score_of(m)['score'] if m else None
    rows.append((sc, f, m))
    print(f"  03 {f.stem}: {sc}", flush=True)
rows.sort(key=lambda x: -(x[0] if x[0] is not None else -999))
print("\n(B) 03_v2《》排名:", flush=True)
for i, (sc, f, m) in enumerate(rows, 1):
    shutil.copy2(f, OUT/f"rank{i:02d}_{f.stem}.flac")
    print(f"  #{i} {sc:6.1f} {f.stem} seg={m.get('seg_t') if m else '-'}", flush=True)
