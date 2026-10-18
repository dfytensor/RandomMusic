# -*- coding: utf-8 -*-
"""交叉验证: emotion_score_v2 独立给 11 seed 排名, 与 v3 排名比一致性"""
import sys, json, warnings
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from emotion_score_v2 import score_audio_v2

files = sorted((ROOT/"listen_test"/"rank01_听测").glob("rank*.flac"))
print(f"{len(files)} files", flush=True)
rows = []
for f in files:
    rank = int(f.name[4:6])
    try:
        sc = score_audio_v2(f)
        total = sc.get("total") if isinstance(sc, dict) else sc
    except Exception as e:
        print(f"  {f.name} ERR {e}", flush=True); total = None
    rows.append((rank, total))
    print(f"  rank{rank:02d}: emo={total}", flush=True)

ok = [(r, t) for r, t in rows if t is not None]
if len(ok) >= 6:
    rho, p = spearmanr([x[0] for x in ok], [x[1] for x in ok])
    # 注意 rank 越小越好, emo 越大越好 → 期望负相关
    print(f"\nv3排名 vs 情绪分: Spearman ρ={rho:+.3f} p={p:.4f} (期望显著为负)", flush=True)
    emo_sorted = sorted(ok, key=lambda x: -x[1])
    print("情绪分排名:", " > ".join(f"r{r}({t:.0f})" for r, t in emo_sorted), flush=True)
