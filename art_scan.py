# -*- coding: utf-8 -*-
"""预热 v3_art_cache.json: 对全库 571 stem 算 5 新维 (断点续跑, N 分批)
跑完后 import score_v3 即可秒级合并 train/test/gold 缓存并重建 norms
"""
import json, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from art_core import ART_DIMS, art_dims

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
CACHE_P = ROOT/"eval/v3_art_cache.json"
cache = json.loads(CACHE_P.read_text(encoding="utf-8")) if CACHE_P.exists() else {}

lib = sorted((ROOT/"eval/library_wavs").glob("*.wav"))
todo = []
for w in lib:
    v = ROOT/"human_stems"/w.stem/"vocals.wav"
    b = ROOT/"human_stems"/w.stem/"beats.json"
    if not v.exists():
        continue
    ent = cache.get(w.stem)
    if ent is not None and all(k in ent for k in ART_DIMS):
        continue
    todo.append((w.stem, v, b if b.exists() else None))
print(f"total={len(lib)} todo={len(todo)}", flush=True)
todo = todo[:N]

for i, (stem, v, b) in enumerate(todo, 1):
    try:
        cache[stem] = art_dims(v, b)
    except Exception as e:
        print(f"  ERR {stem}: {type(e).__name__}: {e}", flush=True)
    if i % 25 == 0:
        CACHE_P.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        print(f"  {i}/{len(todo)}", flush=True)
CACHE_P.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
print("DONE", flush=True)
