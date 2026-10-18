# -*- coding: utf-8 -*-
"""332 个存量渲染 v3 重扫: 每首歌重选冠军 (纯评分, 无渲染)
断点续跑: eval/rescan_v3_cache.json; 每次默认处理 ~50 个新文件
"""
import json, sys, time, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
CACHE = ROOT / "eval" / "rescan_v3_cache.json"
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 50

def _measure_one(path_str):
    import warnings as w
    w.filterwarnings("ignore")
    import sys as s
    s.path.insert(0, r"F:\yinyue")
    import json as j
    from pathlib import Path as P
    rep = j.loads((P(r"F:\yinyue")/"eval"/"v3_validation.json").read_text(encoding="utf-8"))
    from score_v3 import measure, score_of
    m = measure(P(path_str))
    return path_str, (score_of(m) if m else None)

def main():
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    files = []
    for p in (ROOT / "novel_songs").rglob("*.flac"):
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        if "/album_voice/" in rel or "/_stems/" in rel:
            continue
        if rel in cache:
            continue
        files.append((rel, p))
    files.sort()
    todo = files[:BATCH]
    print(f"cache={len(cache)} total_pending={len(files)} this_run={len(todo)}", flush=True)
    if not todo:
        print("DONE_ALL", flush=True)
        return
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_measure_one, str(p)): rel for rel, p in todo}
        n = 0
        for f in as_completed(futs):
            rel = futs[f]
            try:
                _, sc = f.result()
            except Exception as e:
                sc = None
                print(f"  ERR {rel}: {type(e).__name__}", flush=True)
            cache[rel] = sc
            n += 1
            if n % 10 == 0:
                CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                el = (time.time()-t0)/60
                print(f"  progress {n}/{len(todo)} elapsed={el:.1f}min", flush=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"RUN_DONE {n}/{len(todo)} in {(time.time()-t0)/60:.1f}min, cache={len(cache)}", flush=True)

if __name__ == "__main__":
    main()
