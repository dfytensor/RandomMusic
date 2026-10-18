# -*- coding: utf-8 -*-
"""04 三目标重抽卡: v3 六维 (orn/gap/cent/dyn+tension/micro) 做适应度
旧8 seed (90400-90407, 旧4维冠军=90406 分2.5) vs 新8 seed (90410-90417), 双路并行
"""
import json, subprocess, sys, time, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
report = json.loads((ROOT/"eval"/"v3_validation.json").read_text(encoding='utf-8'))
from score_v3 import measure, score_of

OLD = list(range(90400, 90408))
NEW = list(range(90410, 90418))
OUT = ROOT / "novel_songs" / "batch_gacha" / "04_v3"
OUT.mkdir(exist_ok=True)
CACHE = ROOT / "eval" / "gacha04_v3_cache.json"
cache = json.loads(CACHE.read_text(encoding='utf-8')) if CACHE.exists() else {}

def v3score(flac: Path, key: str):
    if key in cache and cache[key] is not None:
        return cache[key]
    for _ in range(2):
        try:
            m = measure(flac)
            if m:
                s = score_of(m)
                cache[key] = s
                CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
                return s
        except Exception as e:
            print(f"    score retry {key}: {type(e).__name__}", flush=True)
            time.sleep(2)
    cache[key] = None
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
    return None

def render(seed: int):
    dst = OUT / f"seed{seed}.flac"
    if dst.exists() and dst.stat().st_size > 10_000_000:
        return seed, True, 0.0, "cached"
    import os
    env = dict(os.environ, PYTHONUTF8="1")
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "render_one.py", "04", str(seed), str(dst)],
        cwd=str(ROOT), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = r.returncode == 0 and dst.exists() and dst.stat().st_size > 10_000_000
    err = "" if ok else (r.stderr or r.stdout or "")[-300:]
    return seed, ok, time.time() - t0, err

print("=== 阶段A: 旧8 seed v3 评分 ===", flush=True)
old_rows = []
for s in OLD:
    p = ROOT / "novel_songs" / "batch_gacha" / "04" / f"seed{s}.flac"
    if not p.exists():
        print(f"  old {s}: 缺文件", flush=True); continue
    sc = v3score(p, f"old_{s}")
    v = sc["score"] if sc else None
    old_rows.append((v, s, sc))
    print(f"  old {s}: {v} [{sc['cat'] if sc else '?'}]", flush=True)

print("=== 阶段B: 新8 seed 双路渲染+评分 ===", flush=True)
results = {}
with ThreadPoolExecutor(max_workers=2) as ex:
    futs = {ex.submit(render, s): s for s in NEW}
    for f in as_completed(futs):
        seed, ok, dt, err = f.result()
        if not ok:
            print(f"  new {seed}: 渲染失败 {err[:200]}", flush=True)
            results[seed] = None
            continue
        print(f"  new {seed}: 渲染OK {dt/60:.1f}min, 评分中 ...", flush=True)
        sc = v3score(OUT / f"seed{seed}.flac", f"new_{seed}")
        results[seed] = sc
        v = sc["score"] if sc else None
        print(f"  new {seed}: {v} [{sc['cat'] if sc else '?'}]", flush=True)

# 失败重试 (单路)
fails = [s for s in NEW if results.get(s) is None]
for s in fails:
    print(f"  重试 {s} ...", flush=True)
    _, ok, dt, err = render(s)
    if ok:
        results[s] = v3score(OUT / f"seed{s}.flac", f"new_{s}")
        print(f"  new {s}: {results[s]['score'] if results[s] else None}", flush=True)
    else:
        print(f"  new {s}: 仍失败", flush=True)

new_rows = [(results[s]["score"], s, results[s]) for s in NEW if results.get(s)]

def top(rows, k=3):
    rows = [r for r in rows if r[0] is not None]
    rows.sort(key=lambda x: -x[0])
    return rows[:k]

print("\n=== 结果 ===", flush=True)
print("旧8 (4维抽卡产物) v3 分:", [f"{s}:{v:.1f}" for v, s, _ in sorted(old_rows, key=lambda x: -x[0] if x[0] is not None else 999)][:8], flush=True)
print("新8 (v3抽卡) 分:", [f"{s}:{v:.1f}" for v, s, _ in sorted(new_rows, key=lambda x: -x[0] if x[0] is not None else 999)][:8], flush=True)
ob, nb = top(old_rows, 1), top(new_rows, 1)
if ob and ob[0][0] is not None:
    print(f"旧8最优: seed{ob[0][1]} = {ob[0][0]:.1f}  (注: 旧4维冠军 seed90406 的 v3 分=2.5)", flush=True)
if nb and nb[0][0] is not None:
    print(f"新8最优: seed{nb[0][1]} = {nb[0][0]:.1f}", flush=True)
