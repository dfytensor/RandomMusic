# -*- coding: utf-8 -*-
"""歌词韵律分 vs 渲染质量(v3.5) 相关性分析 — 检验韵律评价/创作体系有效性"""
import json, sys
from pathlib import Path
ROOT = Path(r"F:\yinyue")

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

NOS = [f"{i:02d}" for i in range(1, 13)]
CHAMPS = {"01": "01_v2", "02": "02_wide", "03": "03_v2", "04": "04", "05": "05_wide",
          "06": "06_wide", "07": "07_adj", "08": "11", "09": "09", "10": "10",
          "11": "11", "12": "12"}

pro_cur = load(ROOT / "eval" / "prosody_lyrics_current.json")   # 旧词韵律
pro_v3 = load(ROOT / "eval" / "prosody_v3.json")                # 新词韵律
rescan = load(ROOT / "eval" / "rescan_v35_cache.json")          # 全库 v3.5 (旧词)
expr = load(ROOT / "eval" / "gacha_ex_v3_stage2.json")          # express 旧词
v3x = load(ROOT / "eval" / "gacha_v3x_v3_stage2.json")          # express 新词

print("=" * 100)
print("1) 旧词: 韵律分 vs 各渠道渲染 v3.5 最佳")
print(f"{'no':>3} {'韵律(旧)':>8} {'全库池top':>9} {'express top':>11} {'新词韵律':>8} {'v3x top':>8}")
rows = []
for no in NOS:
    # 全库 rescan: 该歌所有池(同名前缀 batch_gacha/<pool>/ 但要按歌归属——池名前2位=歌号)
    tops = [v["score"] for k, v in rescan.items() if v and k.split("/")[2][:2] == no]
    pool_top = max(tops) if tops else None
    # express 旧词
    et = [v["score"] for k, v in expr.items() if v and k.startswith(f"{no}_s") and v["score"] > -100]
    ex_top = max(et) if et else None
    # v3x 新词
    vt = [v["score"] for k, v in v3x.items() if v and k.startswith(f"{no}_s") and v["score"] > -100]
    vx_top = max(vt) if vt else None
    pc = pro_cur.get(no, {}).get("total") if isinstance(pro_cur.get(no), dict) else pro_cur.get(no)
    pv = pro_v3.get(no, {}).get("total") if isinstance(pro_v3.get(no), dict) else pro_v3.get(no)
    rows.append((no, pc, pool_top, ex_top, pv, vx_top))
    def f(v, w):
        return f"{v:>{w}.1f}" if v is not None else " " * (w - 1) + "-"
    print(f"{no:>3} {f(pc,8)} {f(pool_top,9)} {f(ex_top,11)} {f(pv,8)} {f(vx_top,8)}")

# 相关性: 旧词韵律 vs 该歌历史最佳 v3.5 (全库池 top 与 express top 取 max)
import statistics
def spearman(x, y):
    n = len(x)
    rx = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: x[i]))}
    ry = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: y[i]))}
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))

xs = [r[1] for r in rows]
ys = [max(v for v in (r[2], r[3]) if v is not None) for r in rows]
print(f"\nSpearman(旧词韵律, 该歌历史最佳v3.5) = {spearman(xs, ys):.3f}  n={len(xs)}")

# 渠道均值对比 (01 池有新旧对照)
print("\n2) express 渠道逐首对照: 旧词 vs 新词 (01 池仅新词已跑)")
for no in NOS:
    et = [v["score"] for k, v in expr.items() if v and k.startswith(f"{no}_s") and v["score"] > -100]
    vt = [v["score"] for k, v in v3x.items() if v and k.startswith(f"{no}_s") and v["score"] > -100]
    if et:
        em = statistics.mean(et)
        if vt:
            print(f"{no}: 旧词 express n={len(et)} mean={em:.1f} top={max(et):.1f} | "
                  f"新词 v3x n={len(vt)} mean={statistics.mean(vt):.1f} top={max(vt):.1f}")
        else:
            print(f"{no}: 旧词 express n={len(et)} mean={em:.1f} top={max(et):.1f} | 新词待跑")

print("\n3) 01 池对照结论数据")
et = [v["score"] for k, v in expr.items() if v and k.startswith("01_s") and v["score"] > -100]
vt = [v["score"] for k, v in v3x.items() if v and k.startswith("01_s") and v["score"] > -100]
print(f"旧词 express: {sorted(et, reverse=True)}")
print(f"新词 v3x:     {sorted(vt, reverse=True)}")
print(f"旧词 mean={statistics.mean(et):.1f} top={max(et):.1f} | 新词 mean={statistics.mean(vt):.1f} top={max(vt):.1f}")
