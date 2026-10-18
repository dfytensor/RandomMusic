# -*- coding: utf-8 -*-
"""聚合重扫: 仅 batch_gacha 种子池(seed*.flac)内重选, 排除 hum/mastered/根目录旧文件"""
import json, re, sys
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
cache = json.loads((ROOT/"eval"/"rescan_v3_cache.json").read_text(encoding="utf-8"))

CHAMPS = {  # 现任冠军 -> 所在池目录
    "01": "01_v2", "02": "02_wide", "03": "03_v2", "04": "04", "05": "05_wide",
    "06": "06_wide", "07": "07_adj", "08": "11", "09": "09", "10": "10",
    "11": "11", "12": "12",
}
CHAMP_FILE = {
    "01": "seed58888", "02": "seed7778", "03": "seed61001", "04": "seed90406",
    "05": "seed1111", "06": "seed3690", "07": "seed1357_r1", "08": "seed91104",
    "09": "seed90904", "10": "seed91007", "11": "seed91103", "12": "seed91202",
}

pools = defaultdict(list)  # pool_dir -> [(score, rel, stem)]
for rel, v in cache.items():
    if v is None:
        continue
    if not rel.startswith("novel_songs/batch_gacha/"):
        continue
    parts = rel.split("/")
    pool, fname = parts[2], parts[3]
    if not re.match(r"seed\d+.*\.flac$", fname):
        continue
    pools[pool].append((v["score"], rel, Path(fname).stem))

print(f"{'池':10s} {'n':>3s} {'现任':>7s} {'重选':>7s} {'Δ':>8s}  冠军")
lines = ["# 全库存量 v3 重扫: 种子池内重选冠军 (2026-10-15)", "",
         "口径: 仅 novel_songs/batch_gacha/*/seed*.flac, 排除 hum/mastered/根目录旧文件", "",
         "| 池 | n | 现任冠军 | 现任v3 | 重选v3 | Δ | 新冠军 |",
         "|---|---|---|---|---|---|---|"]
rows = []
for pool in sorted(pools):
    cands = sorted(pools[pool], key=lambda x: -x[0])
    best_s, best_rel, best_stem = cands[0]
    # 该池对应的歌曲号
    no = next((n for n, p in CHAMPS.items() if p == pool and pool != "11"), None)
    if pool == "11":
        cur_stems = {"08": "seed91104", "11": "seed91103"}
        cur_scores = {n: (cache.get(f"novel_songs/batch_gacha/11/{s}.flac") or {}).get("score")
                      for n, s in cur_stems.items()}
        print(f"{pool:10s} {len(cands):>3d}  08={cur_scores['08']} 11={cur_scores['11']}  "
              f"best={best_s:.1f} ({best_stem})")
        lines.append(f"| 11 (08/11共用) | {len(cands)} | seed91104/91103 | {cur_scores['08']}/{cur_scores['11']} | "
                     f"{best_s:.1f} | 见下 | {best_stem} |")
        continue
    if no is None:
        print(f"{pool:10s} {len(cands):>3d}  (无现任)  best={best_s:.1f} ({best_stem})")
        lines.append(f"| {pool} | {len(cands)} | - | - | {best_s:.1f} | - | {best_stem} |")
        continue
    cur_stem = CHAMP_FILE[no]
    cur_score = (cache.get(f"novel_songs/batch_gacha/{pool}/{cur_stem}.flac") or {}).get("score")
    if cur_score is None:  # 缓存键可能带子路径差异
        for c_sc, c_rel, c_stem in cands:
            if c_stem == cur_stem:
                cur_score = c_sc
                break
    delta = (best_s - cur_score) if cur_score is not None else None
    mark = "★" if best_stem != cur_stem else "="
    dstr = f"{delta:+.1f}" if delta is not None else "?"
    print(f"{pool:10s} {len(cands):>3d} {cur_score if cur_score else '?':>7} {best_s:>7.1f} {dstr:>8}  {mark} {best_stem}")
    lines.append(f"| {pool} | {len(cands)} | {cur_stem} | {cur_score if cur_score else 'NA'} | "
                 f"{best_s:.1f} | {dstr} | {best_stem if mark=='★' else '不变'} |")
    rows.append((no, pool, cur_score, best_s, best_stem))

if rows:
    cur = [r[2] for r in rows if r[2] is not None]
    new = [r[3] for r in rows]
    print(f"\n可比对 {len(rows)} 池: 现任均值={np.mean(cur):.1f} 重选均值={np.mean(new):.1f} "
          f"Δ={np.mean(new)-np.mean(cur):+.1f}")
    lines += ["", f"可比对 {len(rows)} 池: 现任均值 {np.mean(cur):.1f} → 重选 {np.mean(new):.1f} (Δ {np.mean(new)-np.mean(cur):+.1f})"]
lines += ["", "注: 池11 同时是歌08与歌11的种子池(历史共用)。", ""]
(ROOT/"eval"/"v3_rescan_report.md").write_text("\n".join(lines), encoding="utf-8")
print("SAVED eval/v3_rescan_report.md")
