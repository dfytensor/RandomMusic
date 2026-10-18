# -*- coding: utf-8 -*-
import json
from pathlib import Path

d = json.loads(Path(r"F:\yinyue\eval\hit_library_score.json").read_text(encoding="utf-8"))
print("n =", d["n"], "| med", d["med"], "| P90", d["p90"], "| P99", d["p99"], "| MAX", d["mx"])
print("96-100区间真曲数量:", d["zone_96_100_count"])
print("超过该区间(>68.2)真曲数量:", d["above_zone_count"])
print("TOP10:")
for t in d["top10"]:
    print(f"  {t['total']:5.1f}  {t['name'][:26]}  弧线{t['arc']}  首唱{t['fv']}s  副歌段{t['n_chorus']}  高潮位{t['cpos']}")
