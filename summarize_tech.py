# -*- coding: utf-8 -*-
import json
from pathlib import Path

txt = Path(r"F:\yinyue\novel_songs\tech_report.md").read_text(encoding="utf-8")
data = json.loads(txt.split("```json")[1].split("```")[0])
n_ok = 0
for r in data:
    ok = r["pass"] == r["total"]
    n_ok += ok
    print(f"{'✅' if ok else '❌'} {r['file'][:26]:30s} {r['pass']}/{r['total']}  {r['duration_s']:6.1f}s  bpm_win={r['bpm_windows']}")
print(f"\n{n_ok}/{len(data)} 全过")
