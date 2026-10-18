# -*- coding: utf-8 -*-
import json
from pathlib import Path

txt = Path(r"F:\yinyue\novel_songs\final\tech_hum.md").read_text(encoding="utf-8")
data = json.loads(txt.split("json")[1].split("```")[0])
for r in data:
    if r["pass"] != r["total"]:
        print(r["file"], "->", {k: v for k, v in r.items()
                                if k in ("bpm_windows", "bpm_stability_pct",
                                         "key_windows", "key_stable")})
