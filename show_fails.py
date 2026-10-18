# -*- coding: utf-8 -*-
import json
from pathlib import Path

txt = Path(r"F:\yinyue\novel_songs\tech_report.md").read_text(encoding="utf-8")
data = json.loads(txt.split("```json")[1].split("```")[0])
for r in data:
    bad = {k: v for k, v in r["checks"].items() if not v} if "checks" in r else r
    if r["pass"] != r["total"]:
        print(r["file"], "->", bad)
