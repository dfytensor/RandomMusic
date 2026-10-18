# -*- coding: utf-8 -*-
import json
from pathlib import Path

d = json.loads(Path(r"F:\yinyue\eval\ai_flavor_rank.json").read_text(encoding="utf-8"))
for r in d:
    print(f"{r['score']:6.2f}  {r['no']} {r['title']}")
