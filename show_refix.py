# -*- coding: utf-8 -*-
import json
from pathlib import Path

r = json.loads(Path(r"F:\yinyue\novel_songs\v2\refix_report.json").read_text(encoding="utf-8"))
for x in r:
    print(f"{x['no']} {x['title']}: cand_dur={x['cand_dur']}s (v1={x['v1_dur']}s) 截断={x['truncated']}")
    print(f"   edits={x['edits']}")
    print(f"   cand: AI味={x['ai_cand']} cos={x['cos_cand']} | final: AI味={x['ai_final']} cos={x['cos_final']}")
    print(f"   -> {x['chosen']}")
