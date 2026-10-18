# -*- coding: utf-8 -*-
import json
from pathlib import Path

txt = Path(r"F:\yinyue\novel_songs\v2\yuegongri_prosody\tech.md").read_text(encoding="utf-8")
data = json.loads(txt.split("json")[1].split("```")[0])[0]
print(f"技术 {data['pass']}/{data['total']} | dur={data['duration_s']}s | bpm={data['bpm_windows']}")
for k, v in data["checks"].items():
    print(f"  {'✅' if v else '❌'} {k}")
