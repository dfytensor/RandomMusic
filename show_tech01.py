# -*- coding: utf-8 -*-
import json
from pathlib import Path

txt = Path(r"F:\yinyue\novel_songs\final\tech_01.md").read_text(encoding="utf-8")
d = json.loads(txt.split("json")[1].split("```")[0])[0]
print(f"技术 {d['pass']}/{d['total']} | dur={d['duration_s']}s | bpm={d['bpm_windows']} | key={d['key_majority']}")
