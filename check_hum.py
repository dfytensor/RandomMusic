# -*- coding: utf-8 -*-
import json
import os
import subprocess
from pathlib import Path

final = Path(r"F:\yinyue\novel_songs\final")
hum = sorted(final.glob("*_hum.flac"))
subprocess.run([r"F:\yinyue\.venv\Scripts\python.exe",
                r"F:\yinyue\reference_free_check.py",
                *[str(p) for p in hum],
                "--out", str(final / "tech_hum.md")], check=True, capture_output=True)
txt = (final / "tech_hum.md").read_text(encoding="utf-8")
data = json.loads(txt.split("json")[1].split("```")[0])
for r in data:
    print(f"{r['file'][:32]:34s} {r['pass']}/{r['total']}  {r['duration_s']:.0f}s")
flacs = [f for f in os.listdir(final) if f.endswith(".flac")]
print(f"\nhum 版 {len(hum)} 首 | final 全部 flac {len(flacs)} 首")
