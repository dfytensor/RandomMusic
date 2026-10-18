# -*- coding: utf-8 -*-
"""01 重生成后复测：爆款指数 + AI味 + 技术"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS
from hit_score import score_song, BASE
from ai_flavor_check import score_song as flavor_score, load_reference
from remediate_novel import naturalness

s = next(x for x in SONGS if x["no"] == "01")
flac = ROOT / "novel_songs" / "final" / "01_欠天地一笔.flac"

hit = score_song(s)
print(f"爆款指数 v2: {hit['total']} (v1=65.9)")
print("  dims:", json.dumps(hit["dims"], ensure_ascii=False))
print("  facts:", json.dumps(hit["facts"], ensure_ascii=False))

# AI 味：先补转谱
tdir = ROOT / "novel_songs" / "v2" / "novel_01_欠天地一笔" / "verify"
if not (tdir / "realized" / "structure.lab").exists():
    transcribe_realized = __import__("remediate_novel").transcribe_realized
    transcribe_realized(flac, tdir)
REF = load_reference()
fl_sc, _ = flavor_score(flac, tdir / "realized", REF)
maha, cos5 = naturalness(flac)
print(f"AI味 v2: {fl_sc} | 自然度 cos={cos5:.3f} maha={maha:.0f}  (v1: 8.20 / 0.821 / 91)")

# 技术
subprocess.run([str(ROOT / ".venv" / "Scripts" / "python.exe"),
                str(ROOT / "reference_free_check.py"), str(flac), "--out",
                str(ROOT / "novel_songs" / "final" / "tech_01.md")], check=True,
               capture_output=True)
txt = (ROOT / "novel_songs" / "final" / "tech_01.md").read_text(encoding="utf-8")
d = json.loads(txt.split("json")[1].split("```")[0])[0]
print(f"技术: {d['pass']}/{d['total']} dur={d['duration_s']}s")
