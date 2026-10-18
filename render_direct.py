# -*- coding: utf-8 -*-
"""直接渲染《月供日》精修版（主进程，无子进程）"""
import json, sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS

s = next(x for x in SONGS if x["no"] == "08")
work = ROOT / "novel_songs" / "v2" / "yuegongri_prosody"
work.mkdir(parents=True, exist_ok=True)

lyrics_f = work / "lyrics_v2.txt"
lyrics = lyrics_f.read_text(encoding="utf-8") if lyrics_f.exists() else s["lyrics"]

from yue2 import YuE2Pipeline
pipe = YuE2Pipeline.from_pretrained(str(ROOT/"models"/"YuE2-3B"),
                                     vae=str(ROOT/"models"/"YuE2-Vae"),
                                     device="cuda", progress=False)
try:
    out = pipe(style=s["style"], lyrics=lyrics, cot="full", seed=42)
    out_flac = work / "prosody_v2.flac"
    out.save(str(out_flac))
    print(f"saved: {out_flac}", flush=True)
finally:
    pipe.close()

# 母带
from remediate_novel import master
mastered = work / "mastered.flac"
master(str(out_flac), str(mastered))
print(f"mastered: {mastered}", flush=True)

# 技术快检
import subprocess
subprocess.run([str(ROOT/".venv"/"Scripts"/"python.exe"),
                str(ROOT/"reference_free_check.py"), str(mastered), "--out",
                str(work/"tech.md")], check=True, capture_output=True)
print("DONE", flush=True)
