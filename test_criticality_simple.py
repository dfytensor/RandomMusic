# -*- coding: utf-8 -*-
"""临界性测试：guidance 7.4/7.8/8.2 三点扫描"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS

s = next(x for x in SONGS if x["no"] == "08")
work = ROOT / "novel_songs" / "v2" / "criticality"
work.mkdir(parents=True, exist_ok=True)

lyrics = (ROOT / "novel_songs" / "v2" / "yuegongri_prosody" / "lyrics_v2.txt").read_text(encoding="utf-8") \
    if (ROOT / "novel_songs" / "v2" / "yuegongri_prosody" / "lyrics_v2.txt").exists() else s["lyrics"]

from yue2 import YuE2Pipeline
pipe = YuE2Pipeline.from_pretrained(str(ROOT/"models"/"YuE2-3B"),
                                     vae=str(ROOT/"models"/"YuE2-Vae"),
                                     device="cuda", progress=False)
try:
    for gs in (7.4, 7.8, 8.2):
        out = work / f"gs{gs}.flac"
        if out.exists(): continue
        print(f"guidance={gs}...", flush=True)
        song = pipe(style=s["style"], lyrics=lyrics, cot="full", seed=42, guidance_scale=gs)
        song.save(str(out))
finally:
    pipe.close()
print("ALL DONE", flush=True)
