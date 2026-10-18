# -*- coding: utf-8 -*-
"""分离 AI 多 seed flac → ai_stems/<组>/<seed>/vocals.wav"""
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(r"F:\yinyue")
SRC = [
    ("01_v2", ROOT / "novel_songs" / "batch_gacha" / "01_v2"),
    ("04_v3", ROOT / "novel_songs" / "batch_gacha" / "04_v3"),
]
OUT = ROOT / "ai_stems"

todo = []
for tag, d in SRC:
    for f in sorted(d.glob("seed*.flac")):
        vdir = OUT / tag / f.stem
        if not (vdir / "vocals.wav").exists():
            todo.append((tag, f))
print(f"todo={len(todo)}", flush=True)

from demucs.api import Separator
import torch, soundfile as sf

sep = Separator(model="htdemucs", device="cuda")
for i, (tag, f) in enumerate(todo, 1):
    print(f"[{i}/{len(todo)}] {tag}/{f.stem}", flush=True)
    try:
        out = OUT / tag / f.stem
        out.mkdir(parents=True, exist_ok=True)
        origin, stems = sep.separate_audio_file(str(f))
        vocals = stems["vocals"]
        accomp = origin - vocals
        sf.write(str(out / "vocals.wav"), vocals.cpu().numpy().T, sep.samplerate)
        sf.write(str(out / "accomp.wav"), accomp.cpu().numpy().T, sep.samplerate)
    except Exception as e:
        print(f"  ERR {type(e).__name__}: {e}", flush=True)
print("DONE", flush=True)
