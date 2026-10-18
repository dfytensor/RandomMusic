# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path

from yue2 import YuE2Pipeline

ROOT = Path(__file__).parent
MODEL = ROOT / "models" / "YuE2-3B"
VAE = ROOT / "models" / "YuE2-Vae"

demo = json.loads((MODEL / "examples" / "tonight-awake.json").read_text(encoding="utf-8"))
style, lyrics = demo["style"], demo["lyrics"]

print("[1/3] loading pipeline ...", flush=True)
pipe = YuE2Pipeline.from_pretrained(str(MODEL), vae=str(VAE), device="cuda")

print("[2/3] generating song (cot=full) ...", flush=True)
song = pipe(style=style, lyrics=lyrics, cot="full", seed=demo["seed"])

print("[3/3] saving ...", flush=True)
song.save(str(ROOT / "song.flac"))
song.save_artifacts(str(ROOT / "outputs" / "song"))
pipe.close()
print("DONE -> song.flac", flush=True)
