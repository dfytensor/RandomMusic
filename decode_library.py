# -*- coding: utf-8 -*-
"""全库解码：571 首 -> eval/library_wavs/ (100s, 24k, mono)，断点续跑"""
import json
from pathlib import Path

import librosa
import soundfile as sf

ROOT = Path(__file__).parent
OUT = ROOT / "eval" / "library_wavs"
OUT.mkdir(parents=True, exist_ok=True)

manifest = json.loads((ROOT / "corpus" / "sample_manifest.json").read_text(encoding="utf-8"))
done_stems = {Path(m["wav"]).stem for m in manifest}

files = [f for f in sorted(Path(r"D:\音乐").glob("*.mp3")) if f.stat().st_size > 0]
todo = [f for f in files if f"lib_{f.stem.replace(' ', '_')[:60]}" not in
        {p.stem for p in OUT.glob("*.wav")}]

print(f"total={len(files)} todo={len(todo)}", flush=True)
ok = fail = 0
for i, f in enumerate(todo):
    stem = "lib_" + f.stem.replace(" ", "_")[:60]
    wav = OUT / f"{stem}.wav"
    try:
        y, sr = librosa.load(str(f), sr=24000, mono=True, duration=100)
        sf.write(wav, y, 24000, subtype="PCM_16")
        ok += 1
    except Exception as e:
        fail += 1
        print(f"FAIL {f.name}: {str(e)[:60]}", flush=True)
    if (i + 1) % 50 == 0 or i == len(todo) - 1:
        print(f"[{i+1}/{len(todo)}] ok={ok} fail={fail}", flush=True)
print("DECODE DONE", flush=True)
