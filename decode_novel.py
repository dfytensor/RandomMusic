# -*- coding: utf-8 -*-
"""解码 12 首生成曲 -> eval/wavs/novel_*.wav (120s, 供转谱/嵌入/声学分析)"""
import sys
from pathlib import Path

import librosa
import soundfile as sf

ROOT = Path(__file__).parent
OUT = ROOT / "eval" / "wavs"
OUT.mkdir(parents=True, exist_ok=True)

for flac in sorted((ROOT / "novel_songs").glob("*.flac")):
    stem = "novel_" + flac.stem.replace(" ", "_")
    wav = OUT / f"{stem}.wav"
    if wav.exists():
        continue
    y, sr = librosa.load(str(flac), sr=24000, mono=False, duration=120)
    sf.write(wav, y.T if y.ndim > 1 else y, sr, subtype="PCM_16")
    print(stem, f"{len(y[0]) if y.ndim>1 else len(y)}/{sr:.0f}s" if y.ndim > 1 else f"{len(y)/sr:.1f}s")
print("DONE")
