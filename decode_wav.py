# -*- coding: utf-8 -*-
"""Decode mp3 -> 24kHz mono wav (run in main .venv)."""
import sys
from pathlib import Path

import librosa
import soundfile as sf

src, dst, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
y, sr = librosa.load(src, sr=24000, mono=True, duration=seconds)
sf.write(dst, y, 24000, subtype="PCM_16")
print(f"decoded {Path(src).name} -> {dst} ({len(y)/24000:.1f}s @{sr})")
