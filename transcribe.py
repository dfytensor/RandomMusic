# -*- coding: utf-8 -*-
"""SheetSage2 transcription (run in .venv-ss2). Usage: transcribe.py <wav> <outdir> [full_song_wav]"""
import sys
from pathlib import Path

import torch
import torchaudio
from transformers import AutoModel

wav_path, out_dir = sys.argv[1], sys.argv[2]
repo = Path(__file__).parent / "SheetSage2"

model = AutoModel.from_pretrained(
    str(repo), trust_remote_code=True,
    base_model_path=str(Path(__file__).parent / "models" / "MERT-v2-FullSong"),
    local_files_only=True,
)
model = model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
print("model loaded", flush=True)

from scipy.io import wavfile

rate, data = wavfile.read(wav_path)
import numpy as np
waveform = torch.from_numpy(data.astype(np.float32) / 32768.0)
melody_only = not (len(sys.argv) > 3 and sys.argv[3] in ("--full", "full"))
result = model.transcribe(waveform, sampling_rate=rate, output_dir=out_dir, melody_only=melody_only)
print("ABC preview:\n", result["abc"][:600], flush=True)
print("SAVED ->", out_dir, flush=True)
