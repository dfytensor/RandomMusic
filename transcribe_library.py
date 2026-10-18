# -*- coding: utf-8 -*-
"""全库批量转谱（run in .venv-ss2），断点续跑 -> eval/library_transcripts/"""
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from transformers import AutoModel

ROOT = Path(__file__).parent

model = AutoModel.from_pretrained(
    str(ROOT / "SheetSage2"), trust_remote_code=True,
    base_model_path=str(ROOT / "models" / "MERT-v2-FullSong"),
    local_files_only=True,
).eval().to("cuda" if torch.cuda.is_available() else "cpu")
print("model loaded", flush=True)

wavs = sorted((ROOT / "eval" / "library_wavs").glob("*.wav"))
t0 = time.time()
done = fail = 0
for i, w in enumerate(wavs):
    out_dir = ROOT / "eval" / "library_transcripts" / w.stem
    abc = out_dir / "score.abc"
    if abc.exists():
        done += 1
        continue
    try:
        rate, data = wavfile.read(w)
        if data.ndim > 1:
            data = data.mean(axis=1)
        wave = torch.from_numpy(data.astype(np.float32) / 32768.0)
        model.transcribe(wave, sampling_rate=rate, output_dir=str(out_dir), melody_only=True)
        done += 1
    except Exception as e:
        fail += 1
        print(f"FAIL {w.stem[:24]}: {str(e)[:80]}", flush=True)
    if (i + 1) % 25 == 0 or i == len(wavs) - 1:
        print(f"[{i+1}/{len(wavs)}] done={done} fail={fail} elapsed={time.time()-t0:.0f}s", flush=True)
print("BATCH DONE", flush=True)
