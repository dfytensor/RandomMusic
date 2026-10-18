# -*- coding: utf-8 -*-
"""Batch transcribe sample wavs with SheetSage2 (run in .venv-ss2)."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from transformers import AutoModel

ROOT = Path(__file__).parent
repo = ROOT / "SheetSage2"

model = AutoModel.from_pretrained(
    str(repo), trust_remote_code=True,
    base_model_path=str(ROOT / "models" / "MERT-v2-FullSong"),
    local_files_only=True,
).eval().to("cuda" if torch.cuda.is_available() else "cpu")
print("model loaded", flush=True)

manifest = json.loads((ROOT / "corpus" / "sample_manifest.json").read_text(encoding="utf-8"))
t0 = time.time()
done = fail = 0
for i, m in enumerate(manifest):
    out_dir = ROOT / "corpus" / "transcriptions" / Path(m["wav"]).stem
    marker = out_dir / "score.abc"
    if marker.exists():
        done += 1
        continue
    try:
        rate, data = wavfile.read(m["wav"])
        waveform = torch.from_numpy(data.astype(np.float32) / 32768.0)
        model.transcribe(waveform, sampling_rate=rate, output_dir=str(out_dir), melody_only=True)
        done += 1
    except Exception as e:
        fail += 1
        print(f"FAIL {m['file']}: {str(e)[:100]}", flush=True)
    if (i + 1) % 5 == 0 or i == len(manifest) - 1:
        print(f"[{i+1}/{len(manifest)}] done={done} fail={fail} elapsed={time.time()-t0:.0f}s", flush=True)
print("BATCH DONE", flush=True)
