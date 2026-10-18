# -*- coding: utf-8 -*-
"""生成曲批量转谱(run in .venv-ss2) -> eval/novel_transcripts/ + 旋律统计"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from transformers import AutoModel

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from melody_stats import analyze_song

model = AutoModel.from_pretrained(
    str(ROOT / "SheetSage2"), trust_remote_code=True,
    base_model_path=str(ROOT / "models" / "MERT-v2-FullSong"),
    local_files_only=True,
).eval().to("cuda" if torch.cuda.is_available() else "cpu")
print("model loaded", flush=True)

wavs = sorted((ROOT / "eval" / "wavs").glob("novel_*.wav"))
rows = []
for i, w in enumerate(wavs):
    out_dir = ROOT / "eval" / "novel_transcripts" / w.stem
    abc = out_dir / "score.abc"
    if not abc.exists():
        rate, data = wavfile.read(w)
        if data.ndim > 1:
            data = data.mean(axis=1)
        wave = torch.from_numpy(data.astype(np.float32) / 32768.0)
        model.transcribe(wave, sampling_rate=rate, output_dir=str(out_dir), melody_only=True)
    try:
        r = analyze_song(abc)
        if r:
            r["file"] = w.stem
            rows.append(r)
        print(f"[{i+1}/{len(wavs)}] {w.stem} notes-ok", flush=True)
    except Exception as e:
        print(f"FAIL {w.stem}: {str(e)[:90]}", flush=True)

import json
(ROOT / "eval" / "novel_melody.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
print("SAVED eval/novel_melody.json", len(rows), flush=True)
