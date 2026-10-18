# -*- coding: utf-8 -*-
"""MERT-v2 嵌入提取（run in .venv-ss2）-> eval/embeddings.npz"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from transformers import AutoModel

ROOT = Path(__file__).parent
SR = 24000

model = AutoModel.from_pretrained(str(ROOT / "models" / "MERT-v2-FullSong"),
                                  trust_remote_code=True, local_files_only=True)
model = model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
print("MERT loaded", flush=True)


def embed(wav_path):
    rate, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    w = torch.from_numpy(data.astype(np.float32) / 32768.0)[None, :].to(next(model.parameters()).device)
    with torch.no_grad():
        out = model(w)
    return out.last_hidden_state.mean(dim=1)[0].float().cpu().numpy()


entries = []
for w in sorted((ROOT / "corpus" / "sample_wavs").glob("*.wav")):
    entries.append((f"corpus::{w.stem}", str(w)))
for w in sorted((ROOT / "eval" / "wavs").glob("*.wav")):
    entries.append((f"creation::{w.stem}", str(w)))

names, embs, groups = [], [], []
for i, (name, path) in enumerate(entries):
    e = embed(path)
    names.append(name)
    embs.append(e)
    groups.append(name.split("::")[0])
    if (i + 1) % 6 == 0 or i == len(entries) - 1:
        print(f"[{i+1}/{len(entries)}] dim={e.shape[0]}", flush=True)

np.savez(ROOT / "eval" / "embeddings.npz",
         names=np.array(names), embs=np.stack(embs), groups=np.array(groups))
print("SAVED eval/embeddings.npz", flush=True)
