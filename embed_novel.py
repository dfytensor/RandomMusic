# -*- coding: utf-8 -*-
"""生成曲 MERT 嵌入(run in .venv-ss2) -> eval/novel_embeddings.npz"""
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from transformers import AutoModel

ROOT = Path(__file__).parent

model = AutoModel.from_pretrained(str(ROOT / "models" / "MERT-v2-FullSong"),
                                  trust_remote_code=True, local_files_only=True)
model = model.eval().to("cuda" if torch.cuda.is_available() else "cpu")


def embed(wav_path):
    rate, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    w = torch.from_numpy(data.astype(np.float32) / 32768.0)[None, :].to(next(model.parameters()).device)
    with torch.no_grad():
        out = model(w)
    return out.last_hidden_state.mean(dim=1)[0].float().cpu().numpy()


wavs = sorted((ROOT / "eval" / "wavs").glob("novel_*.wav"))
names, embs = [], []
for w in wavs:
    names.append(w.stem)
    embs.append(embed(w))
    print("emb", w.stem, flush=True)
np.savez(ROOT / "eval" / "novel_embeddings.npz",
         names=np.array(names), embs=np.stack(embs))
print("SAVED", flush=True)
