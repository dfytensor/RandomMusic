
# -*- coding: utf-8 -*-
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.io import wavfile
from transformers import AutoModel

ROOT = Path(__file__).parent
model = AutoModel.from_pretrained(str(ROOT/"models"/"MERT-v2-FullSong"),
                                  trust_remote_code=True, local_files_only=True)
model = model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
print("MERT loaded", flush=True)

lib_wav = ROOT / "eval" / "library_wavs"
out_p = ROOT / "eval" / "mert_all_embeddings.npz"

# 加载已有
done = {}
if out_p.exists():
    z = np.load(str(out_p), allow_pickle=True)
    for n, e in zip(z["names"], z["embs"]):
        done[n] = e
    print(f"已有 {len(done)} 首", flush=True)

wavs = sorted(lib_wav.glob("*.wav"))
t0 = time.time()
new = 0
for i, w in enumerate(wavs):
    if w.stem in done:
        continue
    try:
        rate, data = wavfile.read(str(w))
        if data.ndim > 1: data = data.mean(axis=1)
        wav = torch.from_numpy(data.astype(np.float32))[None, :].to(next(model.parameters()).device)
        with torch.no_grad():
            out = model(wav)
        done[w.stem] = out.last_hidden_state.mean(dim=1)[0].float().cpu().numpy()
        new += 1
    except Exception as e:
        print(f"ERR {w.stem[:20]}: {str(e)[:50]}", flush=True)
    if (i+1) % 50 == 0:
        print(f"[{i+1}/{len(wavs)}] new={new} el={time.time()-t0:.0f}s", flush=True)

names = sorted(done.keys())
embs = np.stack([done[n] for n in names])
np.savez(str(out_p), names=np.array(names), embs=embs)
print(f"SAVED {len(names)} embeddings dim={embs.shape[1]}", flush=True)
