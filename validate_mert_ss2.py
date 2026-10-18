# -*- coding: utf-8 -*-
"""指标体系元验证（MERT 窗口嵌入部分，run in .venv-ss2）
同曲两窗口嵌入的余弦 vs 跨曲余弦 -> MERT 自然度指标的噪声底
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from transformers import AutoModel

ROOT = Path(__file__).parent

model = AutoModel.from_pretrained(str(ROOT / "models" / "MERT-v2-FullSong"),
                                  trust_remote_code=True, local_files_only=True)
model = model.eval().to("cuda" if torch.cuda.is_available() else "cpu")


def embed_window(wav_path, t1, t2):
    rate, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    seg = data[t1 * rate:t2 * rate]
    w = torch.from_numpy(seg.astype(np.float32))[None, :].to(next(model.parameters()).device)
    with torch.no_grad():
        out = model(w)
    return out.last_hidden_state.mean(dim=1)[0].float().cpu().numpy()


manifest = json.loads((ROOT / "corpus" / "sample_manifest.json").read_text(encoding="utf-8"))[:8]
rows = []
for m in manifest:
    w = m["wav"]
    e1 = embed_window(w, 5, 30)
    e2 = embed_window(w, 45, 70)
    cos12 = float(e1 @ e2 / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9))
    rows.append(dict(stem=Path(w).stem, e1=e1.tolist(), e2=e2.tolist(), cos_same=cos12))
    print("emb", Path(w).stem[:20], round(cos12, 3), flush=True)

np.savez(ROOT / "eval" / "mert_window_test.npz",
         meta=np.array([r["stem"] for r in rows]),
         e1=np.stack([r["e1"] for r in rows]),
         e2=np.stack([r["e2"] for r in rows]),
         cos_same=np.array([r["cos_same"] for r in rows]))
print("SAVED", flush=True)
