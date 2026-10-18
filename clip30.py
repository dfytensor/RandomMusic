# -*- coding: utf-8 -*-
"""截取 rank01/06/11 的副歌段 30s 对比片段到 只听这3个_30秒副歌/"""
import json
from pathlib import Path
import soundfile as sf
import librosa
import numpy as np

ROOT = Path(__file__).parent
cache = json.loads((ROOT/"eval/seed_rank01_cache.json").read_text(encoding='utf-8'))
SRC = ROOT/"listen_test"/"rank01_听测"
OUT = ROOT/"listen_test"/"只听这3个_30秒副歌"
OUT.mkdir(parents=True, exist_ok=True)

targets = [("rank01", "seed65535"), ("rank06", "seed58888"), ("rank11", "seed90124")]
# rank01 seed65535 的 seg_t=109.8, 但若按其真实 ~92bpm 副歌另算? 用缓存值即可
for rank, seed in targets:
    f = SRC/f"{rank}_{seed}.flac"
    m = cache.get(seed) or {}
    t0 = m.get('seg_t', 60.0)
    # 06/11 的 seg_t 也在缓存; 但 rank01 缓存 seg_t=109.8 有效
    y, sr = librosa.load(str(f), sr=22050, mono=True)
    seg = y[int(t0*sr):int((t0+30)*sr)]
    dst = OUT/f"{rank}_{seed}_{int(t0)}秒起30秒.wav"
    sf.write(dst, seg, sr)
    print(f"{dst.name}  ({len(seg)/sr:.1f}s)", flush=True)
