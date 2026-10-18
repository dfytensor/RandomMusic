# -*- coding: utf-8 -*-
"""验证 final 母带效果: 高频占比 + 立体声宽度 vs 曲库基线"""
import json
import warnings
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
BANDS = [(20, 250), (250, 2000), (2000, 6000), (6000, 11000)]

fc = json.loads((ROOT / "eval" / "feat_corpus.json").read_text(encoding="utf-8"))
base_hi = float(np.median([r["band_share"][3] for r in fc]))
base_w = float(np.median([r["stereo_width"] for r in fc]))

rows = []
for flac in sorted((ROOT / "novel_songs" / "final").glob("*.flac")):
    data, sr = sf.read(str(flac), always_2d=True)
    mid = data.mean(axis=1)
    side = (data[:, 0] - data[:, 1]) / 2
    width = float(np.sqrt((side ** 2).mean()) / (np.sqrt((mid ** 2).mean()) + 1e-9))
    y = librosa.resample(mid.astype(np.float32), orig_sr=sr, target_sr=SR)[: SR * 90]
    S = np.abs(librosa.stft(y)) ** 2
    freqs = librosa.fft_frequencies(sr=SR)
    tot = S.sum() + 1e-9
    hi = float(S[(freqs >= 6000) & (freqs < 11000)].sum() / tot)
    rows.append((flac.stem, hi, width))
    print(f"{flac.stem[:20]:24s} 高频={hi:.4f} (曲库{base_hi}) 宽度={width:.3f} (曲库{base_w})")
mh = float(np.median([r[1] for r in rows]))
mw = float(np.median([r[2] for r in rows]))
print(f"\n生成曲母带后中位: 高频 {mh:.4f} (v1=0.009, 曲库={base_hi}) | 宽度 {mw:.3f} (v1=0.269, 曲库={base_w})")
