# -*- coding: utf-8 -*-
import warnings, json, sys
from pathlib import Path
import numpy as np, librosa
from scipy.spatial.distance import pdist

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
ch = json.loads((ROOT/"eval"/"v33_champions.json").read_text(encoding="utf-8"))

def sig(path, t0=0.0):
    y, sr = librosa.load(str(path), sr=22050, mono=True, offset=t0, duration=30)
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.concatenate([m.mean(1), m.std(1)])

CH = [("01", r"novel_songs\batch_gacha\01_v2\seed58888.flac"),
      ("02", r"novel_songs\batch_gacha\02_wide\seed7778.flac"),
      ("03", r"novel_songs\batch_gacha\03_v2\seed61001.flac"),
      ("04", r"novel_songs\batch_gacha\04\seed90406.flac"),
      ("05", r"novel_songs\batch_gacha\05_wide\seed1111.flac"),
      ("06", r"novel_songs\batch_gacha\06_wide\seed3690.flac"),
      ("07", r"novel_songs\batch_gacha\07_adj\seed1357_r1.flac"),
      ("08", r"novel_songs\batch_gacha\11\seed91104.flac"),
      ("09", r"novel_songs\batch_gacha\09\seed90904.flac"),
      ("10", r"novel_songs\batch_gacha\10\seed91007.flac"),
      ("11", r"novel_songs\batch_gacha\11\seed91103.flac"),
      ("12", r"novel_songs\batch_gacha\12\seed91202.flac")]

vecs = []
for no, rel in CH:
    p = ROOT / rel
    if not p.exists():
        print(no, "MISSING"); continue
    t0 = list((ch[no] or {}).keys())[0] if ch.get(no) else "0.0"
    try:
        vecs.append(sig(p, float(t0)))
        print(no, "ok")
    except Exception as e:
        print(no, "ERR", type(e).__name__)

D = pdist(np.array(vecs))
med = float(np.median(D))
print(f"12 different-song champions timbre dist median={med:.1f} (n={len(D)})")
print(f"vs same-song-11-seed=42.2, human-11-songs=77.2")
verdict = "SAME FACE sings all songs: confirmed" if med < 60 else "different songs have distinct timbre"
print(f"ratio_vs_seed={med/42.2:.2f}x -> {verdict}")
