# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, r'F:\yinyue')
import librosa
from category_score import score_audio, score_vs_category

d = Path(r'F:\yinyue\novel_songs\batch_gacha\01_v2')
for f in sorted(d.glob('*.flac')):
    y, _ = librosa.load(str(f), sr=22050, mono=True, offset=78, duration=30)
    r = score_audio(y)
    sc = score_vs_category(r)
    print(f"{f.stem}: score={sc['score']} ({sc['n_hits']}/4) cat={sc['cat']} "
          f"orn={r['orn']} gap={r['gap']} cent={r['cent']} dyn={r['dyn']}")
