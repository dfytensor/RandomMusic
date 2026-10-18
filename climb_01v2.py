# -*- coding: utf-8 -*-
"""01 v2 爬坡：90125 邻域 ±1（类别感知制）"""
import subprocess, sys
from pathlib import Path
sys.path.insert(0, r'F:\yinyue')
import librosa
from category_score import score_audio, score_vs_category

PY = r'F:\yinyue\.venv\Scripts\python.exe'
outd = Path(r'F:\yinyue\novel_songs\batch_gacha\01_v2')

for sd in [90124, 90126]:
    out = outd / f'seed{sd}.flac'
    if not out.exists():
        r = subprocess.run([PY, r'F:\yinyue\render_one.py', '01', str(sd), str(out)],
                           capture_output=True, text=True, timeout=1500)
        print(sd, 'OK' if 'OK' in (r.stdout or '') else 'FAIL', flush=True)

for f in sorted(outd.glob('seed9012[456].flac')):
    try:
        y, _ = librosa.load(str(f), sr=22050, mono=True, offset=78, duration=30)
        r = score_audio(y)
        sc = score_vs_category(r)
        print(f"{f.stem}: score={sc['score']} ({sc['n_hits']}/4) cat={sc['cat']} "
              f"orn={r['orn']} gap={r['gap']} cent={r['cent']} dyn={r['dyn']}", flush=True)
    except Exception as e:
        print(f"{f.stem}: ERR {type(e).__name__}", flush=True)
