# -*- coding: utf-8 -*-
"""01 v2 大范围粗探 + 爬坡：跳出 901xx，横扫不同量级种子域"""
import subprocess, sys
from pathlib import Path
sys.path.insert(0, r'F:\yinyue')
import librosa
from category_score import score_audio, score_vs_category

PY = r'F:\yinyue\.venv\Scripts\python.exe'
outd = Path(r'F:\yinyue\novel_songs\batch_gacha\01_v2')
outd.mkdir(parents=True, exist_ok=True)

COARSE = [1024, 27182, 58888, 65535]

def render_score(sd):
    out = outd / f'seed{sd}.flac'
    if not out.exists():
        r = subprocess.run([PY, r'F:\yinyue\render_one.py', '01', str(sd), str(out)],
                           capture_output=True, text=True, timeout=1500)
        if 'OK' not in (r.stdout or ''):
            print(f'seed{sd}: RENDER FAIL', flush=True)
            return None
    try:
        y, _ = librosa.load(str(out), sr=22050, mono=True, offset=78, duration=30)
        r = score_audio(y)
        sc = score_vs_category(r)
        print(f'seed{sd}: score={sc["score"]} ({sc["n_hits"]}/4) cat={sc["cat"]} '
              f'orn={r["orn"]} gap={r["gap"]} cent={r["cent"]} dyn={r["dyn"]}', flush=True)
        return (sc["score"], sc["n_hits"], r)
    except Exception as e:
        print(f'seed{sd}: ERR {type(e).__name__}', flush=True)
        return None

results = {}
for sd in COARSE:
    v = render_score(sd)
    if v:
        results[sd] = v

if results:
    best = max(results, key=lambda s: (results[s][0], ))
    print(f'== coarse best: {best} ({results[best][0]}) ==', flush=True)
    for nb in [best-1, best+1]:
        v = render_score(nb)
        if v:
            results[nb] = v
    champ = max(results, key=lambda s: results[s][0])
    print(f'CHAMPION: seed{champ} score={results[champ][0]}', flush=True)
