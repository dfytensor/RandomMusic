# -*- coding: utf-8 -*-
"""03 v2 大范围撒网 + 爬坡（类别感知制）"""
import subprocess, sys
from pathlib import Path
sys.path.insert(0, r'F:\yinyue')
import librosa
from category_score import score_audio, score_vs_category

PY = r'F:\yinyue\.venv\Scripts\python.exe'
NO = '03'
outd = Path(rf'F:\yinyue\novel_songs\batch_gacha\{NO}_v2')
outd.mkdir(parents=True, exist_ok=True)
COARSE = [2048, 31416, 52000, 61001]

def render_score(sd):
    out = outd / f'seed{sd}.flac'
    if not out.exists():
        r = subprocess.run([PY, r'F:\yinyue\render_one.py', NO, str(sd), str(out)],
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
        return sc["score"]
    except Exception as e:
        print(f'seed{sd}: ERR {type(e).__name__}', flush=True)
        return None

results = {}
for sd in COARSE:
    v = render_score(sd)
    if v is not None:
        results[sd] = v
if results:
    best = max(results, key=results.get)
    print(f'== coarse best: {best} ({results[best]}) ==', flush=True)
    for nb in [best-1, best+1]:
        v = render_score(nb)
        if v is not None:
            results[nb] = v
    champ = max(results, key=results.get)
    print(f'CHAMPION: seed{champ} score={results[champ]}', flush=True)
