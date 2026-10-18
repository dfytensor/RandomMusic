# -*- coding: utf-8 -*-
"""通用撒网+爬坡：目标全专辑 ≥85（即 100=4/4）
对 02,03,05,06,07,10,12 各撒 4 个跨量级新种子域 + 对高点 ±1 爬坡
"""
import subprocess, sys
from pathlib import Path
sys.path.insert(0, r'F:\yinyue')
import librosa
from category_score import score_audio, score_vs_category

PY = r'F:\yinyue\.venv\Scripts\python.exe'
SR = 22050
PLAN = {
    '02': [7777, 33333, 44444, 55555],
    '03': [8642, 42424, 53000, 63001],
    '05': [1111, 22222, 48000, 58001],
    '06': [3690, 26000, 47000, 57001],
    '07': [1357, 28000, 46000, 56001],
    '10': [2468, 24000, 45000, 55001],
    '12': [9753, 23000, 44000, 54001],
}

def render_score(no, sd):
    outd = Path(rf'F:\yinyue\novel_songs\batch_gacha\{no}_wide')
    outd.mkdir(parents=True, exist_ok=True)
    out = outd / f'seed{sd}.flac'
    if not out.exists():
        r = subprocess.run([PY, r'F:\yinyue\render_one.py', no, str(sd), str(out)],
                           capture_output=True, text=True, timeout=1500)
        if 'OK' not in (r.stdout or ''):
            print(f'{no} seed{sd}: RENDER FAIL', flush=True)
            return None
    try:
        y, _ = librosa.load(str(out), sr=SR, mono=True, offset=78, duration=30)
        r = score_audio(y)
        sc = score_vs_category(r)
        print(f'{no} seed{sd}: {sc["score"]} ({sc["n_hits"]}/4) orn={r["orn"]} gap={r["gap"]} '
              f'cent={r["cent"]} dyn={r["dyn"]}', flush=True)
        return sc["score"]
    except Exception as e:
        print(f'{no} seed{sd}: ERR {type(e).__name__}', flush=True)
        return None

champs = {}
for no, seeds in PLAN.items():
    res = {}
    for sd in seeds:
        v = render_score(no, sd)
        if v is not None:
            res[sd] = v
        if v == 100:
            break
    if res and max(res.values()) < 100:
        best = max(res, key=res.get)
        for nb in [best-1, best+1]:
            v = render_score(no, nb)
            if v is not None:
                res[nb] = v
    if res:
        c = max(res, key=res.get)
        champs[no] = (c, res[c])
        print(f'== {no} CHAMPION seed{c} = {res[c]} ==', flush=True)

print('FINAL:', {k: v for k, v in champs.items()}, flush=True)
