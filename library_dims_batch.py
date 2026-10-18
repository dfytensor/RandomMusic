# -*- coding: utf-8 -*-
"""曲库全量 4 维扫描: orn/s, gapCV, cent, dyn（多进程 + 断点续跑）
输出 eval/library_dims_cache.json
"""
import json, sys, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
CACHE = ROOT / "eval" / "library_dims_cache.json"
WAVS = sorted((ROOT/"eval"/"library_wavs").glob("*.wav"))

def one_song(w):
    try:
        y, _ = librosa.load(str(w), sr=SR, mono=True, offset=78, duration=30)
        hop = 512
        f0, vflag, _ = librosa.pyin(y, fmin=65, fmax=700, sr=SR,
                                    frame_length=1024, hop_length=hop)
        voiced = vflag & ~np.isnan(f0)
        # orn: 每秒相邻帧F0变化>15c 的次数（voiced 帧）
        orn = None
        if voiced.sum() > 20:
            d = np.abs(np.diff(f0))
            m = voiced[1:] & voiced[:-1]
            orn = round(float(np.sum(d[m] > 15) / (len(y)/SR)), 2)
        # gap_cv
        gap = None
        idx = np.where(voiced)[0]
        if len(idx) > 5:
            gl = np.diff(idx) * hop / SR * 1000
            gl = gl[gl > 1]
            if len(gl) > 3:
                gap = round(float(np.std(gl)/(np.mean(gl)+1e-9)), 2)
        # cent / dyn
        cent = round(float(librosa.feature.spectral_centroid(y=y, sr=SR)[0].mean()))
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        db = librosa.amplitude_to_db(rms+1e-9)
        dyn = round(float(np.percentile(db,95)-np.percentile(db,10)), 1)
        return w.stem, dict(orn=orn, gap=gap, cent=cent, dyn=dyn)
    except Exception:
        return w.stem, None

def main():
    done = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [w for w in WAVS if w.stem not in done]
    print(f"total={len(WAVS)} done={len(done)} todo={len(todo)}", flush=True)
    n = 0
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(one_song, w): w.stem for w in todo}
        for fut in as_completed(futs):
            stem, r = fut.result()
            done[stem] = r
            n += 1
            if n % 20 == 0:
                CACHE.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
                print(f"  {n}/{len(todo)}", flush=True)
    CACHE.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    ok = sum(1 for v in done.values() if v)
    print(f"DONE ok={ok}/{len(done)}", flush=True)

if __name__ == "__main__":
    main()
