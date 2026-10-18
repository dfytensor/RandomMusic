# -*- coding: utf-8 -*-
"""波形级新指标：真曲全库采集（断点续跑）-> eval/wave_metric_cache.json"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import librosa
from scipy.signal import butter, sosfilt
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
sys.path.insert(0, str(ROOT))
from ai_segment_forensic import parse_lab, first_span

CACHE = ROOT / "eval" / "wave_metric_cache.json"


def wave_metrics(y):
    """副歌段波形级指标（mono 22050）"""
    # 1) 峰值因子
    rms_g = float(np.sqrt((y ** 2).mean()) + 1e-9)
    crest = round(float(20 * np.log10(np.abs(y).max() / rms_g)), 2)
    # 2) 包络粗糙度
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    env_rough = round(float(np.std(np.diff(db))), 2)
    # 3) 低频冲击速度：低通包络 波谷->波峰 中位时长
    sos = butter(4, 150 / (SR / 2), btype="low", output="sos")
    low = sosfilt(sos, y)
    lrms = librosa.feature.rms(y=low, frame_length=1024, hop_length=256)[0]
    thr = lrms.mean() + lrms.std()
    attacks = []
    in_rise, start = False, 0
    for i, v in enumerate(lrms):
        if not in_rise and v > thr:
            in_rise, start = True, i
        elif in_rise and v < thr:
            if i > start:
                attacks.append((i - start) * 256 / SR * 1000)
            in_rise = False
    atk = round(float(np.median(attacks)), 1) if len(attacks) >= 3 else None
    # 4) 静音结构
    db_f = librosa.amplitude_to_db(rms + 1e-9)
    sil = round(float((db_f < -50).mean()), 4)
    # 5) 起音间隔变异
    ons = librosa.onset.onset_detect(y=y, sr=SR, hop_length=256, units="time",
                                     delta=0.06, backtrack=False)
    iv = np.diff(ons)
    iv_cv = round(float(iv.std() / iv.mean()), 2) if len(iv) > 5 else None
    # 6) 波形峭度
    from scipy import stats as sps
    kurt = round(float(sps.kurtosis(y)), 2)
    # 7) 响度域（3s 窗 P95-P10）
    w = int(3 * SR / 512)
    n = len(db) // w
    lds = [float(db[i * w:(i + 1) * w].mean()) for i in range(n)] if n else [float(db.mean())]
    lra = round(float(np.percentile(lds, 95) - np.percentile(lds, 10)), 1)
    return dict(crest_db=crest, env_rough=env_rough, low_attack_ms=atk,
                silence_frac=sil, onset_iv_cv=iv_cv, kurtosis=kurt, lra_db=lra)


def main():
    cache = json.loads((ROOT / "eval" / "wave_metric_cache.json").read_text(encoding="utf-8")) \
        if (ROOT / "eval" / "wave_metric_cache.json").exists() else {}
    lib = ROOT / "eval" / "library_wavs"
    trs = ROOT / "eval" / "library_transcripts"
    wavs = sorted(lib.glob("*.wav"))
    t0 = time.time()
    done = 0
    for i, w in enumerate(wavs):
        if w.stem in cache:
            continue
        rows = parse_lab(trs / w.stem, "structure.lab")
        ch = first_span(rows, "chorus", 99)
        if not ch or ch[1] - ch[0] < 8:
            cache[w.stem] = {"skip": True}
            continue
        try:
            y, _ = librosa.load(str(w), sr=SR, mono=True)
            t2 = min(ch[1], ch[0] + 30, len(y) / SR)
            seg = y[int(ch[0] * SR):int(t2 * SR)]
            cache[w.stem] = wave_metrics(seg)
            done += 1
        except Exception as e:
            print("ERR", w.stem[:22], str(e)[:60], flush=True)
        if done % 25 == 0 and done > 0:
            (ROOT / "eval" / "wave_metric_cache.json").write_text(
                json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            print(f"[{i+1}/{len(wavs)}] new={done} el={time.time()-t0:.0f}s", flush=True)
    (ROOT / "eval" / "wave_metric_cache.json").write_text(
        json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"WAVE CACHE SAVED entries={len(cache)} new={done}", flush=True)


if __name__ == "__main__":
    main()
