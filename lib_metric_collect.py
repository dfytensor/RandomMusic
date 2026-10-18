# -*- coding: utf-8 -*-
"""全库指标验证数据采集：每首真曲副歌段 -> beat_R + AI味指纹 + 动态/循环
断点续跑 -> eval/lib_metric_cache.json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import librosa
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
sys.path.insert(0, str(ROOT))
from ai_segment_forensic import parse_lab, first_span
from ai_flavor_check import load_reference

CACHE_P = ROOT / "eval" / "lib_metric_cache.json"


def groove_metrics(y_seg, sr=SR):
    tempo, beats = librosa.beat.beat_track(y=y_seg, sr=sr, units="time", hop_length=512)
    beat_t = np.atleast_1d(beats)
    if len(beat_t) < 8:
        return None
    period = 60.0 / float(np.atleast_1d(tempo)[0])
    sos = butter(4, [300 / (sr / 2), 3400 / (sr / 2)], btype="band", output="sos")
    voc = sosfilt(sos, y_seg)
    ons = librosa.onset.onset_detect(y=voc, sr=sr, hop_length=256, units="time",
                                     delta=0.06, backtrack=False)
    if len(ons) < 10:
        return None
    phases = np.array([((o - beat_t[0]) % period) / period for o in ons])
    ang = phases * 2 * np.pi
    R = float(abs(np.exp(1j * ang).mean()))
    return dict(beat_R=round(R, 3), period_ms=round(period * 1000))


def main():
    cache = json.loads(CACHE_P.read_text(encoding="utf-8")) if CACHE_P.exists() else {}
    lib = ROOT / "eval" / "library_wavs"
    wavs = sorted(lib.glob("*.wav"))
    t0 = __import__("time").time()
    done = 0
    for i, w in enumerate(wavs):
        if w.stem in cache:
            continue
        tdir = ROOT / "eval" / "library_transcripts" / w.stem
        rows = parse_lab(tdir, "structure.lab")
        ch = first_span(rows, "chorus", 99)
        if not ch or ch[1] - ch[0] < 8:
            cache[w.stem] = {"skip": True}
            continue
        try:
            y, _ = librosa.load(str(w), sr=SR, mono=True)
            dur_a = len(y) / SR
            t1, t2 = ch[0], min(ch[1], ch[0] + 30, dur_a)
            seg = y[int(t1 * SR):int(t2 * SR)]

            g = groove_metrics(seg)
            from ai_segment_forensic import f0_metrics, dynamics_metrics, loopiness
            f = f0_metrics(seg)
            d = dynamics_metrics(seg)
            l = loopiness(seg)
            f.update(d)
            f.update(l)
            f["beat_R"] = g["beat_R"] if g else None
            f["period_ms"] = g["period_ms"] if g else None
            cache[w.stem] = f
            done += 1
        except Exception as e:
            print("ERR", w.stem[:24], str(e)[:70], flush=True)
        if done % 20 == 0 and done > 0:
            CACHE_P.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            el = __import__("time").time() - t0
            print(f"[{i+1}/{len(wavs)}] new={done} elapsed={el:.0f}s", flush=True)
    CACHE_P.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"CACHE SAVED entries={len(cache)}", flush=True)


if __name__ == "__main__":
    main()
