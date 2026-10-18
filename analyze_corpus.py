# -*- coding: utf-8 -*-
"""Corpus analysis: 571 Mandarin pop songs -> features.json + summary."""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

warnings.filterwarnings("ignore")

SR = 22050
WINDOW = 90.0
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def detect_key(chroma_mean):
    scores = []
    for rot in range(12):
        c = np.roll(chroma_mean, -rot)
        scores.append((np.corrcoef(c, MAJOR)[0, 1], f"{PITCH[rot]} major"))
        scores.append((np.corrcoef(c, MINOR)[0, 1], f"{PITCH[rot]} minor"))
    best = max(scores, key=lambda x: x[0])
    return best[1], round(float(best[0]), 3)


def analyze(path):
    info = sf.info(str(path))
    duration = round(float(info.frames) / info.samplerate, 1)
    y, _ = librosa.load(str(path), sr=SR, mono=True, duration=WINDOW)
    tempo, _ = librosa.beat.beat_track(y=y, sr=SR, units="time")
    bpm = round(float(np.atleast_1d(tempo)[0]), 1)
    chroma = librosa.feature.chroma_stft(y=y, sr=SR).mean(axis=1)
    key, key_conf = detect_key(chroma / (chroma.max() + 1e-9))
    rms = librosa.feature.rms(y=y)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=SR)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR)[0]
    y30 = y[: SR * 30]
    harm = librosa.effects.harmonic(y30)
    perc = librosa.effects.percussive(y30)
    return {
        "file": path.name,
        "duration_s": duration,
        "bpm": bpm,
        "key": key,
        "key_conf": key_conf,
        "rms_db": round(float(librosa.amplitude_to_db(rms.mean() + 1e-9)), 1),
        "dyn_p95_p10_db": round(float(librosa.amplitude_to_db(np.percentile(rms, 95) + 1e-9)
                                      - librosa.amplitude_to_db(np.percentile(rms, 10) + 1e-9)), 1),
        "hpr": round(float(harm.std() / (perc.std() + 1e-9)), 3),
        "centroid_hz": round(float(cent.mean()), 0),
        "rolloff_hz": round(float(rolloff.mean()), 0),
        "zcr": round(float(zcr.mean()), 4),
    }


def main():
    files = sorted(Path(r"D:\音乐").glob("*.mp3"))
    files = [f for f in files if f.stat().st_size > 0]
    out_json = Path(r"F:\yinyue\corpus\features.json")
    results = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    t0 = time.time()
    errs = []
    for i, f in enumerate(files):
        if f.name in results:
            continue
        try:
            results[f.name] = analyze(f)
        except Exception as e:
            errs.append({"file": f.name, "err": str(e)[:100]})
        if (i + 1) % 20 == 0 or i == len(files) - 1:
            out_json.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[{i+1}/{len(files)}] ok={len(results)} err={len(errs)} elapsed={time.time()-t0:.0f}s",
                  flush=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(r"F:\yinyue\corpus\errors.json").write_text(json.dumps(errs, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
