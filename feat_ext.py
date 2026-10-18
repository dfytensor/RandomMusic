# -*- coding: utf-8 -*-
"""扩展声学特征：曲库样本(32) vs 生成曲(12)
BPM/调/响度/质心/滚降/过零/动态 + 频段能量占比 + 起音密度 + 立体声宽度
-> eval/feat_corpus.json, eval/feat_novel.json
"""
import json
import warnings
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
WINDOW = 90.0
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
BANDS = [(20, 250), (250, 2000), (2000, 6000), (6000, 11000)]  # 低频/中低/中高/高频


def key_of(chroma_mean):
    scores = []
    for rot in range(12):
        c = np.roll(chroma_mean, -rot)
        scores.append((np.corrcoef(c, MAJOR)[0, 1], f"{PITCH[rot]} major"))
        scores.append((np.corrcoef(c, MINOR)[0, 1], f"{PITCH[rot]} minor"))
    return max(scores, key=lambda x: x[0])[1]


def analyze(path):
    info = sf.info(str(path))
    duration = round(float(info.frames) / info.samplerate, 1)
    # 立体声宽度（全曲）
    data, sr0 = sf.read(str(path), always_2d=True)
    if data.shape[1] >= 2:
        mid = data.mean(axis=1)
        side = (data[:, 0] - data[:, 1]) / 2
        width = float(np.sqrt((side ** 2).mean()) / (np.sqrt((mid ** 2).mean()) + 1e-9))
    else:
        width = 0.0
    y, _ = librosa.load(str(path), sr=SR, mono=True, duration=WINDOW)
    tempo, _ = librosa.beat.beat_track(y=y, sr=SR, units="time")
    bpm = round(float(np.atleast_1d(tempo)[0]), 1)
    chroma = librosa.feature.chroma_stft(y=y, sr=SR).mean(axis=1)
    key = key_of(chroma / (chroma.max() + 1e-9))
    rms = librosa.feature.rms(y=y)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=SR)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=SR)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # 频段能量占比（90s 窗口平均谱）
    S = np.abs(librosa.stft(y)) ** 2
    freqs = librosa.fft_frequencies(sr=SR)
    band_share = []
    tot = S.sum() + 1e-9
    for lo, hi in BANDS:
        band_share.append(round(float(S[(freqs >= lo) & (freqs < hi)].sum() / tot), 4))

    # 起音密度与强度
    ons = librosa.onset.onset_strength(y=y, sr=SR)
    onsets = librosa.onset.onset_detect(y=y, sr=SR, units="time", delta=0.08)
    onset_density = round(len(onsets) / (len(y) / SR), 2)
    onset_strength = round(float(ons.mean()), 3)

    y30 = y[: SR * 30]
    harm = librosa.effects.harmonic(y30)
    perc = librosa.effects.percussive(y30)
    return dict(
        file=Path(path).name, duration_s=duration, bpm=bpm, key=key,
        rms_db=round(float(librosa.amplitude_to_db(rms.mean() + 1e-9)), 1),
        dyn_p95_p10_db=round(float(librosa.amplitude_to_db(np.percentile(rms, 95) + 1e-9)
                                    - librosa.amplitude_to_db(np.percentile(rms, 10) + 1e-9)), 1),
        centroid_hz=round(float(cent.mean())), rolloff_hz=round(float(roll.mean())),
        zcr=round(float(zcr.mean()), 4),
        band_share=band_share, onset_density=onset_density, onset_strength=onset_strength,
        stereo_width=round(width, 3),
    )


def run(files, out):
    res = []
    for i, f in enumerate(files):
        try:
            res.append(analyze(f))
        except Exception as e:
            print("FAIL", Path(f).name, str(e)[:80])
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(files)}]", flush=True)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("SAVED", out, len(res))


if __name__ == "__main__":
    corpus = sorted((ROOT / "corpus" / "sample_wavs").glob("*.wav"))
    novel = sorted((ROOT / "eval" / "wavs").glob("novel_*.wav"))
    run(corpus, ROOT / "eval" / "feat_corpus.json")
    run(novel, ROOT / "eval" / "feat_novel.json")
