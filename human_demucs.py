# -*- coding: utf-8 -*-
"""Demucs 批量分离人类库 wav → human_stems/<stem>/vocals.wav + beats.json (断点续跑)
v3.5: 新维(5 articulation dims)按全曲人声 stem 口径测量, 拍点来自 accomp beat_track 存为 beats.json
剩余文件只写 vocals(省磁盘), accomp 不落盘
"""
import sys, json, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(r"F:\yinyue")
LIB = ROOT / "eval" / "library_wavs"
OUT = ROOT / "human_stems"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100

import librosa
import numpy as np

# 1) 回填已有 accomp 的 beats.json
backfill = 0
for d in sorted(OUT.iterdir()):
    bj = d / "beats.json"
    acc = d / "accomp.wav"
    if d.is_dir() and not bj.exists() and acc.exists():
        try:
            ya, _ = librosa.load(str(acc), sr=22050, mono=True)
            tempo, beats = librosa.beat.beat_track(y=ya, sr=22050, hop_length=512)
            bt = librosa.frames_to_time(beats, sr=22050, hop_length=512)
            bj.write_text(json.dumps({
                "tempo": float(np.atleast_1d(tempo)[0]),
                "beat_times": [round(float(x), 4) for x in bt],
            }), encoding="utf-8")
            backfill += 1
        except Exception as e:
            print(f"  backfill ERR {d.name}: {e}", flush=True)
print(f"beats.json backfill={backfill}", flush=True)

# 2) 分离剩余
wavs = sorted(LIB.glob("*.wav"))
todo = []
for w in wavs:
    vdir = OUT / w.stem
    if (vdir / "vocals.wav").exists():
        continue
    todo.append(w)
todo = todo[:N]
print(f"todo={len(todo)}", flush=True)

from demucs.api import Separator
import torch, soundfile as sf

sep = Separator(model="htdemucs", device="cuda")
for i, w in enumerate(todo, 1):
    print(f"[{i}/{len(todo)}] {w.stem}", flush=True)
    try:
        out = OUT / w.stem
        out.mkdir(parents=True, exist_ok=True)
        origin, stems = sep.separate_audio_file(str(w))
        vocals = stems["vocals"]
        accomp = origin - vocals
        sf.write(str(out / "vocals.wav"), vocals.cpu().numpy().T, sep.samplerate)
        # accomp 仅用于拍点, 不落盘
        acc_np = accomp.cpu().numpy().mean(axis=0)
        ya = librosa.resample(acc_np, orig_sr=sep.samplerate, target_sr=22050)
        tempo, beats = librosa.beat.beat_track(y=ya, sr=22050, hop_length=512)
        bt = librosa.frames_to_time(beats, sr=22050, hop_length=512)
        (out / "beats.json").write_text(json.dumps({
            "tempo": float(np.atleast_1d(tempo)[0]),
            "beat_times": [round(float(x), 4) for x in bt],
        }), encoding="utf-8")
    except Exception as e:
        print(f"  ERR {type(e).__name__}: {e}", flush=True)
print("DONE", flush=True)
