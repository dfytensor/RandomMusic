# -*- coding: utf-8 -*-
"""ACE-Step 抽卡：同一首歌多 seed 生成 + NetEase 评分择优"""
import json, sys, os
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

ROOT = Path(r"F:\yinyue")
ACE_ROOT = Path(r"F:\ACE-Step")
MODEL_PATH = ACE_ROOT / "acestep-v15-xl-sft-diffusers"
OUT = ROOT / "ace_step_output" / "gacha"

sys.path.insert(0, str(ROOT))
from netease_direct_compare import compute_all, netease_score, IDEAL

# 歌曲定义（同 YuE2 专辑）
import sys as _s
_s.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS

TARGETS = ["08", "01", "06"]  # YuE2 最强的 3 首
SEEDS = {"08": [100, 200, 300], "01": [400, 500, 600], "06": [700, 800, 900]}

# 从 YuE2 manifest 取歌词和 prompt
from ace_generate import TEST  # 复用之前定义的 3 首

from diffusers import AceStepPipeline
pipe = AceStepPipeline.from_pretrained(str(MODEL_PATH), torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")
pipe.vae.enable_tiling()

all_results = []
for t in TEST:
    no = t["no"]
    title = t["title"]
    dur = t["duration"]
    lyrics = t["lyrics"]
    prompt = t["prompt"]
    
    seeds = SEEDS.get(no, [100, 200, 300])
    best_score = -1
    best_file = None
    
    for seed in seeds:
        out_flac = OUT / f"ace_{no}_{title}_s{seed}.flac"
        wav_tmp = OUT / f"ace_{no}_{title}_s{seed}.wav"
        
        if out_flac.exists():
            print(f"  skip seed{seed} (cached)", flush=True)
        else:
            try:
                output = pipe(prompt=prompt, lyrics=lyrics, audio_duration=dur,
                              seed=seed)
                audio = output.audios[0] if hasattr(output, 'audios') else output.audio[0]
                if hasattr(audio, 'cpu'):
                    audio = audio.cpu().numpy()
                if audio.ndim > 1:
                    audio = audio[0] if audio.shape[0] < audio.shape[1] else audio.T[0]
                sf.write(str(wav_tmp), audio, 22050, subtype="PCM_16")
                import soundfile as sf2
                data, sr = sf2.read(str(wav_tmp))
                sf2.write(str(out_flac), data, sr, format="FLAC")
                wav_tmp.unlink()
            except Exception as e:
                print(f"  seed{seed} FAIL: {str(e)[:60]}", flush=True)
                continue
        
        # NetEase 评分
        import librosa
        y, _ = librosa.load(str(out_flac), sr=22050, mono=True, duration=100)
        
        # 快速算 9 指标
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        db = librosa.amplitude_to_db(rms+1e-9)
        cent = float(librosa.feature.spectral_centroid(y=y, sr=22050)[0].mean())
        w3 = int(3*22050/512)
        smooth = np.convolve(db, np.ones(12)/12, mode="same")
        dips = sum(1 for i in range(w3, len(smooth)-w3)
                   if smooth[i]==smooth[i-w3:i+w3].min() and smooth[i-w3:i+w3].max()-smooth[i]>3)
        
        # 简化 NetEase 分
        s = 0.0
        s += max(0, 1 - abs(dips - 2) / 2) * 33
        s += max(0, 1 - abs(cent - 2050) / 1000) * 33
        s += 34 if 180 <= dur <= 280 else 17
        total = round(s, 1)
        
        if total > best_score:
            best_score = total
            best_file = out_flac
        print(f"  seed{seed}: NetEase分={total}", flush=True)
    
    all_results.append(dict(no=no, title=title, best_score=best_score, best_file=str(best_file) if best_file else ""))
    print(f"  最佳: {best_score}", flush=True)

pipe = None

print(f"\n=== 总结 ===", flush=True)
for r in all_results:
    print(f"  {r['no']} {r['title']}: best={r['best_score']}", flush=True)
