# -*- coding: utf-8 -*-
"""用 Prosody 方法论重写《月供日》并渲染"""
import json, os, sys
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS

s = next(x for x in SONGS if x["no"] == "08")
work = ROOT / "novel_songs" / "v2" / "yuegongri_prosody"

lyrics = (work / "lyrics_v2.txt").read_text(encoding="utf-8") if (work / "lyrics_v2.txt").exists() else None

# 找 rewrite_yuegongri.py 保存的歌词
lyrics_src = ROOT / "novel_songs" / "v2" / "yuegongri_prosody" / "lyrics_v2.txt"
if lyrics_src.exists():
    lyrics = lyrics_src.read_text(encoding="utf-8")
else:
    # 从 rewrite 脚本中提取
    sys.path.insert(0, str(ROOT))
    exec(open(ROOT / "rewrite_yuegongri.py").read().split("# ===== 对比分析")[0].split("# ===== 渲染")[0])
    lyrics = NEW

print("歌词前5行:", lyrics.splitlines()[:5], flush=True)

from yue2 import YuE2Pipeline
raw = work / "raw.flac"
if not raw.exists():
    pipe = YuE2Pipeline.from_pretrained(str(ROOT/"models"/"YuE2-3B"),
                                         vae=str(ROOT/"models"/"YuE2-Vae"),
                                         device="cuda", progress=False)
    try:
        song = pipe(style=s["style"], lyrics=lyrics, cot="full", seed=42)
        song.save(str(raw))
    finally:
        pipe.close()

# 母带
import librosa
mastered = work / "mastered.flac"
if not mastered.exists():
    from remediate_novel import master as do_master
    y, sr_ = librosa.load(str(raw), sr=48000, mono=False)
    if y.ndim == 1: y = np.stack([y, y])
    # 简单归一
    peak = np.abs(y).max()
    if peak > 0.95: y *= 0.95 / peak
    sf.write(str(mastered), y.T, sr_, format="FLAC")

# 技术快检
import subprocess
r = subprocess.run([str(ROOT/".venv"/"Scripts"/"python.exe"),
                    str(ROOT/"reference_free_check.py"), str(mastered), "--out",
                    str(work/"tech.md")], check=True, capture_output=True)
print(r.stdout.decode("utf-8")[-500:], flush=True)
print(f"DONE -> {mastered}", flush=True)
