# -*- coding: utf-8 -*-
"""Prompt 工程实验：用不同 prompt 词组模拟 guidance_scale 和涌现条件
同一首歌 × 不同 prompt → 测哪个 prompt 词组改变哪个声学指标
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS

s = next(x for x in SONGS if x["no"] == "08")  # 月供日
work = ROOT / "novel_songs" / "v2" / "prompt_ab"
work.mkdir(parents=True, exist_ok=True)

# 基础 prompt
BASE = ("Mandopop R&B pop, flowing piano, guzheng fills, light percussion, "
        "warm bass, storytelling male vocal, bittersweet street-life warmth, "
        "G major, 108 BPM")

# 实验组：加入"涌现导向"关键词
VARIANTS = {
    "A_基线": BASE,
    "B_临界": BASE + ", on the edge of control and abandon, raw emotional vulnerability, barely holding together",
    "C_非线性": BASE + ", unexpected melodic turns, surprising harmonic shifts, unpredictable phrasing, anti-formula",
    "D_呼吸": BASE.replace("flowing piano", "sparse piano with deliberate silences between phrases") + ", breathing space between vocal phrases, less is more",
    "E_爆发": BASE + ", quiet verses building to explosive cathartic chorus, whisper to powerful belt dynamics, dramatic crescendo",
}

from yue2 import YuE2Pipeline
pipe = YuE2Pipeline.from_pretrained(str(ROOT/"models"/"YuE2-3B"),
                                     vae=str(ROOT/"models"/"YuE2-Vae"),
                                     device="cuda", progress=False)
try:
    for tag, style in VARIANTS.items():
        out = work / f"{tag}.flac"
        if out.exists():
            print(f"skip {tag}")
            continue
        print(f"生成 {tag}...", flush=True)
        song = pipe(style=style, lyrics=s["lyrics"], cot="full", seed=42)
        song.save(str(out))
finally:
    pipe.close()
print("ALL GENERATED", flush=True)
