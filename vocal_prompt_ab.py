# -*- coding: utf-8 -*-
"""Prompt → 演唱表现力实验：
基线 prompt vs "克制唱腔" prompt，看装饰密度/气息CV 是否下降
"""
import json, sys, warnings, subprocess as sp
from pathlib import Path
import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from vocal_expression import vocal_expression_metrics

SR = 22050
OUT = ROOT / "novel_songs" / "vocal_prompt_ab"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "baseline": "female vocal, gentle ballad, piano and strings, warm tone, emotional",
    "restrained": ("female vocal, restrained singing, minimal vocal ornamentation, "
                   "clean straight tones, natural irregular breathing, understated, "
                   "less is more, no excessive melisma"),
    "raw": ("female vocal, raw unpolished singing, natural human imperfection, "
            "irregular breath, slightly off-pitch moments, authentic emotional delivery"),
}

LYRICS = "[verse]\n月光洒在窗台\n照见未还的账\n[chorus]\n欠天地一笔\n月供日再算"

from yue2 import YuE2Pipeline
pipe = YuE2Pipeline.from_pretrained(str(ROOT/"models"/"YuE2-3B"),
                                     vae=str(ROOT/"models"/"YuE2-Vae"),
                                     device="cuda", progress=False)

try:
    for name, style in VARIANTS.items():
        out = OUT / f"{name}.flac"
        if out.exists():
            print(f"  {name}: 已有", flush=True); continue
        try:
            song = pipe(style=style, lyrics=LYRICS, cot="full", seed=42)
            song.save(str(out))
            print(f"  {name}: ✅ saved", flush=True)
        except Exception as e:
            print(f"  {name}: FAIL {str(e)[:50]}", flush=True)
finally:
    pipe.close()

# ===== 评测 =====
lines = ["# Prompt → 演唱表现力 A/B", "",
         "| 变体 | 装饰密度/s | 气息CV | 音高偏离 | 乐句动态 |", "|---|---|---|---|---|"]
for name in VARIANTS:
    f = OUT / f"{name}.flac"
    if not f.exists(): continue
    try:
        y, _ = librosa.load(str(f), sr=SR, mono=True, duration=45)
        r = vocal_expression_metrics(y, SR)
        lines.append(f"| {name} | {r.get('orn_density','?')} | {r.get('gap_cv','?')} "
                     f"| {r.get('pitch_dev','?')} | {r.get('phrase_dyn','?')} |")
    except Exception as e:
        lines.append(f"| {name} | FAIL | | | |")

(ROOT/"eval"/"vocal_prompt_ab.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
