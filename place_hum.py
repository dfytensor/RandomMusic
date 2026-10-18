# -*- coding: utf-8 -*-
"""归位已渲染的 hum 版本到 final"""
import shutil
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).parent
FINAL = ROOT / "novel_songs" / "final"

for no in sys.argv[1].split(","):
    s = next(x for x in __import__("novel_songs_manifest").SONGS if x["no"] == no)
    seed = s["seed"] + 800
    src = ROOT / "novel_songs" / "gacha" / no / str(seed) / "mastered_hum.flac"
    if not src.exists():
        # 也检查 verify 相关目录
        alt = ROOT / "novel_songs" / "gacha" / no / str(seed) / "raw_hum.flac"
        if alt.exists():
            src = alt
        else:
            print(f"{no}: 无成品，跳过")
            continue
    dst = FINAL / f"{no}_{s['title']}_hum.flac"
    shutil.copy(src, dst)
    i = sf.info(str(dst))
    print(f"{no} {s['title']}: {i.frames/i.samplerate:.0f}s -> {dst.name}")
