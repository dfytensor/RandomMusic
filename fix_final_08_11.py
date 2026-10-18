# -*- coding: utf-8 -*-
"""08 回退到完整版 v1 母带；11 用全曲手术重唱版替换。"""
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from remediate_novel import master, naturalness

FINAL = ROOT / "novel_songs" / "final"
V2 = ROOT / "novel_songs" / "v2"


def dur(p):
    i = __import__("soundfile").info(str(p))
    return i.frames / i.samplerate


# 08: 现役是截断版 -> 用完整 v1 母带
s08 = FINAL / "08_月供日.flac"
print("08 before:", round(dur(s08), 1), "s")
master(ROOT / "novel_songs" / "08_月供日.flac", s08)
print("08 after:", round(dur(s08), 1), "s")
m08, c08 = naturalness(s08)
print(f"08 naturalness: maha={m08:.0f} cos5={c08:.3f}")

# 11: 用全曲手术重唱版（AI味 4.79 vs 12.22）
s11 = FINAL / "11_烂尾劫.flac"
cand = V2 / "novel_11_烂尾劫_full" / "cand_mastered.flac"
shutil.copy(cand, s11)
m11, c11 = naturalness(s11)
print(f"11 replaced: {dur(s11):.1f}s naturalness maha={m11:.0f} cos5={c11:.3f}")
