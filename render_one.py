# -*- coding: utf-8 -*-
"""单曲单seed渲染（独立子进程，OOM 安全）"""
import sys
from pathlib import Path

no, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ROOT = Path(r"F:\yinyue")
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS

s = next(x for x in SONGS if x["no"] == no)
v2 = ROOT / "novel_songs" / "v2" / f"{no}" / "lyrics_v2.txt"
if not v2.exists():
    alt = ROOT / "novel_songs" / "v2" / "yuegongri_prosody" / "lyrics_v2.txt"
    v2 = alt if (no == "08" and alt.exists()) else None
lyrics = v2.read_text(encoding="utf-8") if v2 else s["lyrics"]

STYLE_SUFFIX = ("restrained singing, minimal vocal ornamentation, clean straight tones, "
                "natural irregular breathing, warm dark tone, sparse piano with silences, "
                "breathing space between phrases, intimate, less is more")
style = s["style"] + ", " + STYLE_SUFFIX

from yue2 import YuE2Pipeline
pipe = YuE2Pipeline.from_pretrained(str(ROOT/"models"/"YuE2-3B"),
                                     vae=str(ROOT/"models"/"YuE2-Vae"),
                                     device="cuda", progress=False)
try:
    song = pipe(style=style, lyrics=lyrics, cot="full", seed=seed)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    song.save(out)
    print(f"OK {no} seed{seed}")
finally:
    pipe.close()
