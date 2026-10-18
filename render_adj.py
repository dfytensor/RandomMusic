# -*- coding: utf-8 -*-
"""adj 渲染：基础 suffix 按需裁剪 + 缺维修正词"""
import sys
from pathlib import Path
no, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
mods = sys.argv[4] if len(sys.argv) > 4 else ""      # 追加修正词
drop = sys.argv[5] if len(sys.argv) > 5 else ""      # 需从基础suffix剔除的词组(分号分隔)
ROOT = Path(r"F:\yinyue")
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS
s = next(x for x in SONGS if x["no"] == no)
v2 = ROOT / "novel_songs" / "v2" / no / "lyrics_v2.txt"
lyrics = v2.read_text(encoding="utf-8") if v2.exists() else s["lyrics"]
BASE = ("restrained singing, minimal vocal ornamentation, clean straight tones, "
        "natural irregular breathing, warm dark tone, sparse piano with silences, "
        "breathing space between phrases, intimate, less is more")
for d in [x for x in drop.split(";") if x]:
    BASE = BASE.replace(d, "")
style = s["style"] + ", " + BASE + ((", " + mods) if mods else "")
from yue2 import YuE2Pipeline
pipe = YuE2Pipeline.from_pretrained(str(ROOT/"models"/"YuE2-3B"),
                                     vae=str(ROOT/"models"/"YuE2-Vae"),
                                     device="cuda", progress=False)
try:
    pipe(style=style, lyrics=lyrics, cot="full", seed=seed).save(out)
    print(f"OK {no} seed{seed} adj")
finally:
    pipe.close()
