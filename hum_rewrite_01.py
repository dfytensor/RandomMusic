# -*- coding: utf-8 -*-
"""副歌旋律简化手术：《欠天地一笔》可哼版
保留节奏骨架与 Ins 伴奏；每次副歌的 Vocal 重写为 8 小节可哼动机循环（末句 b12 长音收）
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS
from remediate_novel import master

s = next(x for x in SONGS if x["no"] == "01")
work = ROOT / "novel_songs" / "v2" / "novel_01_hum"
work.mkdir(parents=True, exist_ok=True)
src = (ROOT / "novel_songs/v2/novel_01_欠天地一笔/verify/realized/score.abc").read_text(encoding="utf-8")

PATTERN = ["b4 b4 f4 e4", "d8 b4 d8", "e4 e4 e4 f8-", "f4 e4 d4 c4"]  # 8 小节循环
END_BAR = "b12 z4"

out_lines, in_chorus, pat_i = [], False, 0
for line in src.splitlines():
    st = line.strip()
    if st.startswith("%"):
        name = st[1:].strip().lower()
        if in_chorus and name != "chorus":
            out_lines.append("V: Vocal")
            out_lines.append(END_BAR + "|")
            in_chorus = False
        in_chorus = (name == "chorus")
        out_lines.append(line)
        continue
    if not in_chorus or st.startswith(("V:", "X:", "T:", "M:", "L:", "Q:", "K:")) or not st:
        out_lines.append(line)
        continue
    # Vocal 音乐行（4 小节）-> 输出 4 行模板
    out_lines.append("V: Vocal")
    for _ in range(4):
        out_lines.append(PATTERN[pat_i % len(PATTERN)] + "|")
        pat_i += 1
if in_chorus:
    out_lines.append("V: Vocal")
    out_lines.append(END_BAR + "|")

new_abc = "\n".join(out_lines) + "\n"
(work / "hum.abc").write_text(new_abc, encoding="utf-8")
print(f"rewritten, pattern bars used: {pat_i}", flush=True)

from yue2 import YuE2Pipeline
raw = work / "raw.flac"
if not raw.exists():
    pipe = YuE2Pipeline.from_pretrained(str(ROOT / "models" / "YuE2-3B"),
                                        vae=str(ROOT / "models" / "YuE2-Vae"),
                                        device="cuda", progress=False)
    try:
        song = pipe(style=s["style"], lyrics=s["lyrics"], abc=new_abc, cot="melody", seed=s["seed"])
        song.save(str(raw))
    finally:
        pipe.close()
mastered = work / "mastered.flac"
if not mastered.exists():
    master(raw, mastered)
final_flac = ROOT / "novel_songs" / "final" / f"01_{s['title']}_hum.flac"
shutil.copy(mastered, final_flac)

import soundfile as sf
dur = (lambda i: i.frames / i.samplerate)(sf.info(str(final_flac)))
print(f"OK -> {final_flac.name} {dur:.0f}s", flush=True)
