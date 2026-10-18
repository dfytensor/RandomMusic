# -*- coding: utf-8 -*-
"""01 主题曲：砍前奏手术。实录谱删 intro 段 -> cot=melody 重唱 -> 母带 -> 复测。"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS
from remediate_novel import master, naturalness, transcribe_realized
from ai_flavor_check import load_reference, ai_score
from hit_score import score_song
from melody_stats import analyze_song

s = next(x for x in SONGS if x["no"] == "01")
work = ROOT / "novel_songs" / "v2" / "novel_01_欠天地一笔"
src_abc = (work / "verify" / "realized" / "score.abc").read_text(encoding="utf-8")

# 删 % intro 到下一个 % 段之间的所有行
lines = src_abc.splitlines()
out, skip = [], False
for line in lines:
    st = line.strip()
    if st.startswith("%"):
        if st[1:].strip().lower() == "intro":
            skip = True
            continue
        skip = False
    if not skip:
        out.append(line)
cut_abc = "\n".join(out) + "\n"
(work / "cut_intro.abc").write_text(cut_abc, encoding="utf-8")
print("intro cut, abc lines:", len(lines), "->", len(out), flush=True)

from yue2 import YuE2Pipeline
pipe = YuE2Pipeline.from_pretrained(str(ROOT / "models" / "YuE2-3B"),
                                    vae=str(ROOT / "models" / "YuE2-Vae"),
                                    device="cuda", progress=False)
try:
    song = pipe(style=s["style"], lyrics=s["lyrics"], abc=cut_abc, cot="melody", seed=32191)
    raw = work / "raw_cut.flac"
    song.save(str(raw))
finally:
    pipe.close()

mastered = work / "cut_mastered.flac"
master(raw, mastered)
shutil_copy = mastered
final_flac = ROOT / "novel_songs" / "final" / "01_欠天地一笔.flac"
shutil_copy_dst = final_flac
import shutil
shutil.copy(mastered, shutil_copy_dst)

# 复测
dur = __import__("soundfile").info(str(final_flac)).frames / 48000
tdir = work / "verify_cut"
transcribe_realized(final_flac, tdir)
REF = load_reference()
m = analyze_song(str(tdir / "realized" / "score.abc"))
import os
env_ok = True
hit = score_song(s)
fl_sc, _ = ai_score(*(() ,)) if False else (None, None)
import ai_segment_forensic as asf
seg = asf.read_mono_24k(str(final_flac), 0, min(32, dur))
fm = asf.f0_metrics(seg)
maha, cos5 = naturalness(final_flac)
print(json.dumps(dict(final_dur=round(dur, 1),
                      melody=dict(density=m["density"], range=m["pitch_range"],
                                  lift=m["lift_mean"], motif=m["motif_repeat"]),
                      first32_voiced=fm["voiced_ratio"],
                      naturalness=dict(maha=round(maha), cos5=round(cos5, 3)),
                      ), ensure_ascii=False, indent=1), flush=True)
print("hit score v3:", flush=True)
hit3 = score_song(s)
print(json.dumps(hit3, ensure_ascii=False, indent=1), flush=True)
