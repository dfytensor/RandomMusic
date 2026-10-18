# -*- coding: utf-8 -*-
"""破格实验：01《欠天地一笔》副歌冷开场（打破前奏惯例）
实录谱按 % 段落切块 -> 重排为 Chorus 前置 -> 歌词重排 -> cot=melody 重唱 -> 母带 -> 四方复测
"""
import json
import re
import shutil
import subprocess
import sys
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS
from remediate_novel import master, naturalness, transcribe_realized
from ai_flavor_check import load_reference
from melody_stats import analyze_song

s = next(x for x in SONGS if x["no"] == "01")
title = s["title"]
work = ROOT / "novel_songs" / "v2" / "novel_01_coldopen"
work.mkdir(parents=True, exist_ok=True)
src_abc = (ROOT / "novel_songs/v2/novel_01_欠天地一笔/verify/realized/score.abc").read_text(encoding="utf-8")

# ---------- 1. 按 % 标记切块 ----------
header, blocks, cur_name, cur_lines = [], [], None, []
for line in src_abc.splitlines():
    st = line.strip()
    if st.startswith("%"):
        if cur_name is not None:
            blocks.append((cur_name, cur_lines))
        cur_name = st[1:].strip().lower()
        cur_lines = []
        continue
    if cur_name is None:
        header.append(line)
    else:
        cur_lines.append(line)
if cur_name is not None:
    blocks.append((cur_name, cur_lines))

names = [b[0] for b in blocks]
print("原段落顺序:", names, flush=True)

# ---------- 2. 重排：去 intro，副歌前置 ----------
chorus_blocks = [b for b in blocks if b[0] == "chorus"]
first_chorus = chorus_blocks[0] if chorus_blocks else None
new_order = []
if first_chorus:
    new_order.append(first_chorus)
for b in blocks:
    if b[0] != "chorus" and b[0] != "intro":
        new_order.append(b)
# 去掉最后的 outro（冷开场版直接收）
new_order = [b for b in new_order if b[0] != "outro"] or new_order
new_text = "\n".join(header) + "\n" + "\n".join(
    f"% {name}\n" + "\n".join(ls) for name, ls in new_order) + "\n"
(work / "coldopen.abc").write_text(new_text, encoding="utf-8")
print("新段落顺序:", [b[0] for b in new_order], flush=True)

# ---------- 3. 歌词重排（与 ABC 段落一一对应） ----------
lyrics = s["lyrics"]
secs, cur = [], None
for raw in lyrics.splitlines():
    st = raw.strip()
    m = re.match(r"\[([^\]]+)\]", st)
    if m:
        cur = m.group(1)
        secs.append({"tag": cur, "lines": []})
        continue
    if st and cur:
        secs[-1]["lines"].append(st)
by_tag = {}
for sec in secs:
    by_tag.setdefault(sec["tag"], []).append(sec["lines"])

# ABC 新顺序 -> 歌词段落（intro/outro 跳过）
LYRIC_TAG = {"verse": "Verse", "pre-chorus": "Pre-Chorus", "chorus": "Chorus",
             "bridge": "Bridge", "interlude": "Interlude"}
used = {}
new_lyrics_parts = []
for name, _ in new_order:
    tag = LYRIC_TAG.get(name)
    if not tag:
        continue
    idx = used.get(tag, 0)
    pool = by_tag.get(tag, [])
    block_lines = pool[idx] if idx < len(pool) else (pool[-1] if pool else [])
    used[tag] = idx + 1
    new_lyrics_parts.append(f"[{tag}]")
    if block_lines:
        new_lyrics_parts.extend(block_lines)
    new_lyrics_parts.append("")
new_lyrics = "\n".join(new_lyrics_parts).strip() + "\n"
(work / "coldopen_lyrics.txt").write_text(new_lyrics, encoding="utf-8")
print("歌词重排完成", flush=True)

# ---------- 4. 重唱 + 母带 ----------
from yue2 import YuE2Pipeline
raw = work / "raw.flac"
if not raw.exists():
    pipe = YuE2Pipeline.from_pretrained(str(ROOT / "models" / "YuE2-3B"),
                                        vae=str(ROOT / "models" / "YuE2-Vae"),
                                        device="cuda", progress=False)
    try:
        song = pipe(style=s["style"], lyrics=new_lyrics, abc=new_text, cot="melody",
                    seed=s["seed"] + 900)
        song.save(str(raw))
    finally:
        pipe.close()
mastered = work / "coldopen_mastered.flac"
if not mastered.exists():
    master(raw, mastered)

# ---------- 5. 复测 ----------
import soundfile as sf
dur = (lambda i: i.frames / i.samplerate)(sf.info(str(mastered)))
tdir = work / "verify"
transcribe_realized(mastered, tdir)
r = analyze_song(str(tdir / "realized" / "score.abc"))
maha, cos5 = naturalness(mastered)
# 冷开场判定：首个人声段起点
first_voice_sec = next((n for n, t1, t2 in
                        [(rr[2], rr[0], rr[1]) for rr in
                         [l.split("\t") for l in (tdir / "realized" / "structure.lab").read_text(encoding="utf-8").splitlines() if len(l.split("\t")) >= 3]]
                        ), None)
# 简化：取 structure.lab 第二行段落名
lab_lines = [l.split("\t") for l in (tdir / "realized" / "structure.lab").read_text(encoding="utf-8").splitlines() if len(l.split("\t")) >= 3]
entry_section = next((q[2] for q in lab_lines if q[2].strip().lower() not in ("silence", "intro")), "?")

REF = load_reference()
from ai_flavor_check import ai_score
seg_y = None
import librosa
seg_y, _ = librosa.load(str(mastered), sr=24000, mono=True, duration=30)
import ai_segment_forensic as asf
fm = asf.f0_metrics(seg_y)
m30 = dict(fm)
m30["notes"] = None
ai30, _ = ai_score(m30, REF)

print(json.dumps(dict(final_dur=round(dur, 1), entry_section=entry_section,
                      melody=dict(density=r["density"], range=r["pitch_range"],
                                  lift=r["lift_mean"], motif=r["motif_repeat"]),
                      first30_voiced=fm["voiced_ratio"],
                      first30_ai=ai30,
                      naturalness=dict(maha=round(maha), cos5=round(cos5, 3))),
                 ensure_ascii=False, indent=1), flush=True)
print(f"final -> {mastered}", flush=True)
