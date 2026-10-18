# -*- coding: utf-8 -*-
"""专辑对应性审计：类型/BPM/调式/韵辙-情绪/字音对应 -> eval/album_audit.md"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS
from lyrics_check import parse_lyrics, line_final
from abc_compare import parse_abc

# 十三辙响度三级（传统曲艺理论）
LOUD = {"发花", "江阳", "言前", "人辰", "中东"}      # 洪亮级：激昂/宏伟/直抒
SOFT = {"怀来", "梭波", "遥条", "由求"}             # 柔和级：舒展/温柔/流动
THIN = {"衣期", "灰堆", "姑苏", "乜斜"}             # 细微级：内敛/哀愁/呜咽

# 每首歌的意图情绪 -> 适配响度级
SONG_EMOTION = {
    "01": ("激昂壮阔", {"洪亮"}),
    "02": ("冷峻内敛", {"洪亮", "柔和", "细微"}),
    "03": ("哀而不伤", {"细微", "柔和"}),
    "04": ("谐趣轻快", {"柔和", "洪亮"}),
    "05": ("温情催泪", {"洪亮", "柔和"}),
    "06": ("哀恸挽歌", {"细微", "柔和", "洪亮"}),
    "07": ("冷酷压迫", {"细微", "洪亮"}),
    "08": ("温暖流转", {"柔和", "细微"}),
    "09": ("危机反抗", {"洪亮"}),
    "10": ("治愈温情", {"柔和"}),
    "11": ("胜利坦荡", {"洪亮"}),
    "12": ("史诗终局", {"洪亮"}),
}

# 实测 BPM（final tech report）与实测调式
tech = json.loads((ROOT / "novel_songs" / "final" / "tech_report.md").read_text(encoding="utf-8").split("json")[1].split("```")[0])
MEAS = {r["file"][:2]: r for r in tech}


def norm_bpm(b, ref):
    while b < ref / 1.5:
        b *= 2
    while b > ref * 1.5:
        b /= 2
    return b


def chorus_rhyme(lyrics):
    secs = parse_lyrics(lyrics)
    chorus = next((s for s in secs if s["tag"] == "Chorus" and len(s["lines"]) >= 4), None)
    if not chorus:
        return None, None, None
    groups = [line_final(l) for l in chorus["lines"]]
    groups = [g for g in groups if g]
    if not groups:
        return None, None, None
    top, cnt = Counter(groups).most_common(1)[0]
    level = "洪亮" if top in LOUD else ("柔和" if top in SOFT else "细微")
    return top, round(cnt / len(groups), 2), level


def syllable_note_ratio(no, title, lyrics, tdir):
    """歌词正文字数(Verse+PC+Chorus 前三段) vs ABC 人声音符数(verse起~chorus末)"""
    secs = parse_lyrics(lyrics)
    syll = 0
    taken = {"Verse": 0, "Pre-Chorus": 0, "Chorus": 0}
    for s in secs:
        if s["tag"] in taken and taken[s["tag"]] < 1:
            for l in s["lines"]:
                syll += len(re.findall(r"[\u4e00-\u9fff]", l))
            taken[s["tag"]] += 1
    meta, vocal, sp = parse_abc(str(Path(tdir) / "score.abc"))
    # verse 首个起点
    v_start = next((t for n, t in sp if n.lower() == "verse"), None)
    c_end = None
    chorus_times = [t for n, t in sp if n.lower() == "chorus"]
    if chorus_times:
        c_end = chorus_times[0] + 30
    if v_start is None or c_end is None:
        return None
    notes = [1 for o, m, d in vocal if v_start <= o <= c_end]
    if syll == 0:
        return None
    return round(len(notes) / syll, 2)


TRANSCRIPT = {
    "01": ROOT / "eval/novel_transcripts/novel_01_欠天地一笔",
    "02": ROOT / "eval/novel_transcripts/novel_02_平账人",
    "03": ROOT / "eval/novel_transcripts/novel_03_白户",
    "04": ROOT / "novel_songs/redo/04/verify/realized",
    "05": ROOT / "eval/novel_transcripts/novel_05_心疼说成省钱",
    "06": ROOT / "novel_songs/redo/06/verify/realized",
    "07": ROOT / "novel_songs/v2/novel_07_规矩/verify/realized",
    "08": ROOT / "eval/novel_transcripts/novel_08_月供日",
    "09": ROOT / "eval/novel_transcripts/novel_09_断供",
    "10": ROOT / "eval/novel_transcripts/novel_10_糕与药包",
    "11": ROOT / "novel_songs/v2/novel_11_烂尾劫_full/verify/realized",
    "12": ROOT / "novel_songs/v2/novel_12_新契_full/verify/realized",
}

GF = json.loads((ROOT / "corpus" / "guofeng_profile.json").read_text(encoding="utf-8"))
lines = ["# 《现贷修仙传》12 首对应性审计", "",
         f"国风基准（曲库24首）：BPM P25-P75 {GF['bpm_p25']}-{GF['bpm_p75']}（中位 {GF['bpm_med']}）· "
         f"小调 {GF['minor_pct']}% · 常用调 {GF['top_keys']}", "",
         "韵辙响度三级：洪亮(发花/江阳/言前/人辰/中东)=激昂宏伟 · 柔和(怀来/梭波/遥条/由求)=温柔流动 · "
         "细微(一七/灰堆/姑苏/乜斜)=内敛哀愁", "",
         "| # | 曲目 | 角色 | 意图情绪 | 规格BPM | 实测BPM(归一) | 实测调式 | 副歌韵辙(级) | 韵-情匹配 | 字/音比 |", "|---|---|---|---|---|---|---|---|---|---|"]
issues = []
for s in SONGS:
    no = s["no"]
    emo, ok_levels = SONG_EMOTION[no]
    top, cov, level = chorus_rhyme(s["lyrics"])
    rhyme_ok = level in ok_levels
    m = MEAS.get(no, {})
    bpm_meas = m.get("bpm_windows", [None])[0]
    bpm_n = norm_bpm(bpm_meas, s["bpm"]) if bpm_meas else None
    bpm_off = abs(bpm_n - s["bpm"]) / s["bpm"] * 100 if bpm_n else None
    key_maj = m.get("key_majority", "?")
    mode_meas = "minor" if "minor" in key_maj else ("major" if "major" in key_maj else "?")
    mode_ok = mode_meas == s["mode"]
    tdir = TRANSCRIPT.get(no)
    ratio = syllable_note_ratio(no, s["title"], s["lyrics"], tdir) if tdir and tdir.exists() else None
    ratio_ok = ratio is not None and 0.75 <= ratio <= 1.45
    if not rhyme_ok:
        issues.append(f"{no} {s['title']}：副歌韵辙[{top}·{level}级] 与意图情绪[{emo}]不适配（适配级：{'/'.join(sorted(ok_levels))}）")
    if bpm_off and bpm_off > 10:
        issues.append(f"{no} {s['title']}：BPM 偏离规格 {bpm_off:.0f}%")
    if not mode_ok:
        issues.append(f"{no} {s['title']}：调式 实测{mode_meas} vs 规格{s['mode']}")
    if ratio is not None and not ratio_ok:
        issues.append(f"{no} {s['title']}：字/音比 {ratio} 超出 0.75-1.45（歌词与旋律疏密度不匹配）")
    lines.append(f"| {no} | {s['title']} | {s['kind']} | {emo} | {s['bpm']} | "
                 f"{bpm_n if bpm_n else '?'} ({bpm_off:.0f}%) | {key_maj} | {top}({level}) 覆盖{cov} | "
                 f"{'✅' if rhyme_ok else '❌'} | {ratio if ratio is not None else '?'} |")

lines += ["", "## 发现的问题", ""]
if issues:
    for i in issues:
        lines.append(f"- {i}")
else:
    lines.append("- 全部对应 ✅")
(Path(ROOT / "eval" / "album_audit.md")).write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print("\nISSUES:", len(issues))
