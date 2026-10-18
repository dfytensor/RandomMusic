# -*- coding: utf-8 -*-
"""12 个类别感知冠军歌词 × Prosody 门禁体检
冠军唱腔已满分/高分，检查歌词是否配得上
"""
import json, sys
from pathlib import Path
sys.path.insert(0, r"F:\yinyue")
from lyrics_check import check
from novel_songs_manifest import SONGS

ROOT = Path(r"F:\yinyue")
CHAMPIONS = {"01":"90107","02":"90200","03":"90302","04":"90406","05":"90505",
             "06":"90607","07":"90708","08":"91104","09":"90904","10":"91007",
             "11":"91103","12":"91202"}
V2 = {"08": ROOT/"novel_songs/v2/yuegongri_prosody/lyrics_v2.txt"}

lines = ["# 冠军歌词 Prosody 门禁体检", "",
         "| 歌 | 词版本 | 门禁通过 | 总检查项 | 韵辙覆盖 | 句长中位 | 段落数 | 短板 |",
         "|---|---|---|---|---|---|---|---|"]
for no in sorted(CHAMPIONS):
    s = next(x for x in SONGS if x["no"] == no)
    if no in V2 and V2[no].exists():
        text, ver = V2[no].read_text(encoding="utf-8"), "v2-Prosody"
    else:
        text, ver = s["lyrics"], "v1-原稿"
    try:
        checks, info = check(text)
        npass = sum(1 for c in checks if c[1])
        names = [str(c[0])[:12] for c in checks if not c[1]]
        lines.append(f"| {no} | {ver} | **{npass}/{len(checks)}** | {len(checks)} "
                     f"| {info['rhyme_coverage']} | {info['line_len_median']} | {info['n_sections']} "
                     f"| {'、'.join(names) if names else '-'} |")
    except Exception as e:
        lines.append(f"| {no} | {ver} | ERR {type(e).__name__} | | | | | |")
(ROOT/"eval"/"champion_lyrics_gate.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
