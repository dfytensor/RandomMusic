# -*- coding: utf-8 -*-
"""逐曲择优：v1(母带化) vs v2手术版 -> novel_songs/final/ + 终检"""
import shutil
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS
from remediate_novel import master, naturalness

FINAL = ROOT / "novel_songs" / "final"
FINAL.mkdir(exist_ok=True)
V2 = ROOT / "novel_songs" / "v2"


def master_to_temp(src_flac, dst_flac):
    tmp = dst_flac.parent / (dst_flac.stem + "_tmp" + dst_flac.suffix)
    master(src_flac, tmp)
    return tmp


rows = []
for s in SONGS:
    stem = f"novel_{s['no']}_{s['title']}"
    v1_flac = ROOT / "novel_songs" / f"{s['no']}_{s['title']}.flac"
    v2_flac = V2 / stem / f"{s['no']}_{s['title']}_mastered.flac"
    if not v2_flac.exists():
        # v2 手术版可能生成失败：退化用 v1
        v2_flac = None
    cands = []
    # v1 母带化到临时，测自然度
    tmp = FINAL / f"__tmp_{s['no']}.flac"
    shutil.copy(v1_flac, tmp)
    m1, c1 = naturalness(tmp)
    cands.append(("v1", m1, c1, tmp))
    if v2_flac and v2_flac.exists():
        m2, c2 = naturalness(v2_flac)
        cands.append(("v2", m2, c2, v2_flac))
    # 择优: 自然度 cos 优先；差 <0.003 时比马氏
    cands.sort(key=lambda t: (-t[2], t[1]))
    tag, maha, cos, src = cands[0]
    final_flac = FINAL / f"{s['no']}_{s['title']}.flac"
    if src == tmp:
        shutil.move(str(tmp), final_flac)
    else:
        shutil.copy(v2_flac, final_flac)
        tmp.unlink(missing_ok=True)
    rows.append((s["no"], s["title"], tag, cos, maha))
    print(f"{s['no']} {s['title']}: choose {tag} cos={cos:.3f} maha={maha:.0f}", flush=True)

# 技术终检
import subprocess
subprocess.run([str(ROOT / ".venv" / "Scripts" / "python.exe"),
                str(ROOT / "reference_free_check.py"),
                *[str(FINAL / f"{s['no']}_{s['title']}.flac") for s in SONGS],
                "--out", str(FINAL / "tech_report.md")], check=True)

# 汇总
txt = (FINAL / "tech_report.md").read_text(encoding="utf-8")
data = json.loads(txt.split("```json")[1].split("```")[0])
tech_ok = sum(1 for r in data if r["pass"] == r["total"])
print(f"FINAL: {len(rows)} 首，择优 v2={sum(1 for r in rows if r[2]=='v2')} v1={sum(1 for r in rows if r[2]=='v1')}，"
      f"自然度均值 {np.mean([r[3] for r in rows]):.3f}，技术全过 {tech_ok}/{len(data)}")
