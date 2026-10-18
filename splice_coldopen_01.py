# -*- coding: utf-8 -*-
"""01 音频拼接冷开场：副歌段(59.9-78s)剪到开头 + 交叉淡化 -> final
（绕开模型重渲染的塌缩问题，DAW 思路）
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS

s = next(x for x in SONGS if x["no"] == "01")
src = ROOT / "novel_songs" / "final" / f"01_{s['title']}.flac"
work = ROOT / "novel_songs" / "v2" / "novel_01_coldopen"

y, sr = sf.read(str(src), always_2d=True)
CH1_T1, CH1_T2 = 59.9, 78.5   # 第一副歌段
XFADE = int(0.6 * sr)

a1, a2 = int(CH1_T1 * sr), int(CH1_T2 * sr)
hook = y[a1:a2].copy()

# 交叉淡化：hook 尾部 vs 正曲头部
n = min(XFADE, len(hook))
ramp = np.linspace(0, 1, n)[:, None]
head = y[: n + int(0.2 * sr)]
tail_fade = hook[-n:] * (1 - ramp)
head_fade = head[:n] * ramp
spliced_head = np.vstack([hook[:-n], tail_fade + head_fade, head[n:]])

# hook 头部 0.2s 淡入
fi = int(0.2 * sr)
spliced_head[:fi] *= np.linspace(0.2, 1, fi)[:, None]

out = np.vstack([spliced_head, y[n + int(0.2 * sr):]])
peak = max(np.abs(out).max(), 1e-9)
out = out / peak * 0.89

dst = ROOT / "novel_songs" / "final" / f"01_{s['title']}_coldopen.flac"
sf.write(dst, out, sr, format="FLAC")
dur = len(out) / sr
print(f"cold open -> {dst.name}  {dur:.1f}s", flush=True)

# 复测：入场 + 前30s AI味 + 技术
sys.path.insert(0, str(ROOT))
from ai_flavor_check import load_reference, ai_score
import ai_segment_forensic as asf
from remediate_novel import naturalness

REF = load_reference()
seg = asf.read_mono_24k(str(dst), 0, 30)
fm = asf.f0_metrics(seg)
m30 = dict(fm)
m30["notes"] = None
ai30, _ = ai_score(m30, REF)
maha, cos5 = naturalness(dst)
print(f"前30s: 人声占比={fm['voiced_ratio']} AI味={ai30}", flush=True)
print(f"自然度: maha={maha:.0f} cos5={cos5:.3f}", flush=True)

# 技术快检
subprocess.run([str(ROOT / ".venv" / "Scripts" / "python.exe"),
                str(ROOT / "reference_free_check.py"), str(dst), "--out",
                str(work / "tech_cold.md")], check=True, capture_output=True)
txt = (work / "tech_cold.md").read_text(encoding="utf-8")
import json as j
d = j.loads(txt.split("json")[1].split("```")[0])[0]
print(f"技术: {d['pass']}/{d['total']} bpm_win={d['bpm_windows']}", flush=True)
