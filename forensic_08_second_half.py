# -*- coding: utf-8 -*-
"""08《月供日》100s 后 AI 味取证：全曲转谱 + 分段指纹 vs 曲库副歌基线"""
import json
import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from ai_segment_forensic import analyze_segment, parse_lab, read_mono_24k
from ai_flavor_check import load_reference, ai_score

SS2_PY = ROOT / ".venv-ss2" / "Scripts" / "python.exe"
flac = ROOT / "novel_songs" / "final" / "08_月供日.flac"
work = ROOT / "novel_songs" / "v2" / "novel_08_月供日" / "fullcheck"
work.mkdir(parents=True, exist_ok=True)

# 1) 全曲解码 + 转谱
full_wav = work / "full.wav"
if not full_wav.exists():
    y, _ = librosa.load(str(flac), sr=24000, mono=True)
    sf.write(full_wav, y, 24000, subtype="PCM_16")
tdir = work / "realized_full"
if not (tdir / "structure.lab").exists():
    env = dict(os.environ)
    env.update(PYTHONUTF8="1", HF_HUB_OFFLINE="1")
    subprocess.run([str(SS2_PY), str(ROOT / "transcribe.py"), str(full_wav), str(tdir)],
                   check=True, capture_output=True, env=env)

dur = sf.info(str(flac)).frames / sf.info(str(flac)).samplerate
rows = parse_lab(tdir, "structure.lab")
merged = {}
order = []
for t1, t2, n in rows:
    n = n.lower()
    if n == "silence":
        continue
    if n not in merged:
        merged[n] = [t1, t2]
        order.append(n)
    else:
        merged[n][1] = max(merged[n][1], t2)
print("全曲结构:", {k: [round(v[0], 1), round(v[1], 1)] for k, v in merged.items()}, f"dur={dur:.0f}s")

# 2) 100s 之后的段落逐个取证
ref = load_reference()
spans = [(n, v[0], v[1]) for n, v in merged.items() if v[0] >= 95 and v[1] - v[0] >= 8]
report = []
for name, t1, t2 in spans:
    t2c = min(t2, t1 + 30, dur)
    m = analyze_segment(str(flac), tdir, t1, t2c, want_symbolic=True)
    sc, detail = ai_score(m, ref)
    report.append(dict(section=name, t=[round(t1, 1), round(t2c, 1)], score=sc, metrics=m))
    print(f"{name:12s} {t1:6.1f}-{t2c:6.1f}s  AI味={sc}  voiced={m['voiced_ratio']} "
          f"runs={m.get('n_vocal_runs')} notes={m.get('notes')} "
          f"vib={m.get('vibrato_rate_hz')}/{m.get('vibrato_extent_cents')}c "
          f"drift={m.get('intra_note_drift_cents')} dynstd={m.get('macro_dyn_std_db')} "
          f"range={m.get('dyn_range_db')}", flush=True)
    worst = sorted([d for d in detail.values() if d and d["z"] > 1], key=lambda d: -d["z"])[:3]
    for d in worst:
        print(f"    +{d['z']:.1f}σ {d['x']} (真曲中位 {d['med']})")

# 3) 与自身前半对照：前 95s 内的同段落
for name, v in merged.items():
    if v[0] < 90 and v[1] - v[0] >= 8 and name in [r["section"] for r in report]:
        t2c = min(v[1], v[0] + 30, 93)
        m = analyze_segment(str(flac), tdir, v[0], t2c, want_symbolic=True)
        sc, _ = ai_score(m, ref)
        report.append(dict(section="前半-" + name, t=[round(v[0], 1), round(t2c, 1)], score=sc, metrics=m))
        print(f"前半-{name:8s} {v[0]:6.1f}-{t2c:6.1f}s  AI味={sc}", flush=True)

(work / "second_half_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print("SAVED", work / "second_half_report.json")
