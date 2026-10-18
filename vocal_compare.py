# -*- coding: utf-8 -*-
"""演唱表现力：真曲 vs AI 快速对比"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from vocal_expression import vocal_expression_metrics

SR = 22050

# ===== 曲库前 10 首 =====
lib_wav = ROOT / "eval" / "library_wavs"
corpus = []
for w in sorted(lib_wav.glob("*.wav"))[:10]:
    try:
        y, _ = librosa.load(str(w), sr=SR, mono=True, offset=78, duration=30)
        r = vocal_expression_metrics(y, SR)
        r["file"] = w.stem[:16]
        corpus.append(r)
    except: pass

# ===== AI 前 5 首 =====
ai = []
for f in sorted((ROOT/"novel_songs/final").glob("*.flac"))[:5]:
    try:
        y, _ = librosa.load(str(f), sr=SR, mono=True, offset=60, duration=30)
        r = vocal_expression_metrics(y, SR)
        r["file"] = f.stem[:16]
        ai.append(r)
    except: pass

# ===== 对比 =====
def cohen_d(a, b):
    a, b = np.array(a,float), np.array(b,float)
    if len(a)<2 or len(b)<2: return 0
    s = np.sqrt(((len(a)-1)*a.var()+(len(b)-1)*b.var())/(len(a)+len(b)-2))
    return float((a.mean()-b.mean())/(s+1e-9))

lines = ["# 演唱表现力：真曲 vs AI", "",
         f"真曲 n={len(corpus)} · 生成 n={len(ai)}", "",
         "| 指标 | 真曲中位 | AI中位 | d | 判读 |", "|---|---|---|---|---|"]

for key, note in [("phrase_dyn","乐句动态弧线dB"),("orn_density","装饰密度/s"),
                  ("gap_cv","气息自然度CV"),("pitch_dev","音高偏离(音分)"),
                  ("slide_frac","滑音比例")]:
    cv = [r[key] for r in corpus if r.get(key) is not None]
    nv = [r[key] for r in ai if r.get(key) is not None]
    if not cv or not nv: continue
    cm, nm = np.median(cv), np.median(nv)
    d = cohen_d(nv, cv)
    lines.append(f"| {note} | {cm:.2f} | {nm:.2f} | {d:+.2f} |")

for r in ai:
    lines.append(f"\n**{r['file']}**: phrase_dyn={r.get('phrase_dyn','?')} "
                 f"orn={r.get('orn_density','?')} pitch_dev={r.get('pitch_dev','?')} "
                 f"gap_cv={r.get('gap_cv','?')}")

(ROOT/"eval"/"vocal_expression_report.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
