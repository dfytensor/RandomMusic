# -*- coding: utf-8 -*-
"""二分法抽卡：利用种子局部性
已证实: 相邻 seed 得分相似（91101≈91102, 91104≈91105, 91108≈91109）
策略:
  Phase 1: 在已知峰值(91102)左侧 stride=2 粗探（91094-91100）→ 找上升坡
  Phase 2: 在最高点邻域 stride=1 精搜 → 收敛局部峰值
  Phase 3: 对其他峰区(91104)边界同样二分
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from vocal_expression import vocal_expression_metrics

SR = 22050
OUT = ROOT / "novel_songs" / "triple_emergence"

STYLE = ("female vocal, restrained singing, minimal vocal ornamentation, "
         "clean straight tones, natural irregular breathing, warm dark tone, "
         "sparse piano with silences, breathing space between phrases, "
         "intimate, less is more, subtle emotional depth")
LYR_FILE = ROOT / "novel_songs/v2/yuegongri_prosody/lyrics_v2.txt"
LYRICS = LYR_FILE.read_text(encoding="utf-8")

def triple_score_file(f):
    y, _ = librosa.load(str(f), sr=SR, mono=True, duration=60)
    v = vocal_expression_metrics(y, SR)
    orn, gap = v.get("orn_density"), v.get("gap_cv")
    vocal = 0.0
    if orn is not None: vocal += max(0, 1 - abs(orn-0.09)/0.5)*50
    if gap is not None: vocal += max(0, 1 - abs(gap-1.38)/1.0)*50
    cent = float(librosa.feature.spectral_centroid(y=y, sr=SR)[0].mean())
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms+1e-9)
    dyn = float(np.percentile(db,95)-np.percentile(db,10))
    acoustic = max(0,1-abs(cent-2135)/1200)*50 + max(0,1-abs(dyn-14.6)/8)*50
    return dict(vocal=round(vocal,1), acoustic=round(acoustic,1),
                total=round((vocal+acoustic)/2,1), orn=orn, gap=gap,
                cent=round(cent), dyn=round(dyn,1))

# ===== 已知 landscape =====
KNOWN = {91101:52.8, 91102:56.4, 91103:37.5, 91104:48.4, 91105:47.4,
         91106:42.5, 91107:41.8, 91108:28.3, 91109:27.0, 91110:43.7,
         91111:33.2, 91112:45.0, 91094:30.9, 91096:47.1, 91098:28.0}

# Phase 1: 峰值左侧 stride=2 二分粗探
PHASE1 = [91100, 91095, 91097]

from yue2 import YuE2Pipeline
pipe = YuE2Pipeline.from_pretrained(str(ROOT/"models"/"YuE2-3B"),
                                     vae=str(ROOT/"models"/"YuE2-Vae"),
                                     device="cuda", progress=False)
scores = dict(KNOWN)
try:
    for seed in PHASE1:
        out = OUT / f"seed{seed}.flac"
        if not out.exists():
            try:
                song = pipe(style=STYLE, lyrics=LYRICS, cot="full", seed=seed)
                song.save(str(out))
                print(f"  seed{seed}: saved", flush=True)
            except Exception as e:
                print(f"  seed{seed}: FAIL {str(e)[:40]}", flush=True); continue
        r = triple_score_file(out)
        scores[seed] = r["total"]
        print(f"  seed{seed}: total={r['total']} vocal={r['vocal']} acou={r['acoustic']}", flush=True)
finally:
    pipe.close()

# Phase 2: 找到 >56 的新高点则向其邻域 stride=1 精搜
sorted_seeds = sorted([s for s in scores if s not in KNOWN] )
best_new = max((s for s in sorted_seeds), key=lambda s: scores.get(s,0), default=None)
lines = ["# 二分法抽卡 Phase1（峰值左侧 stride=2）", "",
         "| Seed | 综合 | 备注 |", "|---|---|---|"]
for s in sorted(scores):
    mark = " (已知)" if s in KNOWN else ""
    lines.append(f"| {s} | {scores[s]}{mark} | |")
best_all = max(scores, key=scores.get)
lines += ["", f"当前最优: seed {best_all} = {scores[best_all]}",
          "", "## 下一步: 若 91094-91100 出现 >56.4 的高点，向其 stride=1 邻域精搜"]
(ROOT/"eval"/"binary_gacha_phase2.md").write_text("\n".join(lines), encoding="utf-8")
print("\n" + "\n".join(lines[-6:]), flush=True)
