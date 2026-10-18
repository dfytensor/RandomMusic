# -*- coding: utf-8 -*-
"""多维度二分爬坡抽卡
已知 landscape（15 seed × 4 维）：
  每维找冠军 → 对维度冠军的未探邻域 stride=1 爬坡 → 帕累托判优
维度冠军：
  装饰 91096(0.23) | 气息 91103(1.31)/91104(1.14) | 质心 91105(2093) | 动态 91101(17.7)
未探 frontier：91099, 91100(OOM×2), 91093
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
LYRICS = (ROOT / "novel_songs/v2/yuegongri_prosody/lyrics_v2.txt").read_text(encoding="utf-8")
TGT = dict(orn=0.09, gap=1.38, cent=2135, dyn=14.6)

# seed → (orn, gap, cent, dyn)
DATA = {
    91094:(None,None,None,None), 91095:(0.30,0.58,1845,20.5), 91096:(0.23,0.91,1698,25.1),
    91097:(0.33,1.08,1813,19.4), 91098:(None,None,None,None),
    91101:(0.28,0.71,1596,17.7), 91102:(0.33,0.81,1851,18.3), 91103:(0.62,1.31,1620,22.9),
    91104:(1.17,1.14,2048,20.6), 91105:(0.40,0.93,2093,23.5), 91106:(0.45,0.84,1622,19.5),
    91107:(0.23,0.69,1706,23.9), 91108:(0.52,0.66,1789,24.1), 91109:(0.60,0.74,1531,20.8),
    91110:(0.33,0.82,1881,23.0), 91111:(0.47,0.78,1591,21.5), 91112:(0.30,0.58,1845,20.5),
}

def dims_score(orn, gap, cent, dyn):
    v = (max(0,1-abs(orn-0.09)/0.5)*50 + max(0,1-abs(gap-1.38)/1.0)*50) if orn and gap else 0
    a = max(0,1-abs(cent-2135)/1200)*50 + max(0,1-abs(dyn-14.6)/8)*50
    return round(v,1), round(a,1), round((v+a)/2,1)

def score_file(f):
    y,_ = librosa.load(str(f), sr=SR, mono=True, duration=60)
    r = vocal_expression_metrics(y, SR)
    cent = float(librosa.feature.spectral_centroid(y=y, sr=SR)[0].mean())
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms+1e-9)
    dyn = float(np.percentile(db,95)-np.percentile(db,10))
    return r.get("orn_density"), r.get("gap_cv"), round(cent), round(dyn,1)

# ===== Phase 3: frontier 探测 =====
PROBES = [91099, 91093, 91100]
from yue2 import YuE2Pipeline
pipe = YuE2Pipeline.from_pretrained(str(ROOT/"models"/"YuE2-3B"),
                                     vae=str(ROOT/"models"/"YuE2-Vae"),
                                     device="cuda", progress=False)
try:
    for seed in PROBES:
        out = OUT / f"seed{seed}.flac"
        if seed in DATA and DATA[seed][0] is not None: continue
        if not out.exists():
            try:
                song = pipe(style=STYLE, lyrics=LYRICS, cot="full", seed=seed)
                song.save(str(out))
                print(f"  seed{seed}: saved", flush=True)
            except Exception as e:
                print(f"  seed{seed}: FAIL {str(e)[:36]}", flush=True)
                DATA[seed] = (None,None,-1,-1); continue
        DATA[seed] = score_file(out)
finally:
    pipe.close()

# ===== 多维排名 + 帕累托 =====
rows = []
for s,(o,g,c,d) in DATA.items():
    if o is None or c == -1: continue
    v,a,t = dims_score(o,g,c,d)
    rows.append(dict(seed=s, orn=o, gap=g, cent=c, dyn=d, vocal=v, acou=a, total=t))

def dist_tgt(r):
    return (abs(r['orn']-0.09)/0.5 + abs(r['gap']-1.38)/1.0 +
            abs(r['cent']-2135)/1200 + abs(r['dyn']-14.6)/8)

for r in rows: r['dist'] = round(dist_tgt(r),3)

by_total = sorted(rows, key=lambda r:-r['total'])
by_dist  = sorted(rows, key=lambda r:r['dist'])
pareto = [r for r in rows if not any(
    (o['orn']<=r['orn'] and o['gap']>=r['gap'] and abs(o['cent']-2135)<=abs(r['cent']-2135)
     and abs(o['dyn']-14.6)<=abs(r['dyn']-14.6)) and
    (o['orn']<r['orn'] or o['gap']>r['gap'] or abs(o['cent']-2135)<abs(r['cent']-2135)
     or abs(o['dyn']-14.6)<abs(r['dyn']-14.6))
    for o in rows if o is not r)]

lines = ["# 多维二分爬坡 Phase3", "",
         "| Seed | orn/s | gapCV | cent | dyn | 演唱 | 声学 | 综合 | 距目标Σ |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in sorted(rows, key=lambda x:-x['total']):
    lines.append(f"| {r['seed']} | {r['orn']} | {r['gap']} | {r['cent']} | {r['dyn']} "
                 f"| {r['vocal']} | {r['acou']} | **{r['total']}** | {r['dist']} |")
lines += ["", f"帕累托前沿: {[r['seed'] for r in pareto]}",
          f"综合冠军: {by_total[0]['seed']}={by_total[0]['total']}",
          f"距目标最近: {by_dist[0]['seed']} (dist={by_dist[0]['dist']})"]
(ROOT/"eval"/"multidim_gacha.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[-16:]), flush=True)
