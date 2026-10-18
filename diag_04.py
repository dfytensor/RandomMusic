# -*- coding: utf-8 -*-
"""诊断冠军窗分爆炸原因: 每窗打印 bpm/类别/各维原值 vs norms"""
import json, sys, warnings, tempfile
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
sys.path.insert(0, str(ROOT))
report = json.loads((ROOT/"eval"/"v3_validation.json").read_text(encoding='utf-8'))
norms = report['norms']
DIMS = ["orn", "gap", "cent", "dyn", "tension", "micro"]
from score_v3 import measure

def cat_of(bpm):
    if bpm < 100: return "C1"
    if bpm < 120: return "C2"
    if bpm < 135: return "C3"
    return "C4"

p = ROOT/r"novel_songs\batch_gacha\04\seed90406.flac"
y, sr = librosa.load(str(p), sr=SR, mono=True)
print(f"dur={len(y)/sr:.1f}s", flush=True)
hop = 512
rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
ons = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
win = int(30*SR/hop); step = int(5*SR/hop)
best_t, best_sc = 0.0, -1
for t in range(0, max(1, len(rms)-win), step):
    sc = rms[t:t+win].mean()*(1+ons[t:t+win].mean())
    if sc > best_sc: best_sc, best_t = sc, t*hop/sr
max_off = max(0.0, len(y)/sr-30)
tiles = sorted({0.0, round(max_off/2,1), round(max_off,1)})
tops = sorted({round(min(max_off, max(0.0, best_t)),1), round(min(max_off, len(y)/sr/2-15),1)})
order = [t for t in tops + tiles if t <= len(y)/sr-25][:5]
print(f"windows: {order}", flush=True)
for t0 in order:
    seg = y[int(t0*sr):int((t0+30)*sr)]
    if len(seg) < SR*10: print(f"t0={t0}: too short"); continue
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, seg, SR)
        m = measure(Path(f.name))
    Path(f.name).unlink(missing_ok=True)
    if not m: print(f"t0={t0}: measure FAIL"); continue
    cat = cat_of(m['bpm'])
    line = [f"t0={t0:5.1f} bpm={m['bpm']:.0f}[{cat}]"]
    for d in DIMS:
        n = norms[cat][d]; v = m[d]
        s = 100*(1-abs(v-n['med'])/n['scale'])
        if v < n['p5']: s -= (n['p5']-v)/n['scale']*100
        if v > n['p95']: s -= (v-n['p95'])/n['scale']*100
        flag = "!!" if s < -30 else "  "
        line.append(f"{d}={v:.2f}[{s:.0f}]{flag}")
    print(" ".join(line), flush=True)
print("\n对标 C3 norms:", {d: {k: round(norms['C3'][d][k],3) for k in ['p5','med','p95','scale']} for d in DIMS}, flush=True)
