# -*- coding: utf-8 -*-
"""对齐检验: AI 前100s 内 3 平铺窗(0/35/70) vs 曲库同位置窗
排除"曲库截断100s vs AI全长"的测量不对等 confound
"""
import json, sys, warnings, tempfile
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
sys.path.insert(0, str(ROOT))
report = json.loads((ROOT/"eval"/"v3_validation.json").read_text(encoding='utf-8'))
norms = report['norms']
DIMS = ["orn", "gap", "cent", "dyn", "tension", "micro"]
CACHE = ROOT/"eval/v33_champ_100s.json"

def cat_of(bpm):
    if bpm < 100: return "C1"
    if bpm < 120: return "C2"
    if bpm < 135: return "C3"
    return "C4"

def score_of(m):
    if not m: return None
    cat = cat_of(m['bpm']); parts = []
    for d in DIMS:
        n = norms.get(cat, {}).get(d); v = m.get(d)
        if not n or v is None: continue
        s = 100*(1 - abs(v-n['med'])/n['scale'])
        if v < n['p5']: s -= (n['p5']-v)/n['scale']*100
        if v > n['p95']: s -= (v-n['p95'])/n['scale']*100
        parts.append(s)
    return round(float(np.mean(parts)),1) if parts else None

from score_v3 import measure

CHAMPS = [
    ("01", r"novel_songs\batch_gacha\01_v2\seed58888.flac"),
    ("02", r"novel_songs\batch_gacha\02_wide\seed7778.flac"),
    ("03", r"novel_songs\batch_gacha\03_v2\seed61001.flac"),
    ("04", r"novel_songs\batch_gacha\04\seed90406.flac"),
    ("05", r"novel_songs\batch_gacha\05_wide\seed1111.flac"),
    ("06", r"novel_songs\batch_gacha\06_wide\seed3690.flac"),
    ("07", r"novel_songs\batch_gacha\07_adj\seed1357_r1.flac"),
    ("08", r"novel_songs\batch_gacha\11\seed91104.flac"),
    ("09", r"novel_songs\batch_gacha\09\seed90904.flac"),
    ("10", r"novel_songs\batch_gacha\10\seed91007.flac"),
    ("11", r"novel_songs\batch_gacha\11\seed91103.flac"),
    ("12", r"novel_songs\batch_gacha\12\seed91202.flac"),
]

cache = json.loads(CACHE.read_text(encoding='utf-8')) if CACHE.exists() else {}
ai_spreads, ai_medians = {}, {}
for no, rel in CHAMPS:
    p = ROOT/rel
    if not p.exists(): continue
    entry = cache.get(no, {})
    if len(entry) < 3:
        y, sr = librosa.load(str(p), sr=SR, mono=True)
        for t0 in (0.0, 35.0, 70.0):
            sk = str(t0)
            if sk in entry and entry[sk] is not None: continue
            seg = y[int(t0*sr):int((t0+30)*sr)]
            if len(seg) < SR*10: entry[sk] = None; continue
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, seg, SR)
                m = measure(Path(f.name))
            Path(f.name).unlink(missing_ok=True)
            entry[sk] = score_of(m)
        cache[no] = entry
        CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
    vs = [v for v in entry.values() if v is not None]
    if len(vs) >= 3:
        ai_spreads[no] = max(vs)-min(vs)
        ai_medians[no] = float(np.median(vs))
    print(f"{no}: {[round(v,1) for v in vs]} 极差={ai_spreads.get(no, 0):.1f}", flush=True)

# 曲库同位置窗 (v33_window_scores 的 0/35/70 tile)
lb = json.loads((ROOT/"eval/v33_window_scores.json").read_text(encoding='utf-8'))
lib_spreads, lib_medians = {}, {}
for k, d in lb.items():
    sel = {t: v for t, v in d.items() if float(t) in (0.0, 35.0, 70.0) and v is not None}
    if len(sel) >= 3:
        vs = list(sel.values())
        lib_spreads[k] = max(vs)-min(vs)
        lib_medians[k] = float(np.median(vs))

a_s = list(ai_spreads.values()); l_s = list(lib_spreads.values())
print(f"\n对齐后 (同为 0/35/70 三平铺窗):", flush=True)
print(f"  AI n={len(a_s)}  窗分极差中位={np.median(a_s):.1f}  分中位={np.median(list(ai_medians.values())):.1f}", flush=True)
print(f"  曲库 n={len(l_s)} 窗分极差中位={np.median(l_s):.1f}  分中位={np.median(list(lib_medians.values())):.1f}", flush=True)
if a_s and l_s:
    u, p = mannwhitneyu(a_s, l_s, alternative='greater')
    print(f"  极差U检验(AI>人类): p={p:.4f} {'显著' if p<0.05 else '不显著 → 原2.4×主要是confound'}", flush=True)
