# -*- coding: utf-8 -*-
"""12 冠军按 v3.5 标准(11维, 5窗中位)打分, 对照共识组 + v3.3 排名
窗分: 6 旧维(混音窗) + 5 新维(全曲人声 stem, 歌级恒定)
"""
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
DIMS = ["orn", "gap", "cent", "dyn", "tension", "micro", "vib_ratio", "vib_depth",
        "phase_R", "onset_rise_db", "long_smooth"]
CACHE = ROOT/"eval/v35_champions.json"

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
from art_core import art_dims, ART_DIMS

def champ_art(seed):
    """冠军 stem 目录 -> 新维 (带缓存)"""
    cache = json.loads(CACHE.read_text(encoding='utf-8')) if CACHE.exists() else {}
    ck = f"__art__{seed}"
    if ck in cache and cache[ck]: return cache[ck]
    sd = ROOT/"album_voice"/"_stems"/seed
    m = art_dims(sd/"vocals.wav", sd/"beats.json")
    cache[ck] = m
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
    return m

def window_scores(path, key, seed, want=5):
    art = champ_art(seed)  # 先算 art(写盘), 再读 cache 避免覆盖
    cache = json.loads(CACHE.read_text(encoding='utf-8')) if CACHE.exists() else {}
    entry = cache.get(key, {})
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    dur = len(y)/SR
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    ons = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    win = int(30*SR/hop); step = int(5*SR/hop)
    best_t, best_sc = 0.0, -1
    for t in range(0, max(1, len(rms)-win), step):
        sc = rms[t:t+win].mean()*(1+ons[t:t+win].mean())
        if sc > best_sc: best_sc, best_t = sc, t*hop/sr
    max_off = max(0.0, dur-30)
    tiles = sorted({0.0, round(max_off/2,1), round(max_off,1)})
    tops = sorted({round(min(max_off, max(0.0, best_t)),1), round(min(max_off, dur/2-15),1)})
    order = [t for t in tops + tiles if t <= dur-25][:want]
    for t0 in order:
        sk = str(t0)
        if sk in entry and entry[sk] is not None: continue
        seg = y[int(t0*sr):int((t0+30)*sr)]
        if len(seg) < SR*10: entry[sk] = None; continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, seg, SR)
            m = measure(Path(f.name))
        Path(f.name).unlink(missing_ok=True)
        m.update(art)
        entry[sk] = score_of(m)
    cache[key] = entry
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
    return [v for v in entry.values() if v is not None]

CHAMPS = [
    ("01 非我不可", r"novel_songs\batch_gacha\01_v2\seed58888.flac"),
    ("02 平凡的",   r"novel_songs\batch_gacha\02_wide\seed7778.flac"),
    ("03 白狐",     r"novel_songs\batch_gacha\03_v2\seed61001.flac"),
    ("04 龙珠劫",   r"novel_songs\batch_gacha\04\seed90406.flac"),
    ("05 阳光说的对", r"novel_songs\batch_gacha\05_wide\seed1111.flac"),
    ("06 第320次",  r"novel_songs\batch_gacha\06_wide\seed3690.flac"),
    ("07 烟花",     r"novel_songs\batch_gacha\07_adj\seed1357_r1.flac"),
    ("08 童话世界", r"novel_songs\batch_gacha\11\seed91104.flac"),
    ("09 青衫",     r"novel_songs\batch_gacha\09\seed90904.flac"),
    ("10 毒药",     r"novel_songs\batch_gacha\10\seed91007.flac"),
    ("11 尾行",     r"novel_songs\batch_gacha\11\seed91103.flac"),
    ("12 天涯",     r"novel_songs\batch_gacha\12\seed91202.flac"),
]

# v3.3 旧分(对照)
old = {}
old_p = ROOT/"eval/v33_champions.json"
if old_p.exists():
    oc = json.loads(old_p.read_text(encoding='utf-8'))
    for name, _ in CHAMPS:
        scs = [v for v in oc.get(name, {}).values() if v is not None]
        if scs: old[name] = float(np.median(scs))

results = []
for name, rel in CHAMPS:
    p = ROOT/rel
    if not p.exists():
        print(f"{name}: MISSING {rel}", flush=True); continue
    seed = p.stem
    scs = window_scores(p, name, seed)
    med = round(float(np.median(scs)),1) if scs else None
    spread = round(max(scs)-min(scs),1) if len(scs) > 1 else 0
    results.append((name, med, spread, len(scs)))
    o = old.get(name)
    print(f"{name}: v3.5={med}  v3.3={o}  (窗分{[round(x,1) for x in scs]} 极差{spread})", flush=True)

vals = [r[1] for r in results if r[1] is not None]
print(f"\n12冠军 v3.5 分: 均值={np.mean(vals):.1f} 中位={np.median(vals):.1f} min={min(vals)} max={max(vals)}", flush=True)
if old:
    print(f"12冠军 v3.3 分: 中位={np.median(list(old.values())):.1f}", flush=True)
print("对照: 共识经典中位74.2 / 普通中位71.0 / 全库测试中位74.5", flush=True)

# 新维偏离度: 冠军 art vs 人类 train 中位 (按 beats tempo 定 cat)
ch_cache = json.loads(CACHE.read_text(encoding='utf-8'))
print("\n冠军新维 vs 人类 norm (z=|v-med|/scale):", flush=True)
for name, rel in CHAMPS:
    seed = Path(rel).stem
    m = ch_cache.get(f"__art__{seed}")
    if not m: continue
    bj = ROOT/"album_voice"/"_stems"/seed/"beats.json"
    tempo = json.loads(bj.read_text(encoding="utf-8"))["tempo"] if bj.exists() else 120.0
    cat = cat_of(tempo)
    parts = []
    for d in ART_DIMS:
        n = report['norms'].get(cat, {}).get(d)
        v = m.get(d)
        if n and v is not None:
            z = abs(v-n['med'])/n['scale']
            parts.append(f"{d}={z:.2f}")
    print(f"  {name}: {' '.join(parts)}", flush=True)
