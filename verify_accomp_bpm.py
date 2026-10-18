# -*- coding: utf-8 -*-
"""验证 v3.4: 伴奏轨定 BPM — 12 首换嗓版类别是否不再漂移"""
import json, sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
report = json.loads((ROOT/"eval"/"v3_validation.json").read_text(encoding='utf-8'))
from score_v3 import measure, score_of, cat_of

CH = [("01", r"novel_songs\batch_gacha\01_v2\seed58888.flac"),
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
      ("12", r"novel_songs\batch_gacha\12\seed91202.flac")]

cache_p = ROOT/"eval"/"v3_accomp_cache.json"
cache = json.loads(cache_p.read_text(encoding='utf-8')) if cache_p.exists() else {}

def get(key, fn):
    if key in cache and cache[key] is not None:
        return cache[key]
    for _ in range(2):
        try:
            m = fn()
            if m:
                cache[key] = m
                cache_p.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
                return m
        except Exception as e:
            print(f"  retry {key}: {type(e).__name__}", flush=True)
    cache[key] = None
    cache_p.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
    return None

print("| # | bpm原 | bpm新 | 类别 | 分原 | 分新 | Δ |", flush=True)
os_, ns_, bad_old = [], [], 0
# 旧换嗓版 (整曲BPM) 的类别漂移记录: 来自上轮实测
old_drift = {"01", "06", "10", "12"}
for no, rel in CH:
    p0 = ROOT/rel
    p1 = ROOT/"album_voice"/f"{no}_seedvc.flac"
    acc = ROOT/"album_voice"/"_stems"/Path(rel).stem/"accomp.wav"
    if not (acc.exists() and p1.exists()):
        print(f"| {no} | - | 缺stem | | | | |"); continue
    m0 = get(f"o_{no}", lambda: measure(p0, bpm_path=acc))
    m1 = get(f"s_{no}", lambda: measure(p1, bpm_path=acc))
    if not m0 or not m1:
        print(f"| {no} | 测量失败 | | | | | |"); continue
    s0, s1 = score_of(m0), score_of(m1)
    cat0, cat1 = cat_of(m0["bpm"]), cat_of(m1["bpm"])
    same = cat0 == cat1
    if not same: bad_old += 1
    print(f"| {no} | {m0['bpm']:.0f} | {m1['bpm']:.0f} | {cat0}{'=' if same else '≠'+cat1} | "
          f"{s0['score']:.1f} | {s1['score']:.1f} | {s1['score']-s0['score']:+.1f} |", flush=True)
    os_.append(s0['score']); ns_.append(s1['score'])

a, b = np.array(os_), np.array(ns_)
print(f"\n伴奏定BPM后: 类别不配对数={bad_old} (整曲BPM时=4: 01/06/10/12)", flush=True)
print(f"原: 中位={np.median(a):.1f}  新: 中位={np.median(b):.1f}  平均Δ={np.mean(b-a):+.1f}", flush=True)
print(f"配对相关ρ={np.corrcoef(a,b)[0,1]:+.3f} (整曲BPM时=+0.284)", flush=True)
from scipy.stats import wilcoxon
try:
    _, p = wilcoxon(a, b); print(f"Wilcoxon p={p:.4f}", flush=True)
except Exception:
    pass
