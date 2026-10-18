# -*- coding: utf-8 -*-
"""类别感知评分：先定类 → 用该类曲库目标打分（供抽卡调用）
用法:
    from category_score import score_audio, classify
    r = score_audio(y, sr)          # 4维原始值
    cat = classify(r)               # 'C1'..'C4'
    sc = score_vs_category(r, cat)  # 各维落 P25~75 计 25 分
"""
import json
from pathlib import Path
import numpy as np
import librosa

_ROOT = Path(__file__).parent
_TG = json.loads((_ROOT/"eval"/"category_targets.json").read_text(encoding="utf-8"))
# C1..C4 中位向量 (cent, dyn, bpm) 用于定类
_CATV = {c: np.array([v["targets"]["cent"], v["targets"]["dyn"],
                      float(v["bpm_med"])]) for c, v in _TG.items()}
_SCALE = np.array([600.0, 8.0, 50.0])

def _rng(c, key):
    lo, hi = _TG[c][key].split("~")
    return float(lo), float(hi)

def score_audio(y, sr=22050, hop=512):
    """与曲库 library_dims_batch.py 完全同款方法（hop=512, 调用方负责段落选择）"""
    f0, vflag, _ = librosa.pyin(y, fmin=65, fmax=700, sr=sr,
                                frame_length=1024, hop_length=hop)
    voiced = vflag & ~np.isnan(f0)
    orn = gap = None
    if voiced.sum() > 20:
        d = np.abs(np.diff(f0)); m = voiced[1:] & voiced[:-1]
        orn = round(float(np.sum(d[m] > 15)/(len(y)/sr)), 2)
    idx = np.where(voiced)[0]
    if len(idx) > 5:
        gl = np.diff(idx)*hop/sr*1000; gl = gl[gl > 1]
        if len(gl) > 3:
            gap = round(float(np.std(gl)/(np.mean(gl)+1e-9)), 2)
    cent = round(float(librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()))
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms+1e-9)
    dyn = round(float(np.percentile(db,95)-np.percentile(db,10)), 1)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(float(np.atleast_1d(tempo)[0]), 1)
    return dict(orn=orn, gap=gap, cent=cent, dyn=dyn, bpm=bpm)

def classify(r):
    v = np.array([r["cent"], r["dyn"], r["bpm"]])
    return min(_CATV, key=lambda c: np.abs((v-_CATV[c])/_SCALE).sum())

def score_vs_category(r, cat=None):
    """落同类 P25~75 区间每维 25 分；orn 用 ≤P75 计（越少越好）"""
    cat = cat or classify(r)
    t = _TG[cat]
    def in_range(v, key):
        lo, hi = _rng(cat, key)
        return lo <= v <= hi
    hits = {
        "orn": (r["orn"] is not None and r["orn"] <= _rng(cat,"orn_p25_75")[1]),
        "gap": in_range(r["gap"], "gap_p25_75") if r["gap"] is not None else False,
        "cent": in_range(r["cent"], "cent_p25_75"),
        "dyn": in_range(r["dyn"], "dyn_p25_75"),
    }
    total = round(25*sum(hits.values()), 1)
    return dict(cat=cat, hits=hits, n_hits=sum(hits.values()), score=total,
                targets=t["targets"])

if __name__ == "__main__":
    print("类别目标:", {c: v["targets"] for c, v in _TG.items()})
