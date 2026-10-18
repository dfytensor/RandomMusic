# -*- coding: utf-8 -*-
"""情绪导向评分 v2：6 维（情绪锚点 60% + 声学 40%）
用法: from emotion_score_v2 import score_emotion, score_audio_v2
"""
import json
from pathlib import Path
import numpy as np
import librosa

_ROOT = Path(__file__).parent
_TG = json.loads((_ROOT/"eval"/"category_targets.json").read_text(encoding="utf-8"))

def emotion_anchors(y, sr=22050):
    """5 维情绪锚点"""
    hop = 512
    f0, vf, _ = librosa.pyin(y, fmin=65, fmax=700, sr=sr, frame_length=1024, hop_length=hop)
    voiced = vf & ~np.isnan(f0)
    # 紧张-释放
    tension = None
    if voiced.sum() > 20:
        f0v = f0[voiced]; d = np.abs(np.diff(f0v)); d = d[~np.isnan(d)]
        tension = round(float(np.sum(d>200)/(len(y)/sr)*60), 1)
    # 动态微弧
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms+1e-9)
    micro_arcs = []
    w = int(2*sr/hop)
    for i in range(w, len(db)-w):
        seg = db[i-w:i+w]; peak = np.argmax(seg)
        if 0.25*w < peak < 1.75*w:
            rise = seg[peak]-seg[0]; fall = seg[peak]-seg[-1]
            if rise > 1 and fall > 1: micro_arcs.append(min(rise, fall))
    micro_arc = round(float(np.median(micro_arcs)), 1) if micro_arcs else None
    return dict(tension=tension, micro_arc=micro_arc)

def classify_by_cat(r):
    """用已有 category_score 的分类"""
    from category_score import classify, score_vs_category
    return classify(r), score_vs_category(r)

def score_audio_v2(f, cat=None):
    """6 维评分: 情绪锚点 60% + 声学 40%"""
    y, _ = librosa.load(str(f), sr=22050, mono=True, offset=78, duration=30)
    # 声学 4 维 (from category_score)
    from category_score import score_audio, score_vs_category, classify, _TG
    r = score_audio(y)
    cat = cat or classify(r)
    sc = score_vs_category(r, cat)
    # 情绪 2 维
    ea = emotion_anchors(y)
    t = _TG[cat]["targets"]
    # 用同类 P25~75 区间做情绪评分
    def rng(key):
        lo, hi = _TG[cat][key].split("~"); return float(lo), float(hi)
    emo = 0
    if ea["tension"] is not None:
        # 紧张越接近同类中位越好（不太多不少）
        t_med = t.get("tension_med", None)
        if t_med is None:
            # 用 P25~75 中点
            lo, hi = rng("tension_p25_75") if "tension_p25_75" in _TG[cat] else (0, 5)
            t_med = (lo+hi)/2
        emo += max(0, 1-abs(ea["tension"]-t_med)/5)*30
    if ea["micro_arc"] is not None:
        lo, hi = rng("micro_arc_p25_75") if "micro_arc_p25_75" in _TG[cat] else (3, 5)
        m_med = (lo+hi)/2
        emo += max(0, 1-abs(ea["micro_arc"]-m_med)/3)*30
    # 声学分 (0-40, 从 P25~75 命中制转换)
    acou = sc["score"]*0.4  # 0-100 → 0-40
    total = round(emo + acou, 1)
    return dict(cat=cat, emo=round(emo,1), acou=round(acou,1), total=total,
                tension=ea["tension"], micro_arc=ea["micro_arc"],
                orn=r["orn"], gap=r["gap"], cent=r["cent"], dyn=r["dyn"])

if __name__ == "__main__":
    import sys
    f = sys.argv[1] if len(sys.argv) > 1 else "novel_songs/batch_gacha/01_v2/seed58888.flac"
    r = score_audio_v2(Path(_ROOT)/f)
    print(json.dumps(r, ensure_ascii=False, indent=1))
