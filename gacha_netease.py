# -*- coding: utf-8 -*-
"""NetEase 评分导向抽卡：最差的 3 首重抽 + 择优"""
import json, shutil, subprocess, sys, time, os
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS
from remediate_novel import transcribe_realized
from netease_score import compute_netease_metrics, score_netease

REDO = {"05": 3, "08": 3, "11": 3}  # 歌号: 抽几个
FINAL = ROOT / "novel_songs" / "final"


def netease_score_audio(flac_path):
    """计算音频的 NetEase 评分"""
    import librosa
    y, _ = librosa.load(str(flac_path), sr=22050, mono=True, duration=100)
    from netease_score import compute_netease_metrics, score_netease
    m = compute_netease_metrics(y, 22050)
    s, details = score_netease(m)
    return s, m


def render_one(no, seed, s):
    """子进程渲染一首"""
    tag = f"_g{seed}"
    work = ROOT / "novel_songs" / "gacha" / no / str(seed)
    work.mkdir(parents=True, exist_ok=True)
    raw = work / f"raw{tag}.flac"
    mastered = work / f"mastered{tag}.flac"
    if raw.exists() and mastered.exists():
        return mastered
    
    from yue2 import YuE2Pipeline
    from remediate_novel import master as do_master
    pipe = YuE2Pipeline.from_pretrained(str(ROOT / "models" / "YuE2-3B"),
                                         vae=str(ROOT / "models" / "YuE2-Vae"),
                                         device="cuda", progress=False)
    try:
        song = pipe(style=s["style"], lyrics=s["lyrics"], cot="full", seed=seed)
        song.save(str(raw))
    finally:
        pipe.close()
    do_master(str(raw), str(mastered))
    return mastered


def main():
    results = []
    for no, n_gacha in REDO.items():
        s = next(x for x in SONGS if x["no"] == no)
        title = s["title"]
        final_flac = FINAL / f"{no}_{title}.flac"
        
        # 当前版本的 NetEase 分
        import librosa
        y_cur, _ = librosa.load(str(final_flac), sr=22050, mono=True, duration=100)
        from netease_score import compute_netease_metrics, score_netease
        m_cur = compute_netease_metrics(y_cur, 22050)
        old_sc, _ = score_netease(m_cur)
        print(f"\n=== {no} {title} 现役分: {old_sc} ===", flush=True)
        
        # 抽卡
        best_flac, best_score = final_flac, old_sc
        for i in range(n_gacha):
            seed = s["seed"] + 2000 + i * 111
            print(f"  seed {seed} ...", flush=True)
            try:
                mastered = render_one(no, seed, s)
            except Exception as e:
                print(f"    fail: {str(e)[:60]}")
                continue
            
            # 评分
            import librosa as _lb
            y_new, _ = _lb.load(str(mastered), sr=22050, mono=True, duration=100)
            m_new = compute_netease_metrics(y_new, 22050)
            new_sc, _ = score_netease(m_new)
            print(f"    score: {new_sc}", flush=True)
            
            if new_sc > best_score:
                best_score = new_sc
                best_flac = mastered
                print(f"    ✅ 更优!", flush=True)
        
        if str(best_flac) != str(final_flac):
            shutil.copy(str(best_flac), str(final_flac))
            print(f"  → 替换 final", flush=True)
        results.append(dict(no=no, title=title, old_score=old_sc, best_score=best_score))
    
    print("\n=== 总结 ===", flush=True)
    for r in results:
        print(f"  {r['no']} {r['title']}: {r['old_score']} → {r['best_score']}")


if __name__ == "__main__":
    main()
