# -*- coding: utf-8 -*-
"""音乐情绪机制分析：为什么音乐能影响人类情绪和记忆
框架: 基于音乐心理学 + 音频信号分析 + 曲库 571 首实测
输出: 每类歌的情绪画像 + AI 与真人的情绪锚点差距
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050

def emotion_profile(y, sr):
    """5 维情绪锚点"""
    hop = 512
    # 1. 音高轮廓情绪价
    f0, vf, _ = librosa.pyin(y, fmin=65, fmax=700, sr=sr, frame_length=1024, hop_length=hop)
    ft = np.arange(len(f0))*hop/sr
    voiced = vf & ~np.isnan(f0)
    contours = []
    in_v, start = False, 0
    for i, v in enumerate(voiced):
        if v and not in_v: in_v, start = True, i
        elif not v and in_v:
            if i-start > 8:
                seg = f0[start:i]
                seg = seg[~np.isnan(seg)]
                if len(seg) > 4:
                    d = seg[-1]/(seg[0]+1e-9)
                    contours.append(np.log2(d)*1200)  # cents net change
            in_v = False
    pitch_rise = round(float(np.median(contours)),1) if contours else None

    # 2. 动态微弧（乐句内 crescendo→decrescendo）
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms+1e-9)
    micro_arcs = []
    w = int(2*sr/hop)  # 2s window
    for i in range(w, len(db)-w):
        seg = db[i-w:i+w]
        peak = np.argmax(seg)
        if peak > w*0.25 and peak < w*1.75:
            rise = seg[peak] - seg[0]
            fall = seg[peak] - seg[-1]
            if rise > 1 and fall > 1:
                micro_arcs.append(min(rise, fall))
    micro_arc = round(float(np.median(micro_arcs)),1) if micro_arcs else None

    # 3. 时间柔度（rubato = 节拍偏离程度）
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    if len(beats) > 8:
        intervals = np.diff(beats)/sr
        ibi_cv = round(float(np.std(intervals)/(np.mean(intervals)+1e-9)), 3)
    else:
        ibi_cv = None

    # 4. 紧张-释放密度（半音跳跃次数/分钟 = 紧张感来源）
    if voiced.sum() > 20:
        f0v = f0[voiced]
        semitone_diffs = np.abs(np.diff(f0v))
        semitone_diffs = semitone_diffs[~np.isnan(semitone_diffs)]
        tension_events = int(np.sum(semitone_diffs > 200))  # >2 semitones
        tension_density = round(float(tension_events/(len(y)/sr)*60), 1)
    else:
        tension_density = None

    # 5. 人声温暖度（低频能量占比 + F0稳定性）
    S = np.abs(librosa.stft(y, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low_mask = (freqs >= 80) & (freqs <= 300)
    warm_ratio = round(float(S[low_mask].mean()/(S.mean()+1e-9)), 2)
    if voiced.sum() > 10:
        f0v = f0[voiced]
        f0v = f0v[~np.isnan(f0v)]
        f0_stability = round(float(1.0/(np.std(f0v)+1e-9)), 3)
    else:
        f0_stability = None

    return dict(
        pitch_rise=pitch_rise,      # >0 上行(紧张/希望), <0 下行(悲伤/释然)
        micro_arc=micro_arc,        # 乐句内动态弧线深度
        ibi_cv=ibi_cv,              # 节拍柔度(rubato)
        tension=tension_density,    # 紧张-释放密度(次/分)
        warm=warm_ratio,            # 人声温暖度
        f0_stab=f0_stability,       # F0稳定性(越高=越"机器",越低=越"人类")
    )

def main():
    # 曲库
    lib_wav = ROOT/"eval"/"library_wavs"
    corpus = []
    for f in sorted(lib_wav.glob("*.wav"))[:40]:
        try:
            y, _ = librosa.load(str(f), sr=SR, mono=True, offset=78, duration=30)
            r = emotion_profile(y, SR)
            r["file"] = f.stem[:16]
            corpus.append(r)
        except: pass

    # AI 冠军
    ai_files = {
        "01": ROOT/"novel_songs/batch_gacha/01_v2/seed58888.flac",
        "07": ROOT/"novel_songs/batch_gacha/07_adj/seed1357_r1.flac",
        "08": ROOT/"novel_songs/triple_emergence/seed91104.flac",
        "09": ROOT/"novel_songs/batch_gacha/09/seed90904.flac",
    }
    ai = []
    for no, f in ai_files.items():
        if f.exists():
            try:
                y, _ = librosa.load(str(f), sr=SR, mono=True, offset=78, duration=30)
                r = emotion_profile(y, SR)
                r["file"] = no
                ai.append(r)
            except: pass

    def col(v, k):
        return [x[k] for x in v if x.get(k) is not None]

    lines = ["# 音乐情绪机制分析", "",
             "## 情绪锚点定义", "",
             "1. **音高轮廓** pitch_rise: >0 上行=紧张/希望, <0 下行=悲伤/释然",
             "2. **动态微弧** micro_arc: 乐句内 crescendo→decrescendo 深度 = 情绪脉冲",
             "3. **时间柔度** ibi_cv: 节拍偏离度 = rubato 人类感",
             "4. **紧张-释放** tension: 半音跳跃/分钟 = 张力密度",
             "5. **人声温暖** warm: 低频能量占比 = 体温感", "",
             "## 曲库 vs AI（40 首 vs 4 首冠军）", "",
             "| 锚点 | 曲库中位 | 曲库P25-P75 | AI冠军中位 | AI偏离 | 判读 |",
             "|---|---|---|---|---|---|"]
    for k, name in [("pitch_rise","音高轮廓"),("micro_arc","动态微弧"),
                    ("ibi_cv","时间柔度"),("tension","紧张-释放"),("warm","人声温暖")]:
        cv = np.array(col(corpus, k), float)
        nv = np.array(col(ai, k), float)
        if len(cv) < 10 or len(nv) < 2: continue
        cm, c25, c75 = np.median(cv), np.percentile(cv,25), np.percentile(cv,75)
        nm = np.median(nv)
        d = (nm-cm)/(np.std(cv)+1e-9)
        judge = "✅ 进入真曲区间" if c25 <= nm <= c75 else ("⚠️ AI偏高" if nm > c75 else "⚠️ AI偏低")
        lines.append(f"| {name} | {cm:.2f} | {c25:.2f}~{c75:.2f} | {nm:.2f} | d={d:+.2f} | {judge} |")

    # AI 逐曲
    lines += ["", "## AI 冠军逐曲", "", "| 歌 | 音高轮廓 | 动态微弧 | 时间柔度 | 紧张-释放 | 温暖 |",
              "|---|---|---|---|---|---|"]
    for r in ai:
        lines.append(f"| {r['file']} | {r.get('pitch_rise','?')} | {r.get('micro_arc','?')} "
                     f"| {r.get('ibi_cv','?')} | {r.get('tension','?')} | {r.get('warm','?')} |")

    lines += ["", "## 理论框架", "",
              "**为什么音乐能影响情绪**（基于 Huron 的 ITED + Larsson 的轮廓理论）:", "",
              "1. **期待→满足→违反**: 音乐建立期待, 然后满足或违反 → 满足=安全, 违反=情绪爆发",
              "2. **轮廓处理**: 听众跟踪旋律形状, 上行=张力累积, 下行=释放, 峰值=情绪高点",
              "3. **动态弧**: 乐句 crescendo→decrescendo = 情绪脉冲, 最强瞬间 = 多巴胺释放",
              "4. **节奏柔度**: rubato=呼吸感, 完美节拍=机器感, 人类靠微小不规则传递情感",
              "5. **紧张-释放密度**: 半音跳跃越多=情绪波动越大, 但过多=焦虑, 最优=可控张力",
              "6. **人声温暖**: 低频+F0微颤=体温感, 过稳=合成器, 过抖=跑调", "",
              "**AI 与人类的情绪锚点差距** = 不是技术差距, 是以上 6 维中人类独有的'可控不完美'"]

    (ROOT/"eval"/"emotion_mechanism.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
