# -*- coding: utf-8 -*-
"""演唱表现力指标：测量唱法中"人味"的声学证据
1. 乐句动态弧线（crescendo/decrescendo 深度）
2. 音高装饰密度（弯音/颤音/转音 小波动）
3. 气息自然度（呼吸间隔分布的变异系数）
4. 音高精准度偏离（AI 过准 vs 真人的自然偏差）
对曲库真曲 vs AI 生成曲 → 效应量 → 排名
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import librosa
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
sys.path.insert(0, str(ROOT))
from ai_segment_forensic import parse_lab, first_span


def vocal_expression_metrics(y, sr):
    """演唱表现力指标（副歌段 30s）"""
    hop = 256
    try:
        f0, vflag, vprob = librosa.pyin(y, fmin=65, fmax=700, sr=sr,
                                        frame_length=1024, hop_length=hop)
    except Exception:
        return {}
    ft = np.arange(len(f0)) * hop / sr
    voiced = vflag & ~np.isnan(f0)

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)

    phrase_dynamics = []
    in_voice, start = False, 0
    for i, v in enumerate(voiced):
        if v and not in_voice: in_voice, start = True, i
        elif not v and in_voice:
            if i - start > 5 and start < len(db) and i <= len(db):
                seg_db = db[start:min(i, len(db))]
                if len(seg_db) > 0:
                    phrase_dynamics.append(float(seg_db.max() - seg_db.min()))
            in_voice = False
    if in_voice and start < len(db):
        seg_db = db[start:]
        if len(seg_db) > 0:
            phrase_dynamics.append(float(seg_db.max() - seg_db.min()))
    phrase_dyn = round(float(np.median(phrase_dynamics)), 1) if phrase_dynamics else None

    smooth_f0 = np.convolve(np.nan_to_num(f0, nan=0), np.ones(20)/20, mode="same")
    micro = np.abs(np.diff(f0))
    voiced_mask = voiced[1:] & voiced[:-1]
    micro_voiced = micro[voiced_mask]
    ornaments = int(np.sum(micro_voiced > 15))
    orn_density = round(float(ornaments / (len(y)/sr)), 2) if len(y) > 0 else None

    voiced_idx = np.where(voiced)[0]
    if len(voiced_idx) > 3:
        gaps = np.diff(voiced_idx)
        gap_lens = gaps[gaps > 1] * hop / sr * 1000
        gap_cv = round(float(np.std(gap_lens) / (np.mean(gap_lens)+1e-9)), 2) if len(gap_lens) > 3 else None
    else:
        gap_cv = None

    devs = []
    for i in np.where(voiced)[0]:
        if i < len(f0) and not np.isnan(f0[i]):
            midi = 69 + 12 * np.log2(f0[i] / 440)
            nearest = round(midi)
            devs.append(abs(midi - nearest) * 100)
    pitch_dev = round(float(np.mean(devs)), 1) if devs else None

    if len(f0) > 1:
        f0_diff = np.abs(np.diff(f0))
        vd = voiced[1:] & voiced[:-1]
        vd = vd[:len(f0_diff)]
        voiced_diff = f0_diff[vd]
        slide_frac = round(float(np.mean(voiced_diff > 30)), 3) if len(voiced_diff) > 0 else None
    else:
        slide_frac = None

    return dict(
        phrase_dyn=phrase_dyn,
        orn_density=orn_density,
        gap_cv=gap_cv,
        pitch_dev=pitch_dev,
        slide_frac=slide_frac,
    )


def main():
    # ===== 曲库 =====
    lib_wav = ROOT / "eval" / "library_wavs"
    lib_tr = ROOT / "eval" / "library_transcripts"
    corpus = []
    for w in sorted(lib_wav.glob("*.wav"))[:30]:
        rows = parse_lab(lib_tr / w.stem, "structure.lab") if 'parse_lab' in dir() else []
        # 简化：直接取 78-108s（典型副歌区域）
        try:
            y, _ = librosa.load(str(w), sr=SR, mono=True, offset=78, duration=30)
            r = vocal_expression_metrics(y, SR)
            r["tier"] = tier_of(w.stem)
            corpus.append(r)
        except: pass

    # ===== 生成曲 =====
    novel = []
    final = ROOT / "novel_songs" / "final"
    for f in sorted(final.glob("*.flac")):
        if "_hum" in f.name: continue
        try:
            y, _ = librosa.load(str(f), sr=SR, mono=True, offset=60, duration=30)
            r = vocal_expression_metrics(y, SR)
            r["file"] = f.stem
            novel.append(r)
        except: pass

    # ===== 效应量 =====
    def d_of(key):
        cv = np.array([r[key] for r in corpus if r.get(key) is not None], float)
        nv = np.array([r[key] for r in novel if r.get(key) is not None], float)
        if len(cv) < 10 or len(nv) < 3: return None, None, None
        d = (nv.mean() - cv.mean()) / (np.sqrt(((len(nv)-1)*nv.var()+(len(cv)-1)*cv.var())/(len(nv)+len(cv)-2))+1e-9)
        return d, float(np.median(cv)), float(np.median(nv))

    def grade(d):
        d = abs(d)
        return "A 可靠" if d >= 0.8 else ("B 方向性" if d >= 0.5 else "C 仅趋势")

    # ===== 输出 =====
    lines = ["# 演唱表现力指标：真曲 vs AI 生成", "",
             f"真曲 n={len(corpus)} · 生成 n={len(novel)}", "",
             "| 指标 | 真曲中位 | 生成中位 | Cohen's d | 级别 | 含义 |",
             "|---|---|---|---|---|---|"]

    metrics_list = [
        ("phrase_dyn", "乐句动态弧线(dB)", "真人唱法有呼吸起伏"),
        ("orn_density", "音高装饰密度", "弯音/转音多=表现力强"),
        ("gap_cv", "气息自然度(CV)", "变异大=呼吸不机械"),
        ("pitch_dev", "音高偏离(音分)", "AI过准=合成指纹"),
        ("slide_frac", "滑音比例", "滑音多=表现力强"),
    ]

    for key, cname, meaning in metrics_list:
        cv = [r[key] for r in corpus if r.get(key) is not None]
        nv = [r[key] for r in novel if r.get(key) is not None]
        if not cv or not nv: continue
        d = cohen_d(nv, cv)
        g = grade(d)
        lines.append(f"| {cname} | {np.median(cv):.2f} | {np.median(nv):.2f} | {d:+.2f} | {g} | {meaning} |")

    lines += ["", "## YuE2 唱法差距总结", ""]
    for key, cname, meaning in metrics_list:
        cm = np.median([r[key] for r in corpus if r.get(key) is not None]) if cv else 0
        nm = np.median([r[key] for r in novel if r.get(key) is not None]) if nv else 0
        if cm and nm:
            direction = "AI偏高" if nm > cm else "AI偏低"
            diff_pct = round(abs(nm-cm)/max(abs(cm),0.01)*100)
            lines.append(f"- {cname}: 真曲{cm:.1f} vs AI{nm:.1f} ({direction} {diff_pct}%)")

    (ROOT/"eval"/"vocal_expression_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
