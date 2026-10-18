# -*- coding: utf-8 -*-
"""YuE2 vs ACE-Step 对比评测（同一首歌，两个模型）"""
import json, math, sys, warnings
from pathlib import Path
import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050

MEGA_A = {"周杰伦","林俊杰","五月天","张学友","邓紫棋","陈奕迅","王菲","张惠妹",
          "孙燕姿","梁静茹","S.H.E","王心凌","张韶涵","胡歌"}
KNOWN_A = {"许嵩","汪苏泷","后弦","黄雅莉","张芸京","山野","花儿乐队","信乐团",
           "183Club","5566","By2","吴克群","薛之谦","毛不易","陈琳","香香","唐磊","汤潮"}
MEGA_S = {"青花瓷","东风破","菊花台","发如雪","江南","曹操","隐形的翅膀","偏爱",
          "死了都要爱","突然的自我","温柔","知足","Super Star","不想长大","遇见","天黑黑","绿光"}

def tier(stem):
    clean = stem.replace(" ", "_")
    for a in MEGA_A:
        if a in clean: return 3
    for h in MEGA_S:
        if h in clean or h.replace(" ","_") in clean: return 3
    for a in KNOWN_A:
        if a in clean: return 2
    return 1


def full_metrics(flac_path):
    """计算全套音频指标"""
    y, sr = librosa.load(str(flac_path), sr=SR, mono=True, duration=100)
    dur = len(y) / SR
    
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    S = np.abs(librosa.stft(y))**2
    freqs = librosa.fft_frequencies(sr=sr)
    
    # 节拍
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time", hop_length=512)
    bpm = float(np.atleast_1d(tempo)[0])
    
    # 呼吸间隙
    w3 = int(3 * sr / 512)
    smooth = np.convolve(db, np.ones(12)/12, mode="same")
    dips = 0
    for i in range(w3, len(smooth)-w3):
        window = smooth[i-w3:i+w3]
        if smooth[i] == window.min() and (window.max()-window.min()) > 3:
            if i == w3 or smooth[i] <= smooth[i-w3//2]: dips += 1
    
    # 动态范围
    dyn = float(np.percentile(db, 95) - np.percentile(db, 10))
    
    # 质心
    centroid = float(cent.mean())
    
    # RMS
    rms_db = float(librosa.amplitude_to_db(np.sqrt((y**2).mean()) + 1e-9))
    
    # 峰值因子
    crest = float(20*np.log10(np.abs(y).max()/(np.sqrt((y**2).mean())+1e-9)))
    
    # 过零率
    zcr = float(np.mean(np.abs(np.diff(y) > 0)))
    
    # 时长
    return dict(bpm=round(bpm,1), duration=round(dur,1), centroid=round(centroid),
                rms_db=round(rms_db,1), dyn_range=round(dyn,1),
                breath_dips=dips, crest_db=round(crest,1), zcr=round(zcr,4))


def main():
    ace_dir = ROOT / "ace_step_output"
    yue_dir = ROOT / "novel_songs" / "final"
    
    # 曲库 MERT 嵌入（自然度代理）
    mert_npz = ROOT / "eval" / "mert_all_embeddings.npz"
    mert_names = []
    mert_embs = []
    if mert_npz.exists():
        z = np.load(str(mert_npz), allow_pickle=True)
        mert_names = [str(n) for n in z["names"]]
        mert_embs = z["embs"]
    corpus_embs = mert_embs
    
    pairs = []
    for ace_f in sorted(ace_dir.glob("*.flac")):
        no = ace_f.stem.split("_")[1]
        # 找对应的 YuE2 版本
        yue_f = None
        for f in yue_dir.glob(f"{no}_*.flac"):
            if "_hum" not in f.name:
                yue_f = f
                break
        if yue_f:
            pairs.append((no, ace_f, yue_f))
    
    lines = ["# YuE2 vs ACE-Step 对比评测", "",
             f"对比歌曲数: {len(pairs)}", "",
             "| # | 曲目 | 模型 | 时长 | 质心Hz | 动态dB | 呼吸间隙 | 峰值因子 | RMS |", 
             "|---|---|---|---|---|---|---|---|---|"]
    
    for no, ace_f, yue_f in sorted(pairs):
        title = ace_f.stem.replace("ace_", "")
        ace_m = full_metrics(ace_f)
        yue_m = full_metrics(yue_f)
        lines.append(f"| {no} {title[:6]} | ACE {ace_m['duration']}s | 质心{ace_m['centroid']} | "
                    f"动态{ace_m['dyn_range']} | 呼吸{ace_m['breath_dips']} | 峰值{ace_m['crest_db']} | RMS{ace_m['rms_db']} |")
        lines.append(f"| {'':24s} | YuE2 {yue_m['duration']}s | 质心{yue_m['centroid']} | "
                    f"动态{yue_m['dyn_range']} | 呼吸{yue_m['breath_dips']} | 峰值{yue_m['crest_db']} | RMS{yue_m['rms_db']} |")
    
    # 汇总统计
    ace_feats = [full_metrics(f) for f in [p[1] for p in pairs]]
    yue_feats = [full_metrics(f) for f in [p[2] for p in pairs]]
    
    lines += ["", "## 汇总对比", ""]
    for key in ("centroid", "dyn_range", "breath_dips", "crest_db", "rms_db", "duration"):
        a = [f[key] for f in ace_feats]
        b = [f[key] for f in yue_feats]
        lines.append(f"- **{key}**: ACE 均值 {np.mean(a):.1f} | YuE2 均值 {np.mean(b):.1f}")
    
    # 曲库真曲基准
    lib_wav = ROOT / "eval" / "library_wavs"
    lib_tr = ROOT / "eval" / "library_transcripts"
    corpus_feats = []
    for w in sorted(lib_wav.glob("*.wav"))[:20]:
        tdir = lib_tr / w.stem
        if (tdir / "score.abc").exists():
            corpus_feats.append(full_metrics(w))
    if corpus_feats:
        lines += ["", "### 曲库基准（前 20 首）", ""]
        for key in ("centroid", "dyn_range", "rms_db"):
            vals = [f[key] for f in corpus_feats]
            lines.append(f"- {key}: 真曲中位 {np.median(vals):.1f}")
    
    (ROOT / "eval" / "model_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
