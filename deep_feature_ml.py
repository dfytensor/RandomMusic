# -*- coding: utf-8 -*-
"""深度特征扩展：MFCC / 色度 / 谱对比度 / 节拍特征 / 律动 / 能量曲线
目标：把 34 维提升到 80+ 维，提升 8 分类准确率
"""
import json, math, re, sys, warnings
from pathlib import Path
import numpy as np
import librosa
from scipy.stats import kurtosis, skew
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
sys.path.insert(0, str(ROOT))
from ai_segment_forensic import parse_lab, first_span
from novel_songs_manifest import SONGS

MEGA_A = {"周杰伦","林俊杰","五月天","张学友","邓紫棋","陈奕迅","王菲","张惠妹",
          "孙燕姿","梁静茹","S.H.E","王心凌","张韶涵","胡歌"}
KNOWN_A = {"许嵩","汪苏泷","后弦","黄雅莉","张芸京","山野","花儿乐队","信乐团",
           "183Club","5566","By2","吴克群","薛之谦","毛不易","陈琳","香香","唐磊","汤潮"}
MEGA_S = {"青花瓷","东风破","菊花台","发如雪","江南","曹操","隐形的翅膀","偏爱",
          "死了都要爱","突然的自我","温柔","知足","Super Star","不想长大","遇见",
          "天黑黑","绿光","月亮代表我的心","甜蜜蜜","后来","为爱痴狂","朋友",
          "第一次","勇气","老鼠爱大米","两只蝴蝶","秋天不回来","求佛","香水有毒",
          "该死的温柔","专属天使","北极星的眼泪","我难过","倒带","说爱你","那些花儿",
          "生如夏花","平凡之路","董小姐","成都","南山南","小苹果","最炫民族风"}

def tier(name):
    clean = name.replace(" ", "_")
    for a in MEGA_A:
        if a in clean: return 3
    for h in MEGA_S:
        if h in clean: return 3
    for a in KNOWN_A:
        if a in clean: return 2
    return 1

def lib_stem(name):
    name_clean = name.replace(".mp3", "").replace(".flac", "")
    return "lib_" + name_clean.replace(" ", "_")[:60]


def deep_audio_features(y, sr):
    """深度音频特征提取：MFCC/色度/谱对比/节拍/能量曲线"""
    feats = {}

    # ===== MFCC (13×4=52 维统计) =====
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        feats[f"mfcc{i}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i}_std"] = float(np.std(mfcc[i]))
        feats[f"mfcc{i}_skew"] = round(float(skew(mfcc[i])), 3)
        feats[f"mfcc{i}_kurt"] = round(float(kurtosis(mfcc[i])), 3)

    # ===== 色度 (12×3=36 维统计) =====
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        feats[f"chroma{i}_mean"] = round(float(np.mean(chroma[i])), 4)
        feats[f"chroma{i}_std"] = round(float(np.std(chroma[i])), 4)
        feats[f"chroma{i}_max"] = round(float(np.max(chroma[i])), 4)

    # ===== 谱对比度 (7×2=14 维) =====
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(contrast.shape[0]):
            feats[f"contrast{i}_mean"] = round(float(np.mean(contrast[i])), 2)
            feats[f"contrast{i}_std"] = round(float(np.std(contrast[i])), 2)
    except:
        pass

    # ===== 节拍/律动 =====
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512, units="time")
    beat_t = np.atleast_1d(beats)
    feats["beat_consistency"] = round(float(np.std(np.diff(beat_t)) / (np.mean(np.diff(beat_t)) + 1e-9)), 4) if len(beat_t) > 4 else 0
    feats["beat_density"] = round(len(beat_t) / dur_a, 3)

    # 节拍相位直方图熵
    if len(beat_t) > 4:
        period = np.mean(np.diff(beat_t))
        onset_t = librosa.onset.onset_detect(y=y, sr=sr, hop_length=256, units="time", delta=0.04)
        if len(onset_t) > 10:
            phases = np.array([(o - beat_t[0]) % period / period for o in onset_t])
            hist, _ = np.histogram(phases, bins=16, range=(0, 1))
            p = hist / (hist.sum() + 1e-9)
            p = p[p > 0]
            feats["beat_phase_entropy"] = round(float(-sum(x * math.log2(x) for x in p) / math.log2(16)), 3)
        else:
            feats["beat_phase_entropy"] = 1.0
    else:
        feats["beat_phase_entropy"] = 1.0

    # ===== 能量曲线形状 =====
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    # 1. 能量变化率
    feats["energy_rate_std"] = round(float(np.std(np.diff(db))), 2)
    # 2. 能量自相关（前10 lag）
    dn = db - db.mean()
    ac = np.correlate(dn, dn, mode="full")[len(dn)-1:]
    ac = ac / (ac[0] + 1e-9)
    for lag in (10, 20, 43):  # ~0.23s, ~0.46s, ~1s
        if lag < len(ac):
            feats[f"energy_ac_{lag}"] = round(float(ac[lag]), 3)
    # 3. 能量峰值统计
    peaks = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, delta=0.08, units="time")
    feats["onset_count"] = len(peaks)
    # 4. 能量曲线斜率（线性回归）
    x = np.arange(len(db), dtype=float)
    slope = np.polyfit(x, db, 1)[0]
    feats["energy_slope"] = round(float(slope), 4)

    # ===== 频谱时间演化 =====
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats["centroid_std"] = round(float(cent.std()), 1)
    feats["centroid_cv"] = round(float(cent.std() / (cent.mean() + 1e-9)), 3)

    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    feats["bandwidth_mean"] = round(float(bw.mean()), 1)
    feats["bandwidth_std"] = round(float(bw.std()), 1)

    flat = librosa.feature.spectral_flatness(y=y)[0]
    feats["flatness_mean"] = round(float(flat.mean()), 5)
    feats["flatness_std"] = round(float(flat.std()), 5)

    # ===== 低频/中频/高频分带 RMS 时间演化 =====
    S = np.abs(librosa.stft(y, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr)
    for band_name, lo, hi in [("low", 20, 300), ("mid", 300, 3000), ("high", 3000, 11000)]:
        mask = (freqs >= lo) & (freqs < hi)
        band_rms = np.sqrt(S[mask].mean(axis=0) + 1e-9)
        band_db = librosa.amplitude_to_db(band_rms + 1e-9)
        feats[f"band_{band_name}_mean_db"] = round(float(band_db.mean()), 1)
        feats[f"band_{band_name}_std"] = round(float(band_db.std()), 1)
        # 带内波动
        feats[f"band_{band_name}_cv"] = round(float(band_rms.std() / (band_rms.mean() + 1e-9)), 3)

    return feats


def sf_info_dur(p):
    import soundfile
    i = soundfile.info(str(p))
    return i.frames / i.samplerate


def main():
    from wave_metric_collect import wave_metrics
    lib_wav = ROOT / "eval" / "library_wavs"
    lib_tr = ROOT / "eval" / "library_transcripts"
    trs = sorted(lib_tr.glob("*/score.abc"))
    feats = json.loads((ROOT / "corpus" / "features.json").read_text(encoding="utf-8"))
    fp_cache = json.loads((ROOT / "eval" / "lib_metric_cache.json").read_text(encoding="utf-8"))
    wv_cache = json.loads((ROOT / "eval" / "wave_metric_cache.json").read_text(encoding="utf-8"))
    ms_list = json.loads((ROOT / "corpus" / "melody_stats.json").read_text(encoding="utf-8"))["per_song"]
    ms_map = {}
    for r in ms_list:
        if r.get("file"):
            ms_map[r["file"]] = r

    cache_p = ROOT / "eval" / "deep_feature_cache.json"
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}

    import time
    t0 = time.time()
    corpus = []
    for i, w in enumerate(sorted(lib_wav.glob("*.wav"))):
        stem = w.stem
        if stem in cache:
            corpus.append(cache[stem])
            continue
        tdir = lib_tr / stem
        if not (tdir / "score.abc").exists():
            cache[stem] = {"skip": True}
            continue
        try:
            y, _ = librosa.load(str(w), sr=SR, mono=True, duration=100)
            dur_a = len(y) / SR
            # 基础
            base = wave_metrics(y)
            base.update(deep_audio_features(y, SR))
            base["file"] = stem
            # 旋律
            sys.path.insert(0, str(ROOT))
            from melody_stats import analyze_song
            try:
                m = analyze_song(str(tdir / "score.abc"))
                for k in ("density","pitch_range","leap_ratio","motif_repeat","lift_mean","rest_ratio"):
                    base["mel_" + k] = m[k]
            except: pass
            base["tier"] = tier(stem.replace("lib_", "", 1))
            cache[stem] = base
            corpus.append(base)
        except Exception as e:
            cache[stem] = {"skip": True}
        if (i + 1) % 50 == 0:
            cache_p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            print(f"[{i+1}/{len(wavs)}] el={time.time()-t0:.0f}s", flush=True)
    cache_p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"CORPUS DONE: {len([c for c in corpus if not c.get('skip')])}", flush=True)

    # ===== ML =====
    valid = [c for c in corpus if not c.get("skip")]
    all_keys = set()
    for r in valid:
        all_keys.update(k for k, v in r.items() if isinstance(v, (int, float)))
    all_keys.discard("skip")
    all_keys.discard("tier")
    fn_all = sorted(all_keys)

    X, y = [], []
    for r in valid:
        vec = [float(r.get(k, 0)) for k in fn_all]
        X.append(vec)
        y.append(r["tier"])
    X = np.array(X); y = np.array(y)

    from collections import Counter
    tc = Counter(y)
    print(f"ML: {X.shape[0]} 样本 × {X.shape[1]} 维 | 分布: {dict(sorted(tc.items()))}", flush=True)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    pred = cross_val_predict(rf, X, y, cv=5)
    acc = float(np.mean(pred == y))
    baseline = max(tc.values()) / len(y)
    print(f"准确率: {acc*100:.1f}% (最大类基线 {baseline*100:.1f}%) 提升: +{(acc-baseline)*100:.1f}%", flush=True)

    rf.fit(X, y)
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1]
    print("\nTop 20 特征:")
    for i in range(min(20, len(idx))):
        fn = fn_all[idx[i]] if idx[i] < len(fn_all) else "?"
        print(f"  {i+1}. {fn}: {imp[idx[i]]:.4f}")

    (ROOT / "eval" / "deep_feature_ml.md").write_text(
        f"# 深度特征 ML\n\n维度: {X.shape[1]}\n准确率: {acc*100:.1f}% (基线 {baseline*100:.0f}%)\n\n"
        + "\n".join(f"| {i+1} | {fn_all[idx[i]] if idx[i]<len(fn_all) else '?'} | {imp[idx[i]]:.4f} |"
                    for i in range(min(25, len(idx)))), encoding="utf-8")


if __name__ == "__main__":
    main()
