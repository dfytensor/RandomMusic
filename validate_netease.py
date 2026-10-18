# -*- coding: utf-8 -*-
"""用 NetEase 真实热度数据验证所有指标（42 首有真实标签的）
关键问题：这些指标和真实热度有关系吗？
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ===== 加载 =====
ne_data = json.loads((ROOT/"eval"/"netease_popularity_v2.json").read_text(encoding="utf-8"))
fp_cache = json.loads((ROOT/"eval"/"lib_metric_cache.json").read_text(encoding="utf-8"))
wv_cache = json.loads((ROOT/"eval"/"wave_metric_cache.json").read_text(encoding="utf-8"))
res_cache = json.loads((ROOT/"eval"/"resonance_cache.json").read_text(encoding="utf-8"))
ms_map = {r["file"]: r for r in json.loads((ROOT/"corpus"/"melody_stats.json").read_text(encoding="utf-8"))["per_song"] if r.get("file")}

# NetEase 42 首
ne_songs = {}
for orig_name, v in ne_data.items():
    if "netease_id" not in v: continue
    clean = orig_name.replace(".mp3","").replace(" ","_")[:60]
    wav_stem = "lib_" + clean
    ne_songs[wav_stem] = dict(
        netease_name=v.get("netease_name",""),
        popularity=v.get("popularity",0),
        comment_count=v.get("commentCount",0),
    )

# ===== 逐曲计算全部指标 =====
results = []
for wav_stem, ne in ne_songs.items():
    tdir = ROOT/"eval"/"library_transcripts"/wav_stem
    abc = tdir/"score.abc"
    wav = ROOT/"eval"/"library_wavs"/(wav_stem+".wav")
    if not abc.exists() or not wav.exists(): continue
    
    try:
        sys.path.insert(0, str(ROOT))
        from melody_stats import analyze_song
        m = analyze_song(str(abc))
    except: m = {}
    
    try:
        import librosa
        y, _ = librosa.load(str(wav), sr=22050, mono=True, duration=100)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        db = librosa.amplitude_to_db(rms+1e-9)
        cent = librosa.feature.spectral_centroid(y=y, sr=22050)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        S = np.abs(librosa.stft(y))**2
        freqs = librosa.fft_frequencies(sr=22050)
        
        row = dict(
            bpm=m.get("density", 0),  # placeholder
            duration_s=m.get("total_bars", 0),
            centroid_hz=round(float(cent.mean())),
            rms_db=round(float(librosa.amplitude_to_db(rms.mean()+1e-9)),1),
            dyn_p95_p10_db=round(float(librosa.amplitude_to_db(np.percentile(rms,95)+1e-9) - librosa.amplitude_to_db(np.percentile(rms,10)+1e-9)),1),
            zcr=round(float(zcr.mean()),4),
            # 指纹
            voiced_ratio=round(float((vflag & ~np.isnan(f0)).mean()), 3) if False else None,
        )
        # 从 fp_cache 拿指纹
        fp = fp_cache.get(wav_stem, {})
        if not fp.get("skip"):
            for k in ("voiced_ratio","n_vocal_runs","vibrato_rate_hz","vibrato_extent_cents",
                      "intra_note_drift_cents","glide_ratio","macro_dyn_std_db","dyn_range_db",
                      "spectral_flux_std","loop_lag_s","loop_peak_corr","loopiness"):
                v = fp.get(k)
                if v is not None:
                    row[k] = float(v)
        # 波形
        wv = wv_cache.get(wav_stem, {})
        for k in ("crest_db","env_rough","low_attack_ms","onset_iv_cv","kurtosis","lra_db"):
            v = wv.get(k)
            if v is not None:
                row[k] = float(v)
        # 共鸣
        rs = res_cache.get(wav_stem, {})
        for k in ("vibrato_cv","long_drift","breath_dips","leap_prep"):
            v = rs.get(k)
            if v is not None:
                row[k] = float(v)
        # 旋律
        from melody_stats import analyze_song as _as
        m2 = _as(str(abc))
        for k in ("density","pitch_range","leap_ratio","motif_repeat","lift_mean","rest_ratio"):
            v = m2.get(k)
            if v is not None:
                row["mel_"+k] = float(v)
        
        row["_ne_pop"] = ne["popularity"]
        row["_ne_comments"] = ne.get("commentCount", 0)
        row["_name"] = ne["netease_name"]
        results.append(row)
    except Exception as e:
        print(f"skip {wav_stem[:20]}: {str(e)[:50]}")

print(f"有效: {len(results)} 首", flush=True)

# ===== Spearman 相关：每个指标 vs NetEase popularity =====
def safe_spearman(xs, ys):
    pairs = [(x, yy) for x, yy in zip(xs, ys) if x is not None and yy is not None and x == x and yy == yy]
    if len(pairs) < 10: return None
    from scipy.stats import spearmanr
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    rho, pval = spearmanr(a, b)
    return round(float(rho), 3), round(float(pval), 4)

ne_pops = [r["_ne_pop"] for r in results]
ne_comments = [r["_ne_comments"] for r in results]

all_metric_keys = set()
for r in results:
    for k, v in r.items():
        if k.startswith("_") or k in ("netease_name",): continue
        if isinstance(v, (int, float)):
            all_metric_keys.add(k)

lines = ["# NetEase 真实热度 × 指标 相关性（42 首有真实标签的歌）", "",
         f"样本: {len(results)} 首 | 标签: NetEase popularity (0-100)", "",
         "| 指标 | ρ(热度) | p值 | 判定 |", "|---|---|---|---|"]

sig_results = []
for key in sorted(all_metric_keys):
    vals = [r.get(key) for r in results]
    pops = [r["_ne_pop"] for r in results]
    cmts = [r.get("_ne_comments", 0) for r in results]
    
    if key == "_ne_pop" or key == "_ne_comments": continue
    
    # vs popularity
    pairs = [(v, p) for v, p in zip(vals, pops) if v is not None]
    if len(pairs) < 10: continue
    rho, pval = spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
    
    # vs comments
    pairs_c = [(v, c) for v, c in zip(vals, cmts) if v is not None and c is not None]
    rho_c = None
    if len(pairs_c) >= 10:
        rc, pc = spearmanr([p[0] for p in pairs_c], [p[1] for p in pairs_c])
        rho_c = round(float(rc), 3)
    
    sig = "✅" if abs(rho) >= 0.3 else ("🟡" if abs(rho) >= 0.15 else "❌")
    lines.append(f"| {key} | ρ_pop={rho:+.3f} (p={pval:.3f}) | ρ_cmt={rho_c if rho_c else 'N/A'} | {sig} |")
    sig_results.append(dict(metric=key, rho_pop=rho, pval=pval, rho_comments=rho_c))

(ROOT / "eval" / "netease_validation.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines), flush=True)

# 保存完整结果
(ROOT / "eval" / "netease_validation.json").write_text(
    json.dumps(dict(n=len(results), results=sig_results), ensure_ascii=False, indent=1),
    encoding="utf-8")
print("SAVED", flush=True)
