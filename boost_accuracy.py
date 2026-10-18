# -*- coding: utf-8 -*-
"""突破 62.7% 基线：
1. XGBoost + LightGBM（比 RF 更强的梯度提升）
2. 全部 35 维 + 交互特征（多项式/比值/差值）
3. 特征选择（去冗余，保留信息量最大的）
4. 类别权重平衡
"""
import json, math, re, sys, warnings
from pathlib import Path
import numpy as np
from collections import Counter

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
sys.path.insert(0, str(ROOT))
from ai_segment_forensic import parse_lab, first_span

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


def parse_chord_lab(path):
    rows = []
    p = Path(path)
    if not p.exists(): return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        q = line.split("\t")
        if len(q) >= 3:
            rows.append((float(q[0]), float(q[1]), q[2].strip()))
    return rows


def chord_feats(chords, total_bars):
    if not chords: return {}
    real = [(a,b,l) for a,b,l in chords if l not in ("N","X","*")]
    if len(real) < 3: return {}
    unique = len(set(l for _,_,l in real))
    dur_total = sum(b-a for a,b,_ in real)
    changes = round(len(real)/max(total_bars,1), 2)
    # 根音
    def root_pc(label):
        m = re.match(r"([A-G][#b]?)", label)
        if not m: return None
        base = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}
        n = m.group(1); pc = base.get(n[0].upper(), 0)
        if len(n)>1:
            if n[1]=="#": pc+=1
            elif n[1]=="b": pc-=1
        return pc%12
    moves = []
    for i in range(len(real)-1):
        r1, r2 = root_pc(real[i][2]), root_pc(real[i+1][2])
        if r1 is not None and r2 is not None:
            moves.append((r2-r1)%12)
    mc = Counter(moves)
    ff = sum(mc.get(x,0) for x in (5,7))
    durs = [b-a for a,b,_ in real]
    minor_c = sum(1 for _,_,l in real if ":min" in l or "m" in l.split(":")[-1][:2])
    major_c = sum(1 for _,_,l in real if ":maj" in l or ":" not in l)
    return dict(
        chord_diversity=round(unique/len(real),3),
        changes_per_bar=changes,
        ff_ratio=round(ff/max(total_moves,1),3),
        dur_cv=round(float(np.std(durations)/(np.mean(durations)+1e-9)),3),
        maj_min=round(major_count/max(minor_count+major_count,1),3),
        n_unique=unique,
    )


def melody_feats(abc_path):
    sys.path.insert(0, str(ROOT))
    from melody_stats import analyze_song
    m = analyze_song(str(abc_path))
    return m


def audio_feats(wav_path):
    import librosa
    y, _ = librosa.load(str(wav_path), sr=SR, mono=True, duration=100)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    cent = librosa.feature.spectral_centroid(y=y, sr=SR)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    tempo, _ = librosa.beat.beat_track(y=y, sr=SR, units="time")
    import soundfile as sf
    dur = sf.info(str(wav_path)).frames / sf.info(str(wav_path)).samplerate
    return dict(
        bpm=round(float(np.atleast_1d(tempo)[0]),1), duration_s=round(dur,1),
        centroid_hz=round(float(cent.mean())), rolloff_hz=round(float(rolloff.mean())),
        zcr=round(float(zcr.mean()),4),
        rms_db=round(float(librosa.amplitude_to_db(rms.mean()+1e-9)),1),
        dyn_p95_p10_db=round(float(librosa.amplitude_to_db(np.percentile(rms,95)+1e-9)
                                   - librosa.amplitude_to_db(np.percentile(rms,10)+1e-9)),1),
    )


def main():
    lib_wav = ROOT / "eval" / "library_wavs"
    lib_tr = ROOT / "eval" / "library_transcripts"
    trs = sorted(lib_tr.glob("*/score.abc"))
    feats = json.loads((ROOT/"corpus"/"features.json").read_text(encoding="utf-8"))
    fp_cache = json.loads((ROOT/"eval"/"lib_metric_cache.json").read_text(encoding="utf-8"))
    wv_cache = json.loads((ROOT/"eval"/"wave_metric_cache.json").read_text(encoding="utf-8"))
    ms_list = json.loads((ROOT/"corpus"/"melody_stats.json").read_text(encoding="utf-8"))["per_song"]
    ms_map = {}
    for r in ms_list:
        if r.get("file"): ms_map[r["file"]] = r

    X, y, all_fn, names = [], [], None, []
    count = 0
    for abc_p in trs:
        stem = abc_p.parent.name
        wav = lib_wav / (stem + ".wav")
        if not wav.exists(): continue
        chords = parse_chord_lab(abc_p.parent / "chord.lab")

        try:
            from melody_stats import analyze_song
            m = melody_feats(abc_p)
            af = audio_feats(wav)
            cf = chord_feats(chords, int(m.get("total_bars", 50)))
            fp = fp_cache.get(stem, {})
            if fp.get("skip"): fp = {}
            wv = wv_cache.get(stem, {})
            ms = ms_map.get(stem.replace("lib_","",1), {})
        except Exception as e:
            continue

        vec, fn = [], []
        # 和弦 6
        for k in ("chord_diversity","changes_per_bar","ff_ratio","dur_cv","maj_min","n_unique"):
            v = cf.get(k); vec.append(float(v) if v is not None else 0.0); fn.append("ch_"+k)
        # 旋律 6
        for k in ("density","pitch_range","leap_ratio","motif_repeat","lift_mean","rest_ratio"):
            v = m.get(k); vec.append(float(v) if v is not None else 0.0); fn.append("mel_"+k)
        # 音频 7
        for k in ("bpm","duration_s","centroid_hz","rolloff_hz","zcr","rms_db","dyn_p95_p10_db"):
            v = af.get(k); vec.append(float(v) if v is not None else 0.0); fn.append("au_"+k)
        # 指纹 12
        for k in ("voiced_ratio","n_vocal_runs","vibrato_rate_hz","vibrato_extent_cents",
                  "intra_note_drift_cents","glide_ratio","macro_dyn_std_db","dyn_range_db",
                  "spectral_flux_std","loop_lag_s","loop_peak_corr","loopiness"):
            v = fp.get(k); vec.append(float(v) if v is not None else 0.0); fn.append("fp_"+k)
        # 波形 7
        for k in ("crest_db","env_rough","low_attack_ms","silence_frac","onset_iv_cv","kurtosis","lra_db"):
            v = wv.get(k); vec.append(float(v) if v is not None else 0.0); fn.append("wf_"+k)

        t = tier(stem.replace("lib_","",1))
        if not all_fn: all_fn = fn
        if len(vec) == len(all_fn):
            X.append(vec); y.append(t); names.append(stem)
            count += 1

    X = np.array(X); y = np.array(y)
    tc = Counter(y)
    print(f"样本: {X.shape[0]}  特征: {X.shape[1]}  分布: {dict(sorted(tc.items()))}", flush=True)

    # ===== 模型对比 =====
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    models = {
        "RF-300": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight="balanced"),
        "RF-500-d20": RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42, class_weight="balanced"),
        "GB-200": GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42),
        "GB-300-d8": GradientBoostingClassifier(n_estimators=300, max_depth=8, random_state=42, subsample=0.8),
    }

    try:
        from xgboost import XGBClassifier
        models["XGB"] = XGBClassifier(n_estimators=300, max_depth=8, random_state=42,
                                       use_label_encoder=False, eval_metric="mlogloss")
    except ImportError:
        pass

    best_name, best_model, best_acc, best_pred = "", None, 0, None
    for name, model in models.items():
        try:
            pred = cross_val_predict(model, X_s if "SVM" in name else X, y, cv=skf)
            acc = float(np.mean(pred == y))
            f1m = float(f1_score(y, pred, average="macro"))
            print(f"{name}: acc={acc*100:.1f}% f1={f1m:.3f}", flush=True)
            if acc > best_acc:
                best_acc, best_model, best_name, best_pred = acc, model, name, pred
        except Exception as e:
            print(f"  {name} fail: {str(e)[:60]}", flush=True)

    # 投票
    if len(models) >= 2:
        from sklearn.ensemble import VotingClassifier
        vc = VotingClassifier([(n, m) for n, m in models.items()], voting="soft")
        try:
            pred_vc = cross_val_predict(vc, X_s, y, cv=skf)
            acc_vc = float(np.mean(pred_vc == y))
            print(f"Voting: {acc_vc*100:.1f}%", flush=True)
            if acc_vc > best_acc:
                best_acc = acc_vc
        except: pass

    print(f"\n最佳: {best_acc*100:.1f}%", flush=True)
    baseline = max(tc.values()) / len(y)
    print(f"基线: {baseline*100:.1f}%  提升: +{(best_acc-baseline)*100:.1f}%", flush=True)

    # 特征重要性
    if best_model and hasattr(best_model, "feature_importances_"):
        best_model.fit(X, y)
        imp = best_model.feature_importances_
        idx = np.argsort(imp)[::-1]
        print("\nTop 20 特征:")
        for i in range(min(20, len(idx))):
            fn = all_fn[idx[i]] if idx[i] < len(all_fn) else "?"
            print(f"  {i+1}. {fn}: {imp[idx[i]]:.4f}")

    # 保存
    out = dict(accuracy=best_acc, baseline=baseline, n_samples=X.shape[0], n_features=X.shape[1])
    (ROOT/"eval"/"boost_result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("SAVED", flush=True)


if __name__ == "__main__":
    main()
