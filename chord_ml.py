# -*- coding: utf-8 -*-
"""和弦进行分析 + 全特征合并 ML：
1. 从 chord.lab 提取和弦特征（多样性/节奏/常见进行匹配）
2. 合并全部已有缓存
3. RF 分类 3 类热度
"""
import json, re, sys, warnings
from pathlib import Path
from collections import Counter
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def parse_chord_lab(path):
    """chord.lab -> [(start, end, label)]"""
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        q = line.split("\t")
        if len(q) >= 3:
            rows.append((float(q[0]), float(q[1]), q[2].strip()))
    return rows


def chord_features(chords, total_bars):
    """和弦特征提取"""
    if not chords:
        return {}
    # 去静音
    real = [(a, b, l) for a, b, l in chords if l not in ("N", "X", "*")]
    if len(real) < 3:
        return {}

    # 1) 和弦多样性（unique chords / total chords）
    unique = len(set(l for _, _, l in real))
    diversity = round(unique / len(real), 3)

    # 2) 和弦变化率（每小节）
    dur_total = sum(b - a for a, b, _ in real)
    changes_per_bar = round(len(real) / max(total_bars, 1), 2)

    # 3) 最常见和弦进行（相邻和弦对）
    pairs = []
    for i in range(len(real) - 1):
        a_end = real[i][1]
        b_start = real[i + 1][0]
        if abs(a_end - b_start) < 0.5:  # 连续
            pairs.append((real[i][2], real[i + 1][2]))

    # 4) 根音移动模式（简化：只看根音的音程）
    def root_pc(label):
        m = re.match(r"([A-G][#b]?)", label)
        if not m: return None
        base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        n = m.group(1)
        pc = base.get(n[0].upper(), 0)
        if len(n) > 1:
            if n[1] == "#": pc += 1
            elif n[1] == "b": pc -= 1
        return pc % 12

    root_moves = []
    for i in range(len(real) - 1):
        r1 = root_pc(real[i][2])
        r2 = root_pc(real[i + 1][2])
        if r1 is not None and r2 is not None:
            root_moves.append((r2 - r1) % 12)

    # 根音移动分布
    move_counter = Counter(root_moves) if root_moves else Counter()
    # 4度/5度（经典进行）
    fourth_fifth = sum(move_counter.get(x, 0) for x in (5, 7))
    total_moves = sum(move_counter.values())
    fourth_fifth_ratio = round(fourth_fifth / max(total_moves, 1), 3)

    # 5) 和弦时值分布（和弦持续时长的 CV）
    durations = [b - a for a, b, _ in real]
    dur_cv = round(float(np.std(durations) / (np.mean(durations) + 1e-9)), 3)

    # 6) 小调/大调比例
    minor_count = sum(1 for _, _, l in real if ":min" in l or "m" in l.split(":")[-1][:2])
    major_count = sum(1 for _, _, l in real if ":maj" in l or ":" not in l)
    maj_min_ratio = round(major_count / max(minor_count + major_count, 1), 3)

    return dict(
        chord_diversity=diversity,
        chord_changes_per_bar=changes_per_bar,
        fourth_fifth_ratio=fourth_fifth_ratio,
        chord_dur_cv=dur_cv,
        maj_min_ratio=maj_min_ratio,
        n_unique_chords=unique,
    )


def main():
    lib_wav = ROOT / "eval" / "library_wavs"
    trs = sorted((ROOT / "eval" / "library_transcripts").glob("*/score.abc"))
    feats = json.loads((ROOT / "corpus" / "features.json").read_text(encoding="utf-8"))
    fp_cache = json.loads((ROOT / "eval" / "lib_metric_cache.json").read_text(encoding="utf-8"))
    wv_cache = json.loads((ROOT / "eval" / "wave_metric_cache.json").read_text(encoding="utf-8"))
    ms_list = json.loads((ROOT / "corpus" / "melody_stats.json").read_text(encoding="utf-8"))["per_song"]
    ms_map = {}
    for r in ms_list:
        if r.get("file"):
            ms_map[r["file"]] = r

    # 热度
    MEGA_A = {"周杰伦","林俊杰","五月天","张学友","邓紫棋","陈奕迅","王菲","张惠妹",
              "孙燕姿","梁静茹","S.H.E","王心凌","张韶涵","胡歌"}
    KNOWN_A = {"许嵩","汪苏泷","后弦","黄雅莉","张芸京","山野","花儿乐队","信乐团",
               "183Club","5566","By2","吴克群","薛之谦","毛不易","陈琳","香香","唐磊","汤潮"}
    MEGA_S = {"青花瓷","东风破","菊花台","发如雪","江南","曹操","隐形的翅膀","偏爱",
              "死了都要爱","突然的自我","温柔","知足","Super Star","不想长大","遇见",
              "天黑黑","绿光"}

    def tier(stem):
        clean = stem.replace(" ", "_").replace("lib_", "")
        for a in MEGA_A:
            if a in clean: return 3
        for h in MEGA_S:
            if h in clean: return 3
        for a in KNOWN_A:
            if a in clean: return 2
        return 1

    # ===== 逐曲合并 =====
    X, y, all_fn = [], [], None
    count = 0
    for abc_p in trs:
        stem = abc_p.parent.name
        wav = lib_wav / (stem + ".wav")
        if not wav.exists():
            continue
        chords = parse_chord_lab(abc_p.parent / "chord.lab")

        # 旋律
        try:
            from melody_stats import analyze_song
            m = analyze_song(str(abc_p))
        except:
            m = {}

        total_bars = int(m.get("total_bars", 50)) if m else 50

        vec, fn = [], []
        # 1. 和弦特征 (6)
        cf = chord_features(chords, total_bars)
        for k in ("chord_diversity", "chord_changes_per_bar", "fourth_fifth_ratio",
                  "chord_dur_cv", "maj_min_ratio", "n_unique_chords"):
            v = cf.get(k)
            vec.append(float(v) if v is not None else 0.0); fn.append("ch_" + k)

        # 2. 旋律 (6)
        for k in ("density", "pitch_range", "leap_ratio", "motif_repeat", "lift_mean", "rest_ratio"):
            v = m.get(k)
            vec.append(float(v) if v is not None else 0.0); fn.append("mel_" + k)

        # 3. 指纹 (12)
        fp = fp_cache.get(wav.stem if hasattr(wav, 'stem') else stem, {})
        if not fp:
            # 尝试从 stem 反查
            for k2, v2 in fp_cache.items():
                if k2.endswith(stem[-15:]) or stem[-15:] in k2:
                    fp = v2; break
        for k in ("voiced_ratio", "n_vocal_runs", "vibrato_rate_hz", "vibrato_extent_cents",
                  "intra_note_drift_cents", "glide_ratio", "macro_dyn_std_db", "dyn_range_db",
                  "spectral_flux_std", "loop_lag_s", "loop_peak_corr", "loopiness"):
            v = fp.get(k)
            vec.append(float(v) if v is not None else 0.0); fn.append("fp_" + k)

        # 4. 波形 (7)
        wv = wv_cache.get(stem, {})
        for k in ("crest_db", "env_rough", "low_attack_ms", "silence_frac", "onset_iv_cv", "kurtosis", "lra_db"):
            v = wv.get(k)
            vec.append(float(v) if v is not None else 0.0); fn.append("wf_" + k)

        # 5. 基础 (4)
        f = feats.get(stem.replace("lib_", "", 1) + ".mp3", feats.get(stem + ".mp3", {}))
        for k in ("duration_s", "centroid_hz", "dyn_p95_p10_db", "rms_db"):
            v = f.get(k)
            vec.append(float(v) if v is not None else 0.0); fn.append(k)

        t = tier(stem)
        if not all_fn:
            all_fn = fn
        if len(vec) == len(all_fn):
            X.append(vec); y.append(t); count += 1

    X = np.array(X); y = np.array(y)
    from collections import Counter
    tc = Counter(y)
    print(f"样本: {X.shape[0]}  特征: {X.shape[1]}  分布: {dict(sorted(tc.items()))}", flush=True)

    # ===== ML =====
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight="balanced")
    pred = cross_val_predict(rf, X, y, cv=5)
    acc = float(np.mean(pred == y))
    baseline = max(tc.values()) / len(y)
    print(f"准确率: {acc*100:.1f}% (最大类基线 {baseline*100:.1f}%) 提升: +{(acc-baseline)*100:.1f}%", flush=True)

    from sklearn.metrics import f1_score
    f1 = f1_score(y, pred, average="macro")
    print(f"F1-macro: {f1:.3f}", flush=True)

    rf.fit(X, y)
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1]
    fn_names = all_fn if all_fn else [f"f{i}" for i in range(X.shape[1])]

    print("\nTop 20 特征:")
    for i in range(min(20, len(idx))):
        fn = fn_names[idx[i]] if idx[i] < len(fn_names) else "?"
        print(f"  {i+1}. {fn}: {imp[idx[i]]:.4f}")

    # 保存
    lines = [f"# 全特征 ML（{X.shape[1]} 维 × {X.shape[0]} 首）", "",
             f"准确率: {acc*100:.1f}% | F1: {f1:.3f} | 基线: {baseline*100:.1f}%", "",
             "| 排名 | 特征 | 重要性 |", "|---|---|---|"]
    for i in range(min(30, len(idx))):
        fn = fn_names[idx[i]] if idx[i] < len(fn_names) else "?"
        lines.append(f"| {i+1} | {fn} | {imp[idx[i]]:.4f} |")
    (ROOT / "eval" / "chord_ml_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("SAVED", flush=True)


if __name__ == "__main__":
    main()
