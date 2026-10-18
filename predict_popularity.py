# -*- coding: utf-8 -*-
"""42 首真曲 + NetEase 真实热度 → 训练分类/回归模型 → 用 12 首 AI 曲做预测"""
import json, sys, warnings
from pathlib import Path
import numpy as np
from collections import Counter

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ===== 1. 加载 42 首真曲（有 NetEase 标签 + 全特征） =====
ne_data = json.loads((ROOT/"eval"/"netease_popularity_v2.json").read_text(encoding="utf-8"))
fp_cache = json.loads((ROOT/"eval"/"lib_metric_cache.json").read_text(encoding="utf-8"))
wv_cache = json.loads((ROOT/"eval"/"wave_metric_cache.json").read_text(encoding="utf-8"))
res_cache = json.loads((ROOT/"eval"/"resonance_cache.json").read_text(encoding="utf-8"))
ms_map = {r["file"]: r for r in json.loads((ROOT/"corpus"/"melody_stats.json").read_text(encoding="utf-8"))["per_song"] if r.get("file")}

ne_songs = {}
for orig, v in ne_data.items():
    if "netease_id" not in v: continue
    clean = orig.replace(".mp3","").replace(" ","_")[:60]
    wav_stem = "lib_" + clean
    if wav_stem in fp_cache and "crest_db" in wv_cache.get(wav_stem, {}):
        ne_songs[wav_stem] = dict(v, stem=wav_stem)

print(f"真曲+NetEase: {len(ne_songs)} 首", flush=True)

# ===== 构建特征向量 =====
def build_features(stem):
    fp = fp_cache.get(stem, {})
    wv = wv_cache.get(stem, {})
    rs = res_cache.get(stem, {})
    ms = ms_map.get(stem.replace("lib_","",1), {})
    vec = {}
    for src, prefix in [(fp, "fp_"), (wv, "wf_"), (rs, "res_")]:
        for k, v in src.items():
            if isinstance(v, (int, float)) and v == v:
                vec[prefix+k] = float(v)
    for k in ("density","pitch_range","leap_ratio","motif_repeat","lift_mean","rest_ratio"):
        v = ms.get(k)
        if v is not None: vec["mel_"+k] = float(v)
    return vec

# 训练集（真曲）
X_train_rows, y_train_pop, y_train_tier = [], [], []
for stem, ne in ne_songs.items():
    vec = build_features(stem)
    if vec:
        X_train_rows.append(vec)
        y_train_pop.append(ne["popularity"])

fn_all = sorted(set(k for row in X_train_rows for k in row))
X_train = np.array([[row.get(k, 0.0) for k in fn_all] for row in X_train_rows])
y_pop = np.array(y_train_pop, dtype=float)
y_tier = np.array(y_train_tier)
print(f"训练集: {X_train.shape[0]} × {X_train.shape[1]}", flush=True)

# ===== 2. 加载 MERT 嵌入（拼接） =====
mert_npz = ROOT/"eval"/"mert_all_embeddings.npz"
mert_map = {}
if mert_npz.exists():
    z = np.load(str(mert_npz), allow_pickle=True)
    for n, e in zip(z["names"], z["embs"]):
        mert_map[str(n)] = e

# 给真曲训练集加 MERT
X_train_mert = []
for stem in ne_songs:
    e = mert_map.get(stem, np.zeros(1024))
    X_train_mert.append(e)
X_train_mert = np.array(X_train_mert)

# 拼接
X_train_full = np.hstack([X_train, X_train_mert])
print(f"拼接后: {X_train_full.shape}", flush=True)

# ===== 3. 训练 =====
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, f1_score

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_train_full_s = scaler.fit_transform(X_train_full)
y_pop_s = y_pop  # 真实 popularity 0-100

# 回归模型：预测真实 popularity
print("\n=== 热度回归（预测 NetEase popularity）===", flush=True)
for name, model in [
    ("RF", RandomForestRegressor(n_estimators=300, random_state=42)),
    ("ET", ExtraTreesRegressor(n_estimators=300, random_state=42)),
    ("GB", __import__("sklearn.ensemble", fromlist=["GradientBoostingRegressor"]).GradientBoostingRegressor(n_estimators=200, random_state=42)),
    ("SVR", __import__("sklearn.svm", fromlist=["SVR"]).SVR(kernel="rbf", C=10)),
]:
    for feat_set, feat_name in [(X_train, "35维"), (X_train_full, "35+MERT")]:
        pred = cross_val_predict(__import__("sklearn.base", fromlist=["clone"]).clone(model),
                                 feat_set, y_pop_s, cv=5)
        r2 = float(np.corrcoef(pred, y_pop_s)[0, 1]**2)
        mae = float(np.mean(np.abs(pred - y_pop_s)))
        print(f"  {feat_name}+{name}: R²={r2:.3f} MAE={mae:.1f}", flush=True)

# ===== 4. 用最优模型预测 12 首 AI 曲 =====
print("\n=== 预测 12 首 AI 曲 ===", flush=True)

# 加载 12 首 final 的特征
FINAL = ROOT / "novel_songs" / "final"
predictions = []

# 从 manifest 加载
from novel_songs_manifest import SONGS
for s in SONGS:
    flac = FINAL / f"{s['no']}_{s['title']}.flac"
    tdir = TRANSCRIPTS.get(s["no"])
    if not tdir or not Path(tdir).exists():
        # 尝试 humauto
        alt = ROOT / "novel_songs" / "v2" / f"novel_{s['no']}_{s['title']}_humauto" / "realized_full"
        if (alt / "score.abc").exists():
            tdir = alt
        else:
            tdir = ROOT / "eval" / "novel_transcripts" / f"novel_{s['no']}_{s['title']}"
    if not (Path(tdir) / "score.abc").exists():
        continue
    
    try:
        vec = build_features(tdir.name)
        if not vec:
            continue
        row = [vec.get(k, 0.0) for k in fn_all]
        
        # MERT
        mert_e = mert_map.get(tdir.name, np.zeros(1024))
        row_full = list(row) + mert_e.tolist()
        
        # 预测
        X_row = np.array(row).reshape(1, -1)
        X_row_full = np.array(row_full).reshape(1, -1)
        
        # 用最佳模型预测
        model = ExtraTreesRegressor(n_estimators=300, random_state=42)
        model.fit(X_train_full_s, y_pop_s)
        X_row_full_s = scaler.transform(X_row_full)
        pred_pop = model.predict(X_row_full_s)[0]
        
        predictions.append(dict(no=s["no"], title=s["title"], predicted_pop=round(float(pred_pop), 1)))
        print(f"  {s['no']} {s['title']}: 预测热度 {pred_pop:.0f}", flush=True)
    except Exception as e:
        print(f"  FAIL {s['no']}: {str(e)[:60]}", flush=True)

# ===== 输出 =====
predictions.sort(key=lambda p: -p["predicted_pop"])
print("\n=== 排名（预测热度从高到低）===", flush=True)
for p in predictions:
    print(f"  {p['no']} {p['title']}: {p['predicted_pop']}", flush=True)
