# -*- coding: utf-8 -*-
"""用 42 首有真实 NetEase 热度的歌检验之前保存的最优分类模型"""
import json, pickle, sys, warnings
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ===== 1. 加载保存的最优模型 =====
with open(ROOT/"eval"/"best_popularity_model.pkl", "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
pca = artifact["pca"]
scaler_mert = artifact["scaler_mert"]
scaler_manual = artifact["scaler_manual"]
fn_manual = artifact["fn_manual"]
print(f"模型加载成功 (训练时准确率: {artifact['accuracy']*100:.1f}%)", flush=True)

# ===== 2. 加载 42 首的 NetEase 数据 =====
ne_data = json.loads((ROOT/"eval"/"netease_popularity_v2.json").read_text(encoding="utf-8"))

# ===== 3. 加载所有缓存 =====
fp_cache = json.loads((ROOT/"eval"/"lib_metric_cache.json").read_text(encoding="utf-8"))
wv_cache = json.loads((ROOT/"eval"/"wave_metric_cache.json").read_text(encoding="utf-8"))
res_cache = json.loads((ROOT/"eval"/"resonance_cache.json").read_text(encoding="utf-8"))
ms_list = json.loads((ROOT/"corpus"/"melody_stats.json").read_text(encoding="utf-8"))["per_song"]
ms_map = {r["file"]: r for r in ms_list if r.get("file")}

# ===== 4. 构建特征 =====
X_test, y_real_pop, test_names = [], [], []
for orig_name, v in ne_data.items():
    if "netease_id" not in v: continue
    clean = orig_name.replace(".mp3","").replace(" ","_")[:60]
    wav_stem = "lib_" + clean
    tdir = ROOT/"eval"/"library_transcripts"/wav_stem
    abc_p = tdir/"score.abc"
    if not abc_p.exists(): continue

    fp = fp_cache.get(wav_stem, {})
    if fp.get("skip"): continue
    wv = wv_cache.get(wav_stem, {})
    rs = res_cache.get(wav_stem, {})
    ms = ms_map.get(wav_stem.replace("lib_","",1), {})

    row = {}
    for prefix, src in [("fp_",fp),("wf_",wv),("res_",rs)]:
        for k, vv in src.items():
            if isinstance(vv,(int,float)): row[prefix+k] = float(vv)
    for k in ("density","pitch_range","leap_ratio","motif_repeat","lift_mean","rest_ratio"):
        vv = ms.get(k)
        if vv is not None: row["mel_"+k] = float(vv)

    vec = [row.get(k, 0.0) for k in fn_manual]
    X_test.append(vec)
    y_real_pop.append(float(v.get("popularity",0)))
    test_names.append(v.get("netease_name", orig_name))

X_test = np.array(X_test)
print(f"测试样本: {X_test.shape[0]}", flush=True)

# ===== 5. MERT 嵌入 =====
mert_map = {}
if (ROOT/"eval"/"mert_all_embeddings.npz").exists():
    z = np.load(str(ROOT/"eval"/"mert_all_embeddings.npz"), allow_pickle=True)
    for n, e in zip(z["names"], z["embs"]):
        mert_map[str(n)] = e

X_mert_test = []
for stem_clean in [Path(n).stem.replace("lib_","",1) for n in test_names]:
    # 反查 wav stem
    found = None
    for mn, e in mert_map.items():
        if stem_clean[:8] in mn.replace("lib_","",1):
            found = e
            break
    if found is None:
        found = np.zeros(1024)
    X_mert_test.append(found)
X_mert_test = np.array(X_mert_test)

# PCA + 拼接
mert_pca_test = pca.transform(X_mert_test)
X_test_combined = np.hstack([X_test, mert_pca_test])
print(f"拼接后维度: {X_test_combined.shape[1]}", flush=True)

# ===== 6. 预测 =====
pred_tier = model.predict(X_test_combined)
pred_proba = model.predict_proba(X_test_combined)

# ===== 7. 与真实热度对比 =====
tier_names = {1: "普通(B)", 2: "热曲(A)", 3: "爆款(S)"}

print(f"\n{'='*60}")
print(f"{'曲目':24s} {'NetEase热度':>10s} {'预测层级':>8s} {'置信':>6s}")
print(f"{'='*60}")

for i in range(len(test_names)):
    pt = pred_tier[i]
    conf = float(np.max(pred_proba[i]))
    pop = y_real_pop[i]
    tn = tier_names.get(pt, str(pt))
    print(f"  {test_names[i][:22]:24s} {pop:8.0f}   {pt} ({tn})  {conf:.0%}")

# ===== 8. 热度分层 vs 预测层级 相关性 =====
from scipy.stats import spearmanr
rho, pval = spearmanr(y_real_pop, pred_tier)
print(f"\nSpearman ρ(真实热度, 预测层级) = {rho:+.3f} (p={pval:.4f})")
if abs(rho) > 0.3:
    print("✅ 模型预测的层级与真实热度有显著正相关")
elif abs(rho) > 0.15:
    print("🟡 弱正相关")
else:
    print("❌ 无显著相关")

# 按真实热度分组：高热(P75+) vs 低热(P25-)
pop_arr = np.array(y_real_pop)
p25, p75 = np.percentile(pop_arr, [25, 75])
high_mask = pop_arr >= p75
low_mask = pop_arr <= p25
if high_mask.sum() >= 3 and low_mask.sum() >= 3:
    high_tiers = pred_tier[high_mask]
    low_tiers = pred_tier[low_mask]
    print(f"\n高热组平均预测层级: {high_tiers.mean():.2f}")
    print(f"低热组平均预测层级: {low_tiers.mean():.2f}")
    if high_tiers.mean() > low_tiers.mean():
        print("✅ 模型正确：高热歌曲的预测层级更高")
    else:
        print("❌ 模型方向错误")

(ROOT/"eval"/"model_validation_42.md").write_text(
    "\n".join([f"NetEase热度: {y_real_pop}", f"预测层级: {pred_tier}", f"ρ={rho:+.3f}"]),
    encoding="utf-8")
print("SAVED eval/model_validation_42.md")
