# -*- coding: utf-8 -*-
"""保存最优模型 + 预测 12 首 AI 歌的爆款等级"""
import json, pickle, sys, warnings
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ===== 热度 =====
MEGA_A = {"周杰伦","林俊杰","五月天","张学友","邓紫棋","陈奕迅","王菲","张惠妹",
          "孙燕姿","梁静茹","S.H.E","王心凌","张韶涵","胡歌"}
KNOWN_A = {"许嵩","汪苏泷","后弦","黄雅莉","张芸京","山野","花儿乐队","信乐团",
           "183Club","5566","By2","吴克群","薛之谦","毛不易","陈琳","香香","唐磊","汤潮"}
MEGA_S = {"青花瓷","东风破","菊花台","发如雪","江南","曹操","隐形的翅膀","偏爱",
          "死了都要爱","突然的自我","温柔","知足","Super Star","不想长大","遇见","天黑黑","绿光"}

def tier(stem):
    clean = stem.replace("lib_","",1).replace("_"," ")
    for a in MEGA_A:
        if a in clean: return 3
    for h in MEGA_S:
        if h in clean or h.replace(" ","_") in clean: return 3
    for a in KNOWN_A:
        if a in clean: return 2
    return 1

# ===== 加载 MERT =====
z = np.load(str(ROOT/"eval"/"mert_all_embeddings.npz"), allow_pickle=True)
mert_names = [str(n) for n in z["names"]]
mert_embs = z["embs"]
y = np.array([tier(n) for n in mert_names])

# ===== 加载手工特征（35 维） =====
feats = json.loads((ROOT/"corpus"/"features.json").read_text(encoding="utf-8"))
fp_cache = json.loads((ROOT/"eval"/"lib_metric_cache.json").read_text(encoding="utf-8"))
wv_cache = json.loads((ROOT/"eval"/"wave_metric_cache.json").read_text(encoding="utf-8"))
res_cache = json.loads((ROOT/"eval"/"resonance_cache.json").read_text(encoding="utf-8"))
ms_list = json.loads((ROOT/"corpus"/"melody_stats.json").read_text(encoding="utf-8"))["per_song"]
ms_map = {r["file"]: r for r in ms_list if r.get("file")}

manual_keys = set()
manual_rows = []
for stem in mert_names:
    row = {}
    fp = fp_cache.get(stem, {})
    if not fp.get("skip"):
        for k, v in fp.items():
            if isinstance(v, (int, float)): row["fp_"+k] = float(v)
    wv = wv_cache.get(stem, {})
    for k, v in wv.items():
        if isinstance(v, (int, float)): row["wf_"+k] = float(v)
    rs = res_cache.get(stem, {})
    for k, v in rs.items():
        if isinstance(v, (int, float)): row["res_"+k] = float(v)
    ms = ms_map.get(stem.replace("lib_","",1), {})
    for k in ("density","pitch_range","leap_ratio","motif_repeat","lift_mean","rest_ratio"):
        v = ms.get(k)
        if v is not None: row["mel_"+k] = float(v)
    manual_rows.append(row)
    manual_keys.update(row.keys())

fn_manual = sorted(manual_keys)
X_manual = np.array([[r.get(k, 0.0) for k in fn_manual] for r in manual_rows])

# ===== PCA64 + 拼接 =====
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
X_mert = mert_embs
pca = PCA(n_components=64, random_state=42)
scaler_mert = StandardScaler()
X_mert_s = scaler_mert.fit_transform(X_mert)
mert_pca = pca.fit_transform(X_mert_s)

scaler_man = StandardScaler()
X_man_s = scaler_man.fit_transform(X_manual)

X_combined = np.hstack([X_man_s, mert_pca])

# ===== 训练最优模型 =====
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
rf = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pred = cross_val_predict(rf, X_combined, y, cv=skf)
acc = float(np.mean(pred == y))
rf.fit(X_combined, y)  # 用全部数据重新拟合，供后续预测使用
acc = float(np.mean(pred == y))
print(f"验证准确率: {acc*100:.1f}%", flush=True)

# 保存
artifact = {
    "model": rf,
    "pca": pca,
    "scaler_mert": scaler_mert,
    "scaler_manual": scaler_man,
    "fn_manual": fn_manual,
    "accuracy": acc,
}
with open(ROOT/"eval"/"best_popularity_model.pkl", "wb") as f:
    pickle.dump(artifact, f)
print("✅ 模型已保存: eval/best_popularity_model.pkl", flush=True)

# ===== 预测 12 首 AI 歌 =====
# 加载手工特征
from novel_songs_manifest import SONGS
from hit_score import TRANSCRIPTS

ai_results = []
for s in SONGS:
    no = s["no"]
    title = s["title"]
    flac = final_flac = ROOT/"novel_songs"/"final"/f"{no}_{title}.flac"
    
    # 找转谱
    tdir = TRANSCRIPTS.get(no)
    if not tdir:
        # 尝试其他路径
        for alt in [
            ROOT/"novel_songs/v2"/f"novel_{no}_{title}"/"verify"/"realized",
            ROOT/"novel_songs/v2"/f"novel_{no}_{title}_humauto"/"realized_full",
            ROOT/"covers/jiangnan"/"score_full",
        ]:
            if Path(alt)/"score.abc".replace("score.abc",""): 
                if (Path(alt)/"score.abc").exists():
                    tdir = alt
                    break
    
    if not tdir or not (Path(tdir)/"score.abc" if isinstance(tdir, Path) else Path(str(tdir))/("score.abc")).exists():
        # 用 gacha 目录
        for g in sorted((ROOT/"novel_songs"/"gacha"/no).glob("*/verify/realized")) if (ROOT/"novel_songs"/"gacha"/no).exists() else []:
            if (g/"score.abc").exists():
                tdir = g
                break
    
    if not tdir:
        # 最后手段：用 eval/novel_transcripts
        alt2 = ROOT/"eval"/"novel_transcripts"/f"novel_{no}_{title}"
        if (alt2/"score.abc").exists():
            tdir = alt2
    
    if not tdir or not (Path(str(tdir))/"score.abc").exists():
        print(f"  skip {no} {title} (no transcript)")
        continue
    
    # 手工特征
    sys.path.insert(0, str(ROOT))
    from melody_stats import analyze_song
    m = analyze_song(str(Path(tdir)/"score.abc"))
    
    row = {}
    fp = fp_cache.get(Path(tdir).name, {})
    if not fp.get("skip"):
        for k, v in fp.items():
            if isinstance(v, (int, float)): row["fp_"+k] = float(v)
    wv = wv_cache.get(Path(tdir).name, {})
    for k, v in wv.items():
        if isinstance(v, (int, float)): row["wf_"+k] = float(v)
    rs = res_cache.get(Path(tdir).name, {})
    for k, v in rs.items():
        if isinstance(v, (int, float)): row["res_"+k] = float(v)
    ms = ms_map.get(Path(tdir).name, {})
    for k in ("density","pitch_range","leap_ratio","motif_repeat","lift_mean","rest_ratio"):
        v = ms.get(k)
        if v is not None: row["mel_"+k] = float(v)
    
    vec = [row.get(k, 0.0) for k in fn_manual]
    
    # MERT
    # 找 final flac 对应的 library_wav（如果有）
    # 如果没有，跳过 MERT（用 0 填充）
    mert_vec = np.zeros(1024)
    # 尝试从已有嵌入找
    for i, mn in enumerate(mert_names):
        if title[:4] in mn or no in mn:
            mert_vec = mert_embs[i]
            break
    
    mert_p = pca.transform(mert_vec.reshape(1, -1))
    man_s = scaler_man.transform(np.array([vec]))
    X_input = np.hstack([man_s, mert_p])
    
    proba = rf.predict_proba(X_input)[0]
    pred_tier = int(np.argmax(proba)) + 1
    confidence = float(np.max(proba))
    
    ai_results.append(dict(no=no, title=title, pred_tier=pred_tier, confidence=round(confidence, 3),
                           proba=dict(zip([1,2,3], [round(float(p),3) for p in proba]))))
    print(f"  {no} {title}: 预测层级 {pred_tier} (置信 {confidence:.1%})", flush=True)

# ===== 输出 =====
ai_results.sort(key=lambda x: (-x["pred_tier"], -x["confidence"]))
print("\n=== 爆款等级预测 ===", flush=True)
for r in ai_results:
    tier_name = {3: "🔥 爆款潜力", 2: "👍 有潜力", 1: "📝 需要打磨"}[r["pred_tier"]]
    print(f"  {r['no']} {r['title']}: {tier_name} (置信 {r['confidence']:.0%})", flush=True)
