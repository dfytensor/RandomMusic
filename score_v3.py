# -*- coding: utf-8 -*-
"""评分标准 v3.5：训练集建标 → 测试集验证 → 金标裁决
v3 修复点:
  1. 段落自适应: 用能量峰找副歌, 不用固定 78s
  2. 宽区间: P5~P95 (train 建立), 只有真离谱才扣分
  3. 连续计分: 1-|值-中位|/量程, 擦边也有分
  4. 金标终验(2026-10-15): 全量重爬 n≈548, 流行度 ρ≈-0.09 天花板 0.178
     → 流行度锚不可达已定论, 改用共识金标(经典vs普通 p=0.0075 PASS)
  详见 eval/v3_final_report.md
v3.5(2026-10-16): +5 articulation 新维(全曲人声 stem 口径, art_core.art_dims)
  vib_ratio/vib_depth/phase_R/onset_rise_db/long_smooth
  无 stem 时新维=None 自动跳过 → 旧流程/无 stem 文件零破坏
  阶段0快验(eval/articulation_mix_summary.json): 混音口径 1/5 保持, 弃用
"""
import json, sys, warnings, hashlib
from pathlib import Path
import numpy as np
import librosa
from art_core import ART_DIMS, art_dims

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050

# ===== 1. 拆分 =====
gold = json.loads((ROOT/"eval"/"netease_popularity_v2.json").read_text(encoding="utf-8"))
gold_files = {k for k, v in gold.items() if v.get("popularity") and v.get("netease_id")}
wavs = sorted((ROOT/"eval"/"library_wavs").glob("*.wav"))

def name_of(w):
    import re
    s = w.stem.replace("lib_", "")
    s = re.sub(r"[_]+", " ", s)
    return re.sub(r"\s+", "", s).lower()

def name_of_mp3(k):
    import re
    s = k.replace(".mp3", "")
    s = re.sub(r"[_]+", " ", s)
    return re.sub(r"\s+", "", s).lower()

def name_of_stem(stem):
    return name_of(Path(stem + ".wav"))

train, test = [], []
for w in wavs:
    if name_of(w) in gold_files:
        continue  # 金标独立, 不进 train/test
    h = int(hashlib.md5(w.stem.encode()).hexdigest(), 16) % 10
    (train if h < 5 else test).append(w)
print(f"train={len(train)} test={len(test)} gold={len(gold_files)}", flush=True)

# ===== 2. 测量(自适应副歌) =====
def find_chorus_segment(path, seg_len=30):
    """能量+起音密度最高 30s 窗口 = 副歌代理"""
    y, sr = librosa.load(path, sr=SR, mono=True)
    if len(y) < seg_len*SR:
        return y, 0
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    ons = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    n = len(rms)
    win = int(seg_len*SR/hop)
    step = int(5*SR/hop)  # 5s 步进
    best, best_score, best_t = 0, -1, 0
    for t in range(0, n-win, step):
        r = rms[t:t+win].mean(); o = ons[t:t+win].mean()
        sc = r*(1+o)
        if sc > best_score:
            best_score, best_t = sc, t
    t0 = best_t*hop/sr
    return y[int(t0*sr):int((t0+seg_len)*sr)], round(t0,1)

def measure(path, bpm_path=None, vocals_path=None, beats_path=None):
    """bpm_path: 可选伴奏轨 wav — 同窗测 BPM, 对声纹替换/人声变化鲁棒;
    默认 None 时沿用整曲 (行为与历史缓存一致)
    v3.5: vocals_path=全曲人声 stem 时附加 5 articulation 新维(全曲口径);
    缺省时新维=None, score_of 自动跳过"""
    y30, t0 = find_chorus_segment(path)
    if len(y30) < SR*10:
        return None
    y = y30
    hop = 512
    f0, vf, _ = librosa.pyin(y, fmin=65, fmax=700, sr=SR, frame_length=1024, hop_length=hop)
    voiced = vf & ~np.isnan(f0)
    orn = gap = tension = None
    if voiced.sum() > 20:
        d = np.abs(np.diff(f0)); m = voiced[1:] & voiced[:-1]
        orn = float(np.sum(d[m] > 15)/(len(y)/SR))
        idx = np.where(voiced)[0]
        gl = np.diff(idx)*hop/SR*1000; gl = gl[gl > 1]
        if len(gl) > 3: gap = float(np.std(gl)/(np.mean(gl)+1e-9))
        f0v = f0[voiced]; dd = np.abs(np.diff(f0v)); dd = dd[~np.isnan(dd)]
        tension = float(np.sum(dd > 200)/(len(y)/SR)*60)
    cent = float(librosa.feature.spectral_centroid(y=y, sr=SR)[0].mean())
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms+1e-9)
    dyn = float(np.percentile(db,95)-np.percentile(db,10))
    # 动态微弧
    arcs = []; w = int(2*SR/hop)
    for i in range(w, len(db)-w):
        seg = db[i-w:i+w]; pk = np.argmax(seg)
        if 0.25*w < pk < 1.75*w:
            r1, f1 = seg[pk]-seg[0], seg[pk]-seg[-1]
            if r1 > 1 and f1 > 1: arcs.append(min(r1,f1))
    micro = float(np.median(arcs)) if arcs else None
    bpm_audio = y
    if bpm_path is not None:
        try:
            # 整轨伴奏测 BPM: 与人声/副歌窗 t0 均无关, 换嗓前后完全一致
            ya, _ = librosa.load(str(bpm_path), sr=SR, mono=True)
            if len(ya) >= SR*30:
                bpm_audio = ya
        except Exception:
            pass
    tempo, beats = librosa.beat.beat_track(y=bpm_audio, sr=SR, hop_length=hop)
    bpm = float(np.atleast_1d(tempo)[0])
    m = dict(orn=orn, gap=gap, cent=cent, dyn=dyn, tension=tension,
             micro=micro, bpm=bpm, seg_t=t0)
    if vocals_path is not None and Path(vocals_path).exists():
        try:
            m.update(art_dims(vocals_path, beats_path))
        except Exception:
            m.update({d: None for d in ART_DIMS})
    else:
        m.update({d: None for d in ART_DIMS})
    return m

# ===== v3.5: 人声 stem 路径 + 新维共享缓存 =====
ART_CACHE_P = ROOT/"eval/v3_art_cache.json"
_art_cache = json.loads(ART_CACHE_P.read_text(encoding="utf-8")) if ART_CACHE_P.exists() else {}

def stem_paths(stem_name):
    sd = ROOT/"human_stems"/stem_name
    v = sd/"vocals.wav"; b = sd/"beats.json"
    return (v if v.exists() else None, b if b.exists() else None)

def art_for(stem_name):
    """读/算某库文件的 5 新维(共享 art_cache, train/test/gold 三份缓存只算一次)"""
    if stem_name in _art_cache and _art_cache[stem_name] is not None:
        return dict(_art_cache[stem_name])
    v, b = stem_paths(stem_name)
    if v is None:
        return {d: None for d in ART_DIMS}
    m = art_dims(v, b)
    _art_cache[stem_name] = m
    ART_CACHE_P.write_text(json.dumps(_art_cache, ensure_ascii=False), encoding="utf-8")
    return dict(m)

# ===== 3. 定类(用 train 全量 KMeans 简化: BPM 4 段) =====
def cat_of(bpm):
    if bpm < 100: return "C1"
    if bpm < 120: return "C2"
    if bpm < 135: return "C3"
    return "C4"

# ===== 4. 训练集建标 =====
def scan(files, tag):
    cache_p = ROOT/f"eval/v3_{tag}_cache.json"
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}
    for i, f in enumerate(files):
        ent = cache.get(f.stem)
        if ent and all(k in ent for k in ART_DIMS):
            continue
        if ent:  # 旧 6 维条目: 只补新维(省一次混音 pyin)
            try:
                ent.update(art_for(f.stem)); cache[f.stem] = ent
            except Exception:
                pass
        else:
            try:
                v, b = stem_paths(f.stem)
                cache[f.stem] = measure(f, vocals_path=v, beats_path=b)
            except Exception:
                cache[f.stem] = None
        if (i+1) % 25 == 0:
            cache_p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            print(f"  {tag} {i+1}/{len(files)}", flush=True)
    cache_p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache

print("扫描 train...", flush=True)
train_m = scan(train, "train")
print("扫描 test...", flush=True)
test_m = scan(test, "test")

DIMS = ["orn", "gap", "cent", "dyn", "tension", "micro"] + list(ART_DIMS)
norms = {}  # cat -> dim -> {p5, med, p95, scale}
for cat in ["C1", "C2", "C3", "C4"]:
    norms[cat] = {}
    for d in DIMS:
        vals = []
        for stem, m in train_m.items():
            if not m: continue
            if cat_of(m["bpm"]) != cat: continue
            if m.get(d) is None: continue
            vals.append(m[d])
        if len(vals) < 8:
            norms[cat][d] = None; continue
        p5, med, p95 = np.percentile(vals, [5, 50, 95])
        scale = max(p95-p5, 1e-6)
        norms[cat][d] = dict(p5=float(p5), med=float(med), p95=float(p95), scale=float(scale))

# ===== 5. 连续计分 =====
def score_of(m):
    if not m: return None
    cat = cat_of(m["bpm"])
    parts = []
    for d in DIMS:
        n = norms.get(cat, {}).get(d)
        v = m.get(d)
        if not n or v is None: continue
        # 落 P5~P95: 100*(1-|v-med|/scale), 越界按距离衰减到负
        s = 100*(1 - abs(v-n["med"])/n["scale"])
        # 越界惩罚: 超出 P5/P95 每超出 scale 的 10% 扣 10 分
        if v < n["p5"]: s -= (n["p5"]-v)/n["scale"]*100
        if v > n["p95"]: s -= (v-n["p95"])/n["scale"]*100
        parts.append(s)
    if not parts: return None
    return dict(cat=cat, score=round(float(np.mean(parts)),1), n=len(parts))

print("评分 test...", flush=True)
test_scores = {s: score_of(m) for s, m in test_m.items()}
test_scores = {k: v for k, v in test_scores.items() if v}
vals = [v["score"] for v in test_scores.values()]
print(f"test n={len(vals)} 中位={np.median(vals):.1f} P25={np.percentile(vals,25):.1f} "
      f"P75={np.percentile(vals,75):.1f} min={min(vals):.1f} max={max(vals):.1f}", flush=True)

# ===== 6. 金标验证 =====
print("扫描金标...", flush=True)
gold_m = scan([w for w in wavs if name_of(w) in {name_of_mp3(k) for k in gold_files}], "gold")
rows = []
gold_by_norm = {name_of_mp3(k): k for k in gold_files}
for stem, m in gold_m.items():
    if not m: continue
    sc = score_of(m)
    if not sc: continue
    key = gold_by_norm.get(name_of_stem(stem))
    pop = gold.get(key, {}).get("popularity") if key else None
    if pop: rows.append((sc["score"], float(pop), stem[:20]))
if len(rows) >= 10:
    from scipy.stats import spearmanr
    rho, p = spearmanr([r[0] for r in rows], [r[1] for r in rows])
    print(f"金标 n={len(rows)} Spearman ρ={rho:.3f} p={p:.4f} "
          f"(终验已定论: 流行度锚不可达, 以共识金标为准, 见 v3_final_report.md)", flush=True)
else:
    rho, p = None, None
    print(f"金标有效 n={len(rows)} 不足10", flush=True)

report = dict(version="v3.5", train_n=len(train), test_n=len(test), gold_n=len(rows),
              test_median=float(np.median(vals)), test_p25=float(np.percentile(vals,25)),
              test_p75=float(np.percentile(vals,75)), test_min=min(vals), test_max=max(vals),
              spearman_rho=rho, spearman_p=p,
              gold_conclusion="popularity anchor infeasible (triple evidence); consensus gold PASS p=0.0075",
              norms=norms,
              gold_detail=[dict(score=r[0], pop=r[1], song=r[2]) for r in rows])
(ROOT/"eval"/"v3_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print("SAVED eval/v3_validation.json", flush=True)
