# -*- coding: utf-8 -*-
"""情绪锚点 × 4 类情绪画像 + 49 指标相关性 + AI 冠军差距"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050

def emotion_profile(y, sr):
    hop = 512
    f0, vf, _ = librosa.pyin(y, fmin=65, fmax=700, sr=sr, frame_length=1024, hop_length=hop)
    voiced = vf & ~np.isnan(f0)
    # 音高轮廓
    contours = []
    in_v, start = False, 0
    for i, v in enumerate(voiced):
        if v and not in_v: in_v, start = True, i
        elif not v and in_v:
            if i-start > 8:
                seg = f0[start:i]; seg = seg[~np.isnan(seg)]
                if len(seg) > 4: contours.append(np.log2(seg[-1]/(seg[0]+1e-9))*1200)
            in_v = False
    pitch_rise = round(float(np.median(contours)),1) if contours else None
    # 动态微弧
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms+1e-9)
    micro_arcs = []
    w = int(2*sr/hop)
    for i in range(w, len(db)-w):
        seg = db[i-w:i+w]; peak = np.argmax(seg)
        if 0.25*w < peak < 1.75*w:
            rise = seg[peak]-seg[0]; fall = seg[peak]-seg[-1]
            if rise > 1 and fall > 1: micro_arcs.append(min(rise,fall))
    micro_arc = round(float(np.median(micro_arcs)),1) if micro_arcs else None
    # 时间柔度
    _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    if len(beats) > 8:
        iv = np.diff(beats)/sr; ibi_cv = round(float(np.std(iv)/(np.mean(iv)+1e-9)),3)
    else: ibi_cv = None
    # 紧张-释放
    if voiced.sum() > 20:
        f0v = f0[voiced]; d = np.abs(np.diff(f0v)); d = d[~np.isnan(d)]
        tension = round(float(np.sum(d>200)/(len(y)/sr)*60),1)
    else: tension = None
    # 温暖
    S = np.abs(librosa.stft(y, n_fft=2048)); freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    warm = round(float(S[(freqs>=80)&(freqs<=300)].mean()/(S.mean()+1e-9)),2)
    return dict(pitch_rise=pitch_rise, micro_arc=micro_arc, ibi_cv=ibi_cv,
                tension=tension, warm=warm)

def _one(args):
    name, f = args
    try:
        y, _ = librosa.load(str(f), sr=SR, mono=True, offset=78, duration=30)
        r = emotion_profile(y, SR)
        r["name"] = name
        return r
    except: return None

# ===== 加载分类 =====
dims = json.loads((ROOT/"eval"/"library_dims_cache.json").read_text(encoding="utf-8"))
libm = json.loads((ROOT/"eval"/"lib_metric_cache.json").read_text(encoding="utf-8"))
rows = []
for name, d in dims.items():
    if not d or d.get("cent") is None: continue
    lm = libm.get(name, {})
    bpm = 60000/lm["period_ms"] if lm.get("period_ms") else 117.5
    rows.append((name, d, bpm))
X = np.array([[r[2], r[1]["cent"], r[1]["dyn"]] for r in rows])
Xn = (X-X.mean(0))/(X.std(0)+1e-9)
rng = np.random.RandomState(42); k = 4
cents = Xn[rng.choice(len(Xn),k,replace=False)]
for _ in range(50):
    lab = ((Xn[:,None]-cents[None])**2).sum(-1).argmin(1)
    for j in range(k):
        if (lab==j).any(): cents[j] = Xn[lab==j].mean(0)
lab = ((Xn[:,None]-cents[None])**2).sum(-1).argmin(1)
order = sorted(range(k), key=lambda j: np.median(X[lab==j,0]))
remap = {j: f"C{i+1}" for i,j in enumerate(order)}

# ===== 每类采样 20 首算情绪锚点 =====
wavs = ROOT/"eval"/"library_wavs"
jobs = []
for j in range(k):
    names = [r[0] for r, l in zip(rows, lab) if l==j][:20]
    for n in names:
        w = wavs/f"{n}.wav"
        if w.exists(): jobs.append((n, w))
emos = []
for jf in jobs:
    r = _one(jf)
    if r: emos.append(r)

# 按类聚合
cat_of = {r[0]: remap[lab[i]] for i, r in enumerate(rows)}
lines = ["# 4 类情绪画像（每类采样 20 首）", "",
         "| 锚点 | C1慢板 | C2中速亮 | C3大动态 | C4快板 |",
         "|---|---|---|---|---|"]
for key, name in [("pitch_rise","音高轮廓"),("micro_arc","动态微弧"),
                  ("ibi_cv","时间柔度"),("tension","紧张-释放"),("warm","人声温暖")]:
    vals = {c: [e[key] for e in emos if cat_of.get(e["name"])==c and e.get(key) is not None] for c in ["C1","C2","C3","C4"]}
    med = {c: f"{np.median(v):.2f}" if v else "N/A" for c, v in vals.items()}
    lines.append(f"| {name} | {med['C1']} | {med['C2']} | {med['C3']} | {med['C4']} |")

# ===== AI 冠军 vs 同类 =====
ai_files = {
    "01": (ROOT/"novel_songs/batch_gacha/01_v2/seed58888.flac", "C3"),
    "07": (ROOT/"novel_songs/batch_gacha/07_adj/seed1357_r1.flac", "C2"),
    "08": (ROOT/"novel_songs/triple_emergence/seed91104.flac", "C3"),
    "09": (ROOT/"novel_songs/batch_gacha/09/seed90904.flac", "C2"),
}
ai_emos = []
for no, (f, cat) in ai_files.items():
    if f.exists():
        try:
            y, _ = librosa.load(str(f), sr=SR, mono=True, offset=78, duration=30)
            r = emotion_profile(y, SR); r["name"]=no; r["cat"]=cat
            ai_emos.append(r)
        except: pass

lines += ["", "## AI 冠军 vs 同类情绪锚点", "",
          "| 锚点 | 01(C3) | 07(C2) | 08(C3) | 09(C2) | C2真曲 | C3真曲 | 判读 |",
          "|---|---|---|---|---|---|---|---|"]
for key, name in [("pitch_rise","音高轮廓"),("micro_arc","动态微弧"),
                  ("ibi_cv","时间柔度"),("tension","紧张-释放"),("warm","人声温暖")]:
    c2v = [e[key] for e in emos if cat_of.get(e["name"])=="C2" and e.get(key) is not None]
    c3v = [e[key] for e in emos if cat_of.get(e["name"])=="C3" and e.get(key) is not None]
    c2m, c3m = np.median(c2v) if c2v else 0, np.median(c3v) if c3v else 0
    ai = {r["name"]: r.get(key) for r in ai_emos}
    # 判断 AI 是否在同类区间
    def in_range(v, arr):
        if v is None or not arr: return "?"
        lo, hi = np.percentile(arr,10), np.percentile(arr,90)
        return "✅" if lo <= v <= hi else "⚠️"
    j01 = in_range(ai.get("01"), c3v); j07 = in_range(ai.get("07"), c2v)
    j08 = in_range(ai.get("08"), c3v); j09 = in_range(ai.get("09"), c2v)
    overall = "✅" if all(x=="✅" for x in [j01,j07,j08,j09]) else "⚠️"
    lines.append(f"| {name} | {ai.get('01','?')} | {ai.get('07','?')} | {ai.get('08','?')} "
                 f"| {ai.get('09','?')} | {c2m:.1f} | {c3m:.1f} | {overall} |")

(ROOT/"eval"/"emotion_by_category.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
