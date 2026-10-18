# -*- coding: utf-8 -*-
"""AI 冠军歌用曲库同款方法重测（offset=78, 30s, hop=512）→ 定类 → 类别目标对比"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
OUT = ROOT / "eval" / "ai_dims_library_method.json"

SONGS = {
    "01": ROOT/"novel_songs/batch_gacha/01/seed90105.flac",
    "02": ROOT/"novel_songs/batch_gacha/02/seed90202.flac",
    "03": ROOT/"novel_songs/batch_gacha/03/seed90301.flac",
    "04": ROOT/"novel_songs/batch_gacha/04/seed90400.flac",
    "05": ROOT/"novel_songs/batch_gacha/05/seed90505.flac",
    "06": ROOT/"novel_songs/batch_gacha/06/seed90607.flac",
    "07": ROOT/"novel_songs/batch_gacha/07/seed90706.flac",
    "08": ROOT/"novel_songs/triple_emergence/seed91097.flac",
    "09": ROOT/"novel_songs/batch_gacha/09/seed90903.flac",
    "10": ROOT/"novel_songs/batch_gacha/10/seed91005.flac",
    "11": ROOT/"novel_songs/batch_gacha/11/seed91101.flac",
    "12": ROOT/"novel_songs/batch_gacha/12/seed91201.flac",
}

def lib_method(y):
    hop = 512
    f0, vflag, _ = librosa.pyin(y, fmin=65, fmax=700, sr=SR, frame_length=1024, hop_length=hop)
    voiced = vflag & ~np.isnan(f0)
    orn = None
    if voiced.sum() > 20:
        d = np.abs(np.diff(f0)); m = voiced[1:] & voiced[:-1]
        orn = round(float(np.sum(d[m] > 15)/(len(y)/SR)), 2)
    gap = None
    idx = np.where(voiced)[0]
    if len(idx) > 5:
        gl = np.diff(idx)*hop/SR*1000; gl = gl[gl > 1]
        if len(gl) > 3:
            gap = round(float(np.std(gl)/(np.mean(gl)+1e-9)), 2)
    cent = round(float(librosa.feature.spectral_centroid(y=y, sr=SR)[0].mean()))
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms+1e-9)
    dyn = round(float(np.percentile(db,95)-np.percentile(db,10)), 1)
    return orn, gap, cent, dyn

res = {}
for no, f in SONGS.items():
    if not f.exists():
        print(f"{no}: MISSING {f.name}"); continue
    try:
        y, _ = librosa.load(str(f), sr=SR, mono=True, offset=78, duration=30)
    except Exception as e:
        print(f"{no}: CORRUPT {f} -> {type(e).__name__}", flush=True)
        res[no] = dict(error=str(f))
        continue
    orn, gap, cent, dyn = lib_method(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=SR)
    bpm = round(float(np.atleast_1d(tempo)[0]), 1)
    res[no] = dict(orn=orn, gap=gap, cent=cent, dyn=dyn, bpm=bpm, file=str(f))
    print(f"{no}: orn={orn} gap={gap} cent={cent} dyn={dyn} bpm={bpm}", flush=True)

# ===== 定类（最近类别中位） =====
tg = json.loads((ROOT/"eval"/"category_targets.json").read_text(encoding="utf-8"))
cats = {c: np.array([v["targets"]["cent"], v["targets"]["dyn"],
                     float(v["bpm_med"])]) for c, v in tg.items()}
lines = ["# AI 冠军 × 曲库同款方法 × 类别目标", "",
         "| 歌 | 类 | orn/s | gapCV | cent | dyn | BPM | 类目标(orn/gap/cent/dyn) | 达标数/4 |",
         "|---|---|---|---|---|---|---|---|---|"]
for no, r in list(res.items()):
    if "error" in r:
        lines.append(f"| {no} | ⚠️文件损坏需重渲染 | | | | | | | | |")
        continue
    v = np.array([r["cent"], r["dyn"], r["bpm"]])
    cid = min(cats, key=lambda c: np.abs((v - cats[c])/np.array([600,8,50])).sum())
    t = tg[cid]["targets"]
    hits = sum([abs(r["orn"]-t["orn"])<=tg[cid]["orn_p25_75"].replace("~",",").count(",")*0 or
                (float(tg[cid]["orn_p25_75"].split("~")[0]) <= r["orn"] <= float(tg[cid]["orn_p25_75"].split("~")[1])),
                float(tg[cid]["gap_p25_75"].split("~")[0]) <= r["gap"] <= float(tg[cid]["gap_p25_75"].split("~")[1]),
                float(tg[cid]["cent_p25_75"].split("~")[0]) <= r["cent"] <= float(tg[cid]["cent_p25_75"].split("~")[1]),
                float(tg[cid]["dyn_p25_75"].split("~")[0]) <= r["dyn"] <= float(tg[cid]["dyn_p25_75"].split("~")[1])])
    r["cat"] = cid; r["hits"] = hits
    lines.append(f"| {no} | {cid} | {r['orn']} | {r['gap']} | {r['cent']} | {r['dyn']} | {r['bpm']} "
                 f"| {t['orn']}/{t['gap']}/{t['cent']}/{t['dyn']} | **{hits}/4** |")

lines += ["", "P25~75 区间命中数 = 该维落入曲库同类正常范围 → 视为'像真人这一类'"]
(ROOT/"eval"/"ai_vs_category.md").write_text("\n".join(lines), encoding="utf-8")
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n" + "\n".join(lines[3:]), flush=True)
