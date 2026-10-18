# -*- coding: utf-8 -*-
"""终版标定：合并指纹缓存+波形缓存，新权重下重算真曲分布/阈值/生成曲百分位"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from ai_flavor_check import METRICS, load_reference  # noqa
from ai_segment_forensic import parse_lab, first_span  # noqa
from novel_songs_manifest import SONGS  # noqa
from hit_score import TRANSCRIPTS  # noqa

fp_cache = json.loads((ROOT / "eval" / "lib_metric_cache.json").read_text(encoding="utf-8"))
wv_cache = json.loads((ROOT / "eval" / "wave_metric_cache.json").read_text(encoding="utf-8"))


def ai_score_merged(m):
    total = 0.0
    for key, sign, w in METRICS:
        x = m.get(key)
        if x is None:
            continue
        vals = None
        total += 0
    return 0


def score_merged(m, corpus_vals_map):
    total, detail = 0.0, {}
    for key, sign, w in METRICS:
        cv = corpus_vals_map.get(key)
        x = m.get(key)
        if cv is None or len(cv) < 10 or x is None:
            continue
        med = float(np.median(cv))
        iqr = float(np.percentile(cv, 75) - np.percentile(cv, 25)) + 1e-9
        z = sign * (x - med) / (0.7413 * iqr)
        pos = min(max(0.0, z), 6.0)
        total += w * pos
        detail[key] = dict(x=round(float(x), 2), med=round(med, 2), z=round(float(z), 2))
    return round(total, 2), detail


# 合并缓存：真曲逐段
merged = {}
for k, v in fp_cache.items():
    if v.get("skip"):
        continue
    wv = wv_cache.get(k, {})
    if "crest_db" not in wv:
        continue
    m = dict(v)
    m.update({kk: wv[kk] for kk in ("crest_db", "env_rough", "low_attack_ms",
                                    "silence_frac", "onset_iv_cv", "kurtosis", "lra_db")
              if kk in wv})
    merged[k] = m

corpus_vals = {}
for key, _, _ in METRICS:
    corpus_vals[key] = [r[key] for r in merged.values() if r.get(key) is not None]

# 生成 12 首
from remediate_novel import naturalness  # noqa
import librosa  # noqa
import soundfile as sf  # noqa

novel_rows = []
for s in SONGS:
    flac = ROOT / "novel_songs" / "final" / f"{s['no']}_{s['title']}.flac"
    tdir = TRANSCRIPTS[s["no"]]
    rows = parse_lab(tdir, "structure.lab")
    ch = first_span(rows, "chorus", 1e9)
    y, _ = librosa.load(str(flac), sr=22050, mono=True)
    dur_a = len(y) / 22050
    t2 = min(ch[1], ch[0] + 30, dur_a)
    m = dict(json.loads((ROOT / "eval" / "ai_flavor_rank.json").read_text(encoding="utf-8")
                        ) [0].get("metrics", {})) if False else None
    # 直接复用 lib_metric_cache（生成曲不在里面）→ 用已算过的 rank json 指标 + 现算波形
    rank = json.loads((ROOT / "eval" / "ai_flavor_rank.json").read_text(encoding="utf-8"))
    entry = next((r for r in rank if r["no"] == s["no"]), None)
    m = entry["metrics"] if entry else {}
    # 波形三指标现算（快速）
    seg = y[int(ch[0] * 22050):int(min(ch[1], ch[0] + 30, dur_a) * 22050)]
    rms_g = float(np.sqrt((seg ** 2).mean()) + 1e-9)
    m["crest_db"] = round(float(20 * np.log10(np.abs(seg).max() / rms_g + 1e-9)), 2)
    from scipy.signal import butter, sosfilt
    sos = butter(4, 150 / 11025, btype="low", output="sos")
    low = sosfilt(sos, seg)
    lr = librosa.feature.rms(y=low, frame_length=1024, hop_length=256)[0]
    thr = lr.mean() + lr.std()
    atks, rise, st = [], False, 0
    for i, v in enumerate(lr):
        if not rise and v > thr:
            rise, st = True, i
        elif rise and v < thr:
            if i > st:
                atks.append((i - st) * 256 / 22050 * 1000)
            rise = False
    m["low_attack_ms"] = round(float(np.median(atks)), 1) if len(atks) >= 3 else None
    ons = librosa.onset.onset_detect(y=seg, sr=22050, hop_length=256, units="time",
                                     delta=0.06, backtrack=False)
    iv = np.diff(ons)
    m["onset_iv_cv"] = round(float(iv.std() / iv.mean()), 2) if len(iv) > 5 else None
    sc, det = score_merged(m, corpus_vals)
    novel_rows.append(dict(no=s["no"], title=s["title"], score=sc, detail=det))

cs = np.array([score_merged(m, corpus_vals)[0] for m in merged.values()])
novel_sorted = sorted(novel_rows, key=lambda r: -r["score"])
lines = ["# 终版 AI 味排名（13 指标新权重 + 463 真曲标定）", "",
         f"真曲分布：中位 {np.median(cs):.2f} · P75 {np.percentile(cs,75):.2f} · "
         f"P90 {np.percentile(cs,90):.2f} · MAX {cs.max():.2f}", "",
         "| 排名 | 曲目 | AI味分(新权重) |", "|---|---|---|"]
for i, r in enumerate(novel_sorted, 1):
    lines.append(f"| {i} | {r['no']} {r['title']} | {r['score']} |")
lines += ["", "### 逐曲超标明细（z>1 的指纹）", ""]
for r in novel_rows:
    big = [(k, d["x"], d["med"], d["z"]) for k, d in r["detail"].items()
           if d and d.get("z", 0) and d["z"] > 1]
    if big:
        lines.append(f"**{r['no']} {r['title']}** (分 {r['score']}):")
        for k, x, med, z in big:
            lines.append(f"  - {k}: {x} (真曲中位 {med}, z=+{z})")
(ROOT / "eval" / "final_ai_rank.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[-30:]))
