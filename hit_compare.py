# -*- coding: utf-8 -*-
"""爆款指数公平性检验：同一音频结构子量表(75分制)打曲库32 vs 生成12
若生成曲超过曲库全部真曲 -> 指标存在 AI 偏置，需重标定
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050

TRANSCRIPTS = {
    "01": ROOT / "eval/novel_transcripts/novel_01_欠天地一笔",
    "02": ROOT / "eval/novel_transcripts/novel_02_平账人",
    "03": ROOT / "eval/novel_transcripts/novel_03_白户",
    "04": ROOT / "novel_songs/redo/04/verify/realized",
    "05": ROOT / "novel_songs/redo/05/verify/realized",
    "06": ROOT / "novel_songs/redo/06/verify/realized",
    "07": ROOT / "novel_songs/v2/novel_07_规矩/verify/realized",
    "08": ROOT / "novel_songs/v2/novel_08_月供日/fullcheck/realized_full",
    "09": ROOT / "eval/novel_transcripts/novel_09_断供",
    "10": ROOT / "eval/novel_transcripts/novel_10_糕与药包",
    "11": ROOT / "novel_songs/v2/novel_11_烂尾劫_full/verify/realized",
    "12": ROOT / "novel_songs/v2/novel_12_新契_full/verify/realized",
}


def parse_lab(tdir):
    rows = []
    p = Path(tdir) / "structure.lab"
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        q = line.split("\t")
        if len(q) >= 3:
            rows.append((float(q[0]), float(q[1]), q[2].strip().lower()))
    return rows


def subscore(path, tdir, dur):
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    dur_a = len(y) / SR
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    fps = len(db) / dur_a
    rows = parse_lab(tdir)

    # 时长 (10)
    s_dur = 10.0 if 180 <= dur <= 280 else max(0.0, 10 - abs(dur - 230) / 12)

    # 入场 (30)：第一个非 intro 段起点
    fv = next((t1 for t1, t2, n in rows if n in ("verse", "pre-chorus", "chorus")), dur_a)
    s_entry = 30.0 * (1.0 if fv <= 27.8 else max(0.0, 1 - (fv - 27.8) / 40))

    # 副歌对比 (20)：chorus-verse 弧线 + 明暗
    sec_db, sec_cent = {}, {}
    cent = librosa.feature.spectral_centroid(y=y, sr=SR)[0]
    t_axis = np.arange(len(db)) / fps
    for i, t in enumerate(t_axis):
        n_now = next((n for t1, t2, n in rows if t1 <= t < t2), None)
        sec_db.setdefault(n_now, []).append(db[i])
        sec_cent.setdefault(n_now, []).append(cent[i])
    ch_db = [e for n, es in sec_db.items() if n == "chorus" for e in es]
    vs_db = [e for n, es in sec_db.items() if n == "verse" for e in es]
    ch_c = [c for n, cs in sec_cent.items() if n == "chorus" for c in cs]
    vs_c = [c for n, cs in sec_cent.items() if n == "verse" for c in cs]
    arc = (np.mean(ch_db) - np.mean(vs_db)) if ch_db and vs_db else 0.0
    blift = (np.mean(ch_c) - np.mean(vs_c)) if ch_c and vs_c else 0.0
    s_contrast = 20.0 * ((min(max(arc, 0), 4) / 4) * 0.6 + (min(max(blift, 0), 400) / 400) * 0.4)

    # 结构 (15)：副歌实例 + 高潮位置
    chor = sorted([(t1, t2) for t1, t2, n in rows if n == "chorus"])
    inst = []
    for t1, t2 in chor:
        if inst and t1 - inst[-1][1] < 1.5:
            inst[-1] = (inst[-1][0], max(inst[-1][1], t2))
        else:
            inst.append((t1, t2))
    k = int(5 * fps)
    smooth = np.convolve(db, np.ones(k) / k, mode="same")
    lo_i, hi_i = int(0.03 * len(smooth)), int(0.92 * len(smooth))
    cpos = float(np.arange(len(smooth))[lo_i:hi_i][np.argmax(smooth[lo_i:hi_i])] / fps / dur_a)
    s_struct = 15.0 * min(1.0, len(inst) / 2) * (1.0 if 0.55 <= cpos <= 0.85 else max(0.3, min(cpos, 0.55) / 0.55))

    return dict(entry=round(s_entry, 1), contrast=round(s_contrast, 1), struct=round(s_struct, 1),
                dur=round(s_dur, 1), total=round(s_entry + s_contrast + s_struct + s_dur, 1),
                arc=round(float(arc), 2), blift=round(float(blift)), fv=round(fv, 1),
                n_chorus=len(inst), cpos=round(cpos, 2))


def main():
    manifest = json.loads((ROOT / "corpus" / "sample_manifest.json").read_text(encoding="utf-8"))
    corpus = []
    for m in manifest:
        try:
            dur = sf_info_dur(m["wav"])
            tdir = ROOT / "corpus" / "transcriptions" / Path(m["wav"]).stem
            r = subscore(Path(m["wav"]), tdir, dur)
            r["name"] = Path(m["wav"]).stem
            corpus.append(r)
        except Exception as e:
            print("skip", Path(m["wav"]).stem[:20], str(e)[:60])

    from novel_songs_manifest import SONGS
    from hit_score import TRANSCRIPTS
    novel = []
    for s in SONGS:
        flac = ROOT / "novel_songs" / "final" / f"{s['no']}_{s['title']}.flac"
        try:
            dur = sf_info_dur(flac)
            r = subscore(flac, TRANSCRIPTS[s["no"]], dur)
            r["name"] = f"{s['no']} {s['title']}"
            novel.append(r)
        except Exception as e:
            print("skip", s["title"], str(e)[:60])

    cs = np.array([r["total"] for r in corpus])
    cs_sorted = sorted(corpus, key=lambda r: -r["total"])
    ns_sorted = sorted(novel, key=lambda r: -r["total"])

    lines = ["# 爆款指数公平性检验（75 分音频子量表，同公式同介质）", "",
             f"曲库真曲 top5：",
             *[f"- {r['name'][:24]}: **{r['total']}** (入场{r['entry']} 对比{r['contrast']} 结构{r['struct']} 时长{r['dur']})"
               for r in cs_sorted[:5]],
             "",
             f"曲库分布：中位 {np.median(cs):.1f} · P75 {np.percentile(cs,75):.1f} · **最大 {cs.max():.1f}** · "
             f"P95 {np.percentile(cs,95):.1f}",
             f"生成曲中位 {np.median([r['total'] for r in novel]):.1f}", "",
             "## 生成曲（75 分子量表排名）", "",
             "| 曲目 | 子量表 | 入场 | 对比 | 结构 | 时长 | 弧线 | 明暗 | 首唱 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in ns_sorted:
        over = " ⚠️超曲库满分" if r["total"] > cs.max() else ""
        lines.append(f"| {r['name']} | **{r['total']}**{over} | {r['entry']} | {r['contrast']} | {r['struct']} "
                     f"| {r['dur']} | {r['arc']} | {r['blift']} | {r['fv']}s |")
    over_n = sum(1 for r in novel if r["total"] > cs.max())
    lines += ["", "## 判定", "",
              f"- 曲库真曲最高 **{cs.max():.1f}** 分；生成曲超过该值的有 **{over_n}/12** 首"]
    if over_n:
        lines.append("- **结论：指数存在 AI 偏置** —— 偏置来源按维度归因：")
        for key, cname in [("entry", "入场"), ("contrast", "对比"), ("struct", "结构"), ("dur", "时长")]:
            cm = np.median([r[key] for r in corpus])
            nm = np.median([r[key] for r in novel])
            if nm > cm + 0.5:
                lines.append(f"  - {cname}: 生成中位 {nm} vs 曲库中位 {cm} → 该维度给 AI 记了高分的成分")
    else:
        lines.append("- 结论：生成曲均未超过真曲满分 → 指数无 AI 偏置")
    (ROOT / "eval" / "hit_fairness_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def sf_info_dur(p):
    i = __import__("soundfile").info(str(p))
    return i.frames / i.samplerate


if __name__ == "__main__":
    main()
