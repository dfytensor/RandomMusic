# -*- coding: utf-8 -*-
"""爆款指数评分器（爆款相似度）：入场效率/Hook工程/副歌对比/结构重复/歌词词法/时长
-> eval/hit_score_report.md
"""
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SR = 22050
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS

BASE = json.loads((ROOT / "eval" / "hit_baseline.json").read_text(encoding="utf-8"))
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
IMAGERY = set("雨夜光手心路灯海风城桥梦雪江河星酒泪眼脸发衣窗门街桥山月影火灰烟")
PRON = re.compile(r"[你您我]")


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


def score_song(s):
    no = s["no"]
    title = s["title"]
    flac = ROOT / "novel_songs" / "final" / f"{no}_{title}.flac"
    tdir = TRANSCRIPTS[no]
    lyrics = s["lyrics"]
    dur = librosa.get_duration(path=str(flac))

    # 1) 入场效率
    rows = parse_lab(tdir)
    fv = next((t1 for t1, t2, n in rows if n in ("verse", "pre-chorus", "chorus")), dur)
    fc = next((t1 for t1, t2, n in rows if n == "chorus"), dur)
    y, _ = librosa.load(str(flac), sr=22050, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    fps = len(db) / dur
    e30 = float(db[: int(30 * fps)].mean()) - float(db.mean())

    entry = 30.0 * (1.0 if fv <= BASE["first_vocal_p75"] else
                    max(0.0, 1 - (fv - BASE["first_vocal_p75"]) / 40))

    # 2) Hook 工程
    lines = [l.strip() for l in lyrics.splitlines() if l.strip() and not l.strip().startswith("[")]
    hook_count = sum(1 for l in lines if title in l)
    hook_per_min = round(hook_count / (dur / 60), 2)
    hook = 25.0 * min(1.0, hook_count / 3) * (1.0 if fc <= BASE["first_chorus_med"] + 20
                                              else max(0.2, 1 - (fc - BASE["first_chorus_med"]) / 120))

    # 3) 副歌对比（用 v1 期 endurance 数据：直接现算简化版）
    k = int(5 * SR / 512)
    smooth = np.convolve(db, np.ones(k) / k, mode="same")
    lo_i, hi_i = int(0.03 * len(smooth)), int(0.92 * len(smooth))
    climax_pos = float(np.arange(len(smooth))[lo_i:hi_i][np.argmax(smooth[lo_i:hi_i])] / fps / dur)
    lm = librosa.feature.melspectrogram(y=y, sr=22050, n_mels=48, hop_length=1024)
    lmdb = np.log(lm + 1e-6)
    cent = librosa.feature.spectral_centroid(y=y, sr=22050)[0]
    rows_sec = []
    cur_n, cur_s, cur_e, cur_c = None, [], [], []
    for i, t in enumerate(np.arange(len(db)) / fps):
        n_now = next((n for t1, t2, n in rows if t1 <= t < t2), cur_n)
        if n_now != cur_n:
            if cur_n and cur_s:
                rows_sec.append((cur_n, float(np.mean(cur_e)), float(np.mean(cur_c))))
            cur_n, cur_s, cur_e, cur_c = n_now, [], [], []
        cur_s.append(t)
        cur_e.append(db[i])
        cur_c.append(cent[min(i, len(cent) - 1)])
    if cur_n and cur_s:
        rows_sec.append((cur_n, float(np.mean(cur_e)), float(np.mean(cur_c))))
    sec_db = {}
    sec_cent = {}
    for n, e, c in rows_sec:
        sec_db.setdefault(n, []).append(e)
        sec_cent.setdefault(n, []).append(c)
    ch_db = [e for n, es in sec_db.items() if n == "chorus" for e in es]
    vs_db = [e for n, es in sec_db.items() if n == "verse" for e in es]
    arc = round(float(np.mean(ch_db) - np.mean(vs_db)), 1) if ch_db and vs_db else None
    ch_c = [c for n, cs in sec_cent.items() if n == "chorus" for c in cs]
    vs_c = [c for n, cs in sec_cent.items() if n == "verse" for c in cs]
    blift = round(float(np.mean(ch_c) - np.mean(vs_c))) if ch_c and vs_c else None
    contrast = 20.0 * ((min(max(arc or 0, 0), 4) / 4) * 0.6 + (min(blift or 0, 400) / 400) * 0.4)

    # 4) 结构重复
    chor = sorted([(t1, t2) for t1, t2, n in rows if n == "chorus"])
    inst = []
    for t1, t2 in chor:
        if inst and t1 - inst[-1][1] < 1.5:
            inst[-1] = (inst[-1][0], max(inst[-1][1], t2))
        else:
            inst.append((t1, t2))
    struct = 15.0 * min(1.0, len(inst) / 2) * (1.0 if climax_pos >= 0.55 else max(0.3, climax_pos / 0.55))

    # 5) 歌词共鸣词法
    body = "".join(lines)
    pron = len(PRON.findall(body)) / max(len(body), 1) * 100
    imagery = len(set(c for c in body if c in IMAGERY))
    lexc = 10.0 * min(1.0, pron / 6) * 0.5 + 10.0 * min(1.0, imagery / 6) * 0.5

    # 6) 时长
    dur_s = 10.0 if 180 <= dur <= 280 else max(0.0, 10 - abs(dur - 230) / 12)

    total = round(entry + hook + contrast + struct + lexc + dur_s, 1)
    return dict(no=no, title=title, total=total,
                dims=dict(入场=round(entry, 1), hook=round(hook, 1), 对比=round(contrast, 1),
                          结构=round(struct, 1), 词法=round(lexc, 1), 时长=round(dur_s, 1)),
                facts=dict(first_vocal=round(fv, 1), first_chorus=round(fc, 1), dur=round(dur),
                           hook_count=hook_count, hook_per_min=hook_per_min,
                           pron_per100=round(pron, 1), imagery=imagery,
                           arc=arc, bright_lift=blift, climax_pos=round(climax_pos, 2),
                           chorus_instances=len(inst)))


def main():
    ref = {k: v for k, v in BASE.items() if not isinstance(v, list)}
    rows = []
    for s in SONGS:
        r = score_song(s)
        rows.append(r)
    rows.sort(key=lambda r: -r["total"])
    lines = ["# 爆款指数报告（爆款相似度，曲库=已验证发行金曲）", "",
             f"基准：开嗓≤{BASE['first_vocal_p75']}s(P75) · 进副歌≤{BASE['first_chorus_med']}s · "
             f"开场能量 {BASE['e30_ratio_med']}dB · 时长 {BASE['dur_med']}s", "",
             "| 排名 | 曲目 | **爆款指数** | 入场 | Hook | 对比 | 结构 | 词法 | 时长 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        d = r["dims"]
        lines.append(f"| {i} | {r['no']} {r['title']} | **{r['total']}** | {d['入场']} | {d['hook']} "
                     f"| {d['对比']} | {d['结构']} | {d['词法']} | {d['时长']} |")
    lines += ["", "## 逐曲事实与前三条处方", ""]
    # ===== 公平性标定：75分音频子量表 vs 曲库 =====
    try:
        from hit_compare import subscore, sf_info_dur
        cs = []
        manifest = json.loads((ROOT / "corpus" / "sample_manifest.json").read_text(encoding="utf-8"))
        for m in manifest:
            try:
                tdir_c = ROOT / "corpus" / "transcriptions" / Path(m["wav"]).stem
                cs.append(subscore(Path(m["wav"]), tdir_c, sf_info_dur(m["wav"]))["total"])
            except Exception:
                pass
        cs = np.array(cs)
        lines.append(f"### 标定（曲库 32 首同子量表：中位 {np.median(cs):.1f} · 最大 {cs.max():.1f}）")
        lines.append("**重要**：指数本质是'流行工程规范度'。AI 按规格生成，结构维度可超真曲满分"
                     "（规整≠爆款）——爆款与否最终由金句+人耳+市场决定。")
        for r in rows:
            sub = 0
            tdir_s = TRANSCRIPTS[r["no"]]
            try:
                flac = ROOT / "novel_songs" / "final" / f"{r['no']}_{r['title']}.flac"
                sub = subscore(flac, tdir_s, sf_info_dur(flac))["total"]
            except Exception:
                pass
            pct = round(float((cs < sub).mean() * 100), 1)
            lines.append(f"- {r['no']} {r['title']}: 音频子量表 {sub}/75 → 超过曲库 **{pct}%** 的真曲")
    except Exception as e:
        lines.append(f"(标定失败: {str(e)[:80]})")
    lines += ["", "## 逐曲事实与前三条处方", ""]
    for r in rows:
        f = r["facts"]
        lines.append(f"### {r['no']} {r['title']}（指数 {r['total']}）")
        lines.append(f"- 事实: 开嗓 {f['first_vocal']}s · 进副歌 {f['first_chorus']}s · 时长 {f['dur']}s · "
                     f"hook×{f['hook_count']} ({f['hook_per_min']}/min) · 你/我 {f['pron_per100']}/百字 · "
                     f"意象 {f['imagery']} 种 · 副歌段 {f['chorus_instances']}")
        # 处方：找最弱两维
        dims = r["dims"]
        weak = sorted(dims.items(), key=lambda kv: kv[1])[:2]
        advice = {
            "入场": "开嗓晚：删短前奏（把 [Intro] 交给 abc 减小节）或第一句主歌提前进",
            "hook": "歌名曝光不足：副歌首次出现前先来一句 hook 预埋（pre-chorus 尾部），或增加结尾重复",
            "对比": "副歌对比弱：plan-edit 抬副歌/压主歌，或 style 加 'dynamic contrast'",
            "结构": "高潮位置/副歌次数不足：增加一次最终副歌，或把 bridge 后的爆发提前",
            "词法": "词法单薄：增加 你/我 对话感与具体意象（雨/灯/手/街…）",
            "时长": "时长偏离 3-4.5 分钟流媒体甜区",
        }
        for wname, wv in weak:
            if wv < 15:
                lines.append(f"- 处方[{wname}]: {advice.get(wname, '见 map')}")
        lines.append("")
    (ROOT / "eval" / "hit_score_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:22]))
    print("SAVED eval/hit_score_report.md")


if __name__ == "__main__":
    main()
