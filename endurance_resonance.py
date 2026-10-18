# -*- coding: utf-8 -*-
"""耐听度 + 情感共鸣评测（final 12 首 vs 曲库基线）-> eval/endurance_report.md
耐听: 副歌一致性(重复中一致) · 循环回归率(重复与变化平衡) · 响度平台占比(听觉疲劳)
共鸣: 情绪弧线(副歌抬升/高潮位置/明暗对比) · 副歌间发展度
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
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS

TRANSCRIPTS = {
    "01": ROOT / "eval/novel_transcripts/novel_01_欠天地一笔",
    "02": ROOT / "eval/novel_transcripts/novel_02_平账人",
    "03": ROOT / "eval/novel_transcripts/novel_03_白户",
    "04": ROOT / "novel_songs/redo/04/verify/realized",
    "05": ROOT / "eval/novel_transcripts/novel_05_心疼说成省钱",
    "06": ROOT / "novel_songs/redo/06/verify/realized",
    "07": ROOT / "novel_songs/v2/novel_07_规矩/verify/realized",
    "08": ROOT / "novel_songs/v2/novel_08_月供日/fullcheck/realized_full",
    "09": ROOT / "eval/novel_transcripts/novel_09_断供",
    "10": ROOT / "eval/novel_transcripts/novel_10_糕与药包",
    "11": ROOT / "novel_songs/v2/novel_11_烂尾劫_full/verify/realized",
    "12": ROOT / "novel_songs/v2/novel_12_新契_full/verify/realized",
}
FULL_TRANSCRIPT = {"04", "06", "07", "08", "11", "12"}


def merged_sections(tdir):
    rows = []
    p = Path(tdir) / "structure.lab"
    if not p.exists():
        return [], []
    for line in p.read_text(encoding="utf-8").splitlines():
        q = line.split("\t")
        if len(q) >= 3:
            rows.append((float(q[0]), float(q[1]), q[2].strip().lower()))
    merged, order = {}, []
    for t1, t2, n in rows:
        if n in ("silence",):
            continue
        if n not in merged:
            merged[n] = [t1, t2]
            order.append(n)
        else:
            merged[n][1] = max(merged[n][1], t2)
    secs = [(n, v[0], v[1]) for n, v in merged.items()]
    # 副歌实例（未合并，相邻 <1.5s 的粘连）
    raw_ch = sorted([(t1, t2) for t1, t2, n in rows if n == "chorus"])
    inst = []
    for t1, t2 in raw_ch:
        if inst and t1 - inst[-1][1] < 1.5:
            inst[-1] = (inst[-1][0], max(inst[-1][1], t2))
        else:
            inst.append((t1, t2))
    return secs, inst


def mel_vectors(y, sr=SR):
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=48, hop_length=1024)
    return np.log(m + 1e-6)


def analyze(path, tdir, limit=None):
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    dur = len(y) / sr
    lim = min(dur, limit) if limit else dur
    y = y[: int(lim * SR)]
    secs, chorus_inst = merged_sections(tdir)
    secs = [(n, t1, min(t2, lim)) for n, t1, t2 in secs if t1 < lim]
    chorus_inst = [(max(t1, 0), min(t2, lim)) for t1, t2 in chorus_inst if t1 < lim]

    # 每秒 RMS dB 曲线
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    fps = SR / 512
    t_axis = np.arange(len(db)) / fps

    # 1) 情绪弧线：分段响度
    sec_db = {}
    for n, t1, t2 in secs:
        m = (t_axis >= t1) & (t_axis < t2)
        if m.sum() > 3:
            sec_db.setdefault(n, []).append(float(db[m].mean()))
    sec_mean = {n: float(np.mean(v)) for n, v in sec_db.items() if v}
    chorus_dbs = [sec_mean[n] for n in sec_mean if n == "chorus"]
    verse_dbs = [sec_mean[n] for n in sec_mean if n == "verse"]
    arc_lift = round(chorus_dbs[0] - verse_dbs[0], 1) if chorus_dbs and verse_dbs else None

    # 高潮位置：最大 5s 平滑响度的时间占比（排除结尾 8% 与开头 3%）
    k = int(5 * fps)
    smooth = np.convolve(db, np.ones(k) / k, mode="same")
    lo_i, hi_i = int(0.03 * len(smooth)), int(0.92 * len(smooth))
    climax_pos = round(float(t_axis[lo_i + int(np.argmax(smooth[lo_i:hi_i]))] / lim), 2)

    # 2) 疲劳：响度贴近峰值的时间占比
    p95 = np.percentile(db, 95)
    plateau = round(float((db > p95 - 2).mean()), 3)

    # 3) 副歌一致性：副歌实例两两 mel 均值向量余弦
    lm = mel_vectors(y)
    fps_m = lm.shape[1] / lim
    chor = [(t1, t2) for t1, t2 in chorus_inst if t2 - t1 > 8]
    cons = []
    for i in range(len(chor)):
        for j in range(i + 1, len(chor)):
            a = lm[:, int(chor[i][0] * fps_m):int(chor[i][1] * fps_m)]
            b = lm[:, int(chor[j][0] * fps_m):int(chor[j][1] * fps_m)]
            nfr = min(a.shape[1], b.shape[1])
            if nfr < 10:
                continue
            va, vb = a[:, :nfr].mean(1), b[:, :nfr].mean(1)
            cons.append(float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)))
    chorus_cons = round(float(np.mean(cons)), 3) if cons else None

    # 4) 回归率（重复与变化平衡）：SSM 高相似比例（lag>5s）
    step = max(1, int(0.5 * fps_m))
    lm_s = lm[:, ::step]
    nfr = lm_s.shape[1]
    if nfr > 40:
        nrm = np.linalg.norm(lm_s, axis=0) + 1e-9
        lmn = lm_s / nrm
        S = lmn.T @ lmn
        lag = int(5 * (fps_m / step))
        iu = np.triu_indices(nfr, k=lag)
        sims = S[iu]
        recurrence = round(float((sims > 0.85).mean()), 3)
    else:
        recurrence = None

    # 5) 明暗对比：副歌 vs 主歌 质心差
    cent = librosa.feature.spectral_centroid(y=y, sr=SR)[0]
    cent_ch, cent_v = [], []
    for n, t1, t2 in secs:
        m = (t_axis >= t1) & (t_axis < t2)
        if m.sum() > 3:
            (cent_ch if n == "chorus" else cent_v).append(float(cent[m].mean()))
    bright_lift = round(float(np.mean(cent_ch) - np.mean(cent_v)), 0) if cent_ch and cent_v else None

    return dict(dur_analyzed=round(lim, 1), arc_lift_db=arc_lift, climax_pos=climax_pos,
                plateau=plateau, chorus_consistency=chorus_cons, recurrence=recurrence,
                bright_lift_hz=bright_lift, n_sections=len(secs),
                full=(lim >= dur - 1))


def main():
    final = ROOT / "novel_songs" / "final"
    novel = {}
    for s in SONGS:
        flac = final / f"{s['no']}_{s['title']}.flac"
        lim = None if s["no"] in FULL_TRANSCRIPT else 120
        r = analyze(flac, TRANSCRIPTS[s["no"]], limit=lim)
        novel[f"{s['no']}_{s['title']}"] = r
        print(s["no"], s["title"], json.dumps(r, ensure_ascii=False), flush=True)

    # 曲库基线（32 首，100s 窗）
    corpus = []
    manifest = json.loads((ROOT / "corpus" / "sample_manifest.json").read_text(encoding="utf-8"))
    for m in manifest:
        stem = Path(m["wav"]).stem
        tdir = ROOT / "corpus" / "transcriptions" / stem
        try:
            corpus.append(analyze(m["wav"], tdir, limit=100))
        except Exception:
            pass

    def base(key):
        v = [r[key] for r in corpus if r.get(key) is not None]
        return dict(p25=round(float(np.percentile(v, 25)), 3), med=round(float(np.median(v)), 3),
                    p75=round(float(np.percentile(v, 75)), 3)) if v else None

    bases = {k: base(k) for k in ["arc_lift_db", "plateau", "chorus_consistency", "recurrence",
                                  "bright_lift_hz", "climax_pos"]}
    nv = {k: [r[k] for r in novel.values() if r.get(k) is not None] for k in bases}
    novel_med = {k: (round(float(np.median(v)), 3) if v else None) for k, v in nv.items()}

    lines = ["# 耐听度与情感共鸣评测（final 12 首 vs 曲库 32 首）", "",
             "## 曲库基线 vs 生成专辑中位", "",
             "| 指标 | 曲库 P25 | 中位 | P75 | 生成中位 |", "|---|---|---|---|---|"]
    for k, b in bases.items():
        if b is None:
            lines.append(f"| {k} | - | - | - | {novel_med.get(k)} |")
        else:
            lines.append(f"| {k} | {b['p25']} | {b['med']} | {b['p75']} | {novel_med.get(k)} |")

    lines += ["", "## 逐曲明细", "",
              "| 曲目 | 弧线抬升dB | 高潮位置 | 响度平台(疲劳) | 副歌一致性 | 回归率 | 明暗提升Hz |",
              "|---|---|---|---|---|---|---|"]
    for name, r in novel.items():
        lines.append(f"| {name} | {r['arc_lift_db']} | {r['climax_pos']} | {r['plateau']} | "
                     f"{r['chorus_consistency']} | {r['recurrence']} | {r['bright_lift_hz']} |")

    lines += ["", "## 判读规则", "",
              "- **耐听**：副歌一致性 0.75-0.97（重复中一致）· 回归率 0.10-0.45（重复与变化平衡）· 响度平台 <0.35（不轰头）",
              "- **共鸣**：弧线抬升 >4dB · 高潮位置 0.55-0.85（黄金分割）· 明暗提升 >200Hz",
              "- 曲库外 = 人工听检重点；本评测为结构代理，最终以人耳为准"]
    (ROOT / "eval" / "endurance_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
