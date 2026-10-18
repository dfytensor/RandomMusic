# -*- coding: utf-8 -*-
"""曲库 vs YuE2 生成曲 典型差异分析（声学/旋律/MERT 分布三层）-> eval/difference_report.md"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent


def stats(vals):
    a = np.array([v for v in vals if v is not None], float)
    if len(a) == 0:
        return None
    return dict(p25=round(float(np.percentile(a, 25)), 3), med=round(float(np.median(a)), 3),
                p75=round(float(np.percentile(a, 75)), 3))


def cmp_table(name, corpus_vals, novel_vals, better="mid", fmt="{:.3f}"):
    cs, nv = stats(corpus_vals), stats(novel_vals)
    nmed = float(np.median([v for v in novel_vals if v is not None]))
    inside = cs["p25"] <= nmed <= cs["p75"]
    return dict(metric=name, corpus=cs, novel_med=round(nmed, 3), in_iqr=inside, fmt=fmt)


def main():
    fc = {r["file"]: r for r in json.loads((ROOT / "eval" / "feat_corpus.json").read_text(encoding="utf-8"))}
    fn = json.loads((ROOT / "eval" / "feat_novel.json").read_text(encoding="utf-8"))
    mc = json.loads((ROOT / "corpus" / "melody_stats.json").read_text(encoding="utf-8"))["per_song"]
    mn = json.loads((ROOT / "eval" / "novel_melody.json").read_text(encoding="utf-8"))
    zc = np.load(ROOT / "eval" / "embeddings.npz", allow_pickle=True)
    zn = np.load(ROOT / "eval" / "novel_embeddings.npz", allow_pickle=True)
    C = zc["embs"][[i for i, g in enumerate(zc["groups"]) if g == "corpus"]]
    N = zn["embs"]

    lines = ["# 生成曲 vs 曲库同类曲：典型差异分析", "",
             f"样本：曲库 {len(fc)} 首（声学）/ {len(mc)} 首（旋律）/ {len(C)} 首（MERT） vs 生成曲 12 首", ""]

    # ---------- 1. 声学制作层 ----------
    def col(rows, key, sub=None):
        if sub:
            return [r[sub][key] if isinstance(r.get(sub), list) else None for r in rows]
        return [r.get(key) for r in rows]

    band_names = ["低频20-250", "中低250-2k", "中高2k-6k", "高频6k+"]
    audio_metrics = [
        ("频段占比·低频", [r["band_share"][0] for r in fc.values()], [r["band_share"][0] for r in fn]),
        ("频段占比·中低", [r["band_share"][1] for r in fc.values()], [r["band_share"][1] for r in fn]),
        ("频段占比·中高", [r["band_share"][2] for r in fc.values()], [r["band_share"][2] for r in fn]),
        ("频段占比·高频", [r["band_share"][3] for r in fc.values()], [r["band_share"][3] for r in fn]),
        ("起音密度(个/秒)", [r["onset_density"] for r in fc.values()], [r["onset_density"] for r in fn]),
        ("起音强度", [r["onset_strength"] for r in fc.values()], [r["onset_strength"] for r in fn]),
        ("立体声宽度", [r["stereo_width"] for r in fc.values()], [r["stereo_width"] for r in fn]),
        ("频谱质心(Hz)", [r["centroid_hz"] for r in fc.values()], [r["centroid_hz"] for r in fn]),
        ("动态p95-p10(dB)", [r["dyn_p95_p10_db"] for r in fc.values()], [r["dyn_p95_p10_db"] for r in fn]),
        ("BPM", [r["bpm"] for r in fc.values()], [r["bpm"] for r in fn]),
    ]
    lines += ["## 1. 声学制作层（90s 窗，librosa）", "",
              "| 指标 | 曲库 P25 | 中位 | P75 | 生成曲中位 | 出区间 |", "|---|---|---|---|---|---|"]
    audio_rows = []
    for name, cv, nv in audio_metrics:
        row = cmp_table(name, cv, nv)
        audio_rows.append(row)
        mark = "" if row["in_iqr"] else " **❗**"
        lines.append(f"| {name} | {row['corpus']['p25']} | {row['corpus']['med']} | {row['corpus']['p75']} "
                     f"| {row['novel_med']} |{mark}|")

    # ---------- 2. 旋律层 ----------
    melody_metrics = [
        ("音符密度(音/小节)", [r["density"] for r in mc], [r["density"] for r in mn]),
        ("人声音域(半音)", [r["pitch_range"] for r in mc], [r["pitch_range"] for r in mn]),
        ("跳进率", [r["leap_ratio"] for r in mc], [r["leap_ratio"] for r in mn]),
        ("motif重复率", [r["motif_repeat"] for r in mc], [r["motif_repeat"] for r in mn]),
        ("副歌抬升(半音)", [r["lift_mean"] for r in mc], [r["lift_mean"] for r in mn]),
        ("人声休止占比", [r["rest_ratio"] for r in mc], [r["rest_ratio"] for r in mn]),
    ]
    lines += ["", "## 2. 旋律层（SheetSage2 转谱）", "",
              "| 指标 | 曲库 P25 | 中位 | P75 | 生成曲中位 | 出区间 |", "|---|---|---|---|---|---|"]
    melody_rows = []
    for name, cv, nv in melody_metrics:
        row = cmp_table(name, cv, nv)
        melody_rows.append(row)
        mark = "" if row["in_iqr"] else " **❗**"
        lines.append(f"| {name} | {row['corpus']['p25']} | {row['corpus']['med']} | {row['corpus']['p75']} "
                     f"| {row['novel_med']} |{mark}|")

    # ---------- 3. MERT 分布层 ----------
    mu_c = C.mean(0)
    X = C - mu_c
    cov = X.T @ X / (len(C) - 1)
    lam = 0.1 * np.trace(cov) / len(mu_c)
    cov_s = 0.9 * cov + lam * np.eye(len(mu_c))
    inv = np.linalg.inv(cov_s)

    def maha(e):
        return float(np.sqrt(max((e - mu_c) @ inv @ (e - mu_c), 0)))

    def top5cos(e):
        cos = (C @ e) / (np.linalg.norm(C, axis=1) * np.linalg.norm(e) + 1e-9)
        return float(np.sort(cos)[-5:].mean())

    # 判别轴：类均值差方向
    axis = (N.mean(0) - C.mean(0))
    axis /= (np.linalg.norm(axis) + 1e-9)
    proj_c = C @ axis
    proj_n = N @ axis
    pooled = np.concatenate([proj_c, proj_n])
    d = (proj_n.mean() - proj_c.mean()) / (pooled.std() + 1e-9)

    # 判别轴与声学特征的相关（归因：这条轴代表什么）
    all_names = list(fc.keys()) + [r["file"] for r in fn]
    all_proj = np.concatenate([proj_c, proj_n])
    attrib = {}
    for key, label in [("onset_density", "起音密度"), ("stereo_width", "立体声宽度"),
                       ("centroid_hz", "频谱质心"), ("rms_db", "响度")]:
        vals = np.array([ (fc.get(n) or fn_get(fn, n)).get(key) for n in all_names], float)
        r = float(np.corrcoef(all_proj, vals)[0, 1])
        attrib[label] = round(r, 2)
    band_arr = np.array([(fc.get(n) or fn_get(fn, n))["band_share"] for n in all_names], float)
    for bi, bn in enumerate(band_names):
        attrib[f"频段·{bn}"] = round(float(np.corrcoef(all_proj, band_arr[:, bi])[0, 1]), 2)

    lines += ["", "## 3. MERT-v2 分布层", ""]
    maha_n = [maha(e) for e in N]
    cos_n = [top5cos(e) for e in N]
    maha_c = [maha(e) for e in C]
    cos_c = [top5cos(e) for e in C]
    lines += [f"- 判别轴分离度 Cohen's d = **{d:.2f}**（0.8=大效应）",
              f"- 曲库内：马氏中位 {np.median(maha_c):.0f} · top5余弦中位 {np.median(cos_c):.3f}",
              f"- 生成曲：马氏中位 **{np.median(maha_n):.0f}** · top5余弦中位 **{np.median(cos_n):.3f}**", "",
              "### 判别轴归因（投影值与声学特征的相关）", "",
              "| 声学特征 | 与判别轴相关 r |", "|---|---|"]
    for k, v in sorted(attrib.items(), key=lambda kv: -abs(kv[1])):
        lines.append(f"| {k} | {v:+.2f} |")
    lines += ["", "### 逐曲自然度", "", "| 曲目 | 马氏距离↓ | top5余弦↑ |", "|---|---|---|"]
    for name, mh, cs in zip(zn["names"], maha_n, cos_n):
        lines.append(f"| {name} | {mh:.0f} | {cs:.3f} |")

    # ---------- 4. 典型差异结论 ----------
    out_rows = [r for r in audio_rows + melody_rows if not r["in_iqr"]]
    lines += ["", "## 4. 典型差异结论（生成曲中位数落在曲库 IQR 之外项）", ""]
    if out_rows:
        for r in out_rows:
            direction = "偏高" if r["novel_med"] > r["corpus"]["med"] else "偏低"
            lines.append(f"- **{r['metric']}** {direction}：生成中位 {r['novel_med']} vs 曲库中位 {r['corpus']['med']}")
    else:
        lines.append("- 一阶统计量均在曲库 IQR 内：差异藏在分布形状与音色微结构（见 MERT 判别轴）")
    lines += [
        "",
        f"- 判别轴最强相关：{max(attrib.items(), key=lambda kv: abs(kv[1]))[0]} "
        f"(r={max(attrib.items(), key=lambda kv: abs(kv[1]))[1]:+.2f}) —— 这是 YuE2 与真曲最大的可解释差异方向",
        "- 注：马氏/余弦为单曲'自然度'；FAD 为批量分布距离（见 quality_report.md）",
    ]
    report = "\n".join(lines)
    (ROOT / "eval" / "difference_report.md").write_text(report, encoding="utf-8")
    print(report)


def fn_get(fn, n):
    stem = "novel_" + Path(n).stem.replace(" ", "_")
    for r in fn:
        if r["file"].startswith(stem.replace("novel_", "")) or Path(n).stem in r["file"]:
            return r
    return fn[0]


if __name__ == "__main__":
    main()
