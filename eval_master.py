# -*- coding: utf-8 -*-
"""综合质量评估 v2：MERT 分布距离(FAD/自然度) + 能量弧线 + 和弦撞音 vs 语料基线。
输出 eval/quality_report.md。语料区间全部来自本地曲库提炼。
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent


def fad(X, Y, eps_scale=1e-4):
    mu1, mu2 = X.mean(0), Y.mean(0)
    s1 = np.cov(X, rowvar=False) + eps_scale * np.trace(np.cov(X, rowvar=False)) / X.shape[1] * np.eye(X.shape[1])
    s2 = np.cov(Y, rowvar=False) + eps_scale * np.trace(np.cov(Y, rowvar=False)) / Y.shape[1] * np.eye(Y.shape[1])
    d2 = float(((mu1 - mu2) ** 2).sum())
    # tr(s1) + tr(s2) - 2*tr(sqrtm(s1 s2)) 用特征值稳定计算
    from scipy.linalg import sqrtm
    tr = float(np.trace(s1) + np.trace(s2) - 2 * np.real(np.trace(sqrtm(s1 @ s2))))
    return d2 + max(tr, 0.0)


def naturalness(e, C, k=5):
    """与语料分布的距离：马氏距离(收缩协方差) + top-k 余弦。"""
    mu = C.mean(0)
    X = C - mu
    cov = X.T @ X / (len(C) - 1)
    lam = 0.1 * np.trace(cov) / len(mu)
    cov_s = 0.9 * cov + lam * np.eye(len(mu))
    try:
        inv = np.linalg.inv(cov_s)
        maha = float(np.sqrt(max((e - mu) @ inv @ (e - mu), 0)))
    except np.linalg.LinAlgError:
        maha = None
    cos = (C @ e) / (np.linalg.norm(C, axis=1) * np.linalg.norm(e) + 1e-9)
    topk = float(np.sort(cos)[-k:].mean())
    return maha, topk


def stats(vals):
    a = np.array([v for v in vals if v is not None], float)
    if len(a) == 0:
        return None
    return dict(p25=round(float(np.percentile(a, 25)), 2), med=round(float(np.median(a)), 2),
                p75=round(float(np.percentile(a, 75)), 2))


def main():
    z = np.load(ROOT / "eval" / "embeddings.npz", allow_pickle=True)
    names, embs, groups = list(z["names"]), z["embs"], list(z["groups"])
    C = embs[[i for i, g in enumerate(groups) if g == "corpus"]]
    ac = json.loads((ROOT / "eval" / "arc_consonance.json").read_text(encoding="utf-8"))

    corpus_arc_ci = [r.get("arc_chorus_minus_intro_db") for r in ac["corpus"]]
    corpus_arc_cv = [r.get("arc_chorus_minus_verse_db") for r in ac["corpus"]]
    corpus_clash = [r["clash_rate"] for r in ac["corpus"]]
    base = dict(
        arc_ci=stats(corpus_arc_ci), arc_cv=stats(corpus_arc_cv), clash=stats(corpus_clash))

    creation_idx = [i for i, g in enumerate(groups) if g == "creation"]
    alias = {
        "creation::create_520": "creation::深夜便利店(520)",
        "creation::create_731": "creation::selfopt731",
        "creation::cover_jiangnan": "creation::cover江南",
        "creation::demo_tonight": "creation::今晚不眠(官方示例曲)",
    }
    Cr = embs[creation_idx]
    fad_score = fad(C, Cr)
    # 语料自一半 vs 另一半 = FAD 天然下限
    half = len(C) // 2
    fad_floor = fad(C[:half], C[half:])

    rows = []
    for i, idx in enumerate(creation_idx):
        name = str(names[idx])
        maha, topk = naturalness(embs[idx], C)
        acr = next((r for r in ac["creation"] if r["name"] == alias.get(name, name)), {})
        in_ci = base["arc_ci"]["p25"] - 5 <= acr.get("arc_chorus_minus_intro_db", 999) <= base["arc_ci"]["p75"] + 5
        in_cv = base["arc_cv"]["p25"] - 3 <= acr.get("arc_chorus_minus_verse_db", 999) <= base["arc_cv"]["p75"] + 3
        in_clash = acr.get("clash_rate", 9) <= base["clash"]["p75"] + 0.05
        rows.append(dict(name=name, maha=round(maha, 1) if maha else None, cos=topk,
                         arc_ci=acr.get("arc_chorus_minus_intro_db"), in_ci=in_ci,
                         arc_cv=acr.get("arc_chorus_minus_verse_db"), in_cv=in_cv,
                         clash=acr.get("clash_rate"), in_clash=in_clash))

    lines = [
        "# YuE2 生成质量综合评估（标尺=本地曲库）", "",
        "## 1. 分布距离（MERT-v2 嵌入，1024 维）", "",
        f"- 曲库内噪声下限 FAD(corpus_half1, corpus_half2) = **{fad_floor:.3f}**",
        f"- 生成曲 vs 曲库 FAD = **{fad_score:.3f}**（越接近下限越像真歌；当前 4 首 vs 31 首小样本，仅供参考）", "",
        "| 曲目 | 马氏距离↓ | 语料top5余弦↑ | 副歌-前奏(dB) | 副歌-主歌(dB) | 撞音率↓ | 判定 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ok = sum([r["in_ci"], r["in_cv"], r["in_clash"]])
        verdict = "✅" if ok == 3 else ("🟡" if ok == 2 else "❌")
        lines.append(f"| {r['name']} | {r['maha']} | {r['cos']:.3f} | {r['arc_ci']} ({'✅' if r['in_ci'] else '❌'}) "
                     f"| {r['arc_cv']} ({'✅' if r['in_cv'] else '❌'}) | {r['clash']} ({'✅' if r['in_clash'] else '❌'}) | {verdict} |")
    lines += [
        "", "## 2. 语料基线区间", "",
        f"- 弧线 副歌-前奏: {base['arc_ci']}", f"- 弧线 副歌-主歌: {base['arc_cv']}",
        f"- 撞音率: {base['clash']}", "",
        "> 撞音率 = 人声音符不在当前和弦三和弦±1半音邻音集内的时长占比（经过音豁免）。", "",
        f"综合判定: " + "; ".join(f"{r['name']} {sum([r['in_ci'], r['in_cv'], r['in_clash']])}/3" for r in rows),
    ]
    report = "\n".join(lines)
    (ROOT / "eval" / "quality_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
