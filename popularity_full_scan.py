# -*- coding: utf-8 -*-
"""全指标 × 热度相关性大扫描：所有缓存指标 vs 热度分层
-> eval/popularity_full_scan.md
"""
import json, re, sys, warnings
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

# ===== 热度函数 =====
MEGA_ARTISTS = {"周杰伦","林俊杰","五月天","张学友","刘德华","邓紫棋","陈奕迅","王菲",
                "张惠妹","孙燕姿","梁静茹","S.H.E","王心凌","张韶涵","Tank","胡歌"}
KNOWN_ARTISTS = {"许嵩","汪苏泷","后弦","黄雅莉","张芸京","山野","花儿乐队","信乐团",
                 "183Club","5566","By2","吴克群","薛之谦","毛不易","陈琳","香香","唐磊","汤潮"}
MEGA = {"青花瓷","东风破","菊花台","发如雪","夜曲","七里香","江南","曹操","一千年以后",
        "隐形的翅膀","偏爱","死了都要爱","离歌","突然的自我","知足","温柔","倔强",
        "Super Star","不想长大","恋人未满","遇见","天黑黑","绿光","月亮代表我的心",
        "甜蜜蜜","后来","为爱痴狂","朋友","第一次","勇气","老鼠爱大米","两只蝴蝶",
        "秋天不回来","求佛","香水有毒","该死的温柔","专属天使","北极星的眼泪","我难过",
        "倒带","说爱你","那些花儿","生如夏花","平凡之路","董小姐","成都","南山南","小苹果","最炫民族风"}

def tier_of(name):
    parts = re.split(r"[-_—]", name)
    artist = parts[0].strip() if parts else ""
    if artist in MEGA_ARTISTS: return 3
    for h in MEGA:
        if h in name: return 3
    if artist in KNOWN_ARTISTS: return 2
    return 1

# ===== 加载所有缓存 =====
feats = json.loads((ROOT/"corpus"/"features.json").read_text(encoding="utf-8"))
fp_cache = json.loads((ROOT/"eval"/"lib_metric_cache.json").read_text(encoding="utf-8"))
wv_cache = json.loads((ROOT/"eval"/"wave_metric_cache.json").read_text(encoding="utf-8"))
res_cache = json.loads((ROOT/"eval"/"resonance_cache.json").read_text(encoding="utf-8"))
ms_data = json.loads((ROOT/"corpus"/"melody_stats.json").read_text(encoding="utf-8"))["per_song"]

# 构建 unified: {orig_name: {metric: value, tier: int}}
unified = {}
for name, r in feats.items():
    u = {}
    for k, v in r.items():
        if isinstance(v, (int, float)) and v == v and not np.isinf(v):
            u[k] = float(v)
    u["tier"] = tier_of(name)
    u["_name"] = name
    unified[name] = u

# 挂载指纹缓存
for k, v in fp_cache.items():
    if v.get("skip") or "voiced_ratio" not in v:
        continue
    orig = k.replace("lib_", "", 1)
    match = next((n for n in unified if orig[:6] in n.replace(" ", "_")), None)
    if match:
        for kk, vv in v.items():
            if vv is not None and vv == vv:
                unified[match][f"fp_{kk}"] = float(vv)

# 挂载波形缓存
for k, v in wv_cache.items():
    orig = k.replace("lib_", "", 1)
    match = next((n for n in unified if orig[:6] in n.replace(" ", "_")), None)
    if match:
        for kk, vv in v.items():
            if vv is not None and vv == vv:
                unified[match][f"wf_{kk}"] = float(vv)

# 挂载共鸣缓存
for k, v in res_cache.items():
    orig = k.replace("lib_", "", 1)
    match = next((n for n in unified if orig[:6] in n.replace(" ", "_")), None)
    if match:
        for kk, vv in v.items():
            if vv is not None and vv == vv:
                unified[match][f"res_{kk}"] = float(vv)

# 挂载旋律统计（通过文件名模糊匹配）
for r in ms_data:
    stem = r["file"]
    match = next((n for n in unified if stem[:8] in n.replace(" ", "_")[:20]), None)
    if match:
        for kk in ("density", "pitch_range", "leap_ratio", "motif_repeat", "lift_mean", "rest_ratio"):
            if r.get(kk) is not None:
                unified[match][f"mel_{kk}"] = float(r[kk])

# ===== 逐指标 Spearman =====
all_metrics = set()
for u in unified.values():
    for k, v in u.items():
        if k not in ("tier", "_name") and isinstance(v, (int, float)):
            all_metrics.add(k)

results = []
for key in sorted(all_metrics):
    xs, ys = [], []
    for u in unified.values():
        t = u.get("tier")
        v = u.get(key)
        if t is None or v is None or v != v:
            continue
        xs.append(t)
        ys.append(float(v))
    if len(xs) < 50:
        continue
    rho, _ = spearmanr(xs, ys)
    # S vs B 效应量
    s_vals = [v for x, v in zip(xs, ys) if x == 3]
    b_vals = [v for x, v in zip(xs, ys) if x == 1]
    if len(s_vals) < 10 or len(b_vals) < 10:
        continue
    sa, ba = np.array(s_vals), np.array(b_vals)
    se = np.sqrt((sa.std()**2)/len(sa) + (ba.std()**2)/len(ba) + 1e-9)
    d = (sa.mean() - ba.mean()) / (se + 1e-9)
    results.append((key, round(rho, 3), round(d, 2), round(float(sa.mean()), 2), round(float(ba.mean()), 2)))

results.sort(key=lambda r: -abs(r[1]))

# ===== 输出 =====
lines = ["# 全指标 × 热度相关性大扫描", "",
         f"样本：{len(unified)} 首 | 热度分层：S={sum(1 for u in unified.values() if u['tier']==3)} "
         f"A={sum(1 for u in unified.values() if u['tier']==2)} B={sum(1 for u in unified.values() if u['tier']==1)}", "",
         "| 排名 | 指标 | ρ(热度) | d(S/B) | S层均值 | B层均值 | 判定 |", "|---|---|---|---|---|---|---|"]
for i, (key, rho, d, s_m, b_m) in enumerate(results, 1):
    if abs(rho) >= 0.15:
        g = "**✅ 显著相关**"
    elif abs(rho) >= 0.08:
        g = "🟡 弱趋势"
    else:
        g = "❌ 无关"
    lines.append(f"| {i} | {key} | {rho:+.3f} | {d:+.2f} | {s_m} | {b_m} | {g} |")

# 分组汇总
lines += ["", "## 显著相关（|ρ|≥0.15）", ""]
for key, rho, d, s_m, b_m in results:
    if abs(rho) >= 0.15:
        lines.append(f"- **{key}**: ρ={rho:+.3f} → {'爆款更高' if rho > 0 else '爆款更低'}")
lines += ["", "## 全部无关确认", ""]
no_corr = [(key, rho) for key, rho, *_ in results if abs(rho) < 0.08]
if no_corr:
    lines.append("以下指标与热度完全无关：")
    for key, rho in no_corr:
        lines.append(f"- {key}: ρ={rho}")

(ROOT / "eval" / "popularity_full_scan.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:50]))
print(f"\nTOTAL: {len(results)} metrics tested")
