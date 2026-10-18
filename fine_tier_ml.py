# -*- coding: utf-8 -*-
"""10-11 细分层热度分类 ML 分析"""
import json, re, sys, warnings
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
import sys as _sys
_sys.path.insert(0, str(ROOT := Path(__file__).parent))

warnings.filterwarnings("ignore")

# ===== 细分层定义（11 类） =====
# S1: 超级国民爆款（国民级现象曲）
# S2: 大热单曲（顶级艺人代表作）
# S3: 热门主打（大牌艺人热门歌）
# A1: 影视/综艺热曲
# A2: 知名热曲（中小艺人热单曲）
# B1: 大牌专辑曲（知名艺人的非主打）
# B2: 良好专辑曲（中等艺人较好专辑曲）
# B3: 网络热曲（网络走红）
# B4: 影视剧OST（非头部艺人）
# B5: 一般专辑曲（小艺人专辑曲）
# B6: 小众曲（基本无热度的非主打）

MEGA_ARTISTS = {"周杰伦","林俊杰","五月天","张学友","刘德华","邓紫棋","陈奕迅","王菲",
                "张惠妹","孙燕姿","梁静茹","S.H.E","王心凌","张韶涵","Tank","胡歌"}
TOP_TIER = {"周杰伦", "林俊杰", "五月天", "张学友"}  # 最顶级艺人
KNOWN_ARTISTS = {"许嵩","汪苏泷","后弦","黄雅莉","张芸京","山野","花儿乐队","信乐团",
                 "183Club","5566","By2","吴克群","薛之谦","毛不易","陈琳","香香","唐磊","汤潮"}
MEGA_SONGS = {"青花瓷","东风破","菊花台","发如雪","夜曲","七里香","江南","曹操",
              "一千年以后","隐形的翅膀","偏爱","死了都要爱","离歌","突然的自我",
              "知足","温柔","倔强","Super Star","不想长大","遇见","天黑黑","绿光",
              "月亮代表我的心","甜蜜蜜","后来","为爱痴狂","朋友","第一次","勇气",
              "老鼠爱大米","两只蝴蝶","秋天不回来","求佛","香水有毒","该死的温柔",
              "专属天使","北极星的眼泪","我难过","倒带","说爱你","那些花儿",
              "生如夏花","平凡之路","董小姐","成都","南山南","小苹果","最炫民族风"}
HIT_SONGS = {"雨蝶","红色高跟鞋","画出我世界","恋爱ing","憨人","记得","我最亲爱的",
             "开口","日不落","说好的幸福呢","轨迹","龙卷风","简单爱","晴天","回到过去",
             "三国恋","六月的雨","逍遥叹","一直很安静","城府","断桥残雪","清明雨上",
             "有何不可","三国杀","小星星","勇气","会呼吸的痛","宁夏","分手快乐",
             "天灰","我爱你","热带雨林","分开旅行"}
OST_KEYWORDS = ["电视剧","电影","主题曲","插曲","片头","片尾","OST"]
NET_SONGS = ["老鼠爱大米","两只蝴蝶","秋天不回来","求佛","香水有毒","该死的温柔"]

def artist_of(name):
    parts = re.split(r"[-_—]", name)
    return parts[0].strip() if parts else ""

def song_of(name):
    parts = re.split(r"[-_—]", name)
    return parts[-1].strip() if len(parts) > 1 else name

def fine_tier(name):
    artist = artist_of(name)
    song = song_of(name)
    is_ost = any(k in name for k in OST_KEYWORDS)
    is_net = any(k in name for k in NET_SONGS)
    is_mega_artist = artist in MEGA_ARTISTS
    is_top_artist = artist in TOP_TIER
    is_known_artist = artist in KNOWN_ARTISTS
    is_mega_song = any(h in name for h in MEGA_SONGS)
    is_hit_song = any(h in name for h in HIT_SONGS)

    # S1: 顶级艺人的超级爆款
    if is_top_artist and (is_mega_song or is_hit_song):
        return "S1"
    # S2: 大牌艺人的热单曲
    if is_mega_artist and is_mega_song:
        return "S2"
    # S3: 大牌艺人的热门歌（非现象级但很热）
    if is_mega_artist and (is_hit_song or is_mega_song):
        return "S3"
    if is_mega_artist:
        return "S3"
    # A1: 影视/综艺热曲（大牌艺人）
    if is_ost and is_mega_artist:
        return "A1"
    # A2: 中小艺人知名热曲
    if is_hit_song or (is_known_artist and is_mega_song):
        return "A2"
    # B1: 大牌艺人非主打
    if is_mega_artist:
        return "B1"
    # B2: 中等艺人较好专辑曲
    if is_known_artist:
        return "B2"
    # B3: 网络热曲
    if is_net:
        return "B3"
    # B4: 影视剧OST（非头部）
    if is_ost:
        return "B4"
    # B5: 一般专辑曲
    return "B5"


# ===== 加载数据 =====
feats = json.loads((ROOT/"corpus"/"features.json").read_text(encoding="utf-8"))
fp_cache = json.loads((ROOT/"eval"/"lib_metric_cache.json").read_text(encoding="utf-8"))
wv_cache = json.loads((ROOT/"eval"/"wave_metric_cache.json").read_text(encoding="utf-8"))

X, y, names, fn_all = [], [], [], None
for name, r in sorted(feats.items()):
    clean = name.replace(" ", "_")
    t = fine_tier(name)
    
    vec, fn = [], []
    for k in sorted(r.keys()):
        if k == "file": continue
        v = r[k]
        if isinstance(v, (int, float)) and v == v and not np.isinf(v):
            vec.append(float(v)); fn.append(k)
    
    wav_stem = "lib_" + clean[:60]
    for cache, prefix in [(fp_cache, "fp_"), (wv_cache, "wf_")]:
        d = cache.get(wav_stem, {})
        if d.get("skip"): continue
        for k in sorted(d.keys()):
            v = d[k]
            if isinstance(v, (int, float)) and v == v:
                vec.append(float(v)); fn.append(prefix + k)
    
    if not fn_all:
        fn_all = fn
    if len(vec) == len(fn_all):
        X.append(vec); y.append(t); names.append(name)

X = np.array(X); y = np.array(y)
from collections import Counter
tc = Counter(y)
print(f"样本: {X.shape[0]}  特征: {X.shape[1]}  类别数: {len(tc)}", flush=True)
print("分布:", dict(sorted(tc.items())), flush=True)

# ===== ML =====
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
pred = cross_val_predict(rf, X, y, cv=5)
acc = float(np.mean(pred == y))
# 随机基线
baseline = max(tc.values()) / len(y)
print(f"准确率: {acc*100:.1f}% (随机最大类基线: {baseline*100:.1f}%)", flush=True)
print(f"提升: +{(acc-baseline)*100:.1f}%", flush=True)

rf.fit(X, y)
imp = rf.feature_importances_
idx = np.argsort(imp)[::-1]

# ===== 逐类别准确率 =====
print("\n逐类别准确率:", flush=True)
for cls in sorted(tc):
    mask = y == cls
    cls_acc = float(np.mean(pred[mask] == cls))
    print(f"  {cls}: {cls_acc*100:.1f}% (n={mask.sum()})", flush=True)

# ===== 输出 =====
fn_names = fn_all if fn_all else [f"f{i}" for i in range(X.shape[1])]
lines = [f"# 细分层热度分类（{len(tc)} 类）", "",
         f"样本: {X.shape[0]} · 特征: {X.shape[1]} · 类别: {len(tc)}",
         f"准确率: **{acc*100:.1f}%**（最大类基线 {baseline*100:.0f}%）", "",
         "| 排名 | 特征 | 重要性 |", "|---|---|---|"]
for i in range(min(25, len(idx))):
    fn = fn_names[idx[i]] if idx[i] < len(fn_names) else "?"
    lines.append(f"| {i+1} | {fn} | {imp[idx[i]]:.4f} |")
lines += ["", "## 逐类别准确率", ""]
for cls in sorted(tc):
    mask = y == cls
    ca = float(np.mean(pred[mask] == cls))
    lines.append(f"- {cls}: {ca*100:.1f}% (n={mask.sum()})")
lines += ["", "## 类别分布", ""]
for cls in sorted(tc):
    lines.append(f"- {cls}: {tc[cls]} 首")

(ROOT/"eval"/"fine_tier_ml.md").write_text("\n".join(lines), encoding="utf-8")
print("\nSAVED", flush=True)
