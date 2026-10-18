# -*- coding: utf-8 -*-
"""热度分层 + 指标-热度相关性分析 -> eval/popularity_correlation.md
热度分层基于艺人国民度+歌曲文化渗透率（互联网可验证）
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent


def cohen_d(a, b):
    a, b = np.array(a, float), np.array(b, float)
    s = np.sqrt(((len(a)-1)*a.var() + (len(b)-1)*b.var()) / (len(a)+len(b)-2))
    return float((a.mean() - b.mean()) / (s + 1e-9))


def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


sys.path.insert(0, str(ROOT))

# ===== 热度分层（S=国民级爆款 / A=知名热曲 / B=专辑曲或较小众） =====
# 依据：互联网公开播放量、排行榜历史、文化渗透度
MEGA_HITS = {  # S 级：国民级，播放量 10 亿+
    "青花瓷", "东风破", "菊花台", "发如雪", "夜曲", "七里香",
    "江南", "曹操", "一千年以后", "美人鱼",
    "隐形的翅膀", "偏爱", "死了都要爱", "离歌",
    "突然的自我", "知足", "温柔", "倔强",
    "Super Star", "不想长大", "恋人未满",
    "遇见", "天黑黑", "绿光",
    "月亮代表我的心", "甜蜜蜜",
    "后来", "为爱痴狂",
    "朋友", "笨鸟",
    "第一次", "勇气",
    "老鼠爱大米", "两只蝴蝶",
    "秋天不回来", "不要在我寂寞的时候说爱我",
    "求佛", "香水有毒",
    "该死的温柔", "专属天使",
    "北极星的眼泪", "我难过",
    "倒带", "说爱你",
    "那些花儿", "生如夏花",
    "平凡之路", "董小姐",
    "成都", "南山南",
    "小苹果", "最炫民族风",
}

KNOWN_HITS = {  # A 级：知名热曲，播放量 1000 万+
    "雨蝶", "红色高跟鞋", "画出我世界", "恋爱ing", "憨人",
    "记得", "我最亲爱的", "开口", "日不落", "说好的幸福呢",
    "轨迹", "龙卷风", "简单爱", "晴天", "回到过去",
    "三国恋", "六月的雨", "逍遥叹", "一直很安静",
    "城府", "断桥残雪", "清明雨上", "有何不可",
    "三国杀", "小星星", "偏爱", "勇气", "会呼吸的痛",
    " 宁夏", "分手快乐", "不想懂得",
    "天灰", "我爱你", "热带雨林",
    "分开旅行", "bertie",
    "我难过", "真爱", "*C'est La Vie",
    "爱就爱了", "腐烂",
    "别在我离开之前离开", "人怕出名猪怕壮",
    "两个人的荒岛", "天亮了",
    "春秋配", "棠梨煎雪", "回到拉萨",
    "野生玫瑰", "孤独患者", "消灭",
    "爱笑的眼睛", "被风吹过的夏天",
    "遇见", "开始懂了", "神奇", "奔",
    "燕尾蝶", "分开不一定分手",
    "猪之歌", "我爱你再见",
}

# 艺人国民度加成（艺人本身就是热度的代理变量）
MEGA_ARTISTS = {"周杰伦", "林俊杰", "五月天", "张学友", "刘德华", "邓紫棋",
                "陈奕迅", "王菲", "张惠妹", "孙燕姿", "梁静茹", "S.H.E",
                "王心凌", "张韶涵", "Tank", "胡歌"}
KNOWN_ARTISTS = {"许嵩", "汪苏泷", "后弦", "黄雅莉", "张芸京", "山野",
                 "花儿乐队", "信乐团", "183Club", "5566", "By2", "吴克群",
                 "薛之谦", "毛不易", "陈琳", "香香", "唐磊", "汤潮"}


def tier_of(name):
    """返回热度层级: 3=国民爆款, 2=知名热曲, 1=一般"""
    # 提取歌名（去艺人名）
    parts = re.split(r"[-_—]", name)
    song_name = parts[-1].strip() if len(parts) > 1 else name
    song_clean = re.sub(r"[\(\（].*?[\)\）]|mp3|flac", "", song_name).strip()
    artist = parts[0].strip() if len(parts) > 1 else ""

    for hit in MEGA_HITS:
        if hit in name:
            return 3
    if artist in MEGA_ARTISTS:
        return 3
    for hit in KNOWN_HITS:
        if hit in name:
            return 2
    if artist in KNOWN_ARTISTS:
        return 2
    return 1


# ===== 加载全部数据 =====
feats = json.loads((ROOT / "corpus" / "features.json").read_text(encoding="utf-8"))
ms = json.loads((ROOT / "corpus" / "melody_stats.json").read_text(encoding="utf-8"))["per_song"]
lib_scores = json.loads((ROOT / "eval" / "hit_library_score.json").read_text(encoding="utf-8"))
res_cache = json.loads((ROOT / "eval" / "resonance_cache.json").read_text(encoding="utf-8"))
lib_cache = json.loads((ROOT / "eval" / "lib_metric_cache.json").read_text(encoding="utf-8"))

# ===== 全库打热度标签 =====
tiers = {}
for name in feats:
    tiers[name] = tier_of(name)

# ===== 分组统计 =====
tier_groups = {3: [], 2: [], 1: []}
for name, r in feats.items():
    t = tiers[name]
    tier_groups[t].append(r)

print(f"分层: S级={len(tier_groups[3])} A级={len(tier_groups[2])} B级={len(tier_groups[1])}")

# ===== 逐指标热度相关性 =====
def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)

lines = ["# 热度-指标相关性分析（568 首，S/A/B 三层）", "",
         f"- S 级（国民爆款）: {len(tier_groups[3])} 首",
         f"- A 级（知名热曲）: {len(tier_groups[2])} 首",
         f"- B 级（一般）: {len(tier_groups[1])} 首", ""]

# 按特征组逐一检验
tests = []

# 基础特征
for key, cname in [("bpm", "BPM"), ("centroid_hz", "频谱质心"), ("rms_db", "响度"),
                   ("dyn_p95_p10_db", "动态范围"), ("duration_s", "时长"), ("zcr", "过零率")]:
    vals = [(tiers[n], r[key]) for n, r in feats.items() if r.get(key) is not None and r[key] == r[key]]
    if len(vals) < 30:
        continue
    xs = [v[0] for v in vals]
    ys = [v[1] for v in vals]
    r = spearman(xs, ys)
    # 层间差异
    s_med = np.median([v[1] for v in vals if v[0] == 3]) if any(v[0] == 3 for v in vals) else 0
    b_med = np.median([v[1] for v in vals if v[0] == 1]) if any(v[0] == 1 for v in vals) else 0
    d = cohen_d([v[1] for v in vals if v[0] == 3], [v[1] for v in vals if v[0] == 1])
    tests.append((cname, r, d, f"S中位={s_med:.1f} B中位={b_med:.1f}"))

# 旋律特征（31 首子集）
ms_data = []
for r in ms:
    stem = r["file"]
    match = [m for m in json.loads((ROOT / "corpus" / "sample_manifest.json").read_text(encoding="utf-8"))
             if Path(m["wav"]).stem == stem]
    if not match:
        continue
    name = match[0]["file"]
    t = tiers.get(name, 1)
    ms_data.append((t, r))

for key, cname in [("density", "音符密度"), ("pitch_range", "音域"), ("leap_ratio", "跳进率"),
                   ("motif_repeat", "motif重复"), ("lift_mean", "副歌抬升"), ("rest_ratio", "休止占比")]:
    vals = [(t, r[key]) for t, r in ms_data if r.get(key) is not None and r[key] == r[key]]
    if len(vals) < 15:
        continue
    xs = [v[0] for v in vals]
    ys = [v[1] for v in vals]
    r = spearman(xs, ys)
    s_med = np.median([v[1] for v in vals if v[0] == 3]) if any(v[0] == 3 for v in vals) else 0
    b_med = np.median([v[1] for v in vals if v[0] == 1]) if any(v[0] == 1 for v in vals) else 0
    d = cohen_d([v[1] for v in vals if v[0] == 3], [v[1] for v in vals if v[0] == 1])
    tests.append((f"旋律·{cname}", r, d, f"S={s_med:.2f} B={b_med:.2f}"))

# 波形指标（463 首全库）
wv_cache = json.loads((ROOT / "eval" / "wave_metric_cache.json").read_text(encoding="utf-8"))
for key, cname in [("crest_db", "峰值因子"), ("low_attack_ms", "低频冲击"),
                   ("onset_iv_cv", "起音间隔CV"), ("lra_db", "响度域")]:
    vals = []
    for stem, r in wv_cache.items():
        if "crest_db" not in r:
            continue
        # 反查热度
        orig = stem.replace("lib_", "", 1)
        match = [n for n in feats if n.replace(" ", "_")[:len(orig)] == orig[:len(orig)]]
        if not match:
            match = [n for n in feats if orig[:8] in n]
        if not match:
            continue
        t = tiers[match[0]]
        vals.append((t, r[key]))
    if len(vals) < 50:
        continue
    xs = [v[0] for v in vals]
    ys = [v[1] for v in vals]
    r = spearman(xs, ys)
    s_med = np.median([v[1] for v in vals if v[0] == 3]) if any(v[0] == 3 for v in vals) else 0
    b_med = np.median([v[1] for v in vals if v[0] == 1]) if any(v[0] == 1 for v in vals) else 0
    d = cohen_d([v[1] for v in vals if v[0] == 3], [v[1] for v in vals if v[0] == 1])
    tests.append((f"波形·{cname}", r, d, f"S={s_med:.1f} B={b_med:.1f}"))

# ===== 输出 =====
lines += ["## 热度-指标相关性（Spearman ρ + S/B 层 Cohen's d）", "",
          "| 指标 | ρ(热度,指标) | d(S vs B) | S/B 中位 | 判定 |", "|---|---|---|---|---|"]
for name, r, d, extra in sorted(tests, key=lambda t: -abs(t[1])):
    if abs(r) >= 0.15:
        g = "**✅ 相关**"
    elif abs(r) >= 0.08:
        g = "🟡 弱趋势"
    else:
        g = "❌ 无关"
    lines.append(f"| {name} | {r:+.3f} | {d:+.2f} | {extra} | {g} |")

lines += ["", "## 发现", ""]
sig = [(n, r, d, e) for n, r, d, e in tests if abs(r) >= 0.12]
if sig:
    for n, r, d, e in sig:
        direction = "爆款更高" if r > 0 else "爆款更低"
        lines.append(f"- **{n}**（ρ={r:+.3f}）: {direction}")
else:
    lines.append("- 在当前特征集内，未发现与热度强相关的基础声学指标")
    lines.append("- **关键洞察：好歌和普通歌在声学特征上几乎无差异——差异在旋律写作、歌词、和声，不在制作层**")

(ROOT / "eval" / "popularity_correlation.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
