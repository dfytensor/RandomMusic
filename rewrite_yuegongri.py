# -*- coding: utf-8 -*-
"""Prosody 方法论示范：重写《月供日》歌词
对比旧版 → 展示三层翻译法 + 感知词替换 + Prosody 的实际效果
"""
import json, re, sys
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS

s = next(x for x in SONGS if x["no"] == "08")

# ===== 旧版歌词 =====
OLD = s["lyrics"]

# ===== 新版（Prosody 方法论重写）=====
# 情绪锚点：在还债的日子里找到了活着的感觉
# 韵方案：主歌由求(ou)透气，副歌江阳(ang)强韵锁记忆
# Prosody：主歌参差（内心散乱）→ 副歌整齐（决心的秩序感）

NEW = """[Intro]

[Verse]
更鼓敲过第三轮
我把灯芯挑亮了一寸
灵砂在掌心里凉了
又焐热了 跟心跳一个节奏
隔壁的咳嗽停了
整条街只剩我在走
水牌上的名字被风吹
吹到哪个去了 也不回头

[Pre-Chorus]
(build) 月供日的月亮特别亮
亮得让人不敢想
明天的事明天再讲
今晚先把利息偿

[Chorus]
月供日 月亮催着我还
还完这期 还有下期的账
可街角的糕点摊还热着
热得我忘了身上有多凉
月供日 活着就是利息
我把自己 按时交上去
换她那碗 还冒着热气的药
(breathy) 药比蜜甜

[Interlude]

[Verse]
石头的坏消息又到了
嚼碎了咽 竟有点回甘
婆婆把药煎成了两碗
一碗给我 一碗给晚归的遗憾
[Pre-Chorus]
(build) 月亮爬上了城墙
照着还没睡的人的窗
我把算盘拨回原位
把叹息也拨回了胸膛

[Chorus]
月供日 月亮催着我还
还完这期 还有下期的账
可街角的糕点摊还热着
热得我忘了身上有多凉
月供日 活着就是利息
我把自己 按时交上去
换她那碗 还冒着热气的药
(breathy) 药比蜜甜

[Bridge]
(whisper) 天快亮了
货架空了 心满了

[Chorus]
月供日 月亮催着我还
还完这期 还有下期的账
可街角的糕点摊还热着
热得我忘了身上有多凉
月供日 活着就是利息
我把自己 按时交上去
换她那碗 还冒着热气的药
(breathy) 药比蜜甜

[Outro]
"""

# ===== 对比分析 =====
def analyze_lyrics(text, label):
    lines = [l.strip() for l in text.splitlines()
             if l.strip() and not l.strip().startswith("[")]
    lens = [len(re.findall(r"[\u4e00-\u9fff]", l)) for l in lines]
    lens = [x for x in lens if x > 0]
    
    # 韵脚
    from collections import Counter
    finals = []
    for l in lines:
        han = re.findall(r"[\u4e00-\u9fff]", l)
        if han:
            try:
                from pypinyin import pinyin, Style
                py = pinyin(han[-1], style=Style.FINALS, strict=False)
                if py and py[0]:
                    finals.append(py[0][0])
            except: pass
    
    # 具体名词（画面感）
    concrete = sum(1 for l in lines
                   for c in ["灯","街","门","桥","手","药","糕","鼓","墙","窗","砂","算盘"]
                   if c in l)
    
    print(f"\n{label}:")
    print(f"  行数: {len(lines)}  行长中位: {np.median(lens):.0f}字")
    print(f"  具体名词命中: {concrete}")
    print(f"  前5行: {lines[:5]}")

import re
analyze_lyrics(OLD, "旧版")
analyze_lyrics(NEW, "新版")

# 保存新版
out = ROOT / "novel_songs" / "v2" / "yuegongri_prosody"
out.mkdir(parents=True, exist_ok=True)
(out / "lyrics_v2.txt").write_text(NEW, encoding="utf-8")
print("\n新歌词已保存", flush=True)
