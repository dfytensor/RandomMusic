# -*- coding: utf-8 -*-
"""精简版渲染：短歌词 + 子进程隔离"""
import json, os, sys, subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent

# 精简歌词（去 V2 + Bridge，保留核心三段）
LYRICS = """[Verse]
更鼓敲过第三轮
我把灯芯挑亮了一寸
灵砂在掌心里凉了
又焐热了 跟心跳一个节奏
隔壁的咳嗽停了
整条街只剩我在走
水牌上的名字被风吹
吹到哪个去了 也不回头

[Pre-Chorus]
月供日的月亮特别亮
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
药比蜜甜

[Interlude]

[Chorus]
月供日 月亮催着我还
还完这期 还有下期的账
可街角的糕点摊还热着
热得我忘了身上有多凉
月供日 活着就是利息
我把自己 按时交上去
换她那碗 还冒着热气的药
药比蜜甜

[Outro]
"""

STYLE = "Mandopop R&B pop, flowing piano, guzheng fills, light percussion, warm bass, storytelling male vocal, bittersweet street-life warmth, G major, 108 BPM"

work = ROOT / "novel_songs" / "v2" / "yuegongri_short"
work.mkdir(parents=True, exist_ok=True)
lyric_f = work / "lyrics.txt"
lyric_f.write_text(LYRICS, encoding="utf-8")

# 子进程渲染（参数走 JSON 文件避免字符串转义问题）
config = work / "config.json"
config.write_text(json.dumps(dict(style=STYLE, lyrics=LYRICS, seed=42), ensure_ascii=False), encoding="utf-8")

script = work / "gen.py"
script.write_text('''
import json, sys, torch, soundfile as sf
from pathlib import Path
from yue2 import YuE2Pipeline
ROOT = Path(r"F:\\yinyue")
cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
pipe = YuE2Pipeline.from_pretrained(str(ROOT/"models"/"YuE2-3B"),
                                     vae=str(ROOT/"models"/"YuE2-Vae"),
                                     device="cuda", progress=False)
try:
    out = pipe(prompt=cfg["style"], lyrics=cfg["lyrics"], cot="full", seed=cfg["seed"])
    out.save(str(Path(sys.argv[2]) / "out.flac"))
finally:
    pipe.close()
print("OK")
''', encoding="utf-8")

env = dict(os.environ)
env.update(PYTHONUTF8="1", HF_HUB_OFFLINE="1")
r = subprocess.run([sys.executable, str(script), str(config)], capture_output=True, text=True, env=env)
if r.returncode != 0:
    print(f"FAIL: {r.stderr[-200:]}", flush=True)
    sys.exit(1)
print("DONE", flush=True)
