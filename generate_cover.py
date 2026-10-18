# -*- coding: utf-8 -*-
"""Generate high-fidelity cover of 江南 with YuE2 (cot=melody + transcribed ABC)."""
import json
from pathlib import Path

from yue2 import YuE2Pipeline

ROOT = Path(__file__).parent
MODEL = ROOT / "models" / "YuE2-3B"
VAE = ROOT / "models" / "YuE2-Vae"
COVER = ROOT / "covers" / "jiangnan"

abc = (COVER / "score_full" / "score.abc").read_text(encoding="utf-8")

style = ("Mandopop R&B ballad, gentle arpeggiated piano, warm string pads, "
         "clean electric guitar, soft groove bass, subtle pop drums, "
         "emotive male lead vocal with airy falsetto in chorus, "
         "nostalgic and romantic, rainy Jiangnan water-town atmosphere, "
         "Bb major, 62 BPM")

lyrics = """[Intro]

[Verse]
风到这里就是黏
黏住过客的思念
雨到了这里缠成线
缠着我们留恋人世间
你在身边就是缘
缘分写在三生石上面
爱有万分之一甜
宁愿我就葬在这一点

[Pre-Chorus]
圈圈圆圆圈圈
天天年年天天的我
深深看你的脸
生气的温柔 埋怨的温柔的脸

[Chorus]
不懂爱恨情仇煎熬的我们
都以为相爱就像风云的善变
相信爱一天 抵过永远
在这一刹那冻结了时间
不懂怎么表现温柔的我们
还以为殉情只是古老的传言
离愁能有多痛 痛有多浓
当梦被埋在江南烟雨中
心碎了才懂

[Interlude]

[Pre-Chorus]
圈圈圆圆圈圈
天天年年天天的我
深深看你的脸
生气的温柔 埋怨的温柔的脸

[Chorus]
不懂爱恨情仇煎熬的我们
都以为相爱就像风云的善变
相信爱一天 抵过永远
在这一刹那冻结了时间
不懂怎么表现温柔的我们
还以为殉情只是古老的传言
离愁能有多痛 痛有多浓
当梦被埋在江南烟雨中
心碎了才懂

[Bridge]
呜

[Chorus]
相信爱一天 抵过永远
在这一刹那冻结了时间
不懂怎么表现温柔的我们
还以为殉情只是古老的传言
离愁能有多痛 痛有多浓
当梦被埋在江南烟雨中
心碎了才懂

[Outro]
"""

(COVER / "gen_inputs.json").write_text(json.dumps(
    {"style": style, "lyrics": lyrics, "cot": "melody", "seed": 411604}, ensure_ascii=False, indent=1),
    encoding="utf-8")

pipe = YuE2Pipeline.from_pretrained(str(MODEL), vae=str(VAE), device="cuda")
song = pipe(style=style, lyrics=lyrics, abc=abc, cot="melody", seed=411604)
song.save(str(COVER / "cover.flac"))
song.save_artifacts(str(COVER / "gen"))
pipe.close()
print("DONE -> covers/jiangnan/cover.flac", flush=True)
