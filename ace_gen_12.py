# -*- coding: utf-8 -*-
"""ACE-Step 生成《现贷修仙传》12 首（按 F:\动物园歌手大赛 的已验证代码模板）"""
import torch, soundfile as sf, time, os, sys
from pathlib import Path
from diffusers import AceStepPipeline

MODEL = r"F:\ACE-Step\acestep-v15-xl-sft-diffusers"
OUT = Path(r"F:\yinyue\ace_step_output")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 555, 888]

SONGS = [
    dict(name="欠天地一笔", dur=180,
         caption=("c-pop R&B ballad, mid-tempo 96 bpm, intimate male vocal close-mic, "
                  "fingerpicked acoustic guitar, warm electric piano, soft groove bass, "
                  "brushed drums with ghost notes, emotional restrained chorus, "
                  "melancholic storytelling, authentic natural human voice, melodic singing, hummable"),
         lyric="""[Verse]
灵气枯了四百年
我出生就没有名字
代书摊一张桌子
替别人写了一千封信
[Pre-Chorus]
更鼓敲了三遍
季查使在门外站
我把恐惧折成灵砂
一个一个数着看
[Chorus]
欠天地一笔
我用寿元还息
阴阳账簿翻开我自己
每一页都是我的姓名
欠天地一笔
不还不行
[Verse]
周婆婆熬药到深夜
石头把坏消息嚼碎了咽
他们不知道我在算
一笔能让所有人活着的账
[Pre-Chorus]
更鼓敲了三遍
季查使在门外站
我把愤怒折成灵砂
一个一个数着看
[Chorus]
欠天地一笔
我用寿元还息
阴阳账簿翻开我自己
每一页都是我的姓名
欠天地一笔
不还不行"""),
    dict(name="月供日", dur=180,
         caption=("c-pop R&B pop, mid-tempo 108 bpm, flowing piano, guzheng fills, "
                  "light percussion, warm bass, storytelling male vocal, "
                  "bittersweet street-life warmth, authentic natural human voice, melodic, hummable"),
         lyric="""[Verse]
十五的更鼓敲过城楼
家家户数灵砂的愁
我提着灯走过桥头
水牌上写满赊与留
[Pre-Chorus]
月供日不许回头
息上加息没有尽头
可街口糕点的香悠悠
人间烟火替我挡愁
[Chorus]
月供日
月亮也来催息
还的是钱
赎的是命
我按时把自己交出去
换她一碗药的热气
[Chorus]
月供日
月亮也来催息
还的是钱
赎的是命
我按时把自己交出去
换她一碗药的热气"""),
    dict(name="第320代记账人", dur=180,
         caption=("Chinese style elegy ballad, slow tempo 76 bpm, solo erhu lead, "
                  "distant guzheng, deep strings, sparse piano, "
                  "weathered male vocal full of regret and love, emotional depth, "
                  "authentic natural human voice, E minor"),
         lyric="""[Verse]
封账之露凝在阵眼中
十二年不能相认的痛
他把名字浮进我的梦
把算学传进我掌中
[Pre-Chorus]
他在阵眼里点了灯
照我走过账雾重重
每一步他都算得从容
除了那行卒字的疼
[Chorus]
第320代记账人
账未记完人已归尘
债在子偿 字字千钧
你算清天下的息
唯独没算 我想你
来世再叫你一声父亲
[Chorus]
第320代记账人
账未记完人已归尘
债在子偿 字字千钧
来世再叫你一声父亲"""),
]

pipe = AceStepPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")
pipe.vae.enable_tiling()

for song in SONGS:
    for seed in SEEDS:
        name = f"{song['name']}_s{seed}"
        out_flac = OUT / f"{name}.flac"
        if out_flac.exists():
            print(f"skip {name}", flush=True)
            continue
        t0 = time.time()
        try:
            out = pipe(
                prompt=song["caption"], lyrics=song["lyric"],
                audio_duration=song["dur"],
                num_inference_steps=75, guidance_scale=7.5, shift=3.0,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            )
            audio = out.audios[0] if hasattr(out, 'audios') else out.audio[0]
            if hasattr(audio, 'cpu'): audio = audio.cpu().numpy()
            if audio.ndim > 1: audio = audio[0] if audio.shape[0] < audio.shape[1] else audio.T[0]
            sf.write(str(out_flac), audio, 22050, format="FLAC")
            print(f"  saved {name} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  FAIL {name}: {str(e)[:80]}", flush=True)
            import traceback; traceback.print_exc()

print("ALL DONE", flush=True)
