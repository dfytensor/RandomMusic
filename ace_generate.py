# -*- coding: utf-8 -*-
"""ACE-Step 生成测试：与 YuE2 同曲对比
生成 3 首歌 -> 用同一套指标评价 -> 对比两个模型
"""
import json, sys, os
from pathlib import Path

import torch
import soundfile as sf

MODEL_PATH = r"F:\ACE-Step\acestep-v15-xl-sft-diffusers"
OUT = Path(r"F:\yinyue\ace_step_output")
OUT.mkdir(parents=True, exist_ok=True)

# 测试歌曲（取 YuE2 最好的 3 首）
TEST = [
    dict(no="08", title="月供日", 
         prompt="Mandopop R&B pop, flowing piano, guzheng fills, light percussion, warm bass, storytelling male vocal, bittersweet street-life warmth, G major, 108 BPM",
         lyrics="[Verse]\n十五的更鼓敲过城楼\n家家户数灵砂的愁\n我提着灯走过桥头\n水牌上写满赊与留\n\n[Pre-Chorus]\n月供日不许回头\n息上加息没有尽头\n可街口糕点的香悠悠\n人间烟火替我挡愁\n\n[Chorus]\n月供日\n月亮也来催息\n还的是钱\n赎的是命\n我按时把自己交出去\n换她一碗药的热气",
         duration=60),
    dict(no="01", title="欠天地一笔",
         prompt="Mandopop R&B pop, fingerpicked acoustic guitar, warm electric piano, soft groove bass, brushed drums, intimate male vocal close-mic, building emotional restrained chorus, D minor, 96 BPM",
         lyrics="[Verse]\n灵气枯了四百年\n我出生就没有名字\n代书摊一张桌子\n替别人写了一千封信\n\n[Pre-Chorus]\n更鼓敲了三遍\n季查使在门外站\n我把恐惧折成灵砂\n一个一个数着看\n\n[Chorus]\n欠天地一笔\n我用寿元还息\n阴阳账簿翻开我自己\n每一页都是我的姓名\n欠天地一笔\n不还不行",
         duration=60),
    dict(no="06", title="第320代记账人",
         prompt="Chinese style elegy ballad, solo erhu lead, distant guzheng, deep strings, sparse piano, weathered male vocal full of regret and love, E minor, 76 BPM",
         lyrics="[Verse]\n封账之露凝在阵眼中\n十二年不能相认的痛\n他把名字浮进我的梦\n把算学传进我掌中\n\n[Chorus]\n第320代记账人\n账未记完人已归尘\n债在子偿 字字千钧\n你算清天下的息\n唯独没算 我想你\n来世再叫你一声父亲",
         duration=45),
]

from diffusers import AceStepPipeline

pipe = AceStepPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to("cuda")
pipe.vae.enable_tiling()

for t in TEST:
    out_flac = OUT / f"ace_{t['no']}_{t['title']}.flac"
    if out_flac.exists():
        print(f"skip {t['title']}")
        continue
    print(f"生成: {t['title']} ({t['duration']}s)...", flush=True)
    output = pipe(
        prompt=t["prompt"],
        lyrics=t["lyrics"],
        audio_duration=t["duration"],
    )
    audio = output.audios[0] if hasattr(output, 'audios') else output.audio[0]
    if hasattr(audio, 'cpu'):
        audio = audio.cpu().numpy()
    if audio.ndim > 1:
        audio = audio[0] if audio.shape[0] < audio.shape[1] else audio.T[0]
    sf.write(str(out_flac.with_suffix(".wav")), audio, 22050, subtype="PCM_16")
    # 转 flac
    import soundfile as sf2
    data, sr = sf.read(str(out_flac.with_suffix(".wav")))
    sf.write(str(out_flac), data, sr, format="FLAC")
    os.remove(str(out_flac.with_suffix(".wav")))
    print(f"  saved {out_flac.name}", flush=True)

print("DONE", flush=True)
