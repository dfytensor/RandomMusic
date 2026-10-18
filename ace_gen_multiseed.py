# -*- coding: utf-8 -*-
"""ACE-Step 多 seed 生成（run in ACE-Step .venv）"""
import torch, sys
from pathlib import Path
import soundfile as sf
import numpy as np

MODEL_PATH = Path(r"F:\ACE-Step\acestep-v15-xl-sft-diffusers")
OUT = Path(r"F:\yinyue\ace_step_output\gacha")
OUT.mkdir(parents=True, exist_ok=True)

from diffusers import AceStepPipeline
pipe = AceStepPipeline.from_pretrained(str(MODEL_PATH), torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")
pipe.vae.enable_tiling()

SONGS = [
    dict(no="08", title="月供日", dur=60,
         prompt="Mandopop R&B pop, flowing piano, guzheng fills, light percussion, warm bass, storytelling male vocal, bittersweet street-life warmth, G major, 108 BPM",
         lyrics="[Verse]\n十五的更鼓敲过城楼\n家家户数灵砂的愁\n我提着灯走过桥头\n水牌上写满赊与留\n\n[Pre-Chorus]\n月供日不许回头\n息上加息没有尽头\n可街口糕点的香悠悠\n人间烟火替我挡愁\n\n[Chorus]\n月供日\n月亮也来催息\n还的是钱\n赎的是命\n我按时把自己交出去\n换她一碗药的热气"),
    dict(no="01", title="欠天地一笔", dur=60,
         prompt="Mandopop R&B pop, fingerpicked acoustic guitar, warm electric piano, soft groove bass, brushed drums, intimate male vocal close-mic, building emotional restrained chorus, D minor, 96 BPM",
         lyrics="[Verse]\n灵气枯了四百年\n我出生就没有名字\n代书摊一张桌子\n替别人写了一千封信\n\n[Pre-Chorus]\n更鼓敲了三遍\n季查使在门外站\n我把恐惧折成灵砂\n一个一个数着看\n\n[Chorus]\n欠天地一笔\n我用寿元还息\n阴阳账簿翻开我自己\n每一页都是我的姓名\n欠天地一笔\n不还不行"),
    dict(no="06", title="第320代记账人", dur=45,
         prompt="Chinese style elegy ballad, solo erhu lead, distant guzheng, deep strings, sparse piano, weathered male vocal full of regret and love, E minor, 76 BPM",
         lyrics="[Verse]\n封账之露凝在阵眼中\n十二年不能相认的痛\n他把名字浮进我的梦\n把算学传进我掌中\n\n[Chorus]\n第320代记账人\n账未记完人已归尘\n债在子偿 字字千钧\n你算清天下的息\n唯独没算 我想你\n来世再叫你一声父亲"),
]

SEEDS = {"08": [100, 200], "01": [400, 500], "06": [700, 800]}

for t in SONGS:
    for seed in SEEDS.get(t["no"], [100]):
        out = OUT / f"ace_{t['no']}_{t['title']}_s{seed}.flac"
        if out.exists():
            continue
        print(f"gen {t['no']} seed{seed}...", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        output = pipe(prompt=t["prompt"], lyrics=t["lyrics"],
                      audio_duration=t["dur"])
        audio = output.audios[0] if hasattr(output, 'audios') else output.audio[0]
        if hasattr(audio, 'cpu'): audio = audio.cpu().numpy()
        if audio.ndim > 1: audio = audio[0] if audio.shape[0] < audio.shape[1] else audio.T[0]
        sf.write(str(out), audio, 22050, format="FLAC")
        print(f"  saved {out.name}", flush=True)

print("ALL DONE", flush=True)
