# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, r"F:\yinyue")
from novel_songs_manifest import SONGS
s = next(x for x in SONGS if x["no"] == "08")
lyrics = '[Intro]\n\n[Verse]\n更鼓敲过第三轮\n灯芯挑亮一寸烬\n灵砂凉透半生恨\n焐热贴着一念温\n隔壁咳嗽都咽尽\n整条街上剩我巡\n水牌名字字迹泯\n吹散天涯不问因\n\n[Pre-Chorus]\n(build) 月供日子亮堂堂\n亮得人不敢去想\n明天的事明天讲\n今夜先把利息偿\n\n[Chorus]\n月供日 月亮催我还\n还完这期还有下期账\n街角糕点摊还热\n热得忘身上有多凉\n月供日 活着是利息\n我把自己按时交上去\n换她那碗冒热气的药\n(breathy) 药比蜜还甜\n\n[Interlude]\n\n[Verse]\n石头坏信准又狠\n嚼碎咽下能回温\n婆婆煎药分两份\n一碗我一碗温夜深\n月亮爬上老城阵\n照着没睡几个人\n我把算盘拨回本\n一声叹息归了尘\n\n[Pre-Chorus]\n(build) 月供日子亮堂堂\n亮得人不敢去想\n明天的事明天讲\n今夜先把利息偿\n\n[Chorus]\n月供日 月亮催我还\n还完这期还有下期账\n街角糕点摊还热\n热得忘身上有多凉\n月供日 活着是利息\n我把自己按时交上去\n换她那碗冒热气的药\n(breathy) 药比蜜还甜\n\n[Bridge]\n(whisper) 天快亮了\n货架空了心满了\n\n[Chorus]\n月供日 月亮催我还\n还完这期还有下期账\n街角糕点摊还热\n热得忘身上有多凉\n月供日 活着是利息\n我把自己按时交上去\n换她那碗冒热气的药\n(breathy) 药比蜜还甜\n\n[Outro]\n'
style = s["style"] + ", " + 'expressive but tasteful singing, gentle natural vibrato on sustained notes, legato consonant transitions, subtle rubato micro-timing, warm intimate tone, sparse piano with silences, breathing space between phrases'
from yue2 import YuE2Pipeline
pipe = YuE2Pipeline.from_pretrained(str(Path(r"F:\yinyue")/"models"/"YuE2-3B"),
                                     vae=str(Path(r"F:\yinyue")/"models"/"YuE2-Vae"),
                                     device="cuda", progress=False)
try:
    song = pipe(style=style, lyrics=lyrics, cot="full", seed=80602)
    dst = Path(r"F:\yinyue\novel_songs\batch_gacha\08_v3x\seed80602.flac")
    dst.parent.mkdir(parents=True, exist_ok=True)
    song.save(str(dst))
    print("OK 08 seed80602 v3x")
finally:
    pipe.close()
