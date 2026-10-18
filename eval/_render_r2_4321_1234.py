# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, r"F:\\yinyue")
lyrics = '[Intro]\n\n[Verse]\n小卖部冰柜两块五的夏\n玻璃瓶汽水冒泡像烟花\n辣条辣出眼泪也不怕\n你一口我一口笑哈哈\n柜台玻璃贴着旧年画\n老王的算盘噼里啪啦\n后来它变成高楼大厦\n两块五的夏天 再也没啦\n\n[Pre-Chorus]\n汽水现在一箱箱往家拿\n却再也喝不出当年的傻\n冰柜更大 灯火更繁华\n却买不回两块五的夏\n\n[Chorus]\n小卖部 小卖部\n冰柜里冻着夏天\n汽水一开\n气泡炸成了烟花\n小卖部 小卖部\n账本上那八毛\n是我唯一没还的\n长大\n\n[Interlude]\n\n[Verse]\n咖啡一杯顶汽水一打\n醒神却不如那声咔哒\n账单像雪片一样落下\n快乐却一直留在老家\n体检单红字一大把\n发际线跟着房价爬\n长大原来标这个价\n我没还价就画了押\n\n[Pre-Chorus]\n老王 再赊我一瓶盛夏\n这回拿长大慢慢还它\n利息算我没出口的话\n下个夏天我准时回家\n\n[Chorus]\n小卖部 小卖部\n冰柜里冻着夏天\n汽水一开\n气泡炸成了烟花\n小卖部 小卖部\n账本上那八毛\n是我唯一没还的\n长大\n\n[Bridge]\n(whisper) 老王 小卖部拆了以后\n你的算盘收进博物馆了吗\n我欠的八毛\n还记着呢\n\n[Chorus]\n小卖部 小卖部\n冰柜里冻着夏天\n汽水一开\n气泡炸成了烟花\n小卖部 小卖部\n账本上那八毛\n是我唯一没还的\n长大\n\n[Outro]\n'
style = 'warm nostalgic folk pop, acoustic guitar, bright but gentle, mid-tempo, clean warm vocals, light percussion, tender and slightly bittersweet'
from yue2 import YuE2Pipeline
pipe = YuE2Pipeline.from_pretrained(r"C:\yue2_model\models\m-a-p--YuE2-3B\snapshots\master", vae=r"F:\yinyue\models\YuE2-Vae",
                                     device="cuda", progress=False, verify_hashes=False)
try:
    song = pipe(style=style, lyrics=lyrics, cot="full", seed=1234)
    song.save(str(Path(r"F:\yinyue\novel_songs\new_hits\gacha\小卖部_seed1234.flac")))
    print("OK")
finally:
    pipe.close()
