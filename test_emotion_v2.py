# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, r'F:\yinyue')
from emotion_score_v2 import score_audio_v2

ROOT = Path(r'F:\yinyue')
champs = {
    '01_v2(58888)': ROOT/'novel_songs/batch_gacha/01_v2/seed58888.flac',
    '07_adj(1357)': ROOT/'novel_songs/batch_gacha/07_adj/seed1357_r1.flac',
    '08_91104': ROOT/'novel_songs/triple_emergence/seed91104.flac',
    '09_90904': ROOT/'novel_songs/batch_gacha/09/seed90904.flac',
}
for name, f in champs.items():
    if f.exists():
        r = score_audio_v2(f)
        print(f"{name}: emo={r['emo']} acou={r['acou']} total={r['total']} "
              f"tension={r['tension']} micro={r['micro_arc']}")
    else:
        print(f"{name}: MISSING")
