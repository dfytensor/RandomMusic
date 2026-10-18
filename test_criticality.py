# -*- coding: utf-8 -*-
"""临界性测试：guidance_scale 7.4→7.8→8.2 是否存在品质相变"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
import librosa

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS
from netease_score import compute_netease_metrics, score_netease

s = next(x for x in SONGS if x["no"] == "08")  # 月供日（已知最耐听）
work = ROOT / "novel_songs" / "v2" / "criticality_test"
work.mkdir(parents=True, exist_ok=True)

render_script = work / "render.py"
render_script.write_text('''
import torch, soundfile as sf, sys
from pathlib import Path
from yue2 import YuE2Pipeline
ROOT = Path(r"F:\\yinyue")
from novel_songs_manifest import SONGS
s = next(x for x in SONGS if x["no"] == "08")
lyrics = Path(r"F:\\yinyue\\novel_songs\\v2\\yuegongri_prosody\\lyrics_v2.txt").read_text(encoding="utf-8")
pipe = YuE2Pipeline.from_pretrained(str(ROOT/"models"/"YuE2-3B"),
                                     vae=str(ROOT/"models"/"YuE2-Vae"),
                                     device="cuda", progress=False)
try:
    gs = float(sys.argv[1])
    seed = int(sys.argv[2])
    out = sys.argv[3]
    out = out.replace(".flac", ".wav")
    song = pipe(style=s["style"], lyrics=lyrics, cot="full", seed=seed, guidance_scale=gs)
    data, sr = sf.read(str(out), always_2d=True) if False else (None, None)
    # 直接用 pipeline 的音频
    import numpy as np
    audio = song.audio if hasattr(song, 'audio') else None
    if audio is None:
        # fallback: save then load
        tmp = out.with_suffix(".tmp.wav")
        song.save(str(tmp))
        data, sr2 = sf.read(str(tmp))
        sf.write(str(out), data, sr2, subtype="PCM_16")
    else:
        pass
    print("OK")
finally:
    pipe.close()
''', encoding="utf-8")

for gs in (7.4, 7.8, 8.2):
    out = work / f"gs{gs}.flac"
    if out.exists(): continue
    print(f"guidance={gs}...", flush=True)
    # 简化：直接用 render_direct 方式（主进程）
    sys.path.insert(0, str(ROOT))
    from yue2 import YuE2Pipeline
    pipe = YuE2Pipeline.from_pretrained(str(ROOT/"models"/"YuE2-3B"),
                                         vae=str(ROOT/"models"/"YuE2-Vae"),
                                         device="cuda", progress=False)
    try:
        song = pipe(style=s["style"], lyrics=lyrics, cot="full",
                    seed=42, guidance_scale=gs)
        song.save(str(out))
    finally:
        pipe.close()
    print(f"  gs={gs} saved", flush=True)
