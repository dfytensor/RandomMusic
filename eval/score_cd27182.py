# -*- coding: utf-8 -*-
"""补评 充电_s27182 的 v3.5 终分（真C1慢板卡）"""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(r"F:\yinyue")
sys.path.insert(0, str(ROOT))

key = "充电_s27182"
dst = ROOT / "novel_songs" / "new_hits" / "gacha" / "充电_seed27182.flac"
V35C = ROOT / "eval" / "new_hits_v35_cache.json"
v35c = json.loads(V35C.read_text(encoding="utf-8"))
if v35c.get(key):
    print("cached", v35c[key]["score"])
    sys.exit()

from score_v3 import measure, score_of
import torch, soundfile as sf, librosa, numpy as np
sd = ROOT / "render_stems" / "new_hits" / key
vpath, bpath = sd / "vocals.wav", sd / "beats.json"
if not (vpath.exists() and bpath.exists()):
    sd.mkdir(parents=True, exist_ok=True)
    from demucs.api import Separator
    sep = Separator(model="htdemucs", device="cuda")
    origin, stems = sep.separate_audio_file(str(dst))
    vocals = stems["vocals"]; accomp = origin - vocals
    sf.write(str(vpath), vocals.cpu().numpy().T, sep.samplerate)
    acc_np = accomp.cpu().numpy().mean(axis=0)
    ya = librosa.resample(acc_np, orig_sr=sep.samplerate, target_sr=22050)
    tempo, beats = librosa.beat.beat_track(y=ya, sr=22050, hop_length=512)
    bt = librosa.frames_to_time(beats, sr=22050, hop_length=512)
    bpath.write_text(json.dumps({"tempo": float(np.atleast_1d(tempo)[0]),
        "beat_times": [round(float(x), 4) for x in bt]}), encoding="utf-8")
    del sep; torch.cuda.empty_cache()
m = measure(dst, vocals_path=vpath, beats_path=bpath)
r = score_of(m)
v35c[key] = r
V35C.write_text(json.dumps(v35c, ensure_ascii=False), encoding="utf-8")
print(f"{key}: v3.5={r['score']:.1f} [{r['cat']}]")
