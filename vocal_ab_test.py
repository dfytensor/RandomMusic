# -*- coding: utf-8 -*-
"""唱法杠杆 A/B：同曲同seed，只换 style 的唱法描述 -> 测 F0音区/颤音/亮度 差异"""
import json
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS
from remediate_novel import naturalness

s = next(x for x in SONGS if x["no"] == "02")
work = ROOT / "novel_songs" / "v2" / "vocal_ab"
work.mkdir(parents=True, exist_ok=True)
SEED = 71502

VARIANTS = {
    "A_calm": "Mandopop R&B pop, piano and clean electric guitar, warm groove bass, soft pop drums with ghost notes, calm sharp male vocal, introspective and precise, D minor, 100 BPM",
    "B_breathy": "Mandopop R&B pop, piano and clean electric guitar, warm groove bass, soft pop drums with ghost notes, airy breathy intimate male vocal close-mic, introspective and fragile, D minor, 100 BPM",
}

from yue2 import YuE2Pipeline
res = {}
pipe = YuE2Pipeline.from_pretrained(str(ROOT / "models" / "YuE2-3B"),
                                    vae=str(ROOT / "models" / "YuE2-Vae"),
                                    device="cuda", progress=False)
try:
    for tag, style in VARIANTS.items():
        out = work / f"{tag}.flac"
        if not out.exists():
            song = pipe(style=style, lyrics=s["lyrics"], cot="full", seed=SEED)
            song.save(str(out))
        y, _ = librosa.load(str(out), sr=24000, mono=True, duration=120)
        # 人声段 F0（60-90s 副歌附近）
        seg = y[int(60 * 24000):int(90 * 24000)]
        f0, vflag, vprob = librosa.pyin(seg, fmin=65, fmax=500, sr=24000, frame_length=1024,
                                        hop_length=256)
        f0v = f0[vflag & ~np.isnan(f0)]
        med_f0 = round(float(np.median(f0v)), 1)
        # 颤音幅度：中心化后 std（cents，近似）
        c = 1200 * np.log2(f0v / 55.0)
        slow = np.convolve(c, np.ones(40) / 40, mode="same")
        vib = round(float(np.std(c - slow)), 1)
        # 高频倾斜（2-6k vs 0-1k 能量比，唱法明亮度代理）
        S = np.abs(librosa.stft(seg)) ** 2
        fr = librosa.fft_frequencies(sr=24000)
        lo = float(S[fr < 1000].sum())
        hi = float(S[(fr >= 2000) & (fr < 6000)].sum())
        tilt = round(float(np.log10((hi + 1e-9) / (lo + 1e-9))), 2)
        maha, cos5 = naturalness(out)
        res[tag] = dict(med_f0_hz=med_f0, vibrato_std_cents=vib, tilt_2k_1k=tilt,
                        maha=round(maha), cos5=round(cos5, 3))
        print(tag, res[tag], flush=True)
finally:
    pipe.close()

print("\n差异: F0音区", res["B_breathy"]["med_f0_hz"] - res["A_calm"]["med_f0_hz"], "Hz | "
      "颤音std", res["B_breathy"]["vibrato_std_cents"] - res["A_calm"]["vibrato_std_cents"], "c | "
      "亮度倾斜", round(res["B_breathy"]["tilt_2k_1k"] - res["A_calm"]["tilt_2k_1k"], 2))
(work / "vocal_ab_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
