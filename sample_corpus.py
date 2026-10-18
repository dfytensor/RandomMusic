# -*- coding: utf-8 -*-
"""Stratified sample from corpus + decode to 24kHz wav for batch transcription."""
import json
import random
import re
from pathlib import Path

import librosa
import soundfile as sf

ROOT = Path(__file__).parent
SR = 24000
SECONDS = 100
QUOTA = {"C1": 8, "C2": 12, "C3": 8, "C4": 4}
KEYWORDS = {
    "C4": r"派|舞|DJ|remix|Remix|Remix|电音|夜店",
    "C1": r"月光|眼泪|后来|遗憾|再见|寂寞|孤单|暗香|传奇|听海|突然好想你|温柔",
}


def cluster(r, name):
    bpm, cent, dyn = r["bpm"], r["centroid_hz"], r["dyn_p95_p10_db"]
    if re.search(KEYWORDS["C4"], name):
        return "C4"
    if bpm < 75:
        return "C1"
    if bpm < 110:
        return "C2"
    if cent >= 2500 and dyn < 13:
        return "C4"
    return "C3"


feats = json.loads((ROOT / "corpus" / "features.json").read_text(encoding="utf-8"))
by_c = {}
for name, r in feats.items():
    by_c.setdefault(cluster(r, name), []).append(name)

rng = random.Random(42)
manifest = []
for c, quota in QUOTA.items():
    pool = sorted(by_c.get(c, []))
    pick = rng.sample(pool, min(quota, len(pool)))
    for name in pick:
        src = Path(r"D:\音乐") / name
        stem = re.sub(r"[^\w\-]", "_", name[:-4])[:60]
        wav = ROOT / "corpus" / "sample_wavs" / f"{stem}.wav"
        manifest.append({"file": name, "cluster": c, "src": str(src), "wav": str(wav)})

wav_dir = ROOT / "corpus" / "sample_wavs"
wav_dir.mkdir(exist_ok=True)
ok, fail = 0, []
for m in manifest:
    try:
        y, _ = librosa.load(m["src"], sr=SR, mono=True, duration=SECONDS)
        sf.write(m["wav"], y, SR, subtype="PCM_16")
        ok += 1
    except Exception as e:
        fail.append({"file": m["file"], "err": str(e)[:80]})
(ROOT / "corpus" / "sample_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"decoded {ok}/{len(manifest)} fail={len(fail)}")
from collections import Counter
print(Counter(m["cluster"] for m in manifest))
