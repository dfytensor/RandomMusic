# -*- coding: utf-8 -*-
"""切副歌片段 (主环境): structure.lab 定位 -> eval/clips/*.wav + refs.json"""
import json
import re
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).parent

SONGS = [
    dict(name="520", wav="eval/wavs/create_520.wav",
         realized="creations/20260920_043058/self_score",
         lyrics=str(Path("creations/lyrics_convenience_store.txt").read_text(encoding="utf-8"))),
    dict(name="731", wav="eval/wavs/create_731.wav",
         realized="creations/selfopt_20260920_061912/round1_seed731/realized",
         lyrics=str(Path("creations/lyrics_convenience_store.txt").read_text(encoding="utf-8"))),
    dict(name="tonight", wav="eval/wavs/demo_tonight.wav",
         realized="eval/transcriptions/demo_tonight",
         lyrics=None),  # 从 YuE2 示例 JSON 取
]


def first_chorus_span(tdir):
    rows = []
    for line in (Path(tdir) / "structure.lab").read_text(encoding="utf-8").splitlines():
        p = line.split("\t")
        if len(p) >= 3 and p[2].strip().lower() == "chorus":
            rows.append((float(p[0]), float(p[1])))
    if not rows:
        return None
    return rows[0][0], rows[0][1]


def chorus_lyrics(lyrics_text):
    m = re.search(r"\[Chorus\]\n(.*?)(?:\n\n|\Z)", lyrics_text, re.S)
    if not m:
        return ""
    return "\n".join(l for l in m.group(1).strip().splitlines() if l.strip())


clips_dir = ROOT / "eval" / "clips"
clips_dir.mkdir(parents=True, exist_ok=True)
refs = {}
for s in SONGS:
    span = first_chorus_span(ROOT / s["realized"])
    if not span:
        print(f"no chorus in {s['name']}")
        continue
    y, sr = sf.read(str(ROOT / s["wav"]), always_2d=True)
    if y.shape[1] > 1:
        y = y.mean(axis=1)
    a, b = int(span[0] * sr), min(len(y), int(span[1] * sr))
    if b - a < sr * 5:
        print(f"clip too short {s['name']}")
        continue
    clip = clips_dir / f"chorus_{s['name']}.wav"
    sf.write(clip, y[a:b], sr, subtype="PCM_16")
    if s["lyrics"] is None:
        demo = json.loads((ROOT / "models" / "YuE2-3B" / "examples" / "tonight-awake.json")
                          .read_text(encoding="utf-8"))["lyrics"]
        refs[s["name"]] = chorus_lyrics(demo)
    else:
        refs[s["name"]] = chorus_lyrics(s["lyrics"])
    print(f"{s['name']}: chorus {span[0]:.1f}-{span[1]:.1f}s -> {clip.name} ({(b-a)/sr:.1f}s)")

(clips_dir / "refs.json").write_text(json.dumps(refs, ensure_ascii=False, indent=1), encoding="utf-8")
print("DONE")
