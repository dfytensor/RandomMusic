# -*- coding: utf-8 -*-
"""从原始立体声 mp3 重测曲库 stereo_width，回写 feat_corpus.json"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).parent
manifest = json.loads((ROOT / "corpus" / "sample_manifest.json").read_text(encoding="utf-8"))
feat = json.loads((ROOT / "eval" / "feat_corpus.json").read_text(encoding="utf-8"))
by_stem = {Path(m["wav"]).stem: m["src"] for m in manifest}

for r in feat:
    stem = Path(r["file"]).stem
    src = by_stem.get(stem)
    if not src:
        continue
    data, sr = sf.read(src, always_2d=True)
    if data.shape[1] >= 2:
        mid = data.mean(axis=1)
        side = (data[:, 0] - data[:, 1]) / 2
        r["stereo_width"] = round(float(np.sqrt((side ** 2).mean()) /
                                        (np.sqrt((mid ** 2).mean()) + 1e-9)), 3)
(ROOT / "eval" / "feat_corpus.json").write_text(
    json.dumps(feat, ensure_ascii=False, indent=1), encoding="utf-8")
widths = [r["stereo_width"] for r in feat]
print("corpus width med:", round(float(np.median(widths)), 3), "min:", min(widths), "max:", max(widths))
