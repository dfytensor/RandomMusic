# -*- coding: utf-8 -*-
"""从曲库提炼'爆款工程'基准：入场效率/Hook结构/对比度 -> eval/hit_baseline.json"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def parse_lab(tdir):
    rows = []
    p = Path(tdir) / "structure.lab"
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        q = line.split("\t")
        if len(q) >= 3:
            rows.append((float(q[0]), float(q[1]), q[2].strip().lower()))
    return rows


def first_vocal(rows):
    """第一个有唱的段落起点（verse/pre-chorus/chorus）"""
    for t1, t2, n in rows:
        if n in ("verse", "pre-chorus", "chorus"):
            return t1
    return None


def first_chorus(rows):
    for t1, t2, n in rows:
        if n == "chorus":
            return t1
    return None


manifest = json.loads((ROOT / "corpus" / "sample_manifest.json").read_text(encoding="utf-8"))
feats = json.loads((ROOT / "corpus" / "features.json").read_text(encoding="utf-8"))

out = []
for m in manifest:
    stem = Path(m["wav"]).stem
    tdir = ROOT / "corpus" / "transcriptions" / stem
    rows = parse_lab(tdir)
    fv, fc = first_vocal(rows), first_chorus(rows)
    f = feats.get(m["file"], {})
    if fv is None:
        continue
    # 30s 能量 vs 全曲(100s)能量
    try:
        y, _ = librosa.load(m["wav"], sr=22050, mono=True, duration=100)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        db = librosa.amplitude_to_db(rms + 1e-9)
        fps = len(db) / (len(y) / 22050)
        e30 = float(db[: int(30 * fps)].mean())
        eall = float(db.mean())
        e_ratio = round(e30 - eall, 1)
    except Exception:
        e_ratio = None
    # 副歌次数（合并相邻）
    chor = sorted([(t1, t2) for t1, t2, n in rows if n == "chorus"])
    inst = []
    for t1, t2 in chor:
        if inst and t1 - inst[-1][1] < 1.5:
            inst[-1] = (inst[-1][0], max(inst[-1][1], t2))
        else:
            inst.append((t1, t2))
    out.append(dict(stem=stem, first_vocal=round(fv, 1), first_chorus=round(fc, 1) if fc else None,
                    chorus_instances=len(inst), e30_ratio=e_ratio,
                    duration=f.get("duration_s")))

res = dict(
    n=len(out),
    first_vocal_p25=round(float(np.percentile([o["first_vocal"] for o in out], 25)), 1),
    first_vocal_med=round(float(np.median([o["first_vocal"] for o in out])), 1),
    first_vocal_p75=round(float(np.percentile([o["first_vocal"] for o in out], 75)), 1),
    first_chorus_med=round(float(np.median([o["first_chorus"] for o in out if o["first_chorus"]])), 1),
    chorus_inst_med=round(float(np.median([o["chorus_instances"] for o in out])), 1),
    e30_ratio_med=round(float(np.median([o["e30_ratio"] for o in out if o["e30_ratio"] is not None])), 1),
    dur_med=round(float(np.median([o["duration"] for o in out if o["duration"]])), 0),
)
res["songs"] = out
(ROOT / "eval" / "hit_baseline.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "songs"}, ensure_ascii=False, indent=1))
