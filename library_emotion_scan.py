# -*- coding: utf-8 -*-
"""全库情绪评分 v2 扫描：抽样 60 首 → 最高/最低/分布"""
import json, sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from emotion_score_v2 import score_audio_v2

wavs = sorted((ROOT/"eval"/"library_wavs").glob("*.wav"))
# 分层抽样：每 9-10 首取 1 → ~60 首
sample = wavs[::9][:60]

res = []
for i, f in enumerate(sample):
    try:
        r = score_audio_v2(f)
        res.append(dict(file=f.stem[:20], **r))
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(sample)}", flush=True)
    except Exception as e:
        print(f"  {f.stem[:16]} ERR {type(e).__name__}", flush=True)

res.sort(key=lambda x: -x["total"])
print(f"\nn={len(res)}")
print(f"最高: {res[0]['file']} total={res[0]['total']} emo={res[0]['emo']} acou={res[0]['acou']} "
      f"cat={res[0]['cat']} tension={res[0]['tension']}")
print(f"最低: {res[-1]['file']} total={res[-1]['total']} emo={res[-1]['emo']} acou={res[-1]['acou']} "
      f"cat={res[-1]['cat']} tension={res[-1]['tension']}")
tots = [x["total"] for x in res]
print(f"中位={np.median(tots):.1f} P25={np.percentile(tots,25):.1f} P75={np.percentile(tots,75):.1f}")
print(f"AI冠军对照: 08=79 09=64 07=62 01=40")

(ROOT/"eval"/"library_emotion_scores.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
