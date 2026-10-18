# 无参考质量自检

- 最佳候选: **mastered.flac** (5/5 项通过)

| 检查项 | mastered.flac |
|---|---|
| bpm_stable_≤8% | ✅ |
| key_stable | ✅ |
| no_clipping | ✅ |
| rms_in_-25~-8 | ✅ |
| vocal_present_≥0.35 | ✅ |

## 明细

```json
[
 {
  "file": "mastered.flac",
  "duration_s": 158.8,
  "clip_ratio": 0.0,
  "bpm_windows": [
   184.6,
   95.7,
   95.7
  ],
  "bpm_stability_pct": 3.6,
  "key_windows": [
   "A# major",
   "D minor",
   "A# major"
  ],
  "key_stable": true,
  "key_majority": "A# major (2/3)",
  "vocal_band_share": 0.418,
  "rms_db": -17.6,
  "pass": 5,
  "total": 5
 }
]
```