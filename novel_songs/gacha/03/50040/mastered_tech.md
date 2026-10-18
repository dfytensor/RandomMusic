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
  "duration_s": 171.1,
  "clip_ratio": 0.0,
  "bpm_windows": [
   143.6,
   152.0,
   143.6
  ],
  "bpm_stability_pct": 5.9,
  "key_windows": [
   "D# minor",
   "D# minor",
   "D# minor"
  ],
  "key_stable": true,
  "key_majority": "F# minor (3/3)",
  "vocal_band_share": 0.543,
  "rms_db": -17.5,
  "pass": 5,
  "total": 5
 }
]
```