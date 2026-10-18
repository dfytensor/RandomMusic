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
  "duration_s": 221.0,
  "clip_ratio": 0.0,
  "bpm_windows": [
   152.0,
   76.0,
   152.0
  ],
  "bpm_stability_pct": 0.0,
  "key_windows": [
   "D# minor",
   "D# major",
   "D# minor"
  ],
  "key_stable": true,
  "key_majority": "F# minor (2/3)",
  "vocal_band_share": 0.466,
  "rms_db": -21.2,
  "pass": 5,
  "total": 5
 }
]
```