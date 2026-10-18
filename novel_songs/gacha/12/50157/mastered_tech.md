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
  "duration_s": 221.8,
  "clip_ratio": 0.0,
  "bpm_windows": [
   123.0,
   123.0,
   123.0
  ],
  "bpm_stability_pct": 0.0,
  "key_windows": [
   "D# minor",
   "D# major",
   "A# minor"
  ],
  "key_stable": true,
  "key_majority": "F# minor (1/3)",
  "vocal_band_share": 0.372,
  "rms_db": -18.9,
  "pass": 5,
  "total": 5
 }
]
```