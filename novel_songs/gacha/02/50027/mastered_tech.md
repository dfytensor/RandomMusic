# 无参考质量自检

- 最佳候选: **mastered.flac** (4/5 项通过)

| 检查项 | mastered.flac |
|---|---|
| bpm_stable_≤8% | ✅ |
| key_stable | ❌ |
| no_clipping | ✅ |
| rms_in_-25~-8 | ✅ |
| vocal_present_≥0.35 | ✅ |

## 明细

```json
[
 {
  "file": "mastered.flac",
  "duration_s": 206.2,
  "clip_ratio": 0.0,
  "bpm_windows": [
   99.4,
   99.4,
   99.4
  ],
  "bpm_stability_pct": 0.0,
  "key_windows": [
   "A# major",
   "A# major",
   "D# minor"
  ],
  "key_stable": false,
  "key_majority": "A# major (2/3)",
  "vocal_band_share": 0.419,
  "rms_db": -18.3,
  "pass": 4,
  "total": 5
 }
]
```