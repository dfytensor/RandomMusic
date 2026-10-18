# 无参考质量自检

- 最佳候选: **mastered.flac** (4/5 项通过)

| 检查项 | mastered.flac |
|---|---|
| bpm_stable_≤8% | ✅ |
| key_stable | ✅ |
| no_clipping | ✅ |
| rms_in_-25~-8 | ✅ |
| vocal_present_≥0.35 | ❌ |

## 明细

```json
[
 {
  "file": "mastered.flac",
  "duration_s": 224.4,
  "clip_ratio": 0.0,
  "bpm_windows": [
   123.0,
   123.0,
   123.0
  ],
  "bpm_stability_pct": 0.0,
  "key_windows": [
   "D# major",
   "D# major",
   "D# minor"
  ],
  "key_stable": true,
  "key_majority": "D# major (2/3)",
  "vocal_band_share": 0.337,
  "rms_db": -19.2,
  "pass": 4,
  "total": 5
 }
]
```