# 无参考质量自检

- 最佳候选: **01_欠天地一笔.flac** (4/5 项通过)

| 检查项 | 01_欠天地一笔.flac |
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
  "file": "01_欠天地一笔.flac",
  "duration_s": 187.0,
  "clip_ratio": 0.0,
  "bpm_windows": [
   184.6,
   95.7,
   95.7
  ],
  "bpm_stability_pct": 3.6,
  "key_windows": [
   "A# minor",
   "A# major",
   "F minor"
  ],
  "key_stable": true,
  "key_majority": "C# minor (1/3)",
  "vocal_band_share": 0.339,
  "rms_db": -18.9,
  "pass": 4,
  "total": 5
 }
]
```