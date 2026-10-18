# 全特征 ML（34 维 × 463 首）

3类准确率: **55.7%**（基线 58%）
跳过: 108

| 排名 | 特征 | 重要性 |
|---|---|---|
| 1 | duration_s | 0.0474 |
| 2 | fp_voiced_ratio | 0.0469 |
| 3 | wf_crest_db | 0.0460 |
| 4 | zcr | 0.0434 |
| 5 | fp_spectral_flux_std | 0.0420 |
| 6 | centroid_hz | 0.0411 |
| 7 | key_conf | 0.0403 |
| 8 | hpr | 0.0392 |
| 9 | rolloff_hz | 0.0385 |
| 10 | wf_kurtosis | 0.0378 |
| 11 | rms_db | 0.0377 |
| 12 | fp_macro_dyn_std_db | 0.0368 |
| 13 | fp_vibrato_extent_cents | 0.0367 |
| 14 | dyn_p95_p10_db | 0.0355 |
| 15 | fp_dyn_range_db | 0.0342 |
| 16 | wf_onset_iv_cv | 0.0341 |
| 17 | fp_loop_peak_corr | 0.0335 |
| 18 | wf_env_rough | 0.0330 |
| 19 | fp_loopiness | 0.0326 |
| 20 | fp_intra_note_drift_cents | 0.0326 |
| 21 | fp_loop_lag_s | 0.0313 |
| 22 | bpm | 0.0309 |
| 23 | fp_vibrato_rate_hz | 0.0309 |
| 24 | fp_glide_ratio | 0.0308 |
| 25 | wf_lra_db | 0.0305 |
| 26 | wf_low_attack_ms | 0.0305 |
| 27 | fp_n_vocal_runs | 0.0215 |
| 28 | mel_leap_ratio | 0.0049 |
| 29 | mel_rest_ratio | 0.0049 |
| 30 | mel_lift_mean | 0.0040 |

## 逐特征热度ρ

- fp_macro_dyn_std_db: ρ=-0.118
- zcr: ρ=-0.113
- centroid_hz: ρ=-0.112
- fp_dyn_range_db: ρ=-0.109
- wf_crest_db: ρ=-0.108
- rolloff_hz: ρ=-0.101
- wf_lra_db: ρ=-0.084
- wf_onset_iv_cv: ρ=-0.081
- wf_kurtosis: ρ=-0.076
- duration_s: ρ=+0.075
- rms_db: ρ=+0.073
- wf_env_rough: ρ=-0.070
- fp_voiced_ratio: ρ=-0.068
- wf_silence_frac: ρ=-0.067
- key_conf: ρ=+0.063

## Top25 重要性明细

- duration_s: 0.0474
- fp_voiced_ratio: 0.0469
- wf_crest_db: 0.0460
- zcr: 0.0434
- fp_spectral_flux_std: 0.0420
- centroid_hz: 0.0411
- key_conf: 0.0403
- hpr: 0.0392
- rolloff_hz: 0.0385
- wf_kurtosis: 0.0378
- rms_db: 0.0377
- fp_macro_dyn_std_db: 0.0368
- fp_vibrato_extent_cents: 0.0367
- dyn_p95_p10_db: 0.0355
- fp_dyn_range_db: 0.0342
- wf_onset_iv_cv: 0.0341
- fp_loop_peak_corr: 0.0335
- wf_env_rough: 0.0330
- fp_loopiness: 0.0326
- fp_intra_note_drift_cents: 0.0326
- fp_loop_lag_s: 0.0313
- bpm: 0.0309
- fp_vibrato_rate_hz: 0.0309
- fp_glide_ratio: 0.0308
- wf_lra_db: 0.0305