# 全维度热度分类（提升版）

样本: 463 · 特征: 70 · 3 类

| 模型 | Accuracy | F1-macro |
|---|---|---|
| RandomForest | 55.1% | F1=0.390 |
| GradientBoosting | 52.1% | F1=0.336 |
| SVM-RBF | 53.8% | F1=0.406 |

## 最佳模型详细报告

```
              precision    recall  f1-score   support

           1       0.63      0.72      0.67       269
           2       0.13      0.07      0.09        44
           3       0.44      0.38      0.41       150

    accuracy                           0.55       463
   macro avg       0.40      0.39      0.39       463
weighted avg       0.52      0.55      0.53       463

```

AUC (OVR macro): **1.000**

## Top 30 特征重要性

| 排名 | 特征 | 重要性 |
|---|---|---|
| 1 | mel_motif_repeat | 0.0471 |
| 2 | key_conf | 0.0420 |
| 3 | centroid_hz | 0.0398 |
| 4 | res_long_drift | 0.0390 |
| 5 | fp_glide_ratio | 0.0366 |
| 6 | duration_s | 0.0365 |
| 7 | fp_dyn_range_db | 0.0362 |
| 8 | fp_loop_lag_s | 0.0359 |
| 9 | hpr | 0.0354 |
| 10 | fp_intra_note_drift_cents | 0.0352 |
| 11 | fp_loop_peak_corr | 0.0342 |
| 12 | dyn_p95_p10_db | 0.0330 |
| 13 | fp_loopiness | 0.0330 |
| 14 | fp_vibrato_extent_cents | 0.0324 |
| 15 | mel_pitch_range | 0.0317 |
| 16 | fp_spectral_flux_std | 0.0314 |
| 17 | mel_lift_mean | 0.0311 |
| 18 | fp_voiced_ratio | 0.0308 |
| 19 | fp_n_vocal_runs | 0.0308 |
| 20 | res_leap_prep | 0.0307 |
| 21 | mel_leap_ratio | 0.0295 |
| 22 | res_vibrato_cv | 0.0294 |
| 23 | mel_rest_ratio | 0.0293 |
| 24 | bpm | 0.0289 |
| 25 | wf_onset_iv_cv | 0.0288 |
| 26 | fp_vibrato_rate_hz | 0.0278 |
| 27 | mel_density | 0.0257 |
| 28 | wf_lra_db | 0.0240 |
| 29 | fp_macro_dyn_std_db | 0.0181 |
| 30 | wf_silence_frac | 0.0160 |