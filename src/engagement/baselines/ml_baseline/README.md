# ML Baseline

Classical all-modality regression baselines over statistical features extracted from native-rate preprocessed sensor samples inside each labeled window.

Models:

- random forest regressor
- linear regression, ridge-stabilized
- LightGBM regressor
- SVR

Each raw sensor channel is summarized at its native sampling rate with a concise statistical feature set: mean, std, kurtosis, min, max, IQR, diff std, and slope. The metadata progress feature is appended once per window as `round(((t_start_video_sec + t_end_video_sec) / 2) / max(t_end_video_sec for that session video), 1)`.

Run fixed 4-fold:

```bash
python ml_baseline/run_ml_baselines.py \
  --models random_forest linear_regression lgbm svm \
  --split-mode fixed4 \
  --output-dir ml_baseline/results/ml_baselines_fixed4
```

Predictions are continuous regressions over internal labels `0..4`, then rounded/clipped back to labels `1..5`.
Reported metrics are MAE, raw-score MAE, exact rounded accuracy, within-1 rounded accuracy, and binary low/high metrics after rebinding rounded labels: `1,2 = low` and `3,4,5 = high`.

On rono, install LightGBM into the existing architecture venv first:

```bash
source /tmp/eeyal3/architecture-venv/bin/activate
pip install lightgbm
```
