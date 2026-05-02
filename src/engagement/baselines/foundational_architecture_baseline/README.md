# Foundational Architecture Baseline

Official-source ports for:

- `deepconvlstm`
- `deepconvlstm_attention`
- `tinyhar`

Inputs are all requested modalities resampled to `[28, 2200]` at 50 Hz, trained as scalar regressors with unweighted MAE/L1 loss on the raw regression output. The metadata channel is `round(((t_start_video_sec + t_end_video_sec) / 2) / max(t_end_video_sec for that session video), 1)` repeated across the window. Predictions are rounded/clipped back to labels `1..5`; reported MAE and within-1 accuracy are computed from those rounded 5-class predictions, and binary metrics rebin the rounded labels with `1,2 = low` and `3,4,5 = high`.

Run fixed 4-fold only:

```bash
python foundational_architecture_baseline/run_architecture_models.py \
  --models deepconvlstm deepconvlstm_attention tinyhar \
  --split-mode fixed4 \
  --device cuda \
  --output-dir foundational_architecture_baseline/results/architecture_models_fixed4_official_full
```
