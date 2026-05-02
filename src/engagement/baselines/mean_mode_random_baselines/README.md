# Mean Mode Random Baselines

Simple fixed4 comparison baselines using only training-label distributions.

- `mean`: predict the rounded training-label mean for every test sample in the fold.
- `mode`: predict the most frequent training label for every test sample in the fold.
- `random_distribution`: sample each prediction from the training-label distribution.

Run:

```bash
python mean_mode_random_baselines/run_baselines.py \
  --output-dir mean_mode_random_baselines/results/mean_mode_random_baselines
```
