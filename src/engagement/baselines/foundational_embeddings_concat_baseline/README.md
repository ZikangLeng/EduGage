# Foundational Embedding Baselines

Strict-intersection baseline for engagement prediction from all 10 generated
embedding modalities.

The concat runner keeps only `window_id`s present in every modality, concatenates the
10 embedding vectors plus a flattened constant metadata channel, standardizes
features inside each train fold, and trains a single `torch.nn.Linear`
classifier with weighted cross-entropy under the fixed 4-fold participant split.

The gated-fusion runner uses the same strict-intersection rows and folds. It
projects each sensor embedding, plus the metadata channel, into a shared latent
space, learns a sigmoid gate per input stream, and fuses streams with a
normalized gated average.

Both runners use the same metadata channel: a rounded normalized video
timestamp, `round(window_center_video_sec / max_video_end_sec, 1)`, repeated
across `--metadata-repeat-dim` positions, defaulting to 512. In concat, this
constant channel is flattened into the concatenated feature vector. In gated
fusion, it is a `metadata` modality that is projected and gated like the sensor
embedding streams.

Run concat:

```bash
python foundational_embeddings_concat_baseline/train_foundational_concat.py
```

Run gated fusion:

```bash
python foundational_embeddings_concat_baseline/train_foundational_gated.py
```

Outputs are written to `foundational_embeddings_concat_baseline/results/` and
`foundational_embeddings_concat_baseline/gated_results/`:

- `manifest.json`
- `metrics_summary.csv`
- `metrics_per_fold.csv`
- `predictions.csv`
- `gates.csv` for gated fusion
