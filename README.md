<p align="center">
  <img src="https://github.com/user-attachments/assets/547b5f1b-b886-4f72-b19e-e9defcc27612" alt="EduGage logo" width="140" />
</p>

<h1 align="center">EduGage</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2605.01238"><img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge" alt="Read the paper on arXiv" /></a>
  <a href="https://doi.org/10.6084/m9.figshare.32145994"><img src="https://img.shields.io/badge/Dataset-Figshare-1F7A8C?style=for-the-badge" alt="Access the dataset on Figshare" /></a>
</p>

## About EduGage

**EduGage** is a multimodal dataset and benchmark for assessing momentary
engagement during self-guided video learning. The paper was **accepted to
[IMWUT](https://dl.acm.org/journal/imwut) in 2026**. A public
[preprint](https://arxiv.org/abs/2605.01238) is available, and this repository
contains the processing and modeling code.

The code discovers raw participant-session folders, builds labeled engagement
windows, exports preprocessed sensor slices, and runs training/evaluation
pipelines and baselines. The raw **EduGage Dataset** (3.05 GB) is accessible
through the [public Figshare dataset page](https://doi.org/10.6084/m9.figshare.32145994).
It contains one folder per participant session; the files are raw collection
outputs, not cleaned/aligned/windowed preprocessing products.

The source code used to collect the raw experiment streams is included in
`src/Data_Collection/`. That folder contains the experiment player and collection
wrappers for the released sensor streams. Proprietary SDK bundles, generated
build outputs, local device identifiers, and raw participant data are
intentionally excluded.

## 1. Set Up Python

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Download The Raw Data

1. Open the public Figshare dataset page:
   <https://doi.org/10.6084/m9.figshare.32145994>
2. Click **Download all**.
3. Extract the downloaded archive.
4. Put the extracted participant folders under `data/` in this repository.

The code expects this layout:

```text
EduGage/
  data/
    P1/
      1_1_engagement_log.csv
      1_1_EEG.csv
      1_1_ACC.csv
      ...
    P2/
      2_1_engagement_log.csv
      2_1_EEG.csv
      2_1_ACC.csv
      ...
```

The `data/` directory is ignored by Git.

## 3. Raw Data Layout

Each participant/session folder should contain one required experiment log and
any available modality files with the same `<participant_id>_<session_id>`
prefix.

Required:

```text
<participant_id>_<session_id>_engagement_log.csv
```

Common optional files:

```text
<participant_id>_<session_id>_EEG.csv
<participant_id>_<session_id>_ACC.csv
<participant_id>_<session_id>_GYRO.csv
<participant_id>_<session_id>_PPG.csv
<participant_id>_<session_id>_Polar_ECG.csv
<participant_id>_<session_id>_msband_gsr.csv
<participant_id>_<session_id>_msband_hr.csv
<participant_id>_<session_id>_BeamEyeTracker.csv
<participant_id>_<session_id>_MARKERS.csv
<participant_id>_<session_id>_eSense.csv
<participant_id>_<session_id>_T-Ring_*.bin
```

The T-Ring `.bin` files are decoded by the preprocessing code; no manual ring
conversion step is required.

## 4. Preprocess The Dataset

Run these stages from the repository root after `data/` exists.

```bash
python scripts/run_pipeline.py --stage data --log-level INFO
python scripts/run_pipeline.py --stage labels --log-level INFO
python scripts/run_pipeline.py --stage preprocessed_data --log-level INFO
```

These commands create:

```text
artifacts/manifests/session_manifest.parquet
artifacts/windows/window_index.parquet
preprocessed_data/P<participant_id>/...
preprocessed_data/export_summary.csv
```

If Parquet support is unavailable, the pipeline writes CSV fallbacks next to the
Parquet target paths.

`preprocessed_data/` and `artifacts/` are generated outputs and are ignored by
Git.

## 5. Train And Evaluate

### Phase-E Embedding Pipeline

The `train_eval` stage uses the embedding table produced by the `features`
stage. The default evaluation uses the repository's four participant folds.

The fixed folds use Figshare participant IDs:

```text
Fold 1: 1, 2, 3, 10
Fold 2: 4, 5, 7, 9
Fold 3: 6, 8, 11, 12
Fold 4: 13, 14, 15, 16
```

```bash
python scripts/run_pipeline.py --stage features --log-level INFO
python scripts/run_pipeline.py --stage train_eval --log-level INFO
```

To recompute embeddings instead of using the cache:

```bash
python scripts/run_pipeline.py --stage features --force-recompute-embeddings --log-level INFO
```

The dedicated foundational embedding extraction scripts require their model
files to be available under `external_models/`.

### Multimodal Optuna Training

The Optuna stages train the native multimodal backend from the preprocessed
sensor windows.

```bash
python scripts/run_pipeline.py \
  --stage tune_optuna \
  --log-level INFO \
  --multimodal-task-mode ordinal \
  --multimodal-split-mode fixed_groups \
  --optuna-trials 20
```

## 6. Baselines

All baseline outputs are generated artifacts. Keep them outside Git or under
ignored output directories. The default baseline evaluation is the repository's
four participant folds.

### Label-Only Mean/Mode/Random Baselines

This runner uses the repository's fixed participant folds.

```bash
python src/engagement/baselines/mean_mode_random_baselines/run_baselines.py \
  --preprocessed-root preprocessed_data \
  --output-dir artifacts/baselines/mean_mode_random \
  --log-level INFO
```

### Classical ML Baselines

```bash
python src/engagement/baselines/ml_baseline/run_ml_baselines.py \
  --preprocessed-root preprocessed_data \
  --output-dir artifacts/baselines/ml_baselines \
  --models random_forest linear_regression svm \
  --split-mode fixed4 \
  --log-level INFO
```

Add `lgbm` to `--models` after installing `lightgbm`.

### Raw-Window Architecture Baselines

```bash
python src/engagement/baselines/foundational_architecture_baseline/run_architecture_models.py \
  --preprocessed-root preprocessed_data \
  --output-dir artifacts/baselines/architecture_models \
  --models deepconvlstm deepconvlstm_attention tinyhar \
  --split-mode fixed4 \
  --device auto \
  --log-level INFO
```

For a quick smoke run:

```bash
python src/engagement/baselines/foundational_architecture_baseline/run_architecture_models.py \
  --preprocessed-root preprocessed_data \
  --output-dir artifacts/baselines/architecture_smoke \
  --models tinyhar \
  --split-mode fixed4 \
  --max-windows 120 \
  --epochs 1 \
  --device cpu \
  --log-level WARNING
```

### Foundational Embedding Concat/Gated Baselines

These require generated embedding `.npz` files under
`output/foundational_embeddings/`.

```bash
python src/engagement/baselines/foundational_embeddings_concat_baseline/train_foundational_concat.py \
  --root . \
  --out-dir artifacts/baselines/foundational_concat
```

```bash
python src/engagement/baselines/foundational_embeddings_concat_baseline/train_foundational_gated.py \
  --root . \
  --out-dir artifacts/baselines/foundational_gated
```

### ConSensus Reference

This project used an unreleased ConSensus-based workflow for multi-agent
multimodal sensing. For methodological background, see
[ConSensus: Multi-Agent Collaboration for Multimodal Sensing](https://arxiv.org/abs/2601.06453)
by Yoon et al.

## 7. Output Locations

Common generated paths:

```text
artifacts/runs/<run_id>/...
artifacts/manifests/session_manifest.parquet
artifacts/windows/window_index.parquet
artifacts/optuna/<study_name>/...
artifacts/baselines/...
preprocessed_data/P<participant_id>/...
preprocessed_data/export_summary.csv
output/foundational_embeddings/...
```

These are intentionally ignored by Git.

## 8. Useful Smoke Checks

After installing dependencies:

```bash
python scripts/run_pipeline.py --help
python src/engagement/baselines/ml_baseline/run_ml_baselines.py --help
python src/engagement/baselines/mean_mode_random_baselines/run_baselines.py --help
python src/engagement/baselines/foundational_architecture_baseline/run_architecture_models.py --help
```

After preprocessing:

```bash
python src/engagement/baselines/ml_baseline/run_ml_baselines.py \
  --preprocessed-root preprocessed_data \
  --output-dir artifacts/baselines/ml_smoke \
  --models linear_regression \
  --split-mode fixed4 \
  --max-windows 240 \
  --log-level WARNING
```
