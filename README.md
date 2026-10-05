<p align="center">
  <img src="https://github.com/user-attachments/assets/547b5f1b-b886-4f72-b19e-e9defcc27612" alt="EduGage logo" width="140" />
</p>

<h1 align="center">EduGage</h1>

This repository contains the processing and modeling code for the EduGage
Dataset. The code discovers raw participant-session folders, builds labeled
engagement windows, exports preprocessed sensor slices, and runs several
training/evaluation pipelines and baselines.

The public raw dataset is hosted on Figshare:

<https://figshare.com/s/cc3a50e1f3724f7ed5f4>

The Figshare item is titled **EduGage Dataset**. It is a 3.05 GB raw dataset
with one folder per participant session. The files are raw collection outputs,
not cleaned/aligned/windowed preprocessing products.

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

1. Open the Figshare private share link:
   <https://figshare.com/s/cc3a50e1f3724f7ed5f4>
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

Parallel GPU workers:

```bash
python scripts/run_pipeline.py \
  --stage tune_optuna_parallel \
  --log-level INFO \
  --multimodal-task-mode ordinal \
  --multimodal-split-mode fixed_groups \
  --optuna-trials 24 \
  --optuna-workers-per-gpu 2 \
  --optuna-gpus 0,1
```

Outputs are written under `artifacts/optuna/<study_name>/`.

For each outer participant fold, 20% of the outer-training windows are held out as
a label-stratified validation split. The epoch/checkpoint is selected by validation
Class MAE (validation binary Macro-F1 in binary-only mode), and the selected
checkpoint is evaluated on the outer test fold once. 

To rerun the fixed configuration that previously produced a mean four-fold test
Class MAE of approximately 0.91:

```bash
python scripts/run_pipeline.py \
  --stage train_eval \
  --run-id reproduce_class_mae_091 \
  --multimodal-reproduction-preset class_mae_091
```

The preset uses seed 42, deterministic single-GPU training, fold seeds 42--45,
40 maximum epochs, patience 8, a 20% stratified validation split, all 11
modalities on complete windows, learning rate 0.0009009827, weight decay
0.0001493374, batch size 16, embedding dimension 32, fusion dimension 256,
ordinal-loss weight 4, regression-loss weight 1. Exact equality can still depend on
using the same PyTorch/CUDA versions, and GPU model.

### Trial Matrices

Run a fixed list of multimodal configurations:

```bash
python scripts/run_pipeline.py \
  --stage trial_matrix \
  --trial-matrix-config configs/trial_matrices/example_full_complete_trials.json \
  --log-level INFO
```

Run Optuna for each trial-matrix subset:

```bash
python scripts/run_pipeline.py \
  --stage trial_matrix_optuna \
  --trial-matrix-config configs/trial_matrices/example_subset_optuna_trials.json \
  --trial-matrix-optuna-trials 24 \
  --log-level INFO
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
