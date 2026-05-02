param(
    [string]$ExternalRoot = "external_models",
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"

$models = @(
    @{
        Modality = "eeg"
        Name = "EEGPT"
        Repo = "https://github.com/BINE022/EEGPT"
        Commit = "a0e0a8fad729e2ecf4eedb3a81548a6e6d48a705"
        Checkpoint = "eegpt/model.ckpt"
    },
    @{
        Modality = "ecg"
        Name = "ECGFounder"
        Repo = "https://github.com/PKUDigitalHealth/ECGFounder"
        Commit = "04edac702b61c91face519774ddcc0cd712fef23"
        Checkpoint = "ecgfounder/model.ckpt"
    },
    @{
        Modality = "ppg"
        Name = "PulsePPG"
        Repo = "https://github.com/maxxu05/pulseppg"
        Commit = "716eaf9cf966e8f76436f2263872ef38b1f90166"
        Checkpoint = "pulseppg/model.ckpt"
    },
    @{
        Modality = "eda_imu"
        Name = "NormWear"
        Repo = "https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear"
        Commit = "f3bc7439efcf30bf9df36566f87b35a277f9cc5e"
        Checkpoint = "normwear/model.ckpt"
    },
    @{
        Modality = "eye"
        Name = "EyeFeaturesMLP"
        Repo = "local"
        Commit = "local"
        Checkpoint = "eye/projector.ckpt"
    }
)

$root = Resolve-Path -Path "."
$externalPath = Join-Path $root $ExternalRoot
New-Item -ItemType Directory -Force -Path $externalPath | Out-Null

$summary = @()

foreach ($model in $models) {
    $repoName = [System.IO.Path]::GetFileNameWithoutExtension($model.Repo)
    if ([string]::IsNullOrWhiteSpace($repoName)) {
        $repoName = $model.Modality
    }

    $repoPath = Join-Path $externalPath $repoName

    if ($model.Repo -ne "local" -and -not (Test-Path $repoPath)) {
        if ($ValidateOnly) {
            Write-Host "[WARN] Missing repo for $($model.Name): $repoPath"
        }
        else {
            git clone $model.Repo $repoPath
        }
    }

    if ((Test-Path $repoPath) -and $model.Commit -ne "local" -and -not $ValidateOnly) {
        git -c safe.directory=$repoPath -C $repoPath fetch --all --tags
        git -c safe.directory=$repoPath -C $repoPath checkout $model.Commit
    }

    $checkpointPath = Join-Path $externalPath $model.Checkpoint
    $checkpointExists = Test-Path $checkpointPath

    $summary += [PSCustomObject]@{
        modality = $model.Modality
        model_name = $model.Name
        repo = $model.Repo
        commit = $model.Commit
        checkpoint = $checkpointPath
        checkpoint_exists = $checkpointExists
    }
}

$summary | Format-Table -AutoSize

$missing = $summary | Where-Object { -not $_.checkpoint_exists }
if ($missing.Count -gt 0) {
    Write-Host "`n[WARN] Some checkpoints are missing:"
    $missing | ForEach-Object { Write-Host ("- {0} -> {1}" -f $_.modality, $_.checkpoint) }
    exit 1
}

Write-Host "`nAll model checkpoints are available."






