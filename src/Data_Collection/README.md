# EduGage Data Collection Code

This directory contains the source code used to collect the raw EduGage
experiment streams released in the public Figshare dataset. It is a curated
source-code snapshot: generated data, build products, local device identifiers,
and redistributable third-party SDK bundles are intentionally excluded.

The public dataset is available at:

<https://doi.org/10.6084/m9.figshare.32145994>

## Included Collection Components

- `experiment_player/`: PsychoPy video player and engagement-report logger.
- `Muse/`: Muse OSC receiver for EEG, PPG, accelerometer, gyroscope, and marker streams.
- `Beam/`: Beam eye-tracker logger wrapper. The Beam SDK itself is not included.
- `Polar/`: Polar H10 ECG BLE logger.
- `MSBand/`: Microsoft Band TCP receiver and UWP streamer source for GSR and heart rate.
- `eSense/`: eSense IMU notebook and requirements.
- `T-Ring/`: Notes for the Android-side T-Ring collection/export used for the raw `.bin` files.

OmniBuds code is not included because OmniBuds data are not part of the public
EduGage Figshare release.

## Configuration

`config.json` is a sanitized example configuration used by the shared helpers
and experiment player. Before collecting a session, edit:

- `participant_id`
- `session_id`
- `video_uids`
- any local video paths in `experiment_player/video_catalog.json`

The collection scripts prefix output filenames with
`<participant_id>_<session_id>`.

## Excluded From This Snapshot

- Raw participant data and generated CSV/BIN outputs.
- Python caches and macOS metadata files.
- Beam Eye Tracker SDK files. Download the SDK from Eyeware and follow
  `Beam/README.md`.
- Microsoft Band build outputs, Visual Studio cache directories, packaged
  installers, and binaries.