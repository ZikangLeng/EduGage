# Beam Eye Tracker Logger

`beam.py` records Beam gaze and head-pose samples to CSV.

The Eyeware Beam SDK is not included in this repository because its license does
not allow uploading the SDK bundle to an internet repository. Install or
download Beam Eye Tracker SDK 2.1.0 from Eyeware, then either:

1. Set `BEAM_SDK_PYTHON_PACKAGE` to the SDK Python package directory, or
2. Place the SDK at `Data_Collection/Beam/beam_eye_tracker_sdk/beam_eye_tracker_sdk-2.1.0/`.

The expected package directory is:

```text
beam_eye_tracker_sdk-2.1.0/python/package
```

In the Beam app, enable **Gaming extensions** before running the logger.

```bash
python Data_Collection/Beam/beam.py
```
