# Engagement Experiment (PsychoPy)

## What this does
- Loads an experiment config (`config.json`) plus a shared video catalog (`video_catalog.json`).
- Experiment config specifies the ordered list of `video_uids` to run.
- Video catalog stores the shared mapping from `uid` to video `path` and trigger timestamps.
- Flow:
  - Welcome screen
  - Prompt description screen
  - Pre-Test screen (waits for `SPACE`)
  - Video playback
  - Break/Post-Test screen between videos (waits for `SPACE`)
  - Experiment Complete screen (waits for `SPACE` then exits)
- During video playback:
  - At each trigger timestamp, sends a TCP message.
  - Shows an opaque on-screen box with `Report engagement now` for 5 seconds (configurable).
  - Pauses video playback until the participant enters an engagement report.
- Logs CSV at 10 Hz (configurable): `video_uid`, `participant_id`, `session_id`, live video timestamp, `time.time()`, flash on/off, and one-shot engagement report (`1-5` or `X`, else `N/A`).

## Requirements
- Python 3.9+
- PsychoPy installed in your experiment environment.

## Run
macOS / Linux:
```bash
python3 Data_Collection/experiment_player/run_engagement_experiment.py
```

Windows:
```bash
python Data_Collection\experiment_player\run_engagement_experiment.py
```

The script automatically loads:
- `Data_Collection/config.json`

## Config notes
- `Data_Collection/config.json` can use:
  - New format (recommended): `video_catalog_path` + `video_uids`
  - Legacy format: inline `videos` list (still supported)
- `video_uids` order controls playback order.
- `video_catalog_path` is resolved relative to `Data_Collection/config.json`.
- Video file `path` values inside `video_catalog.json` are resolved relative to `video_catalog.json`.
- Use forward slashes or escaped backslashes in Windows paths inside `Data_Collection/config.json`.
- Edit `Data_Collection/config.json` for participant/session and run settings.
- Edit `Data_Collection/experiment_player/video_catalog.json` for shared video definitions.
- Optional top-level config fields:
  - `participant_id` (default: `unknown_participant`)
  - `session_id` (default: current timestamp)
  - `allow_no_audio_fallback` (default: `true`; if PsychoPy fails to open a video with audio, retry muted)
  - `window.size` (optional; only used when `window.fullscr` is `false`)
- MSBand haptics (no config needed):
  - The experiment sends `START`/`STOP` to the local BandReceiver command bridge (`127.0.0.1:9899`).
  - `BandReceiver.py` forwards those commands over the existing C# MSBandStreamer socket.
  - This uses the same C# stream connection for commands and sensor data; the experiment does not need MSBand host/port settings.
- Trigger timestamps are in seconds from start of each video.
- `tcp.message_template` supports:
  - `{video_uid}`
  - `{trigger_ts}`

## Keys during experiment
- `SPACE`: advance from non-video screens
- `1`, `2`, `3`, `4`, `5`, `X`: engagement report while video plays
- `ESCAPE`: exit early

## Notes on Video Compatibility (macOS/Windows)
- The script is OS-agnostic, but PsychoPy's movie backend (`ffpyplayer`) can fail on some MP4 encodes.
- If a video fails to open, re-encode to a standard MP4 (H.264 video + AAC audio) and try again.
- FFmpeg warnings about colorspace conversion are common and usually not fatal.
