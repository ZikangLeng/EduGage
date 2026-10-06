#!/usr/bin/env python3
"""Config-driven engagement experiment using PsychoPy."""

from __future__ import annotations

import csv
import inspect
import json
import os
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DATA_COLLECTION_DIR = Path(__file__).resolve().parents[1]
if str(DATA_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_COLLECTION_DIR))
import config as dc_config

from psychopy import core, event, visual
from psychopy.constants import FINISHED


DEFAULT_VIDEO_CATALOG_FILENAME = "video_catalog.json"
DEFAULT_TCP_HOST = "127.0.0.1"
DEFAULT_TCP_PORT = 5001
DEFAULT_TCP_TEMPLATE = "TRIGGER|uid={video_uid}|ts={trigger_ts:.3f}"
DEFAULT_MSBAND_BRIDGE_HOST = "127.0.0.1"
DEFAULT_MSBAND_BRIDGE_PORT = 9899
DEFAULT_MSBAND_START_MESSAGE = "START"
DEFAULT_MSBAND_STOP_MESSAGE = "STOP"
DEFAULT_MSBAND_BRIDGE_TIMEOUT_SEC = 0.25
DEFAULT_FLASH_DURATION_SEC = 5.0
DEFAULT_LOG_HZ = 10.0
MOVIE_END_EPSILON_SEC = 0.05

VALID_REPORT_KEYS = {"1", "2", "3", "4", "5", "x"}
FULLSCREEN_TOGGLE_KEYS = {"f", "f11"}
FULLSCREEN_TOGGLE_DEBOUNCE_SEC = 0.2
_MSBAND_BRIDGE_AVAILABLE: bool | None = None
_LAST_FULLSCREEN_TOGGLE_PERF = -1.0
_FFPY_RESYNC_INFO_PRINTED = False
_VLC_PATHS_CONFIGURED = False

PROMPT_TEXT = (
    "Prompt:\n"
    "How difficult was it to pay attention during the last minute of the lecture? (1-5)\n\n"
    "1: My attention is fused with the lecture; it is completely automatic and effortless.\n"
    "2\n"
    "3\n"
    "4\n"
    "5: I am forcing my attention. It feels like a heavy, conscious struggle to keep up with the lesson.\n"
    "X: External Distraction/Interruption - Use this if something outside the lecture forced your attention away entirely.\n\n"
    "When it is time to report, the band will vibrate, the video will pause, "
    "and the screen will show \"Report engagement now\".\n"
    "Press 1, 2, 3, 4, 5, or X to report. The video will resume after you respond.\n\n"
    "Press SPACE to continue."
)


class ExperimentAbort(Exception):
    """Raised when user exits early."""


@dataclass(frozen=True)
class VideoTrial:
    uid: str
    path: Path
    trigger_timestamps: list[float]


@dataclass(frozen=True)
class ExperimentConfig:
    participant_id: str
    session_id: str
    videos: list[VideoTrial]
    flash_duration_sec: float
    log_hz: float
    tcp_host: str
    tcp_port: int
    tcp_message_template: str
    tcp_timeout_sec: float
    output_csv: Path
    allow_no_audio_fallback: bool
    require_vlc_backend: bool
    fullscr: bool
    screen: int
    window_size: tuple[int, int] | None


def _parse_video_trial_item(item: dict[str, Any], *, base_dir: Path, label: str) -> VideoTrial:
    uid = str(item.get("uid", "")).strip()
    path_raw = str(item.get("path", "")).strip()
    timestamps_raw = item.get("trigger_timestamps_sec", [])

    if not uid:
        raise ValueError(f"{label}.uid is required")
    if not path_raw:
        raise ValueError(f"{label}.path is required")
    if not isinstance(timestamps_raw, list):
        raise ValueError(f"{label}.trigger_timestamps_sec must be a list")

    resolved_path = Path(path_raw)
    if not resolved_path.is_absolute():
        resolved_path = (base_dir / resolved_path).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Video file not found: {resolved_path}")

    trigger_timestamps: list[float] = []
    for ts in timestamps_raw:
        val = float(ts)
        if val < 0:
            raise ValueError(f"{label} has negative timestamp: {val}")
        trigger_timestamps.append(val)
    trigger_timestamps.sort()

    return VideoTrial(
        uid=uid,
        path=resolved_path,
        trigger_timestamps=trigger_timestamps,
    )


def _load_video_catalog(catalog_path: Path) -> dict[str, VideoTrial]:
    with catalog_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    catalog: dict[str, VideoTrial] = {}
    base_dir = catalog_path.parent

    if isinstance(raw, dict) and isinstance(raw.get("videos_by_uid"), dict):
        items = raw["videos_by_uid"]
        for uid_key, item in items.items():
            if not isinstance(item, dict):
                raise ValueError(f"videos_by_uid['{uid_key}'] must be an object")
            merged = dict(item)
            merged.setdefault("uid", uid_key)
            trial = _parse_video_trial_item(
                merged,
                base_dir=base_dir,
                label=f"videos_by_uid['{uid_key}']",
            )
            if trial.uid in catalog:
                raise ValueError(f"Duplicate video uid in catalog: {trial.uid}")
            catalog[trial.uid] = trial
        return catalog

    if isinstance(raw, dict) and isinstance(raw.get("videos"), list):
        items = raw["videos"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(
            "Video catalog must be a list of videos, or an object with 'videos' or 'videos_by_uid'."
        )

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"catalog videos[{idx}] must be an object")
        trial = _parse_video_trial_item(
            item,
            base_dir=base_dir,
            label=f"catalog videos[{idx}]",
        )
        if trial.uid in catalog:
            raise ValueError(f"Duplicate video uid in catalog: {trial.uid}")
        catalog[trial.uid] = trial

    return catalog


def load_config(config_path: Path) -> ExperimentConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    videos: list[VideoTrial] = []
    if isinstance(raw.get("video_uids"), list):
        video_uids = [str(uid).strip() for uid in raw["video_uids"]]
        if not video_uids or any(not uid for uid in video_uids):
            raise ValueError("Config field 'video_uids' must be a non-empty list of strings")

        catalog_path_raw = str(raw.get("video_catalog_path", DEFAULT_VIDEO_CATALOG_FILENAME)).strip()
        if not catalog_path_raw:
            raise ValueError("Config field 'video_catalog_path' cannot be empty when using video_uids")
        catalog_path = Path(catalog_path_raw)
        if not catalog_path.is_absolute():
            catalog_path = (config_path.parent / catalog_path).resolve()
        if not catalog_path.exists():
            raise FileNotFoundError(f"Video catalog file not found: {catalog_path}")

        catalog = _load_video_catalog(catalog_path)
        for idx, uid in enumerate(video_uids):
            if uid not in catalog:
                raise ValueError(f"video_uids[{idx}]='{uid}' not found in catalog {catalog_path}")
            videos.append(catalog[uid])
    elif isinstance(raw.get("videos"), list):
        if not raw["videos"]:
            raise ValueError("Config field 'videos' must not be empty")
        for idx, item in enumerate(raw["videos"]):
            if not isinstance(item, dict):
                raise ValueError(f"videos[{idx}] must be an object")
            videos.append(
                _parse_video_trial_item(
                    item,
                    base_dir=config_path.parent,
                    label=f"videos[{idx}]",
                )
            )
    else:
        raise ValueError(
            "Config must contain either 'videos' (legacy inline format) "
            "or 'video_uids' with 'video_catalog_path'."
        )

    tcp_cfg = raw.get("tcp", {}) if isinstance(raw.get("tcp", {}), dict) else {}
    out_cfg = raw.get("output", {}) if isinstance(raw.get("output", {}), dict) else {}
    win_cfg = raw.get("window", {}) if isinstance(raw.get("window", {}), dict) else {}

    output_csv = out_cfg.get("csv_path")
    if output_csv:
        csv_path = Path(str(output_csv))
        if not csv_path.is_absolute():
            csv_path = (config_path.parent / csv_path).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = (config_path.parent / f"engagement_log_{timestamp}.csv").resolve()
    csv_path = dc_config.prefixed_path(csv_path)

    window_size_raw = win_cfg.get("size")
    window_size: tuple[int, int] | None = None
    if isinstance(window_size_raw, list) and len(window_size_raw) == 2:
        window_size = (int(window_size_raw[0]), int(window_size_raw[1]))

    flash_duration_sec = float(raw.get("flash_duration_sec", DEFAULT_FLASH_DURATION_SEC))
    log_hz = float(raw.get("log_hz", DEFAULT_LOG_HZ))
    if flash_duration_sec < 0:
        raise ValueError("flash_duration_sec must be >= 0")
    if log_hz <= 0:
        raise ValueError("log_hz must be > 0")

    return ExperimentConfig(
        participant_id=str(raw.get("participant_id", "unknown_participant")),
        session_id=str(raw.get("session_id", datetime.now().strftime("%Y%m%d_%H%M%S"))),
        videos=videos,
        flash_duration_sec=flash_duration_sec,
        log_hz=log_hz,
        tcp_host=str(tcp_cfg.get("host", DEFAULT_TCP_HOST)),
        tcp_port=int(tcp_cfg.get("port", DEFAULT_TCP_PORT)),
        tcp_message_template=str(
            tcp_cfg.get("message_template", DEFAULT_TCP_TEMPLATE)
        ),
        tcp_timeout_sec=float(tcp_cfg.get("timeout_sec", 0.25)),
        output_csv=csv_path,
        allow_no_audio_fallback=bool(raw.get("allow_no_audio_fallback", True)),
        require_vlc_backend=bool(raw.get("require_vlc_backend", True)),
        fullscr=bool(win_cfg.get("fullscr", False)),
        screen=int(win_cfg.get("screen", 0)),
        window_size=window_size,
    )


def send_tcp_message(host: str, port: int, timeout_sec: float, payload: str, label: str) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_sec) as sock:
            sock.sendall(payload.encode("utf-8"))
        return True
    except OSError as exc:
        print(
            f"[WARN] Failed {label} TCP send to {host}:{port}: {exc}",
            file=sys.stderr,
        )
        return False


def send_tcp_message_async(host: str, port: int, timeout_sec: float, payload: str, label: str) -> None:
    threading.Thread(
        target=send_tcp_message,
        args=(host, port, timeout_sec, payload, label),
        daemon=True,
    ).start()


def send_tcp_trigger(config: ExperimentConfig, video_uid: str, trigger_ts: float) -> None:
    payload = config.tcp_message_template.format(
        video_uid=video_uid,
        trigger_ts=trigger_ts,
    )
    send_tcp_message_async(
        config.tcp_host,
        config.tcp_port,
        config.tcp_timeout_sec,
        payload,
        "trigger",
    )


def probe_msband_bridge_available() -> bool:
    global _MSBAND_BRIDGE_AVAILABLE
    if _MSBAND_BRIDGE_AVAILABLE is not None:
        return _MSBAND_BRIDGE_AVAILABLE
    try:
        with socket.create_connection(
            (DEFAULT_MSBAND_BRIDGE_HOST, DEFAULT_MSBAND_BRIDGE_PORT),
            timeout=0.05,
        ):
            pass
        _MSBAND_BRIDGE_AVAILABLE = True
    except OSError:
        _MSBAND_BRIDGE_AVAILABLE = False
        print(
            "[INFO] MSBand command bridge not detected on "
            f"{DEFAULT_MSBAND_BRIDGE_HOST}:{DEFAULT_MSBAND_BRIDGE_PORT}; "
            "haptic commands disabled for this run.",
            file=sys.stderr,
        )
    return _MSBAND_BRIDGE_AVAILABLE


def send_msband_haptics_command(config: ExperimentConfig, *, start: bool) -> None:
    global _MSBAND_BRIDGE_AVAILABLE
    if not probe_msband_bridge_available():
        return
    payload = (
        DEFAULT_MSBAND_START_MESSAGE
        if start
        else DEFAULT_MSBAND_STOP_MESSAGE
    )
    ok = send_tcp_message(
        DEFAULT_MSBAND_BRIDGE_HOST,
        DEFAULT_MSBAND_BRIDGE_PORT,
        DEFAULT_MSBAND_BRIDGE_TIMEOUT_SEC,
        payload,
        "MSBand haptics",
    )
    if not ok:
        _MSBAND_BRIDGE_AVAILABLE = False


def _configure_vlc_runtime_paths() -> None:
    global _VLC_PATHS_CONFIGURED
    if _VLC_PATHS_CONFIGURED:
        return
    _VLC_PATHS_CONFIGURED = True

    if os.environ.get("PYTHON_VLC_LIB_PATH"):
        return

    candidates: list[tuple[Path, Path]] = []
    if sys.platform == "darwin":
        candidates.extend(
            [
                (
                    Path("/Applications/VLC.app/Contents/MacOS/lib/libvlc.dylib"),
                    Path("/Applications/VLC.app/Contents/MacOS/plugins"),
                ),
                (
                    Path.home() / "Applications/VLC.app/Contents/MacOS/lib/libvlc.dylib",
                    Path.home() / "Applications/VLC.app/Contents/MacOS/plugins",
                ),
            ]
        )
    elif os.name == "nt":
        for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
            if not base:
                continue
            vlc_dir = Path(base) / "VideoLAN" / "VLC"
            candidates.append((vlc_dir / "libvlc.dll", vlc_dir / "plugins"))

    for libvlc_path, plugins_path in candidates:
        if not libvlc_path.exists():
            continue
        os.environ.setdefault("PYTHON_VLC_LIB_PATH", str(libvlc_path))
        if plugins_path.exists():
            os.environ.setdefault("PYTHON_VLC_MODULE_PATH", str(plugins_path))
            os.environ.setdefault("VLC_PLUGIN_PATH", str(plugins_path))
        print(f"[INFO] Configured VLC runtime from: {libvlc_path}", file=sys.stderr)
        return


def create_movie_stim(
    win: visual.Window,
    video_path: Path,
    *,
    require_vlc_backend: bool,
    allow_no_audio_fallback: bool,
) -> Any:
    _configure_vlc_runtime_paths()
    classes: list[type] = []
    vlc_import_error: Exception | None = None
    try:
        from psychopy.visual.vlcmoviestim import VlcMovieStim  # type: ignore

        classes.append(VlcMovieStim)
    except Exception as exc:
        vlc_import_error = exc
        if require_vlc_backend:
            raise RuntimeError(
                "VLC backend is required for reliable audio/video sync but could not be loaded. "
                "Install VLC and ensure libVLC is available, then rerun. "
                f"Import error: {exc!r}"
            ) from exc
    for name in ("MovieStim", "MovieStim3"):
        if hasattr(visual, name):
            cls = getattr(visual, name)
            if cls not in classes:
                classes.append(cls)

    if not classes:
        raise RuntimeError("No PsychoPy movie stim class found (MovieStim/MovieStim3).")

    last_error: Exception | None = None
    last_error_detail: str | None = None
    for cls in classes:
        try:
            sig = inspect.signature(cls.__init__)
            params = set(sig.parameters.keys())
        except (TypeError, ValueError):
            params = set()

        base_kwargs: dict[str, Any] = {"win": win}
        if "filename" in params:
            base_kwargs["filename"] = str(video_path)
        elif "movieFile" in params:
            base_kwargs["movieFile"] = str(video_path)
        else:
            base_kwargs["filename"] = str(video_path)

        # Using explicit norm units and size avoids some PsychoPy metadata/size inference paths.
        if "units" in params:
            base_kwargs["units"] = "norm"
        if "size" in params:
            base_kwargs["size"] = (2.0, 2.0)
        if "pos" in params:
            base_kwargs["pos"] = (0.0, 0.0)
        if "loop" in params:
            base_kwargs["loop"] = False

        no_audio_options: list[bool | None] = [None]
        if "noAudio" in params:
            no_audio_options = [False, True] if allow_no_audio_fallback else [False]

        for no_audio_value in no_audio_options:
            kwargs = dict(base_kwargs)
            if no_audio_value is not None:
                kwargs["noAudio"] = no_audio_value
            try:
                movie = cls(**kwargs)
                try:
                    if hasattr(movie, "size") and getattr(movie, "size") is None:
                        movie.size = (2.0, 2.0)
                except Exception:
                    pass

                if no_audio_value is True:
                    print(
                        "[WARN] Video opened with noAudio=True fallback. "
                        "Audio will be disabled for this video.",
                        file=sys.stderr,
                    )
                elif getattr(cls, "__name__", "") == "VlcMovieStim":
                    print("[INFO] Using VlcMovieStim backend for video playback.", file=sys.stderr)
                else:
                    print(
                        f"[INFO] Using {_movie_backend_name(movie)} backend for video playback.",
                        file=sys.stderr,
                    )
                return movie
            except Exception as exc:
                last_error = exc
                last_error_detail = (
                    f"class={getattr(cls, '__name__', cls)} kwargs_keys={sorted(kwargs.keys())} "
                    f"noAudio={kwargs.get('noAudio', '<default>')} error={exc!r}"
                )
                continue

    hint = ""
    if last_error is not None and "NoneType" in str(last_error):
        hint = (
            " PsychoPy/ffpyplayer sometimes fails on MP4 metadata/audio init for certain encodes. "
            "Try re-encoding the video to H.264/AAC MP4 with standard timestamps."
        )
    if last_error_detail:
        hint += f" Last attempt: {last_error_detail}"
    if vlc_import_error is not None:
        hint += f" VLC import error: {vlc_import_error!r}"
    raise RuntimeError(f"Unable to open video {video_path}: {last_error}.{hint}")


def _movie_backend_name(movie: Any) -> str:
    return getattr(type(movie), "__name__", "")


def _is_ffpyplayer_movie(movie: Any) -> bool:
    return _movie_backend_name(movie) in {"MovieStim", "MovieStim3"}


def _seek_movie_to_time(movie: Any, target_time: float) -> None:
    seek = getattr(movie, "seek", None)
    if not callable(seek):
        return
    ts = max(0.0, float(target_time))
    try:
        seek(ts, log=False)
        return
    except TypeError:
        pass
    except Exception:
        return
    try:
        seek(ts)
    except Exception:
        return


def _movie_is_playing(movie: Any) -> bool | None:
    attr = getattr(movie, "isPlaying", None)
    try:
        value = attr() if callable(attr) else attr
    except Exception:
        return None
    return value if isinstance(value, bool) else None


def _extract_size_pair(value: Any) -> tuple[float, float] | None:
    try:
        if callable(value):
            value = value()
    except Exception:
        return None
    if value is None:
        return None
    try:
        w = float(value[0])
        h = float(value[1])
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return (w, h)


def _movie_native_size(movie: Any) -> tuple[float, float] | None:
    for name in ("videoSize", "frameSize", "origSize", "_origSize", "size"):
        pair = _extract_size_pair(getattr(movie, name, None))
        if pair is not None:
            return pair
    return None


def _fit_movie_to_window(movie: Any, win: visual.Window) -> None:
    """Fit movie to current window while preserving aspect ratio."""
    try:
        if hasattr(movie, "units"):
            movie.units = "pix"
    except Exception:
        pass
    win_size = _extract_size_pair(getattr(win, "size", None))
    src_size = _movie_native_size(movie)
    target_size: tuple[float, float] | None = None
    if win_size is not None and src_size is not None:
        win_w, win_h = win_size
        src_w, src_h = src_size
        scale = min(win_w / src_w, win_h / src_h)
        if scale > 0:
            target_size = (max(1.0, src_w * scale), max(1.0, src_h * scale))
    if target_size is None and win_size is not None:
        target_size = win_size
    try:
        if target_size is not None:
            movie.size = target_size
    except Exception:
        pass
    try:
        if hasattr(movie, "pos"):
            movie.pos = (0.0, 0.0)
    except Exception:
        pass


def get_movie_time_sec(movie: Any, fallback_clock: core.Clock) -> float:
    accessors = []
    if hasattr(movie, "getCurrentFrameTime"):
        accessors.append(lambda: movie.getCurrentFrameTime())
    if hasattr(movie, "getCurrentFrameTimeSec"):
        accessors.append(lambda: movie.getCurrentFrameTimeSec())
    accessors.append(lambda: getattr(movie, "pts", None))
    accessors.append(lambda: getattr(movie, "t", None))

    for accessor in accessors:
        try:
            value = accessor()
        except Exception:
            continue
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            continue

    return max(0.0, fallback_clock.getTime())


def is_movie_finished(movie: Any, current_time: float) -> bool:
    status = getattr(movie, "status", None)
    if status == FINISHED:
        return True
    # Some PsychoPy movie classes expose boolean end-state flags instead of
    # (or before updating) `status == FINISHED`.
    for name in ("isFinished", "finished"):
        attr = getattr(movie, name, None)
        try:
            value = attr() if callable(attr) else attr
        except Exception:
            value = None
        if value is True:
            return True

    duration = getattr(movie, "duration", None)
    try:
        duration_f = float(duration)
        if duration_f > 0 and current_time >= max(0.0, duration_f - MOVIE_END_EPSILON_SEC):
            return True
    except (TypeError, ValueError):
        pass

    return False


def check_abort() -> None:
    keys = event.getKeys(keyList=["escape"])
    if "escape" in keys:
        raise ExperimentAbort()


def maybe_toggle_fullscreen(win: visual.Window, key: str) -> bool:
    global _LAST_FULLSCREEN_TOGGLE_PERF
    low = key.lower()
    if low not in FULLSCREEN_TOGGLE_KEYS:
        return False

    now_perf = time.perf_counter()
    if (now_perf - _LAST_FULLSCREEN_TOGGLE_PERF) < FULLSCREEN_TOGGLE_DEBOUNCE_SEC:
        return True

    try:
        current = bool(getattr(win, "fullscr", False))
        win.fullscr = (not current)
        _LAST_FULLSCREEN_TOGGLE_PERF = now_perf
        print(
            f"[INFO] Fullscreen {'ON' if not current else 'OFF'}",
            file=sys.stderr,
        )
    except Exception as exc:
        print(f"[WARN] Fullscreen toggle failed: {exc}", file=sys.stderr)
    return True


def wait_for_space(win: visual.Window, stim: visual.TextStim) -> None:
    wait_for_key(win, stim, ["space"])


def wait_for_key(win: visual.Window, stim: visual.TextStim, accepted_keys: list[str]) -> str:
    event.clearEvents()
    key_list = list(dict.fromkeys(accepted_keys + ["escape"] + list(FULLSCREEN_TOGGLE_KEYS)))
    while True:
        check_abort()
        stim.draw()
        win.flip()
        keys = event.getKeys(keyList=key_list)
        if "escape" in keys:
            raise ExperimentAbort()
        for key in keys:
            if maybe_toggle_fullscreen(win, key):
                continue
            if key in accepted_keys:
                return key


def build_text(
    win: visual.Window,
    text: str,
    *,
    height: float = 0.045,
    wrap_width: float = 1.6,
) -> visual.TextStim:
    common = dict(
        win=win,
        text=text,
        color="black",
        height=height,
        wrapWidth=wrap_width,
    )
    try:
        return visual.TextStim(
            **common,
            alignText="left",
            anchorHoriz="center",
            anchorVert="center",
        )
    except TypeError:
        return visual.TextStim(
            **common,
            alignHoriz="left",
        )


def play_video_trial(
    win: visual.Window,
    config: ExperimentConfig,
    trial: VideoTrial,
    csv_writer: csv.writer,
    flash_rect: visual.Rect,
    flash_text: visual.TextStim,
) -> None:
    global _FFPY_RESYNC_INFO_PRINTED
    movie = create_movie_stim(
        win,
        trial.path,
        require_vlc_backend=config.require_vlc_backend,
        allow_no_audio_fallback=config.allow_no_audio_fallback,
    )
    msband_buzzing = False
    movie_paused = False
    manual_pause = False
    report_cue_active = False
    skip_video_requested = False
    last_space_toggle_perf = -1.0
    paused_movie_time_sec: float | None = None
    previous_win_color: Any | None = None

    if _is_ffpyplayer_movie(movie) and not _FFPY_RESYNC_INFO_PRINTED:
        _FFPY_RESYNC_INFO_PRINTED = True
        print(
            "[INFO] ffpyplayer backend detected; enabling seek-on-resume AV resync.",
            file=sys.stderr,
        )

    try:
        try:
            previous_win_color = win.color
            if hasattr(previous_win_color, "copy"):
                previous_win_color = previous_win_color.copy()
        except Exception:
            previous_win_color = None
        try:
            win.color = "black"
        except Exception:
            pass

        _fit_movie_to_window(movie, win)
        last_window_size = tuple(int(v) for v in win.size)
        if hasattr(movie, "play"):
            movie.play()

        video_clock = core.Clock()
        next_log_perf = time.perf_counter()
        log_interval = 1.0 / config.log_hz

        trigger_idx = 0
        pending_reports: deque[str] = deque()

        event.clearEvents()

        while True:
            cue_was_active_at_frame_start = report_cue_active
            keys = event.getKeys(
                keyList=list(VALID_REPORT_KEYS) + ["space", "tab", "escape"] + list(FULLSCREEN_TOGGLE_KEYS)
            )
            if "escape" in keys:
                raise ExperimentAbort()
            report_pressed_this_frame = False
            space_pressed_this_frame = False
            for key in keys:
                low = key.lower()
                if maybe_toggle_fullscreen(win, key):
                    continue
                if low in VALID_REPORT_KEYS:
                    report_pressed_this_frame = True
                    pending_reports.append("X" if low == "x" else low)
                elif low == "space":
                    space_pressed_this_frame = True
                elif low == "tab":
                    skip_video_requested = True

            movie_time = get_movie_time_sec(movie, video_clock)

            while (
                trigger_idx < len(trial.trigger_timestamps)
                and movie_time >= trial.trigger_timestamps[trigger_idx]
            ):
                trigger_ts = trial.trigger_timestamps[trigger_idx]
                send_tcp_trigger(config, trial.uid, trigger_ts)
                report_cue_active = True
                trigger_idx += 1

            # Dismiss the on-screen cue and resume only when a report is entered.
            if report_pressed_this_frame and report_cue_active:
                report_cue_active = False
            if skip_video_requested:
                report_cue_active = False
                manual_pause = False

            if space_pressed_this_frame and (not cue_was_active_at_frame_start) and (not report_cue_active):
                now_perf = time.perf_counter()
                if (now_perf - last_space_toggle_perf) >= 0.15:
                    manual_pause = not manual_pause
                    last_space_toggle_perf = now_perf

            flash_on = int(report_cue_active)
            target_paused = bool(report_cue_active or manual_pause)
            if target_paused != movie_paused:
                if target_paused:
                    paused_movie_time_sec = movie_time
                    for method_name in ("pause",):
                        method = getattr(movie, method_name, None)
                        if callable(method):
                            try:
                                method()
                                break
                            except Exception:
                                pass
                else:
                    if paused_movie_time_sec is not None and _is_ffpyplayer_movie(movie):
                        _seek_movie_to_time(movie, paused_movie_time_sec)
                    for method_name in ("resume", "play"):
                        method = getattr(movie, method_name, None)
                        if callable(method):
                            try:
                                method()
                                break
                            except Exception:
                                pass
                    paused_movie_time_sec = None
                movie_paused = target_paused

            # Keep report/manual pauses indefinite even if a backend resumes on its own.
            if target_paused and _movie_is_playing(movie) is True:
                method = getattr(movie, "pause", None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass

            if flash_on and not msband_buzzing:
                send_msband_haptics_command(config, start=True)
                msband_buzzing = True
            elif (not flash_on) and msband_buzzing:
                send_msband_haptics_command(config, start=False)
                msband_buzzing = False

            if skip_video_requested:
                break

            current_window_size = tuple(int(v) for v in win.size)
            if current_window_size != last_window_size:
                _fit_movie_to_window(movie, win)
                last_window_size = current_window_size

            movie.draw()
            if flash_on:
                flash_rect.draw()
                flash_text.draw()
            win.flip()

            # Sample again after draw/flip so end-of-stream state and final frame
            # timestamp updates are included in the finish check.
            movie_time_after_draw = get_movie_time_sec(movie, video_clock)

            perf_now = time.perf_counter()
            while perf_now >= next_log_perf:
                report = pending_reports.popleft() if pending_reports else "N/A"
                csv_writer.writerow(
                    [
                        trial.uid,
                        config.participant_id,
                        config.session_id,
                        f"{movie_time_after_draw:.6f}",
                        f"{int(time.time() * 1000):d}",
                        1 if movie_paused else 0,
                        flash_on,
                        report,
                    ]
                )
                next_log_perf += log_interval

            if (not report_cue_active) and (not movie_paused) and is_movie_finished(movie, movie_time_after_draw):
                break

            check_abort()
    finally:
        try:
            if previous_win_color is not None:
                win.color = previous_win_color
            else:
                win.color = "white"
        except Exception:
            pass
        if msband_buzzing:
            send_msband_haptics_command(config, start=False)
        if movie_paused:
            for method_name in ("resume", "play"):
                method = getattr(movie, method_name, None)
                if callable(method):
                    try:
                        method()
                        break
                    except Exception:
                        pass
        for method_name in ("stop", "pause", "close", "unload"):
            method = getattr(movie, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass


def run_experiment(config: ExperimentConfig) -> None:
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    probe_msband_bridge_available()

    window_kwargs: dict[str, Any] = {
        # Always start windowed; toggle fullscreen at runtime with F/F11.
        "fullscr": False,
        "color": "white",
        "units": "norm",
        "allowGUI": False,
        "screen": config.screen,
        "winType": "pyglet",
    }
    try:
        if sys.platform == "darwin" and "useRetina" in inspect.signature(visual.Window.__init__).parameters:
            window_kwargs["useRetina"] = False
    except (TypeError, ValueError):
        pass
    if (not config.fullscr) and config.window_size is not None:
        window_kwargs["size"] = config.window_size

    win = visual.Window(**window_kwargs)

    welcome_text = build_text(
        win,
        "Welcome\n\nPress SPACE to continue.\n\nPress F or F11 to toggle fullscreen.\n\n(Press ESCAPE anytime to quit.)",
        height=0.085,
        wrap_width=1.4,
    )
    prompt_text = build_text(win, PROMPT_TEXT, height=0.058, wrap_width=1.8)
    training_intro_text = build_text(
        win,
        "Training Video\n\n"
        "You will now watch a training video to practice responding when prompted.\n"
        "There is no pre-test or post-test for the training video.\n\n"
        "When the band vibrates and the screen says \"Report engagement now\", "
        "enter 1-5 or X.\n\n"
        "Press SPACE to start the training video.",
        height=0.072,
        wrap_width=1.7,
    )
    training_complete_text = build_text(
        win,
        "Training Complete\n\n"
        "Press R to repeat the training video.\n"
        "Press SPACE to continue to the pre-test for the first real video.",
        height=0.078,
        wrap_width=1.55,
    )
    pretest_text = build_text(
        win,
        "Pre-Test\n\nComplete the pre-test on the other device.\n\nPress SPACE when ready to start the first real video.",
        height=0.08,
    )
    break_text = build_text(
        win,
        "Break / Post-Test\n\nComplete the post-test.\n\nPress SPACE when ready for the next video.",
        height=0.08,
    )
    complete_text = build_text(
        win,
        "Experiment Complete\n\nPress SPACE to exit.",
        height=0.09,
        wrap_width=1.4,
    )

    flash_rect = visual.Rect(
        win=win,
        width=1.35,
        height=0.30,
        fillColor="white",
        lineColor="black",
        opacity=1.0,
        pos=(0.0, 0.0),
    )
    flash_text = visual.TextStim(
        win=win,
        text="Report engagement now",
        color="black",
        height=0.08,
        wrapWidth=1.4,
        bold=True,
        pos=(0.0, 0.0),
    )

    with config.output_csv.open("w", newline="", encoding="utf-8", buffering=1) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "video_uid",
                "participant_id",
                "session_id",
                "video_timestamp_sec",
                "system_time_sec",
                "is_paused",
                "flash_on",
                "engagement_report",
            ]
        )

        wait_for_space(win, welcome_text)
        wait_for_space(win, prompt_text)

        if not config.videos:
            raise ValueError("No videos configured.")

        training_trial = config.videos[0]
        real_trials = config.videos[1:]

        wait_for_space(win, training_intro_text)
        while True:
            play_video_trial(win, config, training_trial, writer, flash_rect, flash_text)
            training_next = wait_for_key(win, training_complete_text, ["space", "r"])
            if training_next == "space":
                break

        if real_trials:
            wait_for_space(win, pretest_text)

            for index, trial in enumerate(real_trials):
                play_video_trial(win, config, trial, writer, flash_rect, flash_text)
                if index < len(real_trials) - 1:
                    wait_for_space(win, break_text)

        wait_for_space(win, complete_text)

    win.close()
    core.quit()


def main() -> None:
    config_path = dc_config.CONFIG_JSON_PATH.resolve()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Expected config file at {config_path}. "
            "Create or restore Data_Collection/config.json."
        )
    config = load_config(config_path)
    run_experiment(config)


if __name__ == "__main__":
    try:
        main()
    except ExperimentAbort:
        print("Experiment exited early.")
        core.quit()
