import sys
import time
import csv
import os
from pathlib import Path
from datetime import datetime

DATA_COLLECTION_DIR = Path(__file__).resolve().parents[1]
if str(DATA_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_COLLECTION_DIR))
import config as dc_config

# --- BEAM SDK SETUP ---
sdk_paths = []
if os.environ.get("BEAM_SDK_PYTHON_PACKAGE"):
    sdk_paths.append(Path(os.environ["BEAM_SDK_PYTHON_PACKAGE"]).expanduser())
sdk_paths.append(
    Path(__file__).parent
    / "beam_eye_tracker_sdk"
    / "beam_eye_tracker_sdk-2.1.0"
    / "python"
    / "package"
)

for sdk_path in sdk_paths:
    if sdk_path.exists():
        sys.path.append(str(sdk_path))
        break

try:
    from eyeware.beam_eye_tracker import (
        API,
        ViewportGeometry,
        Point,
        TrackingListener,
        NULL_DATA_TIMESTAMP,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Beam SDK Python package not found. Install Eyeware Beam SDK 2.1.0 "
        "and set BEAM_SDK_PYTHON_PACKAGE to its python/package directory, "
        "or place the SDK under Data_Collection/Beam/beam_eye_tracker_sdk/."
    ) from exc

# --- CSV GLOBALS ---
csv_file = None
csv_writer = None

def setup_csv():
    """Creates a timestamped folder and opens the Beam CSV."""
    global csv_file, csv_writer
    session_name = f"recording_beam_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(session_name, exist_ok=True)
    
    filepath = os.path.join(session_name, dc_config.prefixed_filename("BeamEyeTracker.csv"))
    csv_file = open(filepath, 'w', newline='', buffering=1)
    csv_writer = csv.writer(csv_file)
    
    # Updated Headers
    headers = [
        "sys_time",           # Local system clock (time.time())
        "beam_time",          # Internal SDK timestamp
        "gaze_conf_int",      # Gaze Confidence
        "gaze_por_x",         # Point of Regard X (0-1)
        "gaze_por_y",         # Point of Regard Y (0-1)
        "head_conf_int",      # Head Confidence
        "head_pos_x_m",       # Head Position X (meters)
        "head_pos_y_m",       # Head Position Y
        "head_pos_z_m",       # Head Position Z
        "rot_m11", "rot_m12", "rot_m13", 
        "rot_m21", "rot_m22", "rot_m23", 
        "rot_m31", "rot_m32", "rot_m33"
    ]
    csv_writer.writerow(headers)
    print(f"--- RECORDING TO: {filepath} ---")

def close_csv():
    global csv_file
    if csv_file:
        csv_file.close()
        print("CSV file saved and closed.")

def start_beam_stream():
    viewport = ViewportGeometry(Point(0.0, 0.0), Point(1.0, 1.0))
    setup_csv()

    class TrackingLogger(TrackingListener):
        def on_tracking_state_set_update(self, tracking_state_set, timestamp):
            # Capture system time immediately upon callback
            sys_now = int(time.time() * 1000)
            beam_ts = timestamp.value

            user = tracking_state_set.user_state()
            if user.timestamp_in_seconds.value == NULL_DATA_TIMESTAMP().value:
                return

            gaze = user.unified_screen_gaze
            head = user.head_pose

            # Gaze Data
            gaze_conf_int = gaze.confidence
            if gaze_conf_int == 0:
                por_x = por_y = float('nan')
            else:
                por_x = gaze.point_of_regard.x
                por_y = gaze.point_of_regard.y

            # Head Data
            head_conf_int = head.confidence
            if head_conf_int == 0:
                head_pos_x = head_pos_y = head_pos_z = float('nan')
                rot_flat = [float('nan')] * 9
            else:
                pos = head.translation_from_hcs_to_wcs
                head_pos_x, head_pos_y, head_pos_z = pos.x, pos.y, pos.z
                rot_flat = head.rotation_from_hcs_to_wcs.flatten().tolist()

            # Prepare row with dual timestamps
            row = [
                f"{sys_now:.6f}",
                f"{beam_ts:.6f}",
                float(gaze_conf_int),
                por_x,
                por_y,
                float(head_conf_int),
                head_pos_x,
                head_pos_y,
                head_pos_z,
            ] + rot_flat
            
            if csv_writer:
                csv_writer.writerow(row)

        def on_tracking_data_reception_status_changed(self, status):
            print(f"Tracking reception status: {status}")

    api = API("BeamEyeTracker", viewport)
    api.attempt_starting_the_beam_eye_tracker()

    listener = TrackingLogger()
    handle = api.start_receiving_tracking_data_on_listener(listener)

    print("\nRecording. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        api.stop_receiving_tracking_data_on_listener(handle)
        close_csv()

if __name__ == "__main__":
    start_beam_stream()
