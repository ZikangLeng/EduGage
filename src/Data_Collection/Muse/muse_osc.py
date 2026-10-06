import argparse
import time
import csv
import os
import sys
from pathlib import Path
from datetime import datetime
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

DATA_COLLECTION_DIR = Path(__file__).resolve().parents[1]
if str(DATA_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_COLLECTION_DIR))
import config as dc_config

# --- CONFIGURATION ---
IP = "0.0.0.0"
PORT = 5001

# Global storage for file handles and CSV writers
file_handles = {}
csv_writers = {}

def setup_recording():
    """Creates the folder and opens all CSV files with headers."""
    # 1. Create a unique folder name
    session_name = f"recording_muse_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(session_name, exist_ok=True)
    print(f"--- RECORDING TO FOLDER: {session_name} ---")

    # 2. Define the files we need and their headers
    streams = {
        "EEG":     ["timestamp", "tp9", "af7", "af8", "tp10"], 
        "PPG":     ["timestamp", "ppg1", "ppg2", "ppg3", "ppg4"],
        "ACC":     ["timestamp", "x", "y", "z"],
        "GYRO":    ["timestamp", "x", "y", "z"],
        "MARKERS": ["timestamp", "marker_type"]
    }

    # 3. Open files and write headers
    for name, headers in streams.items():
        filepath = os.path.join(session_name, dc_config.prefixed_filename(f"{name}.csv"))
        f = open(filepath, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # Save to global dictionaries so handlers can use them
        file_handles[name] = f
        csv_writers[name] = writer
        print(f"Created log: {filepath}")

def close_recording():
    """Closes all open file handles safely."""
    print("\nStopping recording...")
    for name, f in file_handles.items():
        f.close()
    print("All files saved and closed.")

# --- DATA HANDLERS ---

def write_to_csv(stream_name, data):
    """Helper to write timestamp + data to the correct CSV."""
    if stream_name in csv_writers:
        ts = int(time.time() * 1000)
        row = [ts] + list(data)
        csv_writers[stream_name].writerow(row)

def eeg_handler(address, *args):
    # Slice to keep only the first 4 channels (TP9, AF7, AF8, TP10)
    write_to_csv("EEG", args[:4])

def ppg_handler(address, *args):
    write_to_csv("PPG", args)

def acc_handler(address, *args):
    write_to_csv("ACC", args)

def gyro_handler(address, *args):
    write_to_csv("GYRO", args)

def blink_handler(address, *args):
    if args[0] == 1:
        write_to_csv("MARKERS", ["blink"])
        print(">> Blink")

def jaw_handler(address, *args):
    if args[0] == 1:
        write_to_csv("MARKERS", ["jaw_clench"])
        print(">> Jaw Clench")

def horseshoe_handler(address, *args):
    write_to_csv("MARKERS", [f"horseshoe_{args}"])

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=IP)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    # 1. Open files
    setup_recording()

    # 2. Map OSC addresses to functions
    dispatcher = Dispatcher()
    dispatcher.map("/muse/eeg", eeg_handler)
    dispatcher.map("/muse/optics", ppg_handler)
    dispatcher.map("/muse/acc", acc_handler)
    dispatcher.map("/muse/gyro", gyro_handler)
    dispatcher.map("/muse/elements/blink", blink_handler)
    dispatcher.map("/muse/elements/jaw_clench", jaw_handler)
    dispatcher.map("/muse/elements/horseshoe", horseshoe_handler)

    # 3. Start Server
    server = BlockingOSCUDPServer((args.ip, args.port), dispatcher)
    print(f"Listening on {args.ip}:{args.port}...")
    print("Press Ctrl+C to stop recording.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        close_recording()
