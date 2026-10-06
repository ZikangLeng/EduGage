import asyncio
import argparse
import sys
import time
import csv
import os
from datetime import datetime
from pathlib import Path
from bleak import BleakClient

DATA_COLLECTION_DIR = Path(__file__).resolve().parents[1]
if str(DATA_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_COLLECTION_DIR))
import config as dc_config

# --- CONFIGURATION ---
# Provide the device address with --address or POLAR_H10_ADDRESS.
DEFAULT_ADDRESS = os.environ.get("POLAR_H10_ADDRESS")

PMD_CONTROL = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_DATA    = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"

# Hex for: ECG, Start, 130Hz (standard Polar H10 protocol)
ECG_WRITE   = bytearray([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0E, 0x00])

# --- CSV GLOBALS ---
file_handle = None
csv_writer = None

def setup_csv():
    """Creates timestamped folder and opens the ECG CSV."""
    global file_handle, csv_writer
    session_name = f"recording_polar_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(session_name, exist_ok=True)
    
    filepath = os.path.join(session_name, dc_config.prefixed_filename("Polar_ECG.csv"))
    file_handle = open(filepath, 'w', newline='', buffering=1)
    csv_writer = csv.writer(file_handle)
    
    # sys_time: Host computer clock (Unix epoch)
    # dev_time: Polar internal clock (Nanoseconds)
    # ecg_val:  Raw microvolt value
    csv_writer.writerow(["sys_time", "dev_time", "ecg_val"])
    print(f"--- RECORDING TO: {filepath} ---")

def close_csv():
    global file_handle
    if file_handle:
        file_handle.close()
        print("CSV file saved and closed.")

async def ecg_data_handler(sender, data: bytearray):
    global csv_writer
    
    # 1. Capture System Time (Wall Clock) immediately
    sys_now = int(time.time() * 1000)

    # 2. Extract Device Time (Polar Hardware Clock)
    # Byte 0 is packet type (0x00 for ECG data)
    # Bytes 1-8 are the timestamp in nanoseconds
    if data[0] == 0x00:
        dev_ts_nanos = int.from_bytes(data[1:9], byteorder='little')
        # Optional: convert to seconds for readability, or keep as nanos
        dev_ts_seconds = dev_ts_nanos / 1_000_000_000.0
        
        # 3. Parse Samples
        # Data starts at byte 10. Each sample is 3 bytes (24-bit integer).
        raw_samples = data[10:]
        step = 3
        
        if csv_writer:
            for i in range(0, len(raw_samples), step):
                sample_bytes = raw_samples[i : i + step]
                if len(sample_bytes) < 3: 
                    break
                
                # Convert 24-bit little-endian to signed int
                val = int.from_bytes(sample_bytes, byteorder="little", signed=True)
                
                # Write row: SystemTime, DeviceTime, Value
                # NOTE: Timestamps will be identical for all samples in this packet
                csv_writer.writerow([f"{sys_now:d}", f"{dev_ts_seconds:.6f}", val])

async def run_ble_client(address):
    setup_csv()
    
    print(f"Searching for Polar device ({address})...")
    async with BleakClient(address) as client:
        if not client.is_connected:
            print("Failed to connect.")
            return

        print(f"Connected to {address}")
        
        # Enable notifications
        await client.start_notify(PMD_DATA, ecg_data_handler)
        
        # Write config to start stream
        await client.write_gatt_char(PMD_CONTROL, ECG_WRITE)
        print("Stream started. Press Ctrl+C to stop.")

        try:
            # Keep alive until user kills it
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            print("Stopping stream...")
            await client.stop_notify(PMD_DATA)

def main():
    parser = argparse.ArgumentParser(description="Record Polar H10 ECG samples.")
    parser.add_argument(
        "--address",
        default=DEFAULT_ADDRESS,
        help="Polar H10 BLE address/identifier. Can also be set with POLAR_H10_ADDRESS.",
    )
    args = parser.parse_args()
    if not args.address:
        raise SystemExit(
            "Missing Polar H10 address. Pass --address or set POLAR_H10_ADDRESS."
        )
    try:
        asyncio.run(run_ble_client(args.address))
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        close_csv()

if __name__ == "__main__":
    main()
