import socket
import time
import csv
import threading
import sys
from pathlib import Path

DATA_COLLECTION_DIR = Path(__file__).resolve().parents[1]
if str(DATA_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_COLLECTION_DIR))
import config as dc_config

# Configuration
HOST = '0.0.0.0'  # Listen on all available network interfaces
PORT = 9898       # Must match the port in your C# app
CONTROL_HOST = "127.0.0.1"  # Local-only command bridge for other local processes (e.g., experiment)
CONTROL_PORT = 9899
VALID_HAPTIC_COMMANDS = {"START", "STOP"}

active_band_conn = None
active_band_conn_lock = threading.Lock()
band_send_lock = threading.Lock()


def set_active_band_conn(conn):
    global active_band_conn
    with active_band_conn_lock:
        active_band_conn = conn


def clear_active_band_conn(conn):
    global active_band_conn
    with active_band_conn_lock:
        if active_band_conn is conn:
            active_band_conn = None


def send_haptic_command_to_band(cmd: str) -> bool:
    cmd = cmd.strip().upper()
    if cmd not in VALID_HAPTIC_COMMANDS:
        print(f"Invalid command. Expected START/STOP, got: {cmd!r}")
        return False

    with active_band_conn_lock:
        conn = active_band_conn

    if conn is None:
        print(f"[WARN] No active Band client connection; dropping command: {cmd}")
        return False

    try:
        with band_send_lock:
            # The C# DataReader expects unpadded strings
            conn.sendall(cmd.encode('utf-8'))
        print(f"-> Sent command: {cmd}")
        return True
    except Exception as e:
        print(f"[WARN] Failed to send command to Band client: {e}")
        return False

def haptic_command_loop():
    """Background thread to send START/STOP haptic commands to the Band."""
    while True:
        try:
            cmd = input().strip().upper()
            if cmd in VALID_HAPTIC_COMMANDS:
                send_haptic_command_to_band(cmd)
            else:
                print("Invalid command. Type START or STOP.")
        except Exception as e:
            print(f"Command loop exited: {e}")
            break


def _handle_local_command_client(cmd_conn, addr):
    print(f"Local command client connected: {addr}")
    buffer = ""
    try:
        while True:
            data = cmd_conn.recv(1024)
            if not data:
                break
            buffer += data.decode("utf-8", errors="ignore")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if line:
                    send_haptic_command_to_band(line)
        # Support single commands sent without newline and then close().
        tail = buffer.strip()
        if tail:
            send_haptic_command_to_band(tail)
    except Exception as e:
        print(f"Local command bridge error ({addr}): {e}")
    finally:
        try:
            cmd_conn.close()
        except Exception:
            pass


def local_command_bridge_loop():
    """Accepts local START/STOP commands and forwards them over the active Band socket."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as cmd_server:
        cmd_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        cmd_server.bind((CONTROL_HOST, CONTROL_PORT))
        cmd_server.listen()
        print(f"Local haptic command bridge listening on {CONTROL_HOST}:{CONTROL_PORT}")

        while True:
            cmd_conn, addr = cmd_server.accept()
            thread = threading.Thread(
                target=_handle_local_command_client,
                args=(cmd_conn, addr),
                daemon=True,
            )
            thread.start()

def main():
    bridge_thread = threading.Thread(target=local_command_bridge_loop, daemon=True)
    bridge_thread.start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"Listening for Microsoft Band on {HOST}:{PORT}...")
        print("Once connected, type START or STOP to control haptics.")
        
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            set_active_band_conn(conn)
            
            # Start the command input thread
            cmd_thread = threading.Thread(target=haptic_command_loop, daemon=True)
            cmd_thread.start()

            # Open CSV files in append mode. flush() will be called to prevent data loss on crash.
            gsr_path = dc_config.prefixed_filename("msband_gsr.csv")
            hr_path = dc_config.prefixed_filename("msband_hr.csv")
            with open(gsr_path, 'a', newline='') as f_gsr, \
                 open(hr_path, 'a', newline='') as f_hr:
                
                gsr_writer = csv.writer(f_gsr)
                hr_writer = csv.writer(f_hr)
                
                # Write headers only for new files.
                if f_gsr.tell() == 0:
                    gsr_writer.writerow(['SystemTime', 'BandTime', 'Resistance_kOhms'])
                if f_hr.tell() == 0:
                    hr_writer.writerow(['SystemTime', 'BandTime', 'HeartRate_bpm', 'Quality'])
                
                buffer = ""
                
                while True:
                    try:
                        data = conn.recv(1024)
                        if not data:
                            print("Client disconnected.")
                            break 
                        
                        # System time captured the millisecond the TCP packet arrives
                        sys_time = int(time.time() * 1000)
                        
                        buffer += data.decode('utf-8')
                        lines = buffer.split('\n')
                        
                        # Keep the last incomplete line in the buffer in case of TCP fragmentation
                        buffer = lines.pop() 
                        
                        for line in lines:
                            parts = line.strip().split(',')
                            if not parts or parts[0] == "":
                                continue
                                
                            if parts[0] == "GSR" and len(parts) == 3:
                                _, band_time, resistance = parts
                                gsr_writer.writerow([sys_time, band_time, resistance])
                                f_gsr.flush() 
                                
                            elif parts[0] == "HR" and len(parts) == 4:
                                _, band_time, hr, quality = parts
                                hr_writer.writerow([sys_time, band_time, hr, quality])
                                f_hr.flush()
                                
                    except ConnectionResetError:
                        print("Connection forcibly closed by the client.")
                        break
                    except Exception as e:
                        print(f"Stream error: {e}")
                        break
            clear_active_band_conn(conn)

if __name__ == "__main__":
    main()
