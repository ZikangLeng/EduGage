"""Parser for ring .bin logs exported as packetized hex text lines.

The Android ring app stores each downloaded file packet as:
- 4-byte response prefix
- 25-byte packet header in compact mode or 29-byte header in legacy mode
- 5 samples of 30 bytes each

For the file type present in this dataset (`*_7.bin`), MainActivity decodes each
30-byte sample as:
- green, red, ir (uint32 each)
- acc_x, acc_y, acc_z (int16 each)
- gyro_x, gyro_y, gyro_z (int16 each)
- temp_0, temp_1, temp_2 (int16 each)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RING_SAMPLES_PER_PACKET = 5
RING_SAMPLE_BYTES = 30
RING_DEFAULT_SAMPLE_PERIOD_MS = 40.0


@dataclass(frozen=True)
class RingPacketLayout:
    name: str
    packet_bytes: int
    header_bytes: int
    timestamp_offset: int


# Legacy ring export: 29-byte header and packet timestamp at [21:29].
RING_LAYOUT_LEGACY = RingPacketLayout(
    name="legacy_v1",
    packet_bytes=179,
    header_bytes=29,
    timestamp_offset=21,
)

# Compact ring export: 25-byte header and packet timestamp at [17:25].
RING_LAYOUT_COMPACT = RingPacketLayout(
    name="compact_v2",
    packet_bytes=175,
    header_bytes=25,
    timestamp_offset=17,
)

# Keep highest-length layout first so longer packets prefer the legacy decode.
RING_PACKET_LAYOUTS = (RING_LAYOUT_LEGACY, RING_LAYOUT_COMPACT)


def _select_packet_layout(packet_len: int) -> RingPacketLayout | None:
    for layout in RING_PACKET_LAYOUTS:
        if packet_len >= layout.packet_bytes:
            return layout
    return None


def _iter_raw_packet_bytes(path: Path):
    marker = b"# Raw data:"
    with path.open("rb") as fp:
        for line_b in fp:
            if marker not in line_b:
                continue

            tail = line_b.split(marker, 1)[1]
            hex_payload = tail.decode("ascii", errors="ignore").strip()
            if not hex_payload or (len(hex_payload) % 2) != 0:
                continue

            try:
                packet = bytes.fromhex(hex_payload)
            except ValueError:
                continue

            layout = _select_packet_layout(len(packet))
            if layout is None:
                continue

            yield packet[: layout.packet_bytes], layout


def _packet_timestamp_ms(packet: bytes, layout: RingPacketLayout) -> int:
    start = layout.timestamp_offset
    end = start + 8
    return int.from_bytes(packet[start:end], byteorder="little", signed=False)


def _parse_sample(sample: bytes) -> tuple[int, int, int, int, int, int, int, int, int, int, int, int]:
    green = int.from_bytes(sample[0:4], byteorder="little", signed=False)
    red = int.from_bytes(sample[4:8], byteorder="little", signed=False)
    ir = int.from_bytes(sample[8:12], byteorder="little", signed=False)
    acc_x = int.from_bytes(sample[12:14], byteorder="little", signed=True)
    acc_y = int.from_bytes(sample[14:16], byteorder="little", signed=True)
    acc_z = int.from_bytes(sample[16:18], byteorder="little", signed=True)
    gyro_x = int.from_bytes(sample[18:20], byteorder="little", signed=True)
    gyro_y = int.from_bytes(sample[20:22], byteorder="little", signed=True)
    gyro_z = int.from_bytes(sample[22:24], byteorder="little", signed=True)
    temp_0 = int.from_bytes(sample[24:26], byteorder="little", signed=True)
    temp_1 = int.from_bytes(sample[26:28], byteorder="little", signed=True)
    temp_2 = int.from_bytes(sample[28:30], byteorder="little", signed=True)
    return green, red, ir, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, temp_0, temp_1, temp_2


def load_ring_dataframe(path: Path) -> pd.DataFrame:
    packets = list(_iter_raw_packet_bytes(path))
    if not packets:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "green",
                "red",
                "ir",
                "acc_x",
                "acc_y",
                "acc_z",
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "temp_0",
                "temp_1",
                "temp_2",
            ]
        )

    packet_ts_ms = np.array(
        [_packet_timestamp_ms(packet, layout) for packet, layout in packets],
        dtype=np.float64,
    )
    n_packets = len(packets)
    total_samples = n_packets * RING_SAMPLES_PER_PACKET

    start_ts = float(packet_ts_ms[0])
    duration_ms = float(packet_ts_ms[-1] - packet_ts_ms[0]) if n_packets > 1 else 0.0
    if total_samples > 1 and duration_ms > 0:
        sample_period_ms = duration_ms / float(total_samples - 1)
    else:
        sample_period_ms = RING_DEFAULT_SAMPLE_PERIOD_MS

    timestamps = start_ts + (np.arange(total_samples, dtype=np.float64) * sample_period_ms)

    green = np.zeros(total_samples, dtype=np.float64)
    red = np.zeros(total_samples, dtype=np.float64)
    ir = np.zeros(total_samples, dtype=np.float64)
    acc_x = np.zeros(total_samples, dtype=np.float64)
    acc_y = np.zeros(total_samples, dtype=np.float64)
    acc_z = np.zeros(total_samples, dtype=np.float64)
    gyro_x = np.zeros(total_samples, dtype=np.float64)
    gyro_y = np.zeros(total_samples, dtype=np.float64)
    gyro_z = np.zeros(total_samples, dtype=np.float64)
    temp_0 = np.zeros(total_samples, dtype=np.float64)
    temp_1 = np.zeros(total_samples, dtype=np.float64)
    temp_2 = np.zeros(total_samples, dtype=np.float64)

    row_idx = 0
    for packet, layout in packets:
        payload_start = layout.header_bytes
        payload_end = payload_start + (RING_SAMPLES_PER_PACKET * RING_SAMPLE_BYTES)
        payload = packet[payload_start:payload_end]
        for sample_idx in range(RING_SAMPLES_PER_PACKET):
            sample = payload[sample_idx * RING_SAMPLE_BYTES : (sample_idx + 1) * RING_SAMPLE_BYTES]
            if len(sample) < RING_SAMPLE_BYTES:
                continue

            (
                v_green,
                v_red,
                v_ir,
                v_ax,
                v_ay,
                v_az,
                v_gx,
                v_gy,
                v_gz,
                v_t0,
                v_t1,
                v_t2,
            ) = _parse_sample(sample)
            green[row_idx] = float(v_green)
            red[row_idx] = float(v_red)
            ir[row_idx] = float(v_ir)
            acc_x[row_idx] = float(v_ax)
            acc_y[row_idx] = float(v_ay)
            acc_z[row_idx] = float(v_az)
            gyro_x[row_idx] = float(v_gx)
            gyro_y[row_idx] = float(v_gy)
            gyro_z[row_idx] = float(v_gz)
            temp_0[row_idx] = float(v_t0)
            temp_1[row_idx] = float(v_t1)
            temp_2[row_idx] = float(v_t2)
            row_idx += 1

    if row_idx < total_samples:
        timestamps = timestamps[:row_idx]
        green = green[:row_idx]
        red = red[:row_idx]
        ir = ir[:row_idx]
        acc_x = acc_x[:row_idx]
        acc_y = acc_y[:row_idx]
        acc_z = acc_z[:row_idx]
        gyro_x = gyro_x[:row_idx]
        gyro_y = gyro_y[:row_idx]
        gyro_z = gyro_z[:row_idx]
        temp_0 = temp_0[:row_idx]
        temp_1 = temp_1[:row_idx]
        temp_2 = temp_2[:row_idx]

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "green": green,
            "red": red,
            "ir": ir,
            "acc_x": acc_x,
            "acc_y": acc_y,
            "acc_z": acc_z,
            "gyro_x": gyro_x,
            "gyro_y": gyro_y,
            "gyro_z": gyro_z,
            "temp_0": temp_0,
            "temp_1": temp_1,
            "temp_2": temp_2,
        }
    )

    df.attrs["ring_packet_count"] = int(n_packets)
    df.attrs["ring_sample_period_ms"] = float(sample_period_ms)
    df.attrs["ring_first_packet_ts_ms"] = float(packet_ts_ms[0])
    df.attrs["ring_last_packet_ts_ms"] = float(packet_ts_ms[-1])
    layout_counts: dict[str, int] = {}
    for _packet, layout in packets:
        layout_counts[layout.name] = layout_counts.get(layout.name, 0) + 1
    df.attrs["ring_packet_layouts"] = layout_counts
    return df


def read_ring_timestamp_series(path: Path) -> pd.Series:
    df = load_ring_dataframe(path)
    if "timestamp" not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df["timestamp"], errors="coerce")
