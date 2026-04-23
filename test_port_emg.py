import csv
import numpy as np
import matplotlib.pyplot as plt
import serial
import struct
import time
from scipy.signal import butter, sosfiltfilt

PORT = "COM11"
BAUD = 115200

# ADS1292R conversion: gain=12, Vref=2.42V
VREF  = 2.42
GAIN  = 12
SCALE = (VREF / GAIN) / (2 ** 23) * 1000   # raw count → mV

def wait_for_ack(ser, timeout=3):
    start = time.time()
    while time.time() - start < timeout:
        b = ser.read(1)
        if b:
            print(f"ACK candidate: 0x{b[0]:02x}")
        if b in (b'\xff', b'\xfe'):   # 0xff = ACK, 0xfe = ACK on some firmware versions
            return True
    return False

def send_cmd(ser, cmd, name):
    ser.write(cmd)
    ok = wait_for_ack(ser)
    print(f"{name}: {'ACK received' if ok else 'ACK timeout'}")
    return ok

def parse_24bit(b0, b1, b2):
    return int.from_bytes([b0, b1, b2], byteorder='big', signed=True)

def run_emg(ser):
    print("Running EMG...")

    # stop any ongoing streaming, give device time to settle
    ser.write(struct.pack('B', 0x20))
    time.sleep(1.0)
    ser.reset_input_buffer()

    send_cmd(ser, struct.pack('BBBB', 0x08, 0x00, 0x00, 0x18), "Enable EXG")

    sampling_freq = 512
    clock_wait = int((2 << 14) / sampling_freq)   # 32768 / 512 = 64
    send_cmd(ser, struct.pack('<BH', 0x05, clock_wait), "Set sample rate")

    # Chip 1: both channels as EMG inputs, gain=12, no RLD
    chip1_cfg = bytes([
        0x61, 0x00, 0x00, 0x0A,
        0x03,   # CONFIG1: 1000 SPS internal (must exceed Shimmer 512 Hz rate)
        0xA0,   # CONFIG2: internal reference enabled
        0x00,   # LOFF: lead-off detection off
        0x60,   # CH1SET: gain=12, normal electrode input
        0x60,   # CH2SET: gain=12, normal electrode input
        0x00,   # RLD_SENS: RLD off
        0x00,   # LOFF_SENS
        0x00,   # LOFF_STAT
        0x02,   # RESP1
        0x01    # RESP2
    ])

    # Chip 2: CH1 as third EMG channel, gain=12
    chip2_cfg = bytes([
        0x61, 0x01, 0x00, 0x0A,
        0x03, 0xA0, 0x00, 0x60, 0x60,
        0x00, 0x00, 0x00, 0x02, 0x01
    ])

    send_cmd(ser, chip1_cfg, "Configure EMG chip 1")
    send_cmd(ser, chip2_cfg, "Configure EMG chip 2")
    send_cmd(ser, struct.pack('B', 0x07), "Start streaming")

    PACKET_SIZE    = 13
    EXPECTED_DELTA = 64    # 32768 / 512

    # --- sync ---
    # Packet layout: [ptype(1)] [ts_lo ts_mid ts_hi (3)] [payload (9)]
    # ts is at bytes 1-3 within each 13-byte packet, so read from pos+1..pos+3
    sync_buf = b""
    while len(sync_buf) < PACKET_SIZE * 10:
        sync_buf += ser.read(PACKET_SIZE * 10 - len(sync_buf))

    best_offset, best_score = 0, -1
    for offset in range(PACKET_SIZE):
        scores, prev, pos = [], None, offset
        while pos + 4 <= len(sync_buf):
            t0, t1, t2 = sync_buf[pos+1], sync_buf[pos+2], sync_buf[pos+3]
            ts = t0 + (t1 << 8) + (t2 << 16)
            if prev is not None:
                scores.append(abs(((ts - prev) & 0xFFFFFF) - EXPECTED_DELTA))
            prev = ts
            pos += PACKET_SIZE
        if scores:
            score = -sum(scores) / len(scores)
            if score > best_score:
                best_score, best_offset = score, offset

    print(f"Sync: best offset={best_offset}, mean delta error={-best_score:.1f} ticks")
    buffer = sync_buf[best_offset:]

    WARMUP_S = 2.0
    RECORD_S = 15.0
    OUT_DIR  = r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop"

    last_ts    = None
    deltas     = []
    records    = []
    byte_count = 0
    t_diag     = time.time()
    dumped     = False
    t_stream_start = time.time()
    recording      = False
    t_record_start = None

    print(f"Warming up for {WARMUP_S:.0f}s...")

    while True:
        chunk = ser.read(ser.in_waiting or PACKET_SIZE)
        if chunk:
            buffer += chunk
            byte_count += len(chunk)

        if not dumped and len(buffer) >= 40:
            print("Raw stream (40 bytes):", buffer[:40].hex(' '))
            dumped = True

        elapsed = time.time() - t_diag
        if elapsed >= 3.0:
            bps = byte_count / elapsed
            print(f"Bytes/sec: {bps:.1f}  => packet size at 512Hz: {bps/512:.2f}")
            byte_count = 0
            t_diag = time.time()

        now = time.time()

        if not recording and (now - t_stream_start) >= WARMUP_S:
            recording      = True
            t_record_start = now
            print("Recording started...")

        if recording and (now - t_record_start) >= RECORD_S:
            break

        while len(buffer) >= PACKET_SIZE:
            # byte 0: packet type (ignored)
            # bytes 1-3: timestamp (little-endian 24-bit)
            # bytes 4-12: payload — 3 channels of 24-bit data (no status bytes)
            t0, t1, t2 = buffer[1], buffer[2], buffer[3]
            ts = t0 + (t1 << 8) + (t2 << 16)
            p  = buffer[4:PACKET_SIZE]   # 9 bytes: CH1(3) CH2(3) CH3(3)
            buffer = buffer[PACKET_SIZE:]

            if last_ts is not None:
                delta = (ts - last_ts) & 0xFFFFFF
                deltas.append(delta)
                if len(deltas) == 512:
                    mean_delta = sum(deltas) / len(deltas)
                    real_fs = 32768 / mean_delta
                    print(f"Real fs: {real_fs:.2f} Hz  (mean delta={mean_delta:.1f} ticks)")
                    deltas.clear()
            last_ts = ts

            if recording:
                # p[3:6]  : EXG1 CH2 → EMG_CH2  (big-endian 24-bit signed)
                # p[6:9]  : EXG2 CH1 → EMG_CH1  (big-endian 24-bit signed)
                ch2 = parse_24bit(p[3], p[4], p[5]) * SCALE
                ch1 = parse_24bit(p[6], p[7], p[8]) * SCALE
                records.append((ts / 32768.0, ch1, ch2))

    print(f"Recording done. {len(records)} samples captured.")

    ts_sec = np.array([r[0] for r in records])
    ch1    = np.array([r[1] for r in records])
    ch2    = np.array([r[2] for r in records])

    # bandpass 20–200 Hz (EMG range; Nyquist at 512 Hz = 256 Hz)
    sos_bp = butter(4, [20.0, 200.0], btype='band', fs=sampling_freq, output='sos')

    channels = {
        "EMG_CH1": sosfiltfilt(sos_bp, ch1),
        "EMG_CH2": sosfiltfilt(sos_bp, ch2),
    }

    for name, sig in channels.items():
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(ts_sec, sig, linewidth=0.7)
        ax.set_title(f"{name} (bandpass 20–200 Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = rf"{OUT_DIR}\{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")

    csv_path = rf"{OUT_DIR}\emg_channels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ts_sec", "EMG_CH1_mV", "EMG_CH2_mV"])
        writer.writerows(zip(
            ts_sec,
            channels["EMG_CH1"],
            channels["EMG_CH2"],
        ))
    print(f"Saved {csv_path}")

def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print("Opened", ser.name)

    try:
        run_emg(ser)
    except KeyboardInterrupt:
        print("\nStopping...")
        ser.write(struct.pack('B', 0x20))
        wait_for_ack(ser)
        ser.close()
        print("Closed")

if __name__ == "__main__":
    main()
