import csv
import numpy as np
import matplotlib.pyplot as plt
import serial
import struct
import time
from scipy.signal import butter, sosfiltfilt

PORT = "COM6"
BAUD = 115200

def wait_for_ack(ser, timeout=3):
    start = time.time()
    while time.time() - start < timeout:
        b = ser.read(1)
        if b:
            print(f"ACK candidate: 0x{b[0]:02x}")
        if b == b'\xff':
            return True
    return False

def send_cmd(ser, cmd, name):
    ser.write(cmd)
    ok = wait_for_ack(ser)
    print(f"{name}: {'ACK received' if ok else 'ACK timeout'}")
    return ok

# =========================
# ===== EXG SECTION =======
# =========================
def run_exg(ser):
    print("Running EXG...")

    send_cmd(ser, struct.pack('BBBB', 0x08, 0x00, 0x00, 0x18), "Enable EXG")

    sampling_freq = 256
    clock_wait = int((2 << 14) / sampling_freq)
    send_cmd(ser, struct.pack('<BH', 0x05, clock_wait), "Set sample rate")

    chip1_cfg = bytes([
        0x61, 0x00, 0x00, 0x0A,
        0x02, 0xA0, 0x10, 0x40, 0x40,
        0x2D, 0x00, 0x00, 0x02, 0x03
    ])

    chip2_cfg = bytes([
        0x61, 0x01, 0x00, 0x0A,
        0x02, 0xA0, 0x10, 0x40, 0x47,
        0x00, 0x00, 0x00, 0x02, 0x01
    ])

    send_cmd(ser, chip1_cfg, "Configure EXG chip 1")
    send_cmd(ser, chip2_cfg, "Configure EXG chip 2")

    send_cmd(ser, struct.pack('B', 0x07), "Start streaming")

    PACKET_SIZE = 14
    EXPECTED_DELTA = 128  # 32768 / 256

    # --- sync: find which byte offset aligns to packet boundaries ---
    sync_buf = b""
    while len(sync_buf) < PACKET_SIZE * 10:
        sync_buf += ser.read(PACKET_SIZE * 10 - len(sync_buf))

    best_offset, best_score = 0, -1
    for offset in range(PACKET_SIZE):
        scores = []
        prev = None
        pos = offset
        while pos + 3 <= len(sync_buf):
            t0, t1, t2 = sync_buf[pos], sync_buf[pos+1], sync_buf[pos+2]
            ts = t0 + (t1 << 8) + (t2 << 16)
            if prev is not None:
                delta = (ts - prev) & 0xFFFFFF
                scores.append(abs(delta - EXPECTED_DELTA))
            prev = ts
            pos += PACKET_SIZE
        if scores:
            score = -sum(scores) / len(scores)
            if score > best_score:
                best_score, best_offset = score, offset

    print(f"Sync: best offset={best_offset}, mean delta error={-best_score:.1f} ticks")
    buffer = sync_buf[best_offset:]

    # ADS1292R conversion constants
    VREF = 2.42        # internal reference voltage (V)
    GAIN = 4           # PGA gain (from CH1SET/CH2SET = 0x40 → bits[6:4]=100 → 4x)
    SCALE = (VREF / GAIN) / (2 ** 23) * 1000  # raw count → mV

    def parse_24bit(b0, b1, b2):
        """24-bit big-endian signed integer from ADS1292R output."""
        return int.from_bytes([b0, b1, b2], byteorder='big', signed=True)

    WARMUP_S = 2.0
    RECORD_S = 15.0
    OUT_DIR  = r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop"

    last_ts = None
    deltas = []
    # records: (ts_sec, lead1_mV, lead2_mV, exg2ch1_mV)
    records = []
    t_stream_start = time.time()
    recording = False
    t_record_start = None

    print(f"Warming up for {WARMUP_S:.0f}s...")

    while True:
        chunk = ser.read(ser.in_waiting or PACKET_SIZE)
        if chunk:
            buffer += chunk

        now = time.time()

        if not recording and (now - t_stream_start) >= WARMUP_S:
            recording = True
            t_record_start = now
            print("Recording started...")

        if recording and (now - t_record_start) >= RECORD_S:
            break

        while len(buffer) >= PACKET_SIZE:
            t0, t1, t2 = buffer[0], buffer[1], buffer[2]
            ts = t0 + (t1 << 8) + (t2 << 16)
            p = buffer[3:PACKET_SIZE]   # 11 payload bytes
            buffer = buffer[PACKET_SIZE:]

            if last_ts is not None:
                delta = (ts - last_ts) & 0xFFFFFF
                deltas.append(delta)
                if len(deltas) == 256:
                    real_fs = 32768 / (sum(deltas) / len(deltas))
                    print(f"Real fs: {real_fs:.4f} Hz")
                    deltas.clear()
            last_ts = ts

            if recording:
                # p[0]      : EXG1 status (skip)
                # p[1:4]    : EXG1 CH1 → Lead II         (big-endian 24-bit signed)
                # p[4:7]    : EXG1 CH2 → RLD/ref, skip
                # p[7]      : EXG2 status (skip)
                # p[8:11]   : EXG2 CH1 → Lead I          (big-endian 24-bit signed)
                lead2 = parse_24bit(p[1], p[2], p[3]) * SCALE
                lead1 = parse_24bit(p[8], p[9], p[10]) * SCALE
                records.append((ts / 32768.0, lead1, lead2))

    print(f"Recording done. {len(records)} samples captured.")

    ts_sec = np.array([r[0] for r in records])
    lead1  = np.array([r[1] for r in records])
    lead2  = np.array([r[2] for r in records])
    lead3  = lead2 - lead1   # Einthoven's law

    # bandpass 0.1–40 Hz, 4th-order Butterworth
    sos_bp = butter(4, [0.1, 40.0], btype='band', fs=sampling_freq, output='sos')

    channels = {
        "Lead_I":   sosfiltfilt(sos_bp, lead1),
        "Lead_II":  sosfiltfilt(sos_bp, lead2),
        "Lead_III": sosfiltfilt(sos_bp, lead3),
    }

    for name, sig in channels.items():
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(ts_sec, sig, linewidth=0.7)
        ax.set_title(f"{name} (bandpass 0.1–40 Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = rf"{OUT_DIR}\{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")

    csv_path = rf"{OUT_DIR}\ecg_leads.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ts_sec", "Lead_I_mV", "Lead_II_mV", "Lead_III_mV"])
        writer.writerows(zip(
            ts_sec,
            channels["Lead_I"],
            channels["Lead_II"],
            channels["Lead_III"],
        ))
    print(f"Saved {csv_path}")

# =========================
# ==== ACCEL SECTION ======
# =========================
def run_accel(ser):
    print("Running ACCEL...")

    send_cmd(ser, struct.pack('BBBB', 0x08, 0x80, 0x00, 0x00), "Enable accel")
    send_cmd(ser, struct.pack('BBB', 0x05, 0x00, 0x19), "Set accel rate")

    send_cmd(ser, struct.pack('B', 0x07), "Start streaming")

    buffer = b""
    FRAME_SIZE = 10

    while True:
        chunk = ser.read(ser.in_waiting or FRAME_SIZE)
        if chunk:
            buffer += chunk

        while len(buffer) >= FRAME_SIZE:
            frame = buffer[:FRAME_SIZE]
            buffer = buffer[FRAME_SIZE:]

            t0, t1, t2 = frame[1:4]
            ts = t0 + (t1 << 8) + (t2 << 16)

            ax, ay, az = struct.unpack("<HHH", frame[4:10])

            print(f"ACCEL | ts={ts} | ax={ax} ay={ay} az={az}")

def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    print("Opened", ser.name)

    try:
        # ============================================
        # 🔥 CHOOSE ONE BLOCK ONLY (comment the other)
        # ============================================

        run_exg(ser)

        #run_accel(ser)

    except KeyboardInterrupt:
        print("\nStopping...")
        ser.write(struct.pack('B', 0x20))
        wait_for_ack(ser)
        ser.close()
        print("Closed")

if __name__ == "__main__":
    main()