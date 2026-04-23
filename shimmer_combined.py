import argparse
import csv
import json
import sys
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe across threads
import matplotlib.pyplot as plt
import serial
import struct
import time
import pylsl
from scipy.signal import butter, sosfiltfilt

ECG_PORT = "COM6"
EMG_PORT = "COM11"
BAUD     = 115200
OUT_DIR  = r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\synchronization"

WARMUP_S = 2.0
RECORD_S =120


class LslTimestampMapper:
    """Map a device-side tick counter into the sender's LSL clock domain."""

    def __init__(self, ticks_per_second, wrap_bits=24, drift_alpha=0.002):
        self.ticks_per_second = float(ticks_per_second)
        self.wrap = 1 << wrap_bits
        self.half_wrap = self.wrap // 2
        self.drift_alpha = drift_alpha
        self.last_raw = None
        self.wrap_count = 0
        self.offset = None

    def _unwrap_ticks(self, raw_ticks):
        if self.last_raw is not None and raw_ticks < self.last_raw:
            if (self.last_raw - raw_ticks) > self.half_wrap:
                self.wrap_count += 1
        self.last_raw = raw_ticks
        return raw_ticks + self.wrap_count * self.wrap

    def to_lsl_time(self, raw_ticks, arrival_lsl):
        unwrapped_ticks = self._unwrap_ticks(raw_ticks)
        device_time = unwrapped_ticks / self.ticks_per_second
        observed_offset = arrival_lsl - device_time

        if self.offset is None:
            self.offset = observed_offset
        elif observed_offset < self.offset:
            # Pull the estimate down immediately when we see a lower-latency sample.
            self.offset = observed_offset
        else:
            # Track slow drift without following short-term transport jitter.
            self.offset += self.drift_alpha * (observed_offset - self.offset)

        return device_time + self.offset, device_time


def parse_24bit(b0, b1, b2):
    return int.from_bytes([b0, b1, b2], byteorder='big', signed=True)


def wait_for_ack(ser, timeout=3, accept=(b'\xff',)):
    start = time.time()
    while time.time() - start < timeout:
        b = ser.read(1)
        if b in accept:
            return True
    return False


def send_cmd(ser, cmd, name, accept=(b'\xff',)):
    ser.write(cmd)
    ok = wait_for_ack(ser, accept=accept)
    print(f"[{ser.name}] {name}: {'ACK' if ok else 'TIMEOUT'}")
    return ok


def create_marker_outlet():
    info = pylsl.StreamInfo("ShimmerMarkers", "Markers", 1, 0,
                            pylsl.cf_string, "shimmer_markers")
    info.desc().append_child_value("manufacturer", "OpenAI/Codex")
    ch = info.desc().append_child("channels").append_child("channel")
    ch.append_child_value("label", "marker")
    ch.append_child_value("type", "Markers")
    return pylsl.StreamOutlet(info)


def emit_marker(outlet, lock, event, stream, timestamp=None, **payload):
    marker_payload = {"event": event, "stream": stream, **payload}
    marker_text = json.dumps(marker_payload, separators=(",", ":"))
    with lock:
        outlet.push_sample([marker_text], timestamp=timestamp or pylsl.local_clock())


def wait_until_lsl(target_lsl):
    while True:
        remaining = target_lsl - pylsl.local_clock()
        if remaining <= 0:
            return
        time.sleep(min(0.05, max(0.001, remaining / 2.0)))


def parse_args():
    parser = argparse.ArgumentParser(description="Stream one or two Shimmer devices to LSL.")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["ecg", "emg", "both"],
        help="Which Shimmer stream(s) to run. If omitted, you will be prompted.",
    )
    parser.add_argument("--ecg-port", default=ECG_PORT, help=f"COM port for ECG Shimmer. Default: {ECG_PORT}.")
    parser.add_argument("--emg-port", default=EMG_PORT, help=f"COM port for EMG Shimmer. Default: {EMG_PORT}.")
    parser.add_argument("--record-seconds", type=float, default=RECORD_S, help=f"Recording duration. Default: {RECORD_S}.")
    return parser.parse_args()


def choose_run_mode(mode):
    if mode is None:
        mode = input("Choose run mode [ecg/emg/both]: ").strip().lower()

    if mode not in {"ecg", "emg", "both"}:
        raise ValueError(f"Invalid run mode '{mode}'. Choose ecg, emg, or both.")
    return mode



def run_ecg(ser, out, marker_outlet, marker_lock, shared_start):
    print(f"[{ser.name}] Running ECG...")

    send_cmd(ser, struct.pack('BBBB', 0x08, 0x00, 0x00, 0x18), "Enable EXG")

    sampling_freq = 256
    send_cmd(ser, struct.pack('<BH', 0x05, int((2 << 14) / sampling_freq)), "Set sample rate")

    send_cmd(ser, bytes([0x61, 0x00, 0x00, 0x0A,
                         0x02, 0xA0, 0x10, 0x40, 0x40,
                         0x2D, 0x00, 0x00, 0x02, 0x03]), "Configure EXG chip 1")
    send_cmd(ser, bytes([0x61, 0x01, 0x00, 0x0A,
                         0x02, 0xA0, 0x10, 0x40, 0x47,
                         0x00, 0x00, 0x00, 0x02, 0x01]), "Configure EXG chip 2")
    send_cmd(ser, struct.pack('B', 0x07), "Start streaming")

    PACKET_SIZE    = 14
    EXPECTED_DELTA = 128
    SCALE = (2.42 / 4) / (2 ** 23) * 1000

    sync_buf = b""
    while len(sync_buf) < PACKET_SIZE * 10:
        sync_buf += ser.read(PACKET_SIZE * 10 - len(sync_buf))

    best_offset, best_score = 0, -1
    for offset in range(PACKET_SIZE):
        scores, prev, pos = [], None, offset
        while pos + 3 <= len(sync_buf):
            t0, t1, t2 = sync_buf[pos], sync_buf[pos+1], sync_buf[pos+2]
            ts = t0 + (t1 << 8) + (t2 << 16)
            if prev is not None:
                scores.append(abs(((ts - prev) & 0xFFFFFF) - EXPECTED_DELTA))
            prev = ts
            pos += PACKET_SIZE
        if scores:
            score = -sum(scores) / len(scores)
            if score > best_score:
                best_score, best_offset = score, offset

    print(f"[{ser.name}] Sync: offset={best_offset}, error={-best_score:.1f} ticks")
    buffer = sync_buf[best_offset:]

    info = pylsl.StreamInfo("ShimmerECG", "ECG", 4, sampling_freq,
                            pylsl.cf_float32, f"ecg_{ser.name}")
    chns = info.desc().append_child("channels")
    for label, unit in [("ts_sec", "seconds"), ("Lead_I", "millivolts"),
                        ("Lead_II", "millivolts"), ("Lead_III", "millivolts")]:
        ch = chns.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", unit)
        ch.append_child_value("type", "ECG")
    outlet = pylsl.StreamOutlet(info)
    print(f"[{ser.name}] LSL outlet: ShimmerECG @ {sampling_freq} Hz")
    emit_marker(marker_outlet, marker_lock, "stream_ready", "ECG",
                device_port=ser.name, nominal_srate=sampling_freq)

    records = []
    ts_mapper = LslTimestampMapper(ticks_per_second=32768)
    last_ts = None
    recording = False
    t_record_start = None
    announced_wait = False

    while True:
        chunk = ser.read(ser.in_waiting or PACKET_SIZE)
        if chunk:
            buffer += chunk

        now_lsl = pylsl.local_clock()
        recording_start_lsl = shared_start["recording_start_lsl"]
        if recording_start_lsl is None:
            if not announced_wait:
                print(f"[{ser.name}] Waiting for scheduled recording start...")
                announced_wait = True
        elif not recording and now_lsl >= recording_start_lsl:
            recording = True
            t_record_start = now_lsl
            print(f"[{ser.name}] Recording started...")
            emit_marker(marker_outlet, marker_lock, "recording_started", "ECG",
                        timestamp=now_lsl, device_port=ser.name)
        if recording and (now_lsl - t_record_start) >= RECORD_S:
            break

        while len(buffer) >= PACKET_SIZE:
            t0, t1, t2 = buffer[0], buffer[1], buffer[2]
            ts = t0 + (t1 << 8) + (t2 << 16)
            p  = buffer[3:PACKET_SIZE]
            buffer = buffer[PACKET_SIZE:]
            last_ts = ts
            lead2 = parse_24bit(p[1], p[2], p[3]) * SCALE
            lead1 = parse_24bit(p[8], p[9], p[10]) * SCALE
            arrival_lsl = pylsl.local_clock()
            lsl_t, device_t = ts_mapper.to_lsl_time(ts, arrival_lsl)
            outlet.push_sample([ts / 32768.0, lead1, lead2, lead2 - lead1], lsl_t)
            if recording:
                records.append((lsl_t, device_t, lead1, lead2))

    print(f"[{ser.name}] Done. {len(records)} ECG samples.")
    emit_marker(marker_outlet, marker_lock, "recording_stopped", "ECG",
                device_port=ser.name, samples=len(records))

    lsl_ts = np.array([r[0] for r in records])
    device_ts = np.array([r[1] for r in records])
    lead1  = np.array([r[2] for r in records])
    lead2  = np.array([r[3] for r in records])
    sos_bp = butter(4, [0.1, 40.0], btype='band', fs=sampling_freq, output='sos')
    lead1_f = sosfiltfilt(sos_bp, lead1)
    lead2_f = sosfiltfilt(sos_bp, lead2)
    out['lsl_ts']   = lsl_ts
    out['device_ts'] = device_ts
    out['Lead_I']   = lead1_f
    out['Lead_II']  = lead2_f
    out['Lead_III'] = lead2_f - lead1_f


def run_emg(ser, out, marker_outlet, marker_lock, shared_start):
    print(f"[{ser.name}] Running EMG...")

    ser.write(struct.pack('B', 0x20))
    time.sleep(1.0)
    ser.reset_input_buffer()

    ack = (b'\xff', b'\xfe')
    send_cmd(ser, struct.pack('BBBB', 0x08, 0x00, 0x00, 0x18), "Enable EXG", ack)

    sampling_freq = 512
    send_cmd(ser, struct.pack('<BH', 0x05, int((2 << 14) / sampling_freq)), "Set sample rate", ack)

    send_cmd(ser, bytes([0x61, 0x00, 0x00, 0x0A,
                         0x03, 0xA0, 0x00, 0x60, 0x60,
                         0x00, 0x00, 0x00, 0x02, 0x01]), "Configure EMG chip 1", ack)
    send_cmd(ser, bytes([0x61, 0x01, 0x00, 0x0A,
                         0x03, 0xA0, 0x00, 0x60, 0x60,
                         0x00, 0x00, 0x00, 0x02, 0x01]), "Configure EMG chip 2", ack)
    send_cmd(ser, struct.pack('B', 0x07), "Start streaming", ack)

    PACKET_SIZE    = 13
    EXPECTED_DELTA = 64
    SCALE = (2.42 / 12) / (2 ** 23) * 1000

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

    print(f"[{ser.name}] Sync: offset={best_offset}, error={-best_score:.1f} ticks")
    buffer = sync_buf[best_offset:]

    info = pylsl.StreamInfo("ShimmerEMG", "EMG", 3, sampling_freq,
                            pylsl.cf_float32, f"emg_{ser.name}")
    chns = info.desc().append_child("channels")
    for label, unit in [("ts_sec", "seconds"), ("EMG_CH1", "millivolts"),
                        ("EMG_CH2", "millivolts")]:
        ch = chns.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", unit)
        ch.append_child_value("type", "EMG")
    outlet = pylsl.StreamOutlet(info)
    print(f"[{ser.name}] LSL outlet: ShimmerEMG @ {sampling_freq} Hz")
    emit_marker(marker_outlet, marker_lock, "stream_ready", "EMG",
                device_port=ser.name, nominal_srate=sampling_freq)

    records = []
    ts_mapper = LslTimestampMapper(ticks_per_second=32768)
    last_ts = None
    recording = False
    t_record_start = None
    announced_wait = False

    while True:
        chunk = ser.read(ser.in_waiting or PACKET_SIZE)
        if chunk:
            buffer += chunk

        now_lsl = pylsl.local_clock()
        recording_start_lsl = shared_start["recording_start_lsl"]
        if recording_start_lsl is None:
            if not announced_wait:
                print(f"[{ser.name}] Waiting for scheduled recording start...")
                announced_wait = True
        elif not recording and now_lsl >= recording_start_lsl:
            recording = True
            t_record_start = now_lsl
            print(f"[{ser.name}] Recording started...")
            emit_marker(marker_outlet, marker_lock, "recording_started", "EMG",
                        timestamp=now_lsl, device_port=ser.name)
        if recording and (now_lsl - t_record_start) >= RECORD_S:
            break

        while len(buffer) >= PACKET_SIZE:
            t0, t1, t2 = buffer[1], buffer[2], buffer[3]
            ts = t0 + (t1 << 8) + (t2 << 16)
            p  = buffer[4:PACKET_SIZE]
            buffer = buffer[PACKET_SIZE:]
            last_ts = ts
            ch2 = parse_24bit(p[3], p[4], p[5]) * SCALE
            ch1 = parse_24bit(p[6], p[7], p[8]) * SCALE
            arrival_lsl = pylsl.local_clock()
            lsl_t, device_t = ts_mapper.to_lsl_time(ts, arrival_lsl)
            outlet.push_sample([ts / 32768.0, ch1, ch2], lsl_t)
            if recording:
                records.append((lsl_t, device_t, ch1, ch2))

    print(f"[{ser.name}] Done. {len(records)} EMG samples.")
    emit_marker(marker_outlet, marker_lock, "recording_stopped", "EMG",
                device_port=ser.name, samples=len(records))

    lsl_ts = np.array([r[0] for r in records])
    device_ts = np.array([r[1] for r in records])
    ch1    = np.array([r[2] for r in records])
    ch2    = np.array([r[3] for r in records])
    sos_bp = butter(4, [20.0, 200.0], btype='band', fs=sampling_freq, output='sos')
    ch1_f = sosfiltfilt(sos_bp, ch1)
    ch2_f = sosfiltfilt(sos_bp, ch2)
    out['lsl_ts']  = lsl_ts
    out['device_ts'] = device_ts
    out['EMG_CH1'] = ch1_f
    out['EMG_CH2'] = ch2_f


def stop_device(ser):
    try:
        ser.write(struct.pack('B', 0x20))
        time.sleep(0.3)
        ser.close()
    except Exception:
        pass


def save_synchronized(ecg_out, emg_out):
    t_start = max(ecg_out['lsl_ts'][0], emg_out['lsl_ts'][0])

    t_ecg_rel = ecg_out['lsl_ts'] - t_start
    t_emg_rel = emg_out['lsl_ts'] - t_start
    t_ecg_dev_rel = ecg_out['device_ts'] - ecg_out['device_ts'][0]
    t_emg_dev_rel = emg_out['device_ts'] - emg_out['device_ts'][0]

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    for ax, (name, t_rel, sig) in zip(axes, [
        ("Lead_I",   t_ecg_rel, ecg_out['Lead_I']),
        ("Lead_II",  t_ecg_rel, ecg_out['Lead_II']),
        ("Lead_III", t_ecg_rel, ecg_out['Lead_III']),
        ("EMG_CH1",  t_emg_rel, emg_out['EMG_CH1']),
        ("EMG_CH2",  t_emg_rel, emg_out['EMG_CH2']),
    ]):
        ax.plot(t_rel, sig, linewidth=0.7)
        ax.set_ylabel(f"{name}\n(mV)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s, LSL-aligned)")
    fig.suptitle("Synchronized ECG + EMG (LSL-aligned)")
    fig.tight_layout()
    path = rf"{OUT_DIR}\synchronized.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # Two CSVs at native rates, both referenced to the same LSL t_start
    ecg_csv = rf"{OUT_DIR}\ecg_synchronized.csv"
    with open(ecg_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lsl_t_rel_s", "device_t_rel_s", "Lead_I_mV", "Lead_II_mV", "Lead_III_mV"])
        writer.writerows(zip(t_ecg_rel, t_ecg_dev_rel, ecg_out['Lead_I'],
                             ecg_out['Lead_II'], ecg_out['Lead_III']))
    print(f"Saved {ecg_csv}")

    emg_csv = rf"{OUT_DIR}\emg_synchronized.csv"
    with open(emg_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lsl_t_rel_s", "device_t_rel_s", "EMG_CH1_mV", "EMG_CH2_mV"])
        writer.writerows(zip(t_emg_rel, t_emg_dev_rel, emg_out['EMG_CH1'], emg_out['EMG_CH2']))
    print(f"Saved {emg_csv}")


def save_single_stream(stream_name, stream_out):
    t_rel = stream_out['lsl_ts'] - stream_out['lsl_ts'][0]
    t_dev_rel = stream_out['device_ts'] - stream_out['device_ts'][0]

    if stream_name == "ECG":
        channels = [
            ("Lead_I", stream_out['Lead_I']),
            ("Lead_II", stream_out['Lead_II']),
            ("Lead_III", stream_out['Lead_III']),
        ]
        csv_path = rf"{OUT_DIR}\ecg_synchronized.csv"
        plot_path = rf"{OUT_DIR}\ecg_synchronized.png"
        header = ["lsl_t_rel_s", "device_t_rel_s", "Lead_I_mV", "Lead_II_mV", "Lead_III_mV"]
        rows = zip(t_rel, t_dev_rel, stream_out['Lead_I'], stream_out['Lead_II'], stream_out['Lead_III'])
    else:
        channels = [
            ("EMG_CH1", stream_out['EMG_CH1']),
            ("EMG_CH2", stream_out['EMG_CH2']),
        ]
        csv_path = rf"{OUT_DIR}\emg_synchronized.csv"
        plot_path = rf"{OUT_DIR}\emg_synchronized.png"
        header = ["lsl_t_rel_s", "device_t_rel_s", "EMG_CH1_mV", "EMG_CH2_mV"]
        rows = zip(t_rel, t_dev_rel, stream_out['EMG_CH1'], stream_out['EMG_CH2'])

    fig, axes = plt.subplots(len(channels), 1, figsize=(14, 3 * len(channels)), sharex=True)
    if len(channels) == 1:
        axes = [axes]

    for ax, (name, signal) in zip(axes, channels):
        ax.plot(t_rel, signal, linewidth=0.7)
        ax.set_ylabel(f"{name}\n(mV)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s, LSL-aligned)")
    fig.suptitle(f"Synchronized {stream_name} (LSL-aligned)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved {plot_path}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved {csv_path}")


def main():
    global RECORD_S

    args = parse_args()
    run_mode = choose_run_mode(args.mode)
    RECORD_S = args.record_seconds
    marker_outlet = create_marker_outlet()
    marker_lock = threading.Lock()
    emit_marker(marker_outlet, marker_lock, "session_started", "SYSTEM", mode=run_mode)

    ecg_ser = None
    emg_ser = None
    if run_mode in {"ecg", "both"}:
        ecg_ser = serial.Serial(args.ecg_port, BAUD, timeout=1)
        ecg_ser.reset_input_buffer()
        ecg_ser.reset_output_buffer()
        print("Opened", ecg_ser.name)

    if run_mode in {"emg", "both"}:
        emg_ser = serial.Serial(args.emg_port, BAUD, timeout=1)
        emg_ser.reset_input_buffer()
        emg_ser.reset_output_buffer()
        print("Opened", emg_ser.name)

    ecg_out = {}
    emg_out = {}
    shared_start = {"recording_start_lsl": None}
    threads = []
    if ecg_ser is not None:
        threads.append(threading.Thread(
            target=run_ecg,
            args=(ecg_ser, ecg_out, marker_outlet, marker_lock, shared_start),
            daemon=True,
        ))
    if emg_ser is not None:
        threads.append(threading.Thread(
            target=run_emg,
            args=(emg_ser, emg_out, marker_outlet, marker_lock, shared_start),
            daemon=True,
        ))

    for thread in threads:
        thread.start()

    print("\nLSL streams are ready to publish once the device setup completes.")
    if run_mode == "both":
        print("Open LabRecorder, wait for ShimmerECG, ShimmerEMG, and ShimmerMarkers, then press Enter here.")
    elif run_mode == "ecg":
        print("Open LabRecorder, wait for ShimmerECG and ShimmerMarkers, then press Enter here.")
    else:
        print("Open LabRecorder, wait for ShimmerEMG and ShimmerMarkers, then press Enter here.")
    input("Press Enter when LabRecorder is recording...")
    recording_start_lsl = pylsl.local_clock() + WARMUP_S
    shared_start["recording_start_lsl"] = recording_start_lsl
    emit_marker(marker_outlet, marker_lock, "recording_armed", "SYSTEM",
                timestamp=pylsl.local_clock(), start_lsl=recording_start_lsl,
                warmup_s=WARMUP_S, record_s=RECORD_S, mode=run_mode)
    print(f"Recording will start in {WARMUP_S:.1f} s and run for {RECORD_S:.1f} s.")

    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\nStopping...")
        emit_marker(marker_outlet, marker_lock, "session_interrupted", "SYSTEM")
        if ecg_ser is not None:
            stop_device(ecg_ser)
        if emg_ser is not None:
            stop_device(emg_ser)
        return

    if ecg_ser is not None:
        ecg_ser.close()
    if emg_ser is not None:
        emg_ser.close()
    emit_marker(marker_outlet, marker_lock, "session_finished", "SYSTEM")
    print("All done.")
    if run_mode == "both":
        save_synchronized(ecg_out, emg_out)
    elif run_mode == "ecg":
        save_single_stream("ECG", ecg_out)
    else:
        save_single_stream("EMG", emg_out)


if __name__ == "__main__":
    main()
