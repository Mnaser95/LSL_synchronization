import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import pyxdf
except ImportError as exc:
    raise SystemExit(
        "pyxdf is required to read XDF files. Install it with: pip install pyxdf"
    ) from exc


DEFAULT_XDF = Path(
    r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\synchronization"
    r"\LabRecorder-1.17.0-Win_amd64\sub-P001\ses-S001\eeg"
    r"\sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
)
MAX_SUBPLOTS_PER_FIGURE = 4


def get_stream_name(stream):
    return stream["info"]["name"][0]


def get_stream_type(stream):
    return stream["info"]["type"][0]


def get_channel_labels(stream):
    channels = (
        stream["info"]
        .get("desc", [{}])[0]
        .get("channels", [{}])[0]
        .get("channel", [])
    )
    labels = []
    for idx, channel in enumerate(channels):
        label = channel.get("label", [f"ch_{idx}"])[0]
        labels.append(label)

    if labels:
        return labels

    data = np.asarray(stream["time_series"])
    if data.ndim == 1:
        return ["value"]
    return [f"ch_{idx}" for idx in range(data.shape[1])]


def print_stream_summary(streams):
    print("Streams found:")
    for idx, stream in enumerate(streams):
        name = get_stream_name(stream)
        stype = get_stream_type(stream)
        nsamp = len(stream["time_stamps"])
        nominal_rate = stream["info"].get("nominal_srate", ["?"])[0]
        print(f"  [{idx}] {name} ({stype}) - {nsamp} samples @ {nominal_rate} Hz")


def print_marker_events(streams, t0):
    marker_streams = [s for s in streams if get_stream_type(s).lower() == "markers"]
    if not marker_streams:
        print("\nNo marker streams found.")
        return

    print("\nMarker events:")
    for stream in marker_streams:
        name = get_stream_name(stream)
        for ts, raw_value in zip(stream["time_stamps"], stream["time_series"]):
            value = raw_value[0] if isinstance(raw_value, (list, tuple, np.ndarray)) else raw_value
            rel_t = ts - t0
            try:
                parsed = json.loads(value)
                print(f"  {name} @ {rel_t:8.3f}s -> {parsed}")
            except (TypeError, json.JSONDecodeError):
                print(f"  {name} @ {rel_t:8.3f}s -> {value}")


def plot_numeric_streams(streams, t0):
    numeric_streams = [s for s in streams if get_stream_type(s).lower() != "markers"]
    if not numeric_streams:
        print("No numeric streams found to plot.")
        return

    total_axes = 0
    plotted = []
    for stream in numeric_streams:
        data = np.asarray(stream["time_series"])
        if data.size == 0:
            continue
        if data.ndim == 1:
            n_channels = 1
        else:
            n_channels = data.shape[1]
        plotted.append((stream, data, n_channels))
        total_axes += n_channels

    if not plotted:
        print("Numeric streams were present but empty.")
        return

    marker_times = []
    for stream in streams:
        if get_stream_type(stream).lower() == "markers":
            marker_times.extend(stream["time_stamps"])

    plot_items = []
    for stream, data, n_channels in plotted:
        name = get_stream_name(stream)
        ts = np.asarray(stream["time_stamps"]) - t0
        labels = get_channel_labels(stream)

        if data.ndim == 1:
            data = data[:, None]

        for ch_idx in range(n_channels):
            label = labels[ch_idx] if ch_idx < len(labels) else f"ch_{ch_idx}"
            plot_items.append((name, label, ts, data[:, ch_idx]))

    figure_count = int(np.ceil(len(plot_items) / MAX_SUBPLOTS_PER_FIGURE))
    for figure_idx, start in enumerate(range(0, len(plot_items), MAX_SUBPLOTS_PER_FIGURE), start=1):
        chunk = plot_items[start : start + MAX_SUBPLOTS_PER_FIGURE]
        fig, axes = plt.subplots(
            len(chunk),
            1,
            figsize=(15, max(6, 2.8 * len(chunk))),
            sharex=True,
        )
        if len(chunk) == 1:
            axes = [axes]

        for ax, (name, label, ts, values) in zip(axes, chunk):
            ax.plot(ts, values, linewidth=0.8)
            ax.set_ylabel(f"{name}\n{label}")
            ax.grid(True, alpha=0.3)
            for marker_ts in marker_times:
                ax.axvline(marker_ts - t0, color="tab:red", alpha=0.15, linewidth=0.8)

        axes[-1].set_xlabel("Time (s, relative to first sample)")
        fig.suptitle(f"XDF Stream Viewer ({figure_idx}/{figure_count})")
        fig.tight_layout()

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Read and visualize an XDF recording.")
    parser.add_argument(
        "xdf_path",
        nargs="?",
        default=str(DEFAULT_XDF),
        help="Path to the XDF file to view.",
    )
    args = parser.parse_args()

    xdf_path = Path(args.xdf_path)
    if not xdf_path.exists():
        raise SystemExit(f"XDF file not found: {xdf_path}")

    streams, header = pyxdf.load_xdf(str(xdf_path))
    if not streams:
        raise SystemExit(f"No streams found in: {xdf_path}")

    print(f"Loaded: {xdf_path}")
    print_stream_summary(streams)

    all_timestamps = [ts for stream in streams for ts in stream["time_stamps"]]
    t0 = min(all_timestamps)
    print_marker_events(streams, t0)
    plot_numeric_streams(streams, t0)


if __name__ == "__main__":
    main()
