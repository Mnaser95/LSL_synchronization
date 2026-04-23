import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import pyxdf
except ImportError as exc:
    raise SystemExit(
        "pyxdf is required to read XDF files. Install it with: pip install pyxdf"
    ) from exc


DEFAULT_XDF_DIR = Path(
    r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\synchronization"
    r"\LabRecorder-1.17.0-Win_amd64\sub-P001\ses-S001\eeg"
)
DEFAULT_STREAM_NAME = "all"


def find_default_xdf() -> Path:
    candidates = sorted(DEFAULT_XDF_DIR.glob("*.xdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"No XDF files found in: {DEFAULT_XDF_DIR}")
    return candidates[0]


def get_stream_name(stream):
    return stream["info"]["name"][0]


def is_emotiv_stream(stream):
    return get_stream_name(stream).startswith("EMOTIV_")


def get_channel_labels(stream):
    desc = stream["info"].get("desc", [{}])
    channels = desc[0].get("channels", [{}])[0].get("channel", []) if desc else []
    labels = [channel.get("label", [f"ch_{idx}"])[0] for idx, channel in enumerate(channels)]
    if labels:
        return labels

    data = np.asarray(stream["time_series"])
    if data.ndim == 1:
        return ["value"]
    return [f"ch_{idx}" for idx in range(data.shape[1])]


def print_stream_summary(stream):
    desc = stream["info"].get("desc", [{}])
    name = get_stream_name(stream)
    stype = stream["info"]["type"][0]
    nsamp = len(stream["time_stamps"])
    srate = stream["info"].get("nominal_srate", ["?"])[0]
    labels = get_channel_labels(stream)

    print(f"Stream: {name} ({stype})")
    print(f"Samples: {nsamp}")
    print(f"Nominal rate: {srate} Hz")
    print(f"Channels: {labels}")

    if desc:
        for key in ["manufacturer", "headset_id", "headset_status", "connected_by", "bridge", "cortex_stream"]:
            values = desc[0].get(key, [])
            if values:
                print(f"{key}: {values[0]}")


def select_emotiv_stream(streams, stream_name: str):
    for stream in streams:
        if get_stream_name(stream) == stream_name:
            return stream
    names = ", ".join(get_stream_name(stream) for stream in streams)
    raise SystemExit(f"Could not find stream '{stream_name}'. Streams present: {names}")


def select_emotiv_streams(streams, stream_name: str):
    if stream_name.lower() == "all":
        selected = [stream for stream in streams if is_emotiv_stream(stream)]
        if not selected:
            names = ", ".join(get_stream_name(stream) for stream in streams)
            raise SystemExit(f"No EMOTIV streams found. Streams present: {names}")
        return selected
    return [select_emotiv_stream(streams, stream_name)]


def plot_emotiv_stream(stream, include_markers: bool):
    ts = np.asarray(stream["time_stamps"])
    data = np.asarray(stream["time_series"])
    labels = get_channel_labels(stream)

    if len(ts) == 0 or data.size == 0:
        print("This XDF contains the EMOTIV stream metadata, but zero recorded samples.")
        print("LabRecorder likely discovered the stream without capturing actual EEG data.")
        return

    if data.ndim == 1:
        data = data[:, None]

    plot_indices = list(range(data.shape[1]))
    if not include_markers:
        plot_indices = [i for i, label in enumerate(labels) if label.upper() != "MARKERS"]

    if not plot_indices:
        raise SystemExit("No plottable EMOTIV channels remained after filtering.")

    t_rel = ts - ts[0]
    fig, axes = plt.subplots(len(plot_indices), 1, figsize=(15, max(6, 2.5 * len(plot_indices))), sharex=True)
    if len(plot_indices) == 1:
        axes = [axes]

    marker_idx = None
    for idx, label in enumerate(labels):
        if label.upper() == "MARKERS":
            marker_idx = idx
            break

    marker_times = []
    if marker_idx is not None and marker_idx < data.shape[1]:
        marker_values = data[:, marker_idx]
        marker_times = t_rel[np.abs(marker_values) > 1e-12]

    for ax, idx in zip(axes, plot_indices):
        label = labels[idx] if idx < len(labels) else f"ch_{idx}"
        ax.plot(t_rel, data[:, idx], linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        for mt in marker_times:
            ax.axvline(mt, color="tab:red", alpha=0.15, linewidth=0.8)

    axes[-1].set_xlabel("Time (s, relative to first EMOTIV sample)")
    fig.suptitle(f"{get_stream_name(stream)} Viewer")
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Read and visualize an EMOTIV XDF recording.")
    parser.add_argument(
        "xdf_path",
        nargs="?",
        default=None,
        help="Path to the XDF file. Defaults to the newest .xdf in the LabRecorder EEG folder.",
    )
    parser.add_argument(
        "--stream-name",
        default=DEFAULT_STREAM_NAME,
        help="LSL stream name to inspect, or 'all' to plot every EMOTIV stream in the file.",
    )
    parser.add_argument(
        "--include-markers-channel",
        action="store_true",
        help="Include the EMOTIV MARKERS channel in the plots.",
    )
    args = parser.parse_args()

    xdf_path = Path(args.xdf_path) if args.xdf_path else find_default_xdf()
    if not xdf_path.exists():
        raise SystemExit(f"XDF file not found: {xdf_path}")

    streams, header = pyxdf.load_xdf(str(xdf_path))
    if not streams:
        raise SystemExit(f"No streams found in: {xdf_path}")

    print(f"Loaded: {xdf_path}")
    selected_streams = select_emotiv_streams(streams, args.stream_name)
    for stream in selected_streams:
        print()
        print_stream_summary(stream)
        plot_emotiv_stream(stream, include_markers=args.include_markers_channel)


if __name__ == "__main__":
    main()
