# LSL Synchronization

Tools for streaming Shimmer and EMOTIV devices into Lab Streaming Layer (LSL), recording them with LabRecorder, and plotting recorded XDF files.

## Scripts

| Script | Purpose |
| --- | --- |
| `run_lsl_streams.py` | Main launcher for choosing Shimmer, EMOTIV, or both. |
| `shimmer_lsl_bridge.py` | Streams Shimmer ECG/EMG data to LSL. |
| `emotiv_lsl_bridge.py` | Streams EMOTIV Cortex data to LSL. |
| `plot_xdf_streams.py` | Loads and plots multi-stream XDF recordings. |

## Setup

Install Python dependencies:

```bat
pip install pylsl pyserial numpy matplotlib scipy websockets pyxdf
```

Keep EMOTIV credentials in local text files that are not committed to GitHub:

```text
CLIENT_ID=your_client_id
CLIENT_SECRET=your_client_secret
```

Example files:

```text
app1_credentials.txt
app2_credentials.txt
```

## Run

Start the interactive launcher:

```bat
python run_lsl_streams.py
```

Run Shimmer ECG and EMG only:

```bat
python run_lsl_streams.py --shimmer both --ecg-port COM6 --emg-port COM11 --emotiv none
```

Run one EMOTIV app/headset only:

```bat
python run_lsl_streams.py --shimmer none --emotiv app1 --credentials-file app1_credentials.txt
```

Run Shimmer plus two EMOTIV apps/headsets:

```bat
python run_lsl_streams.py --shimmer both --ecg-port COM6 --emg-port COM11 --emotiv both --credentials-file app1_credentials.txt --credentials-file-2 app2_credentials.txt
```

## Plot XDF

```bat
python plot_xdf_streams.py path\to\recording.xdf
```

Plots are split so each figure has no more than four subplots.
