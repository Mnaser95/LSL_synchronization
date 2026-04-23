#!/usr/bin/env python3
"""
Choose which Shimmer stream(s) and which EMOTIV app/headset(s) to run together.

Examples:
    python run_lsl_streams.py
    python run_lsl_streams.py --shimmer both --ecg-port COM6 --emg-port COM11 --emotiv app1
    python run_lsl_streams.py --shimmer ecg --ecg-port COM6 --emotiv both

Credentials can be provided with command-line args or environment variables:
    EMOTIV_CLIENT_ID / EMOTIV_CLIENT_SECRET
    EMOTIV2_CLIENT_ID / EMOTIV2_CLIENT_SECRET

Or with text files:
    CLIENT_ID=...
    CLIENT_SECRET=...

    # These also work:
    Client ID: ...
    Client Secret: ...

    # Or two plain lines:
    your_client_id
    your_client_secret
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


ROOT = Path(__file__).resolve().parent
SHIMMER_SCRIPT = ROOT / "shimmer_lsl_bridge.py"
EMOTIV_SCRIPT = ROOT / "emotiv_lsl_bridge.py"


def prompt_choice(label: str, choices: List[str], default: str) -> str:
    choices_text = "/".join(choices)
    while True:
        value = input(f"{label} [{choices_text}] (default {default}): ").strip().lower()
        if not value:
            return default
        if value in choices:
            return value
        print(f"Please choose one of: {choices_text}")


def prompt_text(label: str, default: Optional[str] = None, required: bool = False) -> str:
    suffix = f" (default {default})" if default else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        if not required:
            return ""
        print("This value is required.")


def add_if_value(command: List[str], flag: str, value: Optional[str]) -> None:
    if value:
        command.extend([flag, value])


def read_credential_file(path_text: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not path_text:
        return None, None

    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Credential file not found: {path}")

    values = {}
    plain_lines = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        separator = "=" if "=" in line else ":" if ":" in line else None
        if separator:
            key, value = line.split(separator, 1)
            normalized_key = key.strip().upper().replace(" ", "_").replace("-", "_")
            values[normalized_key] = value.strip().strip('"').strip("'")
        else:
            plain_lines.append(line.strip('"').strip("'"))

    client_id = (
        values.get("CLIENT_ID")
        or values.get("EMOTIV_CLIENT_ID")
        or values.get("CLIENTID")
        or values.get("ID")
        or (plain_lines[0] if len(plain_lines) >= 1 else None)
    )
    client_secret = (
        values.get("CLIENT_SECRET")
        or values.get("EMOTIV_CLIENT_SECRET")
        or values.get("CLIENTSECRET")
        or values.get("SECRET_ID")
        or values.get("SECRET")
        or (plain_lines[1] if len(plain_lines) >= 2 else None)
    )
    return client_id, client_secret


def merge_credentials(
    client_id: Optional[str],
    client_secret: Optional[str],
    file_path: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    file_client_id, file_client_secret = read_credential_file(file_path)
    return client_id or file_client_id, client_secret or file_client_secret


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected Shimmer and EMOTIV LSL streams together.")
    parser.add_argument("--shimmer", choices=["none", "ecg", "emg", "both"], help="Which Shimmer stream(s) to run.")
    parser.add_argument("--ecg-port", help="COM port for ECG Shimmer, e.g. COM6.")
    parser.add_argument("--emg-port", help="COM port for EMG Shimmer, e.g. COM11.")
    parser.add_argument("--record-seconds", type=float, help="Shimmer recording duration in seconds.")

    parser.add_argument("--emotiv", choices=["none", "app1", "app2", "both"], help="Which EMOTIV app/headset session(s) to run.")
    parser.add_argument("--client-id", default=os.getenv("EMOTIV_CLIENT_ID"), help="First EMOTIV app client ID.")
    parser.add_argument("--client-secret", default=os.getenv("EMOTIV_CLIENT_SECRET"), help="First EMOTIV app client secret.")
    parser.add_argument("--client-id-2", default=os.getenv("EMOTIV2_CLIENT_ID"), help="Second EMOTIV app client ID.")
    parser.add_argument("--client-secret-2", default=os.getenv("EMOTIV2_CLIENT_SECRET"), help="Second EMOTIV app client secret.")
    parser.add_argument("--credentials-file", help="Text file containing first EMOTIV app client ID/secret.")
    parser.add_argument("--credentials-file-2", help="Text file containing second EMOTIV app client ID/secret.")
    parser.add_argument("--headset-id", help="Optional first EMOTIV headset ID.")
    parser.add_argument("--headset-id-2", help="Optional second EMOTIV headset ID.")
    parser.add_argument("--streams", default="eeg,mot", help="EMOTIV streams, e.g. eeg or eeg,mot.")
    parser.add_argument(
        "--emotiv-connect-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for an EMOTIV headset to connect. Default: 60.",
    )
    return parser.parse_args()


def build_shimmer_command(args: argparse.Namespace) -> Optional[List[str]]:
    shimmer = args.shimmer or prompt_choice("Choose Shimmer", ["none", "ecg", "emg", "both"], "both")
    if shimmer == "none":
        return None

    command = [sys.executable, str(SHIMMER_SCRIPT), shimmer]
    if shimmer in {"ecg", "both"}:
        ecg_port = args.ecg_port or prompt_text("ECG Shimmer COM port", "COM6")
        command.extend(["--ecg-port", ecg_port])
    if shimmer in {"emg", "both"}:
        emg_port = args.emg_port or prompt_text("EMG Shimmer COM port", "COM11")
        command.extend(["--emg-port", emg_port])
    if args.record_seconds is not None:
        command.extend(["--record-seconds", str(args.record_seconds)])
    return command


def get_app1_credentials(args: argparse.Namespace) -> tuple[str, str]:
    client_id, client_secret = merge_credentials(args.client_id, args.client_secret, args.credentials_file)
    client_id = client_id or prompt_text("APP1 EMOTIV client ID", required=True)
    client_secret = client_secret or prompt_text("APP1 EMOTIV client secret", required=True)
    return client_id, client_secret


def get_app2_credentials(args: argparse.Namespace) -> tuple[str, str]:
    client_id, client_secret = merge_credentials(args.client_id_2, args.client_secret_2, args.credentials_file_2)
    client_id = client_id or prompt_text("APP2 EMOTIV client ID", required=True)
    client_secret = client_secret or prompt_text("APP2 EMOTIV client secret", required=True)
    return client_id, client_secret


def build_emotiv_command(args: argparse.Namespace) -> Optional[List[str]]:
    emotiv = args.emotiv or prompt_choice("Choose EMOTIV", ["none", "app1", "app2", "both"], "app1")
    if emotiv == "none":
        return None

    command = [
        sys.executable,
        str(EMOTIV_SCRIPT),
        "--streams",
        args.streams,
        "--connect-timeout-seconds",
        str(args.emotiv_connect_timeout),
    ]
    if emotiv == "app1":
        client_id, client_secret = get_app1_credentials(args)
        command.extend(["--client-id", client_id, "--client-secret", client_secret])
        headset_id = args.headset_id or prompt_text("APP1 headset ID, or press Enter to auto-discover")
        add_if_value(command, "--headset-id", headset_id)
    elif emotiv == "app2":
        client_id, client_secret = get_app2_credentials(args)
        command.extend(["--client-id", client_id, "--client-secret", client_secret])
        headset_id = args.headset_id_2 or prompt_text("APP2 headset ID, or press Enter to auto-discover")
        add_if_value(command, "--headset-id", headset_id)
    else:
        client_id, client_secret = get_app1_credentials(args)
        client_id_2, client_secret_2 = get_app2_credentials(args)
        command.extend(["--client-id", client_id, "--client-secret", client_secret])
        command.extend(["--client-id-2", client_id_2, "--client-secret-2", client_secret_2])
        headset_id = args.headset_id or prompt_text("APP1 headset ID, or press Enter to auto-discover")
        headset_id_2 = args.headset_id_2 or prompt_text("APP2 headset ID, or press Enter to auto-discover")
        add_if_value(command, "--headset-id", headset_id)
        add_if_value(command, "--headset-id-2", headset_id_2)
    return command


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def main() -> int:
    args = parse_args()
    shimmer_command = build_shimmer_command(args)
    emotiv_command = build_emotiv_command(args)

    if shimmer_command is None and emotiv_command is None:
        print("Nothing selected. Choose at least one Shimmer or EMOTIV stream.")
        return 2

    print("\nSelected commands:")
    if emotiv_command:
        print("EMOTIV: ", subprocess.list2cmdline(emotiv_command))
    if shimmer_command:
        print("Shimmer:", subprocess.list2cmdline(shimmer_command))
    print()

    emotiv_process = None
    try:
        if emotiv_command:
            print("Starting EMOTIV bridge first...")
            emotiv_process = subprocess.Popen(emotiv_command, cwd=str(ROOT))
            time.sleep(2.0)

        if shimmer_command:
            print("Starting Shimmer bridge. Follow its LabRecorder prompt when it appears.")
            shimmer_result = subprocess.run(shimmer_command, cwd=str(ROOT), check=False)
            return_code = shimmer_result.returncode
        elif emotiv_process:
            print("EMOTIV bridge is running. Press Ctrl+C to stop.")
            return_code = emotiv_process.wait()
        else:
            return_code = 0
    except KeyboardInterrupt:
        print("\nStopping selected streams...")
        return_code = 130
    finally:
        if emotiv_process:
            terminate(emotiv_process)

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
