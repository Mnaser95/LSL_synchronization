#!/usr/bin/env python3
"""
Bridge EMOTIV Cortex streams to Lab Streaming Layer (LSL).

Tested by design against the current Cortex API flow:
requestAccess -> controlDevice(refresh/connect) -> queryHeadsets ->
authorize -> createSession(active) -> subscribe.

This script is intended for Insight 2 and other Cortex-supported headsets,
but it discovers the actual channel labels from the Cortex subscribe response
instead of hard-coding them.

Requirements:
    pip install websockets pylsl

Environment variables (recommended for secrets):
    EMOTIV_CLIENT_ID=...
    EMOTIV_CLIENT_SECRET=...
    EMOTIV2_CLIENT_ID=...       # optional second app/device
    EMOTIV2_CLIENT_SECRET=...

Usage example:
    python emotiv_cortex_to_lsl.py --headset-id INSIGHT2-XXXX

Two-app / two-device example:
    python emotiv_lsl_bridge.py --headset-id INSIGHT2-AAAA --headset-id-2 INSIGHT2-BBBB

Notes:
- Cortex uses WSS on localhost:6868.
- The EEG stream layout must be interpreted from the `cols` field returned by
  the subscribe method.
- For Insight-class headsets, raw EEG runs at 128 Hz.
- LSL timestamps should be in the local LSL clock domain, not Unix epoch.
  Therefore this bridge timestamps outgoing samples at receipt time using
  pylsl.local_clock() and keeps Cortex's own `time` value in stream metadata.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import ssl
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import websockets
from pylsl import StreamInfo, StreamOutlet, cf_double64, local_clock

CORTEX_URI = "wss://localhost:6868"
DEFAULT_STREAMS = ["eeg", "mot"]
METHOD_TIMEOUT_S = 20.0
SCAN_TIMEOUT_S = 12.0
CONNECT_TIMEOUT_S = 15.0

# Optional in-code fallback credentials for local development.
# Command-line arguments still take priority over these values.
EMOTIV_CLIENT_ID_FALLBACK = ""
EMOTIV_CLIENT_SECRET_FALLBACK = ""


class CortexError(RuntimeError):
    """Raised when Cortex returns a JSON-RPC error or an unusable response."""


@dataclass
class SubscriptionInfo:
    stream_name: str
    columns: List[str]
    sid: str


@dataclass
class OutletBundle:
    info: SubscriptionInfo
    outlet: StreamOutlet
    sample_indices: List[int]
    stream_type: str


@dataclass
class DeviceConfig:
    label: str
    client_id: str
    client_secret: str
    headset_id: Optional[str]
    connection_type: Optional[str]
    streams: List[str]
    license_id: Optional[str]
    debit: int


class CortexClient:
    def __init__(self, client_id: str, client_secret: str, uri: str = CORTEX_URI) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.uri = uri
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._request_id = 0

    async def connect(self) -> None:
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        self.ws = await websockets.connect(self.uri, ssl=ssl_ctx, max_size=None)

    async def close(self) -> None:
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception as exc:
                logging.debug("Ignoring websocket close error: %s", exc)
            self.ws = None

    async def call(self, method: str, params: Dict[str, Any]) -> Any:
        if self.ws is None:
            raise RuntimeError("WebSocket is not connected.")
        self._request_id += 1
        req_id = self._request_id
        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        await self.ws.send(json.dumps(request))

        deadline = time.monotonic() + METHOD_TIMEOUT_S
        while True:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise TimeoutError(f"Timed out waiting for response to method '{method}'.")
            raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
            msg = json.loads(raw)

            if "warning" in msg:
                logging.warning("Cortex warning: %s", msg["warning"])
                continue

            if msg.get("id") != req_id:
                logging.debug("Ignoring out-of-band message while waiting for %s: %s", method, msg)
                continue

            if "error" in msg:
                raise CortexError(f"{method} failed: {msg['error']}")
            return msg.get("result")

    async def request_access_until_granted(self) -> None:
        while True:
            result = await self.call(
                "requestAccess",
                {"clientId": self.client_id, "clientSecret": self.client_secret},
            )
            if result.get("accessGranted"):
                logging.info("Application access granted in EMOTIV Launcher.")
                return
            logging.warning(
                "Access not granted yet. Approve the app in EMOTIV Launcher, then the script will retry. Message: %s",
                result.get("message"),
            )
            await asyncio.sleep(2.0)

    async def authorize(self, license_id: Optional[str] = None, debit: int = 0) -> str:
        params: Dict[str, Any] = {
            "clientId": self.client_id,
            "clientSecret": self.client_secret,
        }
        if license_id:
            params["license"] = license_id
        if debit:
            params["debit"] = debit

        result = await self.call("authorize", params)
        token = result.get("cortexToken")
        if not token:
            raise CortexError("authorize succeeded but returned no cortexToken.")
        return token

    async def refresh_headsets(self) -> None:
        await self.call("controlDevice", {"command": "refresh"})

    async def query_headsets(self, headset_id: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if headset_id:
            params["id"] = headset_id
        result = await self.call("queryHeadsets", params)
        if not isinstance(result, list):
            raise CortexError(f"queryHeadsets returned unexpected payload: {result!r}")
        return result

    async def connect_headset(self, headset_id: str, connection_type: Optional[str]) -> None:
        params: Dict[str, Any] = {"command": "connect", "headset": headset_id}
        if connection_type:
            params["connectionType"] = connection_type
        await self.call("controlDevice", params)

    async def create_session(self, cortex_token: str, headset_id: str) -> str:
        result = await self.call(
            "createSession",
            {"cortexToken": cortex_token, "headset": headset_id, "status": "active"},
        )
        sid = result.get("id")
        if not sid:
            raise CortexError("createSession succeeded but returned no session id.")
        return sid

    async def subscribe(self, cortex_token: str, session_id: str, streams: Sequence[str]) -> List[SubscriptionInfo]:
        result = await self.call(
            "subscribe",
            {"cortexToken": cortex_token, "session": session_id, "streams": list(streams)},
        )
        success = result.get("success", [])
        failure = result.get("failure", [])
        if failure:
            logging.warning("Some Cortex subscriptions failed: %s", failure)
        if not success:
            raise CortexError("No streams were subscribed successfully.")
        subs: List[SubscriptionInfo] = []
        for item in success:
            subs.append(
                SubscriptionInfo(
                    stream_name=item["streamName"],
                    columns=list(item["cols"]),
                    sid=item["sid"],
                )
            )
        return subs

    async def stream_messages(self):
        if self.ws is None:
            raise RuntimeError("WebSocket is not connected.")
        async for raw in self.ws:
            yield json.loads(raw)


def infer_sampling_rate(stream_name: str, columns: Sequence[str], headset_obj: Dict[str, Any]) -> float:
    if stream_name == "eeg":
        settings = headset_obj.get("settings", {})
        eeg_rate = settings.get("eegRate") or settings.get("eegSamplingRate")
        if isinstance(eeg_rate, (int, float)) and eeg_rate > 0:
            return float(eeg_rate)
        return 128.0
    if stream_name == "dev":
        return 2.0
    return 0.0  # irregular/unknown rate


def stream_type_for(stream_name: str) -> str:
    return {
        "eeg": "EEG",
        "mot": "Motion",
        "dev": "EEGQuality",
        "eq": "EEGQuality",
        "pow": "BandPower",
        "met": "PerformanceMetrics",
        "com": "MentalCommand",
        "fac": "FacialExpression",
        "sys": "Markers",
    }.get(stream_name, stream_name.upper())


def select_channel_indices(stream_name: str, columns: Sequence[str]) -> List[int]:
    if stream_name == "eeg":
        skip = {
            "COUNTER",
            "INTERPOLATED",
            "RAW_CQ",
            "MARKERS",
            "MARKER_HARDWARE",
            "GYROX",
            "GYROY",
            "GYROZ",
            "Q0",
            "Q1",
            "Q2",
            "Q3",
        }
        return [i for i, c in enumerate(columns) if c not in skip]
    return list(range(len(columns)))


def make_stream_info(
    source_id: str,
    stream_name: str,
    stream_type: str,
    columns: Sequence[str],
    sample_rate: float,
    headset_obj: Dict[str, Any],
    app_name: str,
    selected_indices: Sequence[int],
    lsl_prefix: Optional[str] = None,
) -> StreamInfo:
    channel_names = [columns[i] for i in selected_indices]
    stream_prefix = f"EMOTIV_{lsl_prefix}" if lsl_prefix else "EMOTIV"
    info = StreamInfo(
        name=f"{stream_prefix}_{stream_name.upper()}",
        type=stream_type,
        channel_count=len(channel_names),
        nominal_srate=sample_rate,
        channel_format=cf_double64,
        source_id=f"{source_id}_{lsl_prefix or 'primary'}_{stream_name}",
    )

    desc = info.desc()
    desc.append_child_value("manufacturer", "EMOTIV")
    desc.append_child_value("headset_id", str(headset_obj.get("id", "unknown")))
    desc.append_child_value("headset_status", str(headset_obj.get("status", "unknown")))
    desc.append_child_value("connected_by", str(headset_obj.get("connectedBy", "unknown")))
    desc.append_child_value("bridge", app_name)
    desc.append_child_value("cortex_stream", stream_name)
    desc.append_child_value("timestamp_note", "LSL timestamps are assigned on receipt; original Cortex epoch is logged in console only.")

    channels = desc.append_child("channels")
    for ch in channel_names:
        c = channels.append_child("channel")
        c.append_child_value("label", str(ch))
        if stream_name == "eeg":
            c.append_child_value("type", "EEG")
            c.append_child_value("unit", "microvolts")
        else:
            c.append_child_value("type", stream_type)

    return info


async def wait_for_headset(
    cortex: CortexClient,
    desired_headset_id: Optional[str],
    scan_seconds: float,
) -> Dict[str, Any]:
    await cortex.refresh_headsets()
    deadline = time.monotonic() + scan_seconds

    while time.monotonic() < deadline:
        headsets = await cortex.query_headsets(desired_headset_id)
        if desired_headset_id:
            for hs in headsets:
                if hs.get("id") == desired_headset_id:
                    return hs
        elif headsets:
            return headsets[0]
        await asyncio.sleep(1.0)

    raise RuntimeError(
        "No headset discovered. Make sure Insight 2 is powered on, paired, and visible to Cortex."
    )


async def wait_until_connected(
    cortex: CortexClient,
    headset_id: str,
    timeout_s: float,
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        headsets = await cortex.query_headsets(headset_id)
        for hs in headsets:
            if hs.get("id") == headset_id and hs.get("status") == "connected":
                return hs
        await asyncio.sleep(0.75)
    raise RuntimeError(f"Headset {headset_id} did not reach status='connected' in time.")


def build_outlets(
    subs: Sequence[SubscriptionInfo],
    headset_obj: Dict[str, Any],
    app_name: str,
    lsl_prefix: Optional[str] = None,
) -> Dict[str, OutletBundle]:
    bundles: Dict[str, OutletBundle] = {}
    source_id = str(headset_obj.get("id", "emotiv"))

    for sub in subs:
        idx = select_channel_indices(sub.stream_name, sub.columns)
        if not idx:
            logging.warning("Skipping stream %s because no usable channels remained after filtering.", sub.stream_name)
            continue
        srate = infer_sampling_rate(sub.stream_name, sub.columns, headset_obj)
        s_type = stream_type_for(sub.stream_name)
        info = make_stream_info(
            source_id=source_id,
            stream_name=sub.stream_name,
            stream_type=s_type,
            columns=sub.columns,
            sample_rate=srate,
            headset_obj=headset_obj,
            app_name=app_name,
            selected_indices=idx,
            lsl_prefix=lsl_prefix,
        )
        outlet = StreamOutlet(info, chunk_size=0, max_buffered=360)
        bundles[sub.stream_name] = OutletBundle(
            info=sub,
            outlet=outlet,
            sample_indices=idx,
            stream_type=s_type,
        )
        logging.info(
            "Created LSL outlet %-4s | name=%s | channels=%d | srate=%s",
            sub.stream_name,
            info.name(),
            len(idx),
            info.nominal_srate(),
        )
    return bundles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge EMOTIV Cortex streams to LSL.")
    parser.add_argument(
        "--client-id",
        default=os.getenv("EMOTIV_CLIENT_ID") or EMOTIV_CLIENT_ID_FALLBACK,
        help="Cortex client ID. Defaults to EMOTIV_CLIENT_ID, then EMOTIV_CLIENT_ID_FALLBACK.",
    )
    parser.add_argument(
        "--client-secret",
        default=os.getenv("EMOTIV_CLIENT_SECRET") or EMOTIV_CLIENT_SECRET_FALLBACK,
        help="Cortex client secret. Defaults to EMOTIV_CLIENT_SECRET, then EMOTIV_CLIENT_SECRET_FALLBACK.",
    )
    parser.add_argument(
        "--client-id-2",
        default=os.getenv("EMOTIV2_CLIENT_ID"),
        help="Second Cortex app client ID. Defaults to EMOTIV2_CLIENT_ID. Enables two-device mode with --client-secret-2.",
    )
    parser.add_argument(
        "--client-secret-2",
        default=os.getenv("EMOTIV2_CLIENT_SECRET"),
        help="Second Cortex app client secret. Defaults to EMOTIV2_CLIENT_SECRET.",
    )
    parser.add_argument(
        "--headset-id",
        default=None,
        help="Exact headset id from Cortex (recommended).",
    )
    parser.add_argument(
        "--headset-id-2",
        default=None,
        help="Exact headset id for the second Cortex app/device.",
    )
    parser.add_argument(
        "--connection-type",
        choices=["bluetooth", "dongle", "usb cable"],
        default="bluetooth",
        help="Preferred connection type when connecting the headset.",
    )
    parser.add_argument(
        "--connection-type-2",
        choices=["bluetooth", "dongle", "usb cable"],
        default=None,
        help="Preferred connection type for the second headset. Defaults to --connection-type.",
    )
    parser.add_argument(
        "--streams",
        default=",".join(DEFAULT_STREAMS),
        help="Comma-separated Cortex streams to publish to LSL, e.g. eeg or eeg,dev,eq",
    )
    parser.add_argument(
        "--license-id",
        default=None,
        help="Optional Cortex license id to pass to authorize.",
    )
    parser.add_argument(
        "--license-id-2",
        default=None,
        help="Optional Cortex license id for the second app/device.",
    )
    parser.add_argument(
        "--debit",
        type=int,
        default=0,
        help="Optional local session quota to debit during authorize, e.g. 10.",
    )
    parser.add_argument(
        "--debit-2",
        type=int,
        default=0,
        help="Optional local session quota debit for the second app/device.",
    )
    parser.add_argument(
        "--preview-samples",
        type=int,
        default=5,
        help="How many incoming samples per stream to print before sending to LSL. Use 0 to disable preview.",
    )
    parser.add_argument(
        "--status-every",
        type=int,
        default=256,
        help="Log a streaming status line every N samples per stream. Use 0 to disable periodic status logs.",
    )
    parser.add_argument(
        "--preview-raw-messages",
        type=int,
        default=10,
        help="How many raw Cortex websocket messages to print for debugging. Use 0 to disable.",
    )
    parser.add_argument("--scan-seconds", type=float, default=SCAN_TIMEOUT_S, help="How long to wait for headset discovery.")
    parser.add_argument(
        "--connect-timeout-seconds",
        type=float,
        default=60.0,
        help="How long to wait for a headset to reach status='connected' after connect is requested.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    return parser.parse_args()


async def run_device_bridge(
    config: DeviceConfig,
    args: argparse.Namespace,
    stop_event: asyncio.Event,
    lsl_prefix: Optional[str],
) -> None:
    log_prefix = f"[{config.label}]"
    cortex = CortexClient(config.client_id, config.client_secret)

    try:
        logging.info("%s Connecting to Cortex at %s", log_prefix, CORTEX_URI)
        await cortex.connect()

        logging.info("%s Requesting access approval via EMOTIV Launcher...", log_prefix)
        await cortex.request_access_until_granted()

        logging.info("%s Searching for headset%s...", log_prefix, f" {config.headset_id}" if config.headset_id else "")
        headset = await wait_for_headset(cortex, config.headset_id, args.scan_seconds)
        headset_id = headset["id"]
        logging.info(
            "%s Discovered headset id=%s status=%s connectedBy=%s sensors=%s",
            log_prefix,
            headset.get("id"),
            headset.get("status"),
            headset.get("connectedBy"),
            headset.get("sensors"),
        )

        if headset.get("status") != "connected":
            logging.info("%s Connecting headset %s over %s...", log_prefix, headset_id, config.connection_type)
            await cortex.connect_headset(headset_id, config.connection_type)
            headset = await wait_until_connected(cortex, headset_id, args.connect_timeout_seconds)
            logging.info("%s Headset connected via %s", log_prefix, headset.get("connectedBy"))

        logging.info(
            "%s Authorizing with Cortex%s%s...",
            log_prefix,
            f" using license {config.license_id}" if config.license_id else "",
            f" and debit={config.debit}" if config.debit else "",
        )
        token = await cortex.authorize(license_id=config.license_id, debit=config.debit)
        logging.info("%s Authorized with Cortex.", log_prefix)

        session_id = await cortex.create_session(token, headset_id)
        logging.info("%s Created active session: %s", log_prefix, session_id)

        subs = await cortex.subscribe(token, session_id, config.streams)
        logging.info("%s Subscribed to Cortex streams: %s", log_prefix, ", ".join(s.stream_name for s in subs))

        bundles = build_outlets(
            subs,
            headset,
            app_name=f"emotiv_lsl_bridge.py:{config.label}",
            lsl_prefix=lsl_prefix,
        )
        if not bundles:
            raise RuntimeError(f"{log_prefix} No LSL outlets were created.")

        logging.info("%s Streaming. Press Ctrl+C to stop.", log_prefix)
        sample_counts = {stream_name: 0 for stream_name in bundles}
        first_sample_logged = {stream_name: False for stream_name in bundles}
        raw_message_count = 0

        async for msg in cortex.stream_messages():
            if stop_event.is_set():
                break

            raw_message_count += 1
            if args.preview_raw_messages > 0 and raw_message_count <= args.preview_raw_messages:
                logging.info("%s Raw Cortex message #%d: %s", log_prefix, raw_message_count, msg)

            if "warning" in msg:
                logging.warning("%s Cortex warning: %s", log_prefix, msg["warning"])
                continue

            if "error" in msg:
                logging.error("%s Cortex error payload: %s", log_prefix, msg["error"])
                continue

            for stream_name, bundle in bundles.items():
                if stream_name not in msg:
                    continue
                raw_sample = msg[stream_name]
                if not isinstance(raw_sample, list):
                    continue

                try:
                    lsl_sample = [float(raw_sample[i]) for i in bundle.sample_indices]
                except (TypeError, ValueError, IndexError) as exc:
                    logging.debug("%s Skipping malformed %s sample: %s | %s", log_prefix, stream_name, raw_sample, exc)
                    continue

                sample_counts[stream_name] += 1
                if not first_sample_logged[stream_name]:
                    labels = [bundle.info.columns[i] for i in bundle.sample_indices]
                    logging.info(
                        "%s First %s sample received from Cortex | labels=%s | values=%s",
                        log_prefix,
                        stream_name,
                        labels,
                        lsl_sample,
                    )
                    first_sample_logged[stream_name] = True
                elif args.preview_samples > 0 and sample_counts[stream_name] <= args.preview_samples:
                    logging.info("%s Preview %s sample #%d: %s", log_prefix, stream_name, sample_counts[stream_name], lsl_sample)
                elif args.status_every > 0 and sample_counts[stream_name] % args.status_every == 0:
                    logging.info(
                        "%s Stream %s is active | samples forwarded=%d | latest sample head=%s",
                        log_prefix,
                        stream_name,
                        sample_counts[stream_name],
                        lsl_sample[: min(3, len(lsl_sample))],
                    )

                bundle.outlet.push_sample(lsl_sample, timestamp=local_clock())
    finally:
        await cortex.close()


async def main_async() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not args.client_id or not args.client_secret:
        logging.error("Missing Cortex credentials. Set EMOTIV_CLIENT_ID and EMOTIV_CLIENT_SECRET or pass them as arguments.")
        return 2

    requested_streams = [s.strip() for s in args.streams.split(",") if s.strip()]
    if not requested_streams:
        logging.error("No streams requested.")
        return 2

    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:
                pass

    configs = [
        DeviceConfig(
            label="APP1",
            client_id=args.client_id,
            client_secret=args.client_secret,
            headset_id=args.headset_id,
            connection_type=args.connection_type,
            streams=requested_streams,
            license_id=args.license_id,
            debit=args.debit,
        )
    ]

    second_app_requested = bool(args.client_id_2 or args.client_secret_2 or args.headset_id_2)
    if second_app_requested:
        if not args.client_id_2 or not args.client_secret_2:
            logging.error("Two-device mode needs both EMOTIV2_CLIENT_ID and EMOTIV2_CLIENT_SECRET, or --client-id-2 and --client-secret-2.")
            return 2
        configs.append(
            DeviceConfig(
                label="APP2",
                client_id=args.client_id_2,
                client_secret=args.client_secret_2,
                headset_id=args.headset_id_2,
                connection_type=args.connection_type_2 or args.connection_type,
                streams=requested_streams,
                license_id=args.license_id_2,
                debit=args.debit_2,
            )
        )

    if len(configs) == 2 and (not configs[0].headset_id or not configs[1].headset_id):
        logging.warning(
            "Two-device mode works best when both --headset-id and --headset-id-2 are set; otherwise both apps may select the same discovered headset."
        )

    try:
        if len(configs) == 1:
            await run_device_bridge(configs[0], args, stop_event, lsl_prefix=None)
        else:
            logging.info("Starting two EMOTIV app/device bridges concurrently.")
            await asyncio.gather(
                run_device_bridge(configs[0], args, stop_event, lsl_prefix="APP1"),
                run_device_bridge(configs[1], args, stop_event, lsl_prefix="APP2"),
            )
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    except Exception as exc:
        stop_event.set()
        logging.exception("Fatal error: %s", exc)
        return 1

    return 0


def main() -> None:
    rc = asyncio.run(main_async())
    sys.exit(rc)


if __name__ == "__main__":
    main()
