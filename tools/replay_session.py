"""Replay a recorded .mdlg session as TCP streams.

Binds two local TCP servers and serves the recorded packets at the
original timing (or faster, with --fast). Consumers (viewer, calibrator,
future offline SLAM tools) connect as if they were talking to the live
Pi:

    ToF stream     →  localhost:9872   (same wire as tof_streamer.py)
    Digital twin   →  localhost:9870   (same wire as runtime --stream)

Multiple clients per stream are supported. Clients can connect / drop
mid-playback; nothing buffered for the disconnected slot.

Usage:
    uv run tools/replay_session.py session.mdlg
    uv run tools/replay_session.py session.mdlg --fast 4    # 4x speed
    uv run tools/replay_session.py session.mdlg --loop      # repeat
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import threading
import time
from pathlib import Path


_MAGIC = b"MDLG"
_VERSION = 1
_HEADER_FMT = "<4sIQ"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_RECORD_HEADER_FMT = "<QBI"
_RECORD_HEADER_SIZE = struct.calcsize(_RECORD_HEADER_FMT)

_STREAM_TOF  = 0
_STREAM_TWIN = 1


def _load(path: Path):
    with path.open("rb") as f:
        head = f.read(_HEADER_SIZE)
        if len(head) < _HEADER_SIZE:
            raise ValueError("truncated header")
        magic, version, epoch_ms = struct.unpack(_HEADER_FMT, head)
        if magic != _MAGIC:
            raise ValueError(f"bad magic {magic!r}")
        if version != _VERSION:
            raise ValueError(f"unsupported version {version}")
        records: list[tuple[int, int, bytes]] = []
        while True:
            rh = f.read(_RECORD_HEADER_SIZE)
            if len(rh) < _RECORD_HEADER_SIZE:
                break
            ts_us, sid, size = struct.unpack(_RECORD_HEADER_FMT, rh)
            payload = f.read(size)
            if len(payload) < size:
                break
            records.append((ts_us, sid, payload))
        return records, epoch_ms


class _BroadcastServer:
    """Tiny TCP broadcaster: accept-only thread appends to a client list,
    `broadcast()` sends bytes to all of them and drops any that error."""

    def __init__(self, port: int, label: str):
        self._label = label
        self._lock = threading.Lock()
        self._clients: list[socket.socket] = []
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", port))
        self._sock.listen(8)
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self) -> None:
        while True:
            try:
                conn, addr = self._sock.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[{self._label}] client {addr} connected", flush=True)
            with self._lock:
                self._clients.append(conn)

    def broadcast(self, data: bytes) -> None:
        with self._lock:
            current = list(self._clients)
        dropped = []
        for c in current:
            try:
                c.sendall(data)
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                print(f"[{self._label}] client dropped: {e}", flush=True)
                dropped.append(c)
        if dropped:
            with self._lock:
                for c in dropped:
                    if c in self._clients:
                        self._clients.remove(c)
                    try:
                        c.close()
                    except OSError:
                        pass

    def n_clients(self) -> int:
        with self._lock:
            return len(self._clients)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path)
    p.add_argument("--tof-port",  type=int, default=9872)
    p.add_argument("--twin-port", type=int, default=9870)
    p.add_argument("--fast", type=float, default=1.0,
                   help="Playback speed multiplier (default 1.0 = realtime).")
    p.add_argument("--loop", action="store_true",
                   help="Repeat playback when the log ends.")
    args = p.parse_args()

    if args.fast <= 0:
        print("--fast must be > 0", file=sys.stderr)
        return 2

    print(f"loading {args.input} ...", flush=True)
    try:
        records, epoch_ms = _load(args.input)
    except Exception as e:
        print(f"failed to load: {e}", file=sys.stderr)
        return 1
    if not records:
        print("empty session", file=sys.stderr)
        return 1
    duration_us = records[-1][0] - records[0][0]
    n_tof  = sum(1 for _, sid, _ in records if sid == _STREAM_TOF)
    n_twin = sum(1 for _, sid, _ in records if sid == _STREAM_TWIN)
    print(f"  {len(records)} records  ({n_tof} ToF, {n_twin} twin)  "
          f"duration {duration_us / 1e6:.1f} s", flush=True)

    tof_srv  = _BroadcastServer(args.tof_port,  "tof")
    twin_srv = _BroadcastServer(args.twin_port, "twin")
    print(f"serving on TCP {args.tof_port} (ToF) and {args.twin_port} "
          f"(twin). Connect any consumer.", flush=True)

    try:
        loops = 0
        while True:
            wall_start = time.monotonic()
            first_ts_us = records[0][0]
            for ts_us, sid, payload in records:
                target = wall_start + (ts_us - first_ts_us) / 1e6 / args.fast
                slack = target - time.monotonic()
                if slack > 0:
                    time.sleep(slack)
                if sid == _STREAM_TOF:
                    # Re-prepend the size prefix the original wire had.
                    tof_srv.broadcast(struct.pack("<I", len(payload)) + payload)
                elif sid == _STREAM_TWIN:
                    twin_srv.broadcast(payload)
            loops += 1
            if not args.loop:
                print(f"playback complete (1 loop).", flush=True)
                return 0
            print(f"loop {loops} done, restarting ...", flush=True)
    except KeyboardInterrupt:
        print("\nstopped.", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
