"""Record a microduck session to a binary log.

Connects to the Pi's:
  * ToF streamer on port 9872          (raw 8x8 frames + status)
  * Runtime's digital-twin stream 9870 (joint state + IMU + odom)

Both streams are written verbatim to a single .mdlg file with per-packet
timestamps so they can be replayed later (see `replay_session.py`).

Wire format (single file, little-endian throughout):

    header (16 B):
        magic        : 4 bytes  "MDLG"
        version      : u32      (=1)
        epoch_unix_ms: u64      capture start, ms since epoch (informational)

    record (until EOF):
        ts_us        : u64      microseconds since recorder start
        stream_id    : u8       0 = ToF, 1 = digital twin
        size         : u32      payload size
        payload      : u8[size] verbatim bytes from the original TCP wire

The Pi must already be running both `tof_streamer.py` AND
`microduck_runtime --stream`. Use Ctrl-C to stop.

Usage:
    uv run tools/record_session.py --host <pi-ip> -o session.mdlg
    uv run tools/record_session.py --host <pi-ip> -o session.mdlg --duration 60
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue


_MAGIC = b"MDLG"
_VERSION = 1
_HEADER_FMT = "<4sIQ"
_RECORD_HEADER_FMT = "<QBI"

_STREAM_TOF  = 0
_STREAM_TWIN = 1

# Digital-twin packet is fixed-size (see microduck_runtime/fk/viewer.py).
_TWIN_PACKET_SIZE = 8 + 41 * 4   # 172 B


# ── TCP reading helpers ──────────────────────────────────────────────────────


def _read_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"closed after {len(buf)}/{n}")
        buf.extend(chunk)
    return bytes(buf)


def _read_tof_packet(sock: socket.socket) -> bytes:
    """ToF wire = `u32 size | payload`. Stored: just the payload."""
    (size,) = struct.unpack("<I", _read_exact(sock, 4))
    return _read_exact(sock, size)


def _read_twin_packet(sock: socket.socket) -> bytes:
    """Digital-twin packet is fixed 172 B."""
    return _read_exact(sock, _TWIN_PACKET_SIZE)


# ── Threads ──────────────────────────────────────────────────────────────────


def _reader(sock: socket.socket, stream_id: int,
            read_fn, queue: Queue, stop: threading.Event,
            t0: float, label: str) -> None:
    while not stop.is_set():
        try:
            payload = read_fn(sock)
        except (ConnectionError, OSError) as e:
            if not stop.is_set():
                print(f"[record] {label} disconnected: {e}", flush=True)
            return
        ts_us = int((time.monotonic() - t0) * 1e6)
        queue.put((ts_us, stream_id, payload))


def _writer(out_path: Path, queue: Queue, stop: threading.Event,
            stats: dict) -> None:
    epoch_unix_ms = int(time.time() * 1000)
    with out_path.open("wb") as f:
        f.write(struct.pack(_HEADER_FMT, _MAGIC, _VERSION, epoch_unix_ms))
        last_flush = time.monotonic()
        while not (stop.is_set() and queue.empty()):
            try:
                ts_us, sid, payload = queue.get(timeout=0.5)
            except Empty:
                continue
            f.write(struct.pack(_RECORD_HEADER_FMT, ts_us, sid, len(payload)))
            f.write(payload)
            stats[sid] = stats.get(sid, 0) + 1
            now = time.monotonic()
            if now - last_flush > 1.0:
                f.flush()
                last_flush = now


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", required=True)
    p.add_argument("--tof-port",  type=int, default=9872)
    p.add_argument("--twin-port", type=int, default=9870)
    p.add_argument("-o", "--output", type=Path, required=True,
                   help="Output .mdlg path.")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Auto-stop after N seconds (0 = until Ctrl-C).")
    args = p.parse_args()

    print(f"connecting to {args.host}:{args.tof_port} (ToF) "
          f"and :{args.twin_port} (digital twin) ...", flush=True)
    try:
        s_tof = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s_tof.connect((args.host, args.tof_port))
        s_tof.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s_twin = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s_twin.connect((args.host, args.twin_port))
        s_twin.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError as e:
        print(f"connect failed: {e}", file=sys.stderr)
        return 1
    print(f"connected. recording to {args.output}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    queue: Queue = Queue(maxsize=4096)
    stop = threading.Event()
    stats: dict = {}
    threads = [
        threading.Thread(target=_reader, daemon=True,
                         args=(s_tof,  _STREAM_TOF,  _read_tof_packet,
                               queue, stop, t0, "ToF")),
        threading.Thread(target=_reader, daemon=True,
                         args=(s_twin, _STREAM_TWIN, _read_twin_packet,
                               queue, stop, t0, "twin")),
        threading.Thread(target=_writer, daemon=True,
                         args=(args.output, queue, stop, stats)),
    ]
    for t in threads:
        t.start()

    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(2.0)
                el = time.monotonic() - t0
                tof_n = stats.get(_STREAM_TOF, 0)
                twin_n = stats.get(_STREAM_TWIN, 0)
                print(f"  t={el:5.1f}s  ToF: {tof_n:5d} ({tof_n / max(el, 0.1):4.1f}/s)  "
                      f"twin: {twin_n:5d} ({twin_n / max(el, 0.1):4.1f}/s)",
                      flush=True)
    except KeyboardInterrupt:
        print("\nstopping ...", flush=True)
    finally:
        stop.set()
        try:
            s_tof.shutdown(socket.SHUT_RDWR);  s_tof.close()
        except OSError:
            pass
        try:
            s_twin.shutdown(socket.SHUT_RDWR); s_twin.close()
        except OSError:
            pass
        threads[2].join(timeout=5.0)

    el = time.monotonic() - t0
    tof_n = stats.get(_STREAM_TOF, 0)
    twin_n = stats.get(_STREAM_TWIN, 0)
    size_kb = args.output.stat().st_size / 1024
    print(f"\nrecorded {el:.1f}s — {tof_n} ToF frames + {twin_n} twin packets "
          f"→ {args.output} ({size_kb:.1f} KB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
