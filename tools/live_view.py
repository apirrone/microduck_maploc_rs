"""Live PGM viewer — auto-reloads on file change, preserves pan/zoom.

Usage:
    uv run tools/live_view.py /tmp/live_map.pgm
    uv run tools/live_view.py /tmp/live_map.pgm --refresh 0.25

The window stays put: zooming or panning persists across refreshes.
The image content updates whenever the file's mtime changes.

If the image size changes (e.g. the global map grew because a new
submap pushed the bbox out), the view resets — that's unavoidable
since matplotlib's data extent has to match the array shape.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _load_pgm(path: Path) -> np.ndarray:
    """Load a binary (P5) PGM into an (H, W) uint8 array."""
    with path.open("rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError(f"not a P5 PGM (got magic = {magic!r})")
        # Skip comments + read dimensions / max-val.
        def _next_token() -> bytes:
            while True:
                line = f.readline()
                if not line:
                    raise ValueError("unexpected EOF in PGM header")
                line = line.strip()
                if line and not line.startswith(b"#"):
                    return line
        dims = _next_token().split()
        w, h = int(dims[0]), int(dims[1])
        max_val = int(_next_token())
        if max_val > 255:
            raise ValueError(f"expected 8-bit PGM, got max_val={max_val}")
        data = f.read(w * h)
        if len(data) != w * h:
            raise ValueError(
                f"truncated PGM body ({len(data)} != {w * h})")
    return np.frombuffer(data, dtype=np.uint8).reshape(h, w)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path)
    p.add_argument("--refresh", type=float, default=0.25,
                   help="Polling interval in seconds (default 0.25).")
    args = p.parse_args()

    if not args.path.exists():
        print(f"waiting for {args.path} to appear ...", flush=True)
        while not args.path.exists():
            time.sleep(0.5)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title(f"live: {args.path.name}")
    ax.set_facecolor("#222")  # nice contrast for the unknown-gray pixels

    state: dict = {"im": None, "shape": None, "mtime": 0.0}

    def reload_image() -> None:
        try:
            mtime = args.path.stat().st_mtime
        except OSError:
            return
        if mtime == state["mtime"]:
            return
        try:
            data = _load_pgm(args.path)
        except (ValueError, OSError) as e:
            # File is mid-write (e.g. recorder hasn't atomically renamed
            # yet). Try again on the next tick.
            print(f"  (reload skipped: {e})", flush=True)
            return
        state["mtime"] = mtime
        if state["im"] is None or state["shape"] != data.shape:
            ax.cla()
            ax.set_facecolor("#222")
            state["im"] = ax.imshow(
                data, cmap="gray", vmin=0, vmax=255, origin="upper",
                interpolation="nearest")
            state["shape"] = data.shape
            ax.set_title(f"{args.path.name}  ({data.shape[1]}×{data.shape[0]})")
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()
        else:
            state["im"].set_data(data)
        fig.canvas.draw_idle()

    reload_image()

    timer = fig.canvas.new_timer(interval=int(args.refresh * 1000))
    timer.add_callback(reload_image)
    timer.start()

    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
