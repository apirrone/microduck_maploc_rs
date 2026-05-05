"""Interactive ToF mount calibration for the VL53L5CX (8x8).

Outputs the four numbers the runtime needs to project ToF beams correctly
into the body frame:

    --tof-mount-pitch-deg     (rotation about body +y, "nose down")
    --tof-mount-yaw-deg       (rotation about body +z, "to the left")
    --tof-sensor-height-m     (used by the floor filter)
    --tof-floor-safety        (constant, 0.85 by default)

Plus an optional YAML/TOML-ish dump (`tof_calibration.yaml`) so the
runtime can load it instead of relying on long CLI flags.

Procedure overview:

    Step 1 — pitch + height
    ────────────────────────
    Place the duck a known distance D in front of a single flat wall,
    perpendicular and facing it. Run `pitch`. The bottom rows of the
    8x8 will hit the floor; their ranges and our zone geometry let us
    solve mount pitch and sensor height jointly.

    Step 2 — yaw
    ────────────
    Place the duck in a corner formed by two perpendicular walls, body
    +x roughly along the angle bisector. Measure the perpendicular
    distance to each wall (df = front, dl = left). Run `yaw` with the
    pitch from step 1. The yaw value is the one that best aligns the
    predicted corner geometry with the averaged scan.

The default subcommand is `all`, which steps you through both with
prompts. Individual subcommands (`pitch`, `yaw`) are available for
re-running just one step.

Usage:

    uv run tools/calibrate_tof.py all  --host <pi-ip>
    uv run tools/calibrate_tof.py pitch --host <pi-ip> --distance 0.60
    uv run tools/calibrate_tof.py yaw  --host <pi-ip> \\
        --df 1.20 --dl 0.80 --pitch-deg 3.6
"""

from __future__ import annotations

import argparse
import math
import socket
import struct
import sys
from dataclasses import dataclass

import numpy as np


_FOV_DEG  = 45.0
_ROWS     = 8
_COLS     = 8
_HALF_DEG = (_FOV_DEG - _FOV_DEG / _COLS) / 2.0
_ZONE_DEG = _FOV_DEG / _COLS
# Sensor-frame elevation per row (top → bottom, degrees).
_ROW_ELEV_DEG = np.array(
    [_HALF_DEG - r * _ZONE_DEG for r in range(_ROWS)]
)


# ── Wire reading ──────────────────────────────────────────────────────────────


def _read_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("stream closed")
        buf.extend(chunk)
    return bytes(buf)


def _read_frame(sock: socket.socket) -> tuple[np.ndarray, np.ndarray]:
    (size,) = struct.unpack("<I", _read_exact(sock, 4))
    payload = _read_exact(sock, size)
    rows = payload[8]
    cols = payload[9]
    n = rows * cols
    off = 12
    dist = np.frombuffer(payload, dtype=np.float32, count=n, offset=off
                         ).reshape(rows, cols).copy()
    off += n * 4
    status = np.frombuffer(payload, dtype=np.uint8, count=n, offset=off
                           ).reshape(rows, cols).copy()
    return dist, status


def _capture_average(host: str, port: int, n_frames: int) -> np.ndarray:
    print(f"  connecting to {host}:{port} …", flush=True)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    print(f"  averaging {n_frames} frames …", flush=True)
    accum  = np.zeros((_ROWS, _COLS), dtype=np.float64)
    counts = np.zeros((_ROWS, _COLS), dtype=np.int32)
    try:
        for _ in range(n_frames):
            dist, _status = _read_frame(s)
            valid = np.isfinite(dist)
            accum [valid] += dist[valid]
            counts[valid] += 1
    finally:
        s.close()
    if (counts == 0).any():
        print("  warning: some zones had zero valid samples — keep the duck still.",
              file=sys.stderr)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, accum / counts, np.nan)


# ── Step 1 — pitch + height (floor-fit) ───────────────────────────────────────


@dataclass
class PitchResult:
    pitch_deg: float
    height_m:  float
    floor_rows: list[int]
    wall_rows:  list[int]
    notes: list[str]


def _solve_pitch_height(mean_a: float, elev_a_deg: float,
                        mean_b: float, elev_b_deg: float) -> tuple[float, float]:
    """Floor-hit physics for two rows. Each beam looks down at body
    elevation θ_b = elev_sensor − P (P > 0 = nose-down). For a floor
    hit at sensor height h:
        range = h / sin(P − elev_sensor)
    The ratio of two rows eliminates h; bisect for P, then back out h."""
    target = mean_a / mean_b
    def f(P_deg: float) -> float:
        a = math.radians(P_deg - elev_a_deg)
        b = math.radians(P_deg - elev_b_deg)
        if math.sin(a) <= 0 or math.sin(b) <= 0:
            return math.nan
        return math.sin(b) / math.sin(a) - target
    lo, hi = -45.0, 45.0
    f_lo = f(lo)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if not math.isfinite(f_mid):
            return math.nan, math.nan
        if (f_lo < 0) == (f_mid < 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    P = 0.5 * (lo + hi)
    h = mean_a * math.sin(math.radians(P - elev_a_deg))
    return P, h


def calibrate_pitch_height(
    mean: np.ndarray, distance: float, floor_fraction: float = 0.7,
) -> PitchResult:
    notes: list[str] = []
    floor_thresh = floor_fraction * distance
    is_floor = np.isfinite(mean) & (mean < floor_thresh)
    is_wall  = np.isfinite(mean) & (mean >= floor_thresh) & (mean <= 1.5 * distance)
    floor_rows = [r for r in range(_ROWS) if is_floor[r].sum() >= _COLS // 2]
    wall_rows  = [r for r in range(_ROWS) if is_wall [r].sum() >= _COLS // 2]
    if len(floor_rows) < 2:
        notes.append(
            f"need ≥ 2 floor rows; got {floor_rows}. Stand the duck closer "
            f"to the wall, or check that the bottom rows are seeing floor."
        )
        return PitchResult(math.nan, math.nan, floor_rows, wall_rows, notes)
    ra, rb = floor_rows[0], floor_rows[-1]
    ma = float(np.nanmean(np.where(is_floor[ra], mean[ra], np.nan)))
    mb = float(np.nanmean(np.where(is_floor[rb], mean[rb], np.nan)))
    P, h = _solve_pitch_height(
        ma, float(_ROW_ELEV_DEG[ra]),
        mb, float(_ROW_ELEV_DEG[rb]),
    )
    return PitchResult(P, h, floor_rows, wall_rows, notes)


# ── Step 2 — yaw (corner-geometry fit) ────────────────────────────────────────


@dataclass
class YawResult:
    yaw_deg:    float
    rms_mm:     float
    n_used:     int
    notes:      list[str]


def _zone_directions_body(pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """(64, 3) unit body-frame direction vectors per zone, matching
    `microduck_runtime/src/tof.rs::precompute_zone_lookups`."""
    half = math.radians(_HALF_DEG)
    step = (2.0 * half) / (_COLS - 1)
    cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
    cos_y, sin_y = math.cos(yaw_rad),   math.sin(yaw_rad)
    dirs = np.zeros((_ROWS * _COLS, 3), dtype=np.float64)
    k = 0
    for r in range(_ROWS):
        el_s = half - r * step
        for c in range(_COLS):
            az_s = half - c * step
            cx = math.cos(el_s) * math.cos(az_s)
            cy = math.cos(el_s) * math.sin(az_s)
            cz = math.sin(el_s)
            dx_p =  cos_p * cx + sin_p * cz
            dz_p = -sin_p * cx + cos_p * cz
            dy_p = cy
            dx_b = cos_y * dx_p - sin_y * dy_p
            dy_b = sin_y * dx_p + cos_y * dy_p
            dirs[k] = (dx_b, dy_b, dz_p)
            k += 1
    return dirs


def _predicted_horiz_corner(
    yaw_rad: float, pitch_rad: float, df: float, dl: float
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted horizontal-plane range per zone, treating the two walls
    as infinite planes (real walls extend past the corner). For each beam,
    valid_front iff ux > 0 (beam moves toward x=df), valid_left iff uy > 0,
    and the predicted range is the closer of the two valid hits. NaN when
    a beam moves away from both walls (we don't model the back-side walls
    of the room)."""
    dirs = _zone_directions_body(pitch_rad, yaw_rad)
    cos_e = np.linalg.norm(dirs[:, :2], axis=1)
    az    = np.arctan2(dirs[:, 1], dirs[:, 0])
    ux = np.cos(az);  uy = np.sin(az)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_front = np.where(ux > 1e-6, df / ux, np.inf)
        t_left  = np.where(uy > 1e-6, dl / uy, np.inf)
    t = np.minimum(t_front, t_left)
    horiz = np.where(np.isfinite(t), t, np.nan)
    return horiz, cos_e


def _yaw_residual(yaw_rad: float, pitch_rad: float,
                  df: float, dl: float, measured_horiz: np.ndarray,
                  *, beam_mask: np.ndarray | None = None,
                  min_valid: int = 8,
                  ) -> tuple[float, int]:
    """Mean Huber residual per valid beam.

    `beam_mask` (optional, shape (64,)) zeroes out beams that the caller
    knows are floor hits (from the pitch+height step) — they never align
    with any wall geometry and would otherwise just contribute a Huber-
    saturated penalty that biases the optimizer.

    Returns *mean* cost (not sum) so yaws with fewer valid beams aren't
    artificially favoured by the optimizer. If fewer than `min_valid`
    beams contribute, return inf (no signal)."""
    pred, _ = _predicted_horiz_corner(yaw_rad, pitch_rad, df, dl)
    valid = np.isfinite(pred) & np.isfinite(measured_horiz)
    if beam_mask is not None:
        valid = valid & beam_mask
    n_valid = int(valid.sum())
    if n_valid < min_valid:
        return float("inf"), n_valid
    diffs = pred[valid] - measured_horiz[valid]
    huber_k = 0.10
    abs_d = np.abs(diffs)
    mask_q = abs_d <= huber_k
    cost = np.where(mask_q, 0.5 * diffs * diffs,
                            huber_k * (abs_d - 0.5 * huber_k))
    return float(cost.sum() / n_valid), n_valid


def calibrate_yaw(
    mean: np.ndarray, df: float, dl: float, pitch_deg: float,
    *, floor_rows: list[int] | None = None,
    yaw_search_deg: float = 20.0, yaw_step_deg: float = 0.25,
) -> YawResult:
    pitch_rad = math.radians(pitch_deg)
    flat_slant = mean.reshape(-1)
    _, cos_e = _predicted_horiz_corner(0.0, pitch_rad, df, dl)
    measured_horiz = flat_slant * cos_e
    # Build a (64,) bool mask: keep zone iff its row is NOT a floor row.
    # The pitch+height step classifies floor rows; if not provided we
    # default to "skip rows whose body elevation is more than ~10° below
    # horizontal", which is a safe heuristic for sensors mounted ~10 cm
    # above the floor with walls within a couple of metres.
    if floor_rows is None:
        floor_rows = [r for r in range(_ROWS)
                      if (_ROW_ELEV_DEG[r] - pitch_deg) < -10.0]
    keep_row = np.array([r not in floor_rows for r in range(_ROWS)], dtype=bool)
    beam_mask = np.repeat(keep_row, _COLS)  # (64,)

    yaws  = np.arange(-yaw_search_deg, yaw_search_deg + 1e-9, yaw_step_deg)
    costs = np.full_like(yaws, np.inf, dtype=np.float64)
    n_arr = np.zeros_like(yaws, dtype=np.int32)
    for i, y_deg in enumerate(yaws):
        c, n = _yaw_residual(math.radians(y_deg), pitch_rad,
                             df, dl, measured_horiz,
                             beam_mask=beam_mask)
        costs[i] = c; n_arr[i] = n
    if not np.isfinite(costs).any():
        return YawResult(math.nan, math.nan, 0,
                         ["no valid beams matched any wall — check that the "
                          "duck faces into the corner with at least 2 columns "
                          "reaching each wall."])
    i_min = int(np.argmin(costs))
    best_yaw = float(yaws[i_min])
    if 0 < i_min < len(yaws) - 1:
        c_minus, c_zero, c_plus = costs[i_min-1], costs[i_min], costs[i_min+1]
        denom = (c_minus - 2.0 * c_zero + c_plus)
        if abs(denom) > 1e-12:
            shift = 0.5 * (c_minus - c_plus) / denom
            best_yaw = float(yaws[i_min] + shift * yaw_step_deg)
    final_cost, final_n = _yaw_residual(math.radians(best_yaw), pitch_rad,
                                        df, dl, measured_horiz,
                                        beam_mask=beam_mask)
    # `final_cost` is now mean-Huber per beam. RMS-equivalent (in metres)
    # is sqrt(2 * cost) for the quadratic regime — close enough for a
    # readout. We don't expose the cost itself.
    rms_mm = math.sqrt(2.0 * final_cost) * 1000.0
    notes: list[str] = []
    if i_min == 0 or i_min == len(yaws) - 1:
        notes.append(
            f"optimum sits at the edge of the yaw search range "
            f"({yaws[i_min]:+.2f}°); the geometry probably doesn't "
            f"constrain yaw enough — re-check df/dl and that beams reach "
            f"both walls."
        )
    return YawResult(best_yaw, rms_mm, final_n, notes)


# ── Pretty-printing ───────────────────────────────────────────────────────────


def _print_grid(mean: np.ndarray) -> None:
    print("  mean range per zone (m, NaN = no return):")
    for r in range(_ROWS):
        row = "  ".join(
            f"{v:5.2f}" if np.isfinite(v) else "  --  " for v in mean[r])
        print(f"    row {r}: {row}")


def _format_calibration(pitch_deg: float, yaw_deg: float, height_m: float
                        ) -> str:
    return (
        f"# microduck ToF mount calibration\n"
        f"tof:\n"
        f"  mount_pitch_deg:    {pitch_deg:+.3f}\n"
        f"  mount_yaw_deg:      {yaw_deg:+.3f}\n"
        f"  sensor_height_m:    {height_m:.4f}\n"
        f"  floor_safety:       0.85\n"
    )


# ── Pretty diagrams + interactive prompts ────────────────────────────────────


_PITCH_DIAGRAM = r"""
    Top-down view  (body +x is "up" in this drawing).
    The duck faces a flat wall, perpendicular to it:

                        █████████████████████   <- flat wall
                              ▲
                              │
                              │   D = perpendicular distance
                              │   (suggested ~0.6 m)
                              │
                              │
                            ┌─────┐
                            │  ↑  │   duck
                            │     │   (body +x ↑ — faces the wall)
                            └─────┘

    Bottom rows of the 8x8 should hit the FLOOR — that's
    the signal we use to fit pitch + sensor height.
"""

_YAW_DIAGRAM = r"""
    Top-down view  (body +x is "up" in this drawing, +y is "left").
    Two perpendicular walls meet at a 90° corner — the corner is at the
    top-left. The duck sits close to the LEFT wall, far from the FRONT
    wall, body +x parallel to the LEFT wall (so it faces the FRONT wall).

         corner
           ↓
           ┌─────────────────────────────────────
           │                                       <- FRONT WALL
           │                                          (perpendicular
           │                                           to body +x)
           │
           │                ▲
           │                │
           │                │   df  (LARGE, suggested ~1.0 m)
           │                │
           │                ▼
           │         ┌─────────────┐
           │ ←─dl──→ │      ↑      │   duck
           │         │   body +x   │   (body +x faces the FRONT wall)
           │         └─────────────┘
           │
           │
           │
           ▼
        LEFT WALL
        (perpendicular to body +y)

    Goal:  dl / df  ≲  0.25     (e.g. df = 1.0 m, dl = 0.15 m)
    Why: with our ±22.5° FOV, both walls fit in the scan only when
    the corner is asymmetric. A square 1×1 corner does NOT work —
    the front wall would block all beams that would otherwise hit
    the left wall.
"""


def _prompt_float(prompt: str, default: float | None,
                  use_default: bool) -> float:
    """Prompt for a float, accepting <enter> for default."""
    if use_default and default is not None:
        return default
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  not a number, try again.")


# ── Subcommand drivers ────────────────────────────────────────────────────────


def cmd_pitch(args) -> tuple[float, float, list[int]] | None:
    print("\n=== Step 1 — pitch + height ===")
    print(_PITCH_DIAGRAM)
    distance = _prompt_float(
        "Distance D from duck to wall (m)",
        getattr(args, "distance", None),
        not args.interactive,
    )
    if args.interactive:
        input("Press <enter> when the duck is in position ... ")
    mean = _capture_average(args.host, args.port, args.n_frames)
    print()
    _print_grid(mean)
    res = calibrate_pitch_height(mean, distance,
                                 floor_fraction=args.floor_fraction)
    print()
    if math.isnan(res.pitch_deg):
        for note in res.notes:
            print(f"  ! {note}")
        return None
    print(f"  classified rows: floor = {res.floor_rows}, wall = {res.wall_rows}")
    print(f"  fit:  pitch = {res.pitch_deg:+.2f}°    "
          f"sensor height ≈ {res.height_m * 100:.1f} cm")
    return res.pitch_deg, res.height_m, res.floor_rows


def cmd_yaw(args, pitch_deg: float | None = None,
            floor_rows: list[int] | None = None,
            ) -> float | None:
    pitch_deg = pitch_deg if pitch_deg is not None else args.pitch_deg
    if pitch_deg is None:
        print("ERROR: yaw step needs a pitch — pass --pitch-deg or run `all`.",
              file=sys.stderr)
        return None
    print("\n=== Step 2 — yaw ===")
    print(_YAW_DIAGRAM)
    df = _prompt_float(
        "df = distance to FRONT wall (m)",
        getattr(args, "df", None),
        not args.interactive,
    )
    dl = _prompt_float(
        "dl = distance to LEFT wall (m, small)",
        getattr(args, "dl", None),
        not args.interactive,
    )

    # FOV reachability check. With body +x along world +x and the duck
    # at the origin of the corner: a beam at body azimuth α hits the
    # left wall (before the front wall) when tan(α) > dl/df. With our
    # ±22.5° FOV and 5.625°-wide zones, we want enough columns on each
    # wall for the fit to be well-conditioned — at least 2.
    boundary_deg = math.degrees(math.atan2(dl, df))
    half_fov_deg = _HALF_DEG
    n_left  = sum(1 for c in range(_COLS)
                  if (_HALF_DEG - c * _ZONE_DEG) > boundary_deg)
    n_front = _COLS - n_left
    print()
    print(f"  → boundary azimuth (front-vs-left split): {boundary_deg:.1f}°  "
          f"(FOV half: {half_fov_deg:.1f}°)")
    print(f"  → expected zone split (per row): "
          f"{n_left} columns hit the left wall, {n_front} hit the front")
    if n_left < 2 or n_front < 2:
        print()
        print("  ⚠  WARNING: the corner geometry doesn't fit our ±22.5° FOV well.")
        print(f"     With df={df:.2f}, dl={dl:.2f}, only {n_left} column(s) "
              f"reach the left wall and {n_front} reach the front wall.")
        print(f"     Yaw is poorly observable in this setup. Aim for "
              f"`dl/df ≲ 0.25` (e.g. df=1.0 m, dl=0.15 m).")
        print(f"     Continuing anyway — RMS residual at the end will tell "
              f"you if the fit was healthy.")
    print()
    if args.interactive:
        input("Press <enter> when the duck is in position ... ")
    mean = _capture_average(args.host, args.port, args.n_frames)
    print()
    _print_grid(mean)
    res = calibrate_yaw(mean, df, dl, pitch_deg,
                        floor_rows=floor_rows,
                        yaw_search_deg=args.yaw_search_deg,
                        yaw_step_deg=args.yaw_step_deg)
    print()
    if math.isnan(res.yaw_deg):
        for note in res.notes:
            print(f"  ! {note}")
        return None
    if res.notes:
        for note in res.notes:
            print(f"  ! {note}")
    print(f"  beams used: {res.n_used} / {_ROWS * _COLS}")
    print(f"  fit:  yaw = {res.yaw_deg:+.2f}°    "
          f"per-beam RMS residual {res.rms_mm:.0f} mm")
    return res.yaw_deg


def cmd_all(args) -> int:
    print("=== ToF mount calibration — full procedure ===")
    pitch_result = cmd_pitch(args)
    if pitch_result is None:
        return 1
    pitch_deg, height_m, floor_rows = pitch_result
    yaw_deg = cmd_yaw(args, pitch_deg=pitch_deg, floor_rows=floor_rows)
    if yaw_deg is None:
        return 1

    print("\n=== Result ===")
    print(_format_calibration(pitch_deg, yaw_deg, height_m))
    print("Suggested runtime flags:")
    print(f"  --tof-mount-pitch-deg {pitch_deg:+.2f} \\")
    print(f"  --tof-mount-yaw-deg   {yaw_deg:+.2f} \\")
    print(f"  --tof-sensor-height-m {height_m:.3f} \\")
    print(f"  --tof-floor-safety    0.85")
    print()
    if args.out:
        with open(args.out, "w") as f:
            f.write(_format_calibration(pitch_deg, yaw_deg, height_m))
        print(f"  written to {args.out}")
    return 0


# ── Entrypoint ────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, default=9872)
    p.add_argument("--n-frames", type=int, default=30)
    p.add_argument("--no-interactive", dest="interactive",
                   action="store_false", default=True,
                   help="Skip the press-enter prompts (default: prompt).")

    sub = p.add_subparsers(dest="cmd")
    sub.required = False

    # `all` — interactive full procedure. Distances default to interactive
    # prompts; provide on the CLI to skip prompts.
    p_all = sub.add_parser("all", help="Full procedure: pitch then yaw.")
    p_all.add_argument("--distance", type=float, default=None,
                       help="Step 1: perpendicular distance to single wall (m). "
                            "If omitted, prompted interactively.")
    p_all.add_argument("--df", type=float, default=None,
                       help="Step 2: perpendicular distance to FRONT wall (m).")
    p_all.add_argument("--dl", type=float, default=None,
                       help="Step 2: perpendicular distance to LEFT wall (m).")
    p_all.add_argument("--floor-fraction", type=float, default=0.7)
    p_all.add_argument("--yaw-search-deg", type=float, default=20.0)
    p_all.add_argument("--yaw-step-deg",   type=float, default=0.25)
    p_all.add_argument("--out", default=None,
                       help="Write a YAML-style calibration to this file.")

    # `pitch` — only step 1.
    p_pitch = sub.add_parser("pitch", help="Calibrate pitch + height only.")
    p_pitch.add_argument("--distance", type=float, default=None)
    p_pitch.add_argument("--floor-fraction", type=float, default=0.7)

    # `yaw` — only step 2 (needs pitch).
    p_yaw = sub.add_parser("yaw", help="Calibrate yaw only (needs --pitch-deg).")
    p_yaw.add_argument("--df", type=float, default=None)
    p_yaw.add_argument("--dl", type=float, default=None)
    p_yaw.add_argument("--pitch-deg", type=float, required=True)
    p_yaw.add_argument("--yaw-search-deg", type=float, default=20.0)
    p_yaw.add_argument("--yaw-step-deg",   type=float, default=0.25)

    args = p.parse_args()
    if args.cmd is None:
        p.print_help()
        return 2
    if args.cmd == "all":
        return cmd_all(args)
    if args.cmd == "pitch":
        out = cmd_pitch(args)
        return 0 if out is not None else 1
    if args.cmd == "yaw":
        out = cmd_yaw(args)
        return 0 if out is not None else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
