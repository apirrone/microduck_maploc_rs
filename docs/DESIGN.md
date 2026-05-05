# maploc_rs_v2 — design

This document describes the algorithm. It's filled in as each phase
lands; if you only want the work plan and acceptance criteria, see
`docs/PLAN.md`.

## Overview

Submap-based pose-graph 2D SLAM, designed to fit on Pi Zero 2 W
(4 cores @ 1 GHz, 512 MB) alongside a 50 Hz locomotion loop.

```
                       ToF frame                       odometry
                           │                               │
                           ▼                               ▼
                   ┌───────────────┐              ┌────────────────┐
                   │ scan_matcher  │ ◄────────── │  predict pose  │
                   │ (Hector ICP)  │              │  from odom Δ   │
                   └───────┬───────┘              └────────────────┘
                           ▼
                ┌──────────────────────┐
                │   current submap     │ ─── ink scan at corrected pose
                └──────────┬───────────┘
                           │ (submap close)
                           ▼
                ┌──────────────────────┐
        ┌─────► │   submap_manager     │
        │       └──────────┬───────────┘
        │                  ▼
        │       ┌──────────────────────┐
        │       │   loop_closer        │ ─── add scan-match edges
        │       └──────────┬───────────┘
        │                  ▼
        │       ┌──────────────────────┐
        │       │   pose_graph         │
        │       │   + optimizer        │
        │       └──────────┬───────────┘
        │                  ▼
        │       ┌──────────────────────┐
        └────── │   global_render      │ ─── composite global map
                └──────────────────────┘
```

## Data flow

1. **Per-frame (15 Hz)**:
   * Compose the latest odometry delta onto the tracked pose.
     **Tracking is pure odometry** — no per-frame scan matching.
     (See "Why we don't scan-match per-frame" below.)
   * Integrate the scan into the current submap at the odom-tracked pose.

2. **Per-tick (whatever the runtime ticks at)**:
   * `submap_manager` checks whether to close the current submap and
     start a fresh one (time threshold OR in-submap travel threshold).

3. **On submap close**:
   * Freeze the closing submap; record its anchor pose.
   * `loop_closer` searches older submaps within a spatial radius of
     the new anchor; for each candidate, scan-match the new submap's
     first ~10 scans against the candidate's grid (this is where
     `scan_matcher` is used). Below threshold → add a loop edge to the
     pose graph.
   * `optimizer` re-optimizes the pose graph (sparse Gauss-Newton).
   * `global_render` rebuilds the composite map from updated submap
     poses.

## Why we don't scan-match per-frame

Initial v2 design called for Hector ICP per ToF frame. Empirically
on the microduck, this drifts *worse* than the duck's odometry alone
(40 s closed-loop walk: pure odom ended within 1° / 11 cm, Hector
ended at -16° / 37 cm). Single-frame Hector finds spurious local
minima and accumulates bias the odometry doesn't have. The duck's
odometry happens to be good enough on short horizons that scan-match
nudges aren't worth their drift cost.

So we trust odometry between submap closes, and use `scan_matcher`
**only** at submap-to-submap granularity (loop closure). That's the
"large-scale correction" Hector is good at; the per-frame noise it
adds doesn't help us because there's nothing to correct.

## Why submap-based

* **Hector alone has no loop closure** → drift compounds monotonically.
* **Full pose-graph (every scan = node)** → graph blows up, optimization
  too heavy on Pi Zero.
* **Submapping** → tracking inside a submap is bounded-drift (Hector ICP
  against a slowly-evolving local grid), and the global optimization
  runs over **submap poses** (tens of nodes), which is trivial.
* **Loop closure becomes scan-matching at submap granularity**: same
  primitive as per-frame tracking, just used between submaps.

## Memory + CPU envelope

* 50 submaps × 80×80 cells × 4 bytes ≈ 1.3 MB.
* Per-frame: 1 Hector match (~5 ms on Pi Zero 2 W with 64 valid beams).
* Per-close (every 20 s): handful of loop-closure scan matches (~50 ms).
* Per-optimize: sparse GN on ≤ 50 nodes (~10 ms).
* Sustained CPU: ~10 % of one core in steady state.

## Modules

| Module           | Phase | Status | Purpose |
|------------------|:-----:|:------:|--------|
| `grid`           |  —    |   ✓    | Log-odds occupancy grid + distance field. |
| `scan_matcher`   |  —    |   ✓    | Hector ICP against a target grid. |
| `submap`         |  3    |        | Local grid + anchor pose. |
| `submap_manager` |  4    |        | Open/close submaps based on time + travel. |
| `pose_graph`     |  5    |        | SE(2) nodes + edges + sqrt-info. |
| `optimizer`      |  5    |        | Sparse Gauss-Newton on SE(2). |
| `loop_closer`    |  5    |        | Submap-to-submap scan-match for loop edges. |
| `global_render`  |  4    |        | Composite all submaps into a single grid. |
| `replay`         |  1    |        | Offline `.mdlg` session reader. |
| `follower`       |  —    |   ✓    | Waypoint follower (untouched from v1). |
| `planner`        |  —    |   ✓    | A\* path planner (untouched from v1). |
| `wire`/`stream`  |  —    |   ✓    | Telemetry wire format. |

## Coordinate conventions

* World frame: right-handed, +z up. We track only the (x, y, yaw) slice.
* Body frame: +x forward, +y left, +z up.
* ToF mount: rotation about body +y for pitch (positive = nose-down),
  about body +z for yaw (positive = sensor optical axis rotated toward
  body +y / "left"). Calibrated via `tools/calibrate_tof.py`.

## What we explicitly are not building (yet)

* MCL relocalize against a saved map. Comes back at Phase 7, scoped to
  exactly that use case. Not used during mapping.
* Multi-floor / multi-room. Single global frame.
* 3D mapping. The 8×8 ToF is treated as a 2D scan via cos(elev)
  projection.
* Visual SLAM. Camera is unrelated to this stack.
