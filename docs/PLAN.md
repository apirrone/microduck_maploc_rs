# maploc_rs_v2 — clean restart plan

This is the working plan for the v2 rewrite. Each phase has explicit
acceptance criteria; we don't move on until they're met against the
recorded log of the test room. Tick boxes below as work lands.

The v1 history (`mcl.rs`, `accumulator.rs`, the patches stacked on top of
each other) is left on `main` as historical context. v2 starts fresh.

## Goals

1. **Robust full SLAM** running on Pi Zero 2 W alongside the 50 Hz
   locomotion loop, using only the 8×8 ToF (15 Hz) + the runtime's
   odometry.
2. **Reproducible offline iteration** against recorded logs from a known
   room — algorithm work happens on the laptop, not on the duck.
3. **Single-repo discipline**: all maploc code (Rust + Python tooling)
   lives in `microduck_maploc-rs`. The Python sim repo stays untouched
   as historical reference but is not actively used.

## Non-goals (explicit)

- Matching Cartographer's accuracy. We're targeting "good enough to
  navigate".
- Multi-floor, multi-room generalization beyond what fits in a 4×4 m
  grid.
- Visual / semantic SLAM.

---

## Algorithm choice — submap-based pose-graph SLAM

Committed to this rather than wiggle. Reasoning:

- **Hector alone has no loop closure.** Drift compounds monotonically.
  v1 demonstrated this.
- **Full pose-graph (every scan = node)** would work but the graph
  blows up and per-frame optimization is heavy on Pi Zero.
- **Submapping** — each submap has its own grid; scan-match against
  the *current* submap so within-submap drift is bounded; close every
  ~20 s; the pose graph is then over **submap poses** (tens of nodes,
  not thousands), optimizes in milliseconds.
- **Loop closure** = scan-match a new submap against spatially-close
  older submaps. When a match passes, add an edge and re-optimize.
  Same primitive (Hector ICP) at submap-to-submap granularity.
- **Memory**: 50 submaps × 80×80 cells × 4 bytes ≈ 1.3 MB. Trivial.
- **CPU**: per-frame ≈ 1 Hector match (~5 ms). Per-close (every 20 s)
  ≈ handful of matches (~50 ms). Per-optimization ≈ sparse GN on tens
  of nodes (~10 ms). Comfortable on Pi Zero 2 W.

This is essentially Cartographer scoped down. Components:

```
Submap          : OccupancyGrid + world-anchor-pose
SubmapManager   : list of frozen submaps + current submap, switching
ScanMatcher     : Hector ICP against a target grid
PoseGraph       : nodes (submap poses) + edges (relative SE(2) + sqrt-info)
GraphOptimizer  : sparse Gauss-Newton on SE(2)
LoopCloser      : at submap close, scan-match against spatial neighbours
GlobalRender    : composite all submaps into a single grid via their poses
```

Estimated total ~1500 lines of Rust.

---

## Working agreement

- No phase advances until its acceptance criteria are met against the
  recorded log of the test room.
- Every phase produces a small Python visualizer in `tools/` (rerun-
  based) so we can see what's happening, not just numbers.
- Pi-side validation only happens at Phase 6. Until then, everything
  is laptop + recorded logs.
- Branch lives in `maploc_rs_v2` until Phase 6 ships. Then PR-merge
  to main.

---

## Phase 0 — Data quality (no SLAM yet)

**Goal:** be 100 % sure the data we feed the SLAM pipeline is correct.
No algorithm work until this passes.

**Tasks**

- [ ] **Yaw calibration helper.** Stand duck a known distance from two
  perpendicular walls (corner). Measure ranges. Solve analytically for
  mount yaw by minimizing `Σ (predicted_range(yaw) − measured_range)²`
  over the rays. The missing piece from v1's calibrator.
- [ ] **Live body pitch/roll in the ToF projection.** Pipe BMI088
  quaternion into the TCP ToF source (or carry it in `TofFrame`).
  Per-frame, rotate the precomputed zone-direction vectors by the live
  body attitude; recompute the floor filter against live geometry.
  Eliminates the static-mount artifact during gait.
- [ ] **Odometry sanity log.** Walk the duck along marked 1 m straight
  lines and 90° turns. Plot reported (x, y, yaw) vs ground-truth
  gridmarks. Document the actual drift rate (cm/m, °/m) so we know
  what motion-model sigmas to use.
- [ ] **Single canonical recording format defined** (see Phase 1).

**Acceptance criteria**

- ToF point cloud (in `viewer.py` over a 30 s static-then-walking
  session) lines up cleanly with real-room geometry: walls in the
  right places, no smear during rotations, no spurious near-field hits
  during a stride.
- Odom drift documented in numbers, not vibes.

**Deliverables**

- `tools/calibrate_yaw.py`
- Live-attitude ToF projection in the runtime
- `docs/CALIBRATION.md` with procedure + numbers

---

## Phase 1 — Recording infrastructure

**Goal:** capture a session and replay it bit-for-bit so iteration
moves to the laptop.

**Format** (one file per session):

```
header (16 B): magic "MDLG" | u32 version | u64 epoch_unix_ms
records (until EOF):
  u64 ts_us_since_epoch
  u8  stream_id  (0=tof, 1=odom_imu)
  u32 size
  u8[size] payload   (verbatim TCP payload of the original wire)
```

`stream_id=0` carries the existing tof_streamer wire format.
`stream_id=1` carries the runtime's digital-twin packet (172 B,
documented at the top of `microduck_runtime/fk/viewer.py`). Both are
replayed verbatim — no algorithmic re-interpretation at record time.

**Tasks**

- [ ] `tools/record_session.py <pi-host> -o session.mdlg` — opens both
  TCP streams, tags packets, writes the file. Runs on laptop.
- [ ] `tools/replay_session.py session.mdlg [--realtime|--fast]` —
  binds the same ports locally, plays records back. Runtime/viewer
  connect to it as if it were the Pi.
- [ ] Rust helper `crate::replay::SessionReplayer` for offline-first
  SLAM development that bypasses TCP entirely (reads directly from
  file). This is what the algorithm dev loop will use.

**Acceptance criteria**

- Record a 60 s walk, replay it, run the *current* runtime against the
  replay, get the same map as the live run.

---

## Phase 2 — Scaffolding

**Goal:** clean skeleton so we don't fight the existing v1 crate as we
add components.

**Tasks**

- [ ] Top-level layout:
  ```
  src/
    lib.rs
    grid.rs              // keep (occupancy + distance field)
    scan_matcher.rs      // keep, fix the saturation Jacobian bug
    submap.rs            // NEW
    submap_manager.rs    // NEW
    pose_graph.rs        // NEW
    optimizer.rs         // NEW
    loop_closer.rs       // NEW
    global_render.rs     // NEW
    replay.rs            // NEW (offline session reader)
    wire.rs, stream.rs   // keep (telemetry)
    follower.rs, planner.rs  // keep, untouched
  tools/
    record_session.py
    replay_session.py
    calibrate_yaw.py
    visualize_session.py  // rerun-based offline viewer
  docs/
    DESIGN.md            // the algorithm description
    CALIBRATION.md
    PLAN.md              // this file
  ```
- [ ] **Drop** `mcl.rs` and `accumulator.rs` from v2 entirely. They
  were band-aids. MCL comes back at Phase 7 only as the relocalize-
  from-uniform primitive against a saved map.

**Acceptance criteria**

- Branch builds; old tests removed/ported; new modules empty stubs.

---

## Phase 3 — Single-submap tracking

**Goal:** duck builds a single submap during a recorded walk that is
recognisable as the test room.

**Tasks**

- [x] `Submap` struct: `OccupancyGrid` + anchor pose `(x, y, yaw)` (the
  world pose at submap creation).
- [x] Fix `scan_matcher.rs` saturation bug: skip beams where
  `d > sigma_m` entirely (no residual contribution, no Jacobian
  contribution).
- [x] `replay::SessionReplayer` — offline reader for `.mdlg` logs.
- [x] `mount` module — `TofMount` + `precompute_zone_lookups` +
  `project_frame` (raw 8x8 → body azimuth + horizontal-plane range).
- [x] `examples/track_session.rs` — offline tracker that reads a log,
  advances tracked pose by odom delta, integrates scans, dumps a PGM.

**Acceptance criteria**

- [x] Replay the recorded walk; pure-odom integration produces a
  recognisable map of the test room.

**Decision recorded here (2026-05-05):**

Hector scan-matching as the **primary** tracker turned out to drift
*more* than the duck's odometry on the test recording. Walking a 40 s
loop, pure-odom finished within 1° / 11 cm of (0, 0); Hector finished
at -16° / 37 cm off — and the resulting map was a smeared blob while
pure odom's looked clean. Single-frame Hector kept finding spurious
local minima, accumulating bias the duck's odometry doesn't have.

So we **drop Hector as the per-frame tracker for v2**. We keep the
scan_matcher implementation — it's the right primitive for
**submap-to-submap matching** in loop closure (Phase 5). For tracking,
we trust odometry; submap drift is fixed at submap-close time by the
pose graph + loop closer, not by per-frame nudges.

---

## Phase 4 — Submap closure + multi-submap rendering

**Goal:** open and close submaps as the duck moves; render the union.

**Tasks**

- [x] `SubmapManager`: triggers a new submap when current one has been
  active ≥ `max_age_s` OR robot walked ≥ `max_travel_m` from its anchor.
- [x] Cross-submap pose continuity: new submap's anchor pose is exactly
  the tracked pose at switch time (no jump).
- [x] `GlobalRender`: composite all submap grids into a global view via
  each submap's anchor pose. Naive cell-walk; clamp summed log-odds.
- [x] `examples/track_submaps.rs`: offline driver, dumps a global PGM +
  optional path and anchor overlays.

**Acceptance criteria**

- [x] Replay the 40 s loop; multiple submaps are created (3 at default
  thresholds); global render shows the same room outline as the
  pure-odom Phase 3 baseline. (Visual quality will only step up once
  Phase 5 lands loop closure — submaps with pure-odom tracking can't
  improve on the underlying odom.)

---

## Phase 5 — Pose graph + loop closure

**Goal:** detect loops, add constraints, optimize, watch the global map
snap to consistency.

**Tasks**

- [x] `PoseGraph`: SE(2) nodes + edges with 3×3 information matrices.
  `compose`/`between`/`inverse` helpers.
- [x] `LoopCloser`: at each submap close, find candidate older submaps
  within radius `R`. Scan-match the new submap's representative scan
  against each candidate's grid; emit a measurement when residual is
  below threshold.
- [x] `GraphOptimizer`: dense Gauss-Newton on SE(2). N is small enough
  for our scales that pulling in a sparse linear solver isn't worth
  the dependency.
- [x] `examples/track_loop_closure.rs`: end-to-end driver. Adds odom
  edges between consecutive submaps and loop edges from `detect_loops`,
  re-optimizes after each accepted loop, writes corrected anchors back
  to the SubmapManager.

**Acceptance criteria**

- [x] Replay the 40 s loop walk: 3 loop closures detected with low
  residual (0.006 – 0.085 m, 42–46 beams each), graph optimization
  converged after each, anchors shift slightly to be more consistent.
  Map looks more compact than the pure-odom Phase 4 baseline.

  Caveat: the magnitude of the snap is small on this dataset because
  your odom is accurate over 40 s. On a longer / drifter session the
  visible improvement would be larger; the pipeline is in place either
  way.

---

## Phase 6 — Pi integration + perf tuning

**Goal:** confirm it runs on Pi Zero 2 W within budget, alongside 50 Hz
control.

**Tasks**

- [ ] Cross-compile `maploc_rs_v2` for `aarch64-linux-gnu`, integrate
  with `microduck_runtime`.
- [ ] Profile with `perf`: per-frame budget, per-close budget,
  per-optimize budget.
- [ ] Tune submap size, loop-close cadence, optimization frequency to
  fit comfortably (aim ≤ 30 % of one core sustained).

**Acceptance criteria**

- 5-min live walk, 50 Hz control loop holds, map matches the room.

---

## Phase 7 — Relocalize against saved map

**Goal:** start the duck somewhere unknown in a previously-mapped
room, find ourselves.

**Tasks**

- [ ] Bring MCL back, scoped tightly to this use case: uniform cloud
  over the saved global map; scan likelihood pulls it toward truth.
- [ ] On convergence (cloud collapsed + low residual sustained), hand
  off to scan-matching tracking inside the loaded submaps. From then
  on it's the Phase 3–5 pipeline operating on a pre-existing graph.

**Acceptance criteria**

- Load a saved map, place the duck somewhere random, walk a metre,
  watch the arrow lock to the right pose.
