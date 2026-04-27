# microduck_maploc

2D ToF-based mapping, Monte Carlo Localization, A\* path planning, and a streaming protocol — for the [microduck](https://github.com/apirrone/microduck_runtime) robot, running on a Raspberry Pi Zero 2W.

The crate is sensor-agnostic and runtime-agnostic. The duck's runtime feeds it odometry deltas + horizontal-plane ToF scans; the crate maintains an occupancy map, a particle-filter pose estimate, an A\* planner on the map, and (optionally) a TCP server that streams telemetry to a laptop viewer and receives goal clicks back.

## Modules

| Module     | Purpose |
|------------|---------|
| `grid`     | Log-odds occupancy grid (i16 fixed-point), Bresenham ray integration, fast batched ray casting, save/load. |
| `mcl`      | Monte Carlo Localization with augmented-MCL kidnap recovery (searching/tracking modes with persistence counter, free-cell init/inject, rayon-parallel ray casting). |
| `planner`  | A\* on the grid with obstacle inflation + line-of-sight smoothing. Filters single-beam noise via `occ_threshold`. |
| `follower` | Turn-then-go waypoint controller. Inputs: estimated pose. Outputs: body-frame `(forward, dyaw)` to feed motors. |
| `wire`     | Framed binary protocol — 8 messages: Hello, Pose, Map, Path, Scan, Goal, etc. |
| `stream`   | TCP servers (`Telemetry`, `GoalServer`) — single-client, non-blocking. |

## Public API

```rust
use microduck_maploc::{
    OccupancyGrid, GridConfig,
    Localizer, MclConfig,
    plan, PlannerConfig,
    FollowerState, follow_step,
    stream::{Telemetry, GoalServer},
    wire,
};

let mut grid = OccupancyGrid::new(GridConfig::default());
let mut loc  = Localizer::new(&grid, MclConfig::default(), 0);

// Per ToF tick:
loc.predict(dx_body, dy_body, dyaw);             // odometry
loc.update(&grid, &beam_angles, &beam_ranges);   // measurement
let (x, y, yaw) = loc.best();

// Goal arrives:
if let Some(path) = plan(&grid, (x, y), goal, PlannerConfig::default()) {
    let mut follower = FollowerState::new(path[1..].to_vec());
    let cmd = follow_step(&mut follower, (x, y), yaw, 0.20, 1.20, dt, 0.10);
    // cmd.forward + cmd.dyaw → motor velocity command
}
```

## Wire protocol

Every message is `u32 LE length` ++ `u8 tag` ++ payload. All multi-byte fields are little-endian. The `wire` module is the canonical spec; a Python implementation in [microduck_maploc/sim/wire.py](https://github.com/apirrone/microduck_maploc) follows it for the sim and laptop viewer.

| Tag    | Direction      | Body                                                                |
|--------|----------------|---------------------------------------------------------------------|
| `0x01` | server → client | `Hello { version: u32 }` — sent on connect                          |
| `0x02` | server → client | `Pose { x, y, yaw, std, residual, lock, timestamp_ms }`             |
| `0x03` | server → client | Map blob (raw `OccupancyGrid::save` bytes — same on disk and on wire) |
| `0x04` | server → client | `Path { waypoints: Vec<(f32, f32)> }`                               |
| `0x05` | server → client | `Scan { angles_body, ranges, origin }`                              |
| `0x80` | client → server | `Goal { x, y }` — laptop click                                      |

Map blob format (also on-disk): `MDLM` magic, `u32` version, `5 × f32` (x_range, y_range, cell), `2 × u32` (W, H), then `W*H × i16 LE` log-odds × 100. Cross-arch safe.

## Performance

Tuned for Pi Zero 2W (1 GHz quad-core A53). Per 15 Hz tick on the apartment map (130×100 cells, 16 beams, 2000 particles):

- Mapping update: ~3K cell ops along Bresenham rays — sub-millisecond.
- MCL update: ~2.5M cell lookups, parallelized across particles via rayon — ~5–8 ms.
- Memory: ~52 KB map + ~50 KB particles + scratch. Total ~150 KB.

## Tests

```bash
cargo test --release
```

18 unit tests cover round-trips (save/load, wire), planner around obstacles, and MCL convergence/tracking.

## Status

Used in production by the [microduck_runtime](https://github.com/apirrone/microduck_runtime) `--maploc` flag. The sensor-side hookup (VL53L5CX I²C driver) lives on the runtime side; this crate consumes pre-projected horizontal scans through the `TofFrame` shape (angles + ranges in body frame).

The Python reference implementation in [microduck_maploc](https://github.com/apirrone/microduck_maploc) (`sim/`) is the development sandbox: MuJoCo apartment, sim duck, MCL viewer with click-to-goto. It speaks the same wire protocol so the laptop viewer can connect to either the sim or the real duck.
