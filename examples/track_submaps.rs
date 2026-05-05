//! Phase 4 offline tracker — multi-submap.
//!
//! Same input as `track_session`, but uses `SubmapManager` to open and
//! close submaps over the recording. At the end, dumps:
//!
//!   * `<output>.pgm`     — global composite via `render_global`
//!   * `<output>.path.csv`— per-frame `(elapsed_s, tracked_xyz, odom_xyz)`
//!                          (only when `--trajectory-csv` is set)
//!   * stdout: per-submap stats (anchor pose, age, n_cells_known)
//!
//! Tracking is pure-odom (matches the Phase 3 decision).
//!
//! Usage:
//!
//!     cargo run --release --example track_submaps -- \
//!         sessions/test_room_loop1.mdlg \
//!         --pitch-deg 3.36 --yaw-deg 1.24 --height-m 0.109 \
//!         --max-travel-m 1.5 --max-age-s 15 \
//!         -o /tmp/global.pgm

use std::env;
use std::f32::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use microduck_maploc::global_render::{render_global, GlobalRenderConfig};
use microduck_maploc::grid::GridConfig;
use microduck_maploc::mount::{precompute_zone_lookups, project_frame, TofMount};
use microduck_maploc::replay::{Record, SessionReplayer};
use microduck_maploc::submap_manager::{SubmapManager, SubmapManagerConfig};

#[inline]
fn wrap_pi(a: f32) -> f32 {
    let two_pi = 2.0 * PI;
    let mut y = (a + PI).rem_euclid(two_pi) - PI;
    if y == PI { y = -PI; }
    y
}

struct Args {
    input: PathBuf,
    output: PathBuf,
    pitch_deg: f32,
    yaw_deg: f32,
    height_m: f32,
    floor_safety: f32,
    min_range_m: f32,
    submap_half_m: f32,
    cell_m: f32,
    max_age_s: f32,
    max_travel_m: f32,
    margin_m: f32,
    max_seconds: Option<f32>,
    draw_path: bool,
    draw_anchors: bool,
    trajectory_csv: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = env::args().skip(1);
    let input = match args.next() {
        Some(s) => PathBuf::from(s),
        None => return Err("missing <input.mdlg>".into()),
    };
    let mut output = PathBuf::from("/tmp/global.pgm");
    let mut pitch_deg = 0.0_f32;
    let mut yaw_deg = 0.0_f32;
    let mut height_m = 0.0_f32;
    let mut floor_safety = 0.85_f32;
    let mut min_range_m = 0.10_f32;
    let mut submap_half_m = 2.0_f32;
    let mut cell_m = 0.05_f32;
    let mut max_age_s = 20.0_f32;
    let mut max_travel_m = 2.0_f32;
    let mut margin_m = 0.5_f32;
    let mut max_seconds: Option<f32> = None;
    let mut draw_path = false;
    let mut draw_anchors = false;
    let mut trajectory_csv: Option<PathBuf> = None;
    while let Some(flag) = args.next() {
        let mut val = || args.next().ok_or(format!("missing value for {flag}"));
        match flag.as_str() {
            "-o" | "--output"   => output = PathBuf::from(val()?),
            "--pitch-deg"       => pitch_deg = val()?.parse().map_err(|e| format!("{e}"))?,
            "--yaw-deg"         => yaw_deg = val()?.parse().map_err(|e| format!("{e}"))?,
            "--height-m"        => height_m = val()?.parse().map_err(|e| format!("{e}"))?,
            "--floor-safety"    => floor_safety = val()?.parse().map_err(|e| format!("{e}"))?,
            "--min-range-m"     => min_range_m = val()?.parse().map_err(|e| format!("{e}"))?,
            "--submap-half-m"   => submap_half_m = val()?.parse().map_err(|e| format!("{e}"))?,
            "--cell-m"          => cell_m = val()?.parse().map_err(|e| format!("{e}"))?,
            "--max-age-s"       => max_age_s = val()?.parse().map_err(|e| format!("{e}"))?,
            "--max-travel-m"    => max_travel_m = val()?.parse().map_err(|e| format!("{e}"))?,
            "--margin-m"        => margin_m = val()?.parse().map_err(|e| format!("{e}"))?,
            "--max-seconds"     => max_seconds = Some(val()?.parse().map_err(|e| format!("{e}"))?),
            "--draw-path"       => draw_path = true,
            "--draw-anchors"    => draw_anchors = true,
            "--trajectory-csv"  => trajectory_csv = Some(PathBuf::from(val()?)),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(Args {
        input, output, pitch_deg, yaw_deg, height_m, floor_safety,
        min_range_m, submap_half_m, cell_m, max_age_s, max_travel_m,
        margin_m, max_seconds, draw_path, draw_anchors, trajectory_csv,
    })
}

fn save_pgm(grid: &microduck_maploc::grid::OccupancyGrid,
            path_pixels: &[(usize, usize)],
            anchor_pixels: &[(usize, usize)],
            path: &std::path::Path) -> std::io::Result<()> {
    let w = grid.width();
    let h = grid.height();
    let mut buf = vec![0u8; w * h];
    for i in 0..h {
        for j in 0..w {
            buf[i * w + j] = if grid.is_occupied(i, j) {
                0
            } else if grid.is_known_free(i, j) {
                235
            } else {
                128
            };
        }
    }
    for &(i, j) in path_pixels {
        if i < h && j < w { buf[i * w + j] = 60; }
    }
    // Anchors as small 3x3 dark crosses so they pop against free space.
    for &(i, j) in anchor_pixels {
        for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)] {
            let ii = i as i32 + di;
            let jj = j as i32 + dj;
            if ii >= 0 && jj >= 0 && (ii as usize) < h && (jj as usize) < w {
                buf[(ii as usize) * w + (jj as usize)] = 30;
            }
        }
    }
    let f = File::create(path)?;
    let mut bw = BufWriter::new(f);
    writeln!(bw, "P5")?;
    writeln!(bw, "{} {}", w, h)?;
    writeln!(bw, "255")?;
    for i in (0..h).rev() {
        bw.write_all(&buf[i * w..(i + 1) * w])?;
    }
    bw.flush()?;
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mount = TofMount {
        pitch_rad: args.pitch_deg.to_radians(),
        yaw_rad:   args.yaw_deg.to_radians(),
        sensor_height_m: args.height_m,
        floor_safety:    args.floor_safety,
        min_range_m:     args.min_range_m,
    };
    let lut = precompute_zone_lookups(&mount);

    let submap_grid = GridConfig {
        x_range: (-args.submap_half_m, args.submap_half_m),
        y_range: (-args.submap_half_m, args.submap_half_m),
        cell:    args.cell_m,
    };
    let mgr_cfg = SubmapManagerConfig {
        grid:         submap_grid,
        max_age_s:    args.max_age_s,
        max_travel_m: args.max_travel_m,
    };
    let mut mgr = SubmapManager::new(mgr_cfg);

    let mut tracked = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut last_odom: Option<(f32, f32, f32)> = None;
    let mut session_start_us: Option<u64> = None;

    let mut tof_count = 0u64;
    let mut path_world: Vec<(f32, f32, f32, f32, f32, f32, f64)> = Vec::new();

    let replayer = match SessionReplayer::open(&args.input) {
        Ok(r) => r,
        Err(e) => { eprintln!("open: {e}"); return ExitCode::FAILURE; }
    };

    for record in replayer {
        let record = match record {
            Ok(r) => r,
            Err(e) => { eprintln!("decode: {e}"); return ExitCode::FAILURE; }
        };
        let ts_us = record.ts_us();
        if session_start_us.is_none() { session_start_us = Some(ts_us); }
        let elapsed_s = (ts_us - session_start_us.unwrap()) as f64 / 1e6;
        if let Some(max_s) = args.max_seconds {
            if elapsed_s > max_s as f64 { break; }
        }
        // Update submap manager every record (cheap).
        mgr.tick(elapsed_s as f32, tracked);
        match record {
            Record::Twin(t) => {
                let odom = (t.odom_x, t.odom_y, t.odom_yaw);
                if let Some((px, py, pyaw)) = last_odom {
                    let dx_w = odom.0 - px;
                    let dy_w = odom.1 - py;
                    let dyaw = wrap_pi(odom.2 - pyaw);
                    let cp = pyaw.cos(); let sp = pyaw.sin();
                    let dx_b =  cp * dx_w + sp * dy_w;
                    let dy_b = -sp * dx_w + cp * dy_w;
                    let cy = tracked.2.cos(); let sy = tracked.2.sin();
                    tracked.0 += cy * dx_b - sy * dy_b;
                    tracked.1 += sy * dx_b + cy * dy_b;
                    tracked.2  = wrap_pi(tracked.2 + dyaw);
                }
                last_odom = Some(odom);
            }
            Record::Tof(tof) => {
                let (angles, ranges) = project_frame(&tof.ranges_m, &lut, &mount);
                if let Some(cur) = mgr.current_mut() {
                    cur.integrate_scan(tracked, &angles, &ranges);
                }
                tof_count += 1;
                path_world.push((
                    tracked.0, tracked.1, tracked.2,
                    last_odom.map(|o| o.0).unwrap_or(0.0),
                    last_odom.map(|o| o.1).unwrap_or(0.0),
                    last_odom.map(|o| o.2).unwrap_or(0.0),
                    elapsed_s,
                ));
            }
        }
    }

    // Render global composite.
    let render_cfg = GlobalRenderConfig {
        cell_m:   args.cell_m,
        margin_m: args.margin_m,
    };
    let global = match render_global(mgr.all(), &render_cfg) {
        Some(g) => g,
        None => {
            eprintln!("no submaps to render — empty session?");
            return ExitCode::FAILURE;
        }
    };

    // Map world-frame samples to global pixels for overlays.
    let mut path_pixels: Vec<(usize, usize)> = Vec::new();
    if args.draw_path {
        for (tx, ty, _, _, _, _, _) in &path_world {
            if let Some((i, j)) = global.world_to_idx(*tx, *ty) {
                path_pixels.push((i, j));
            }
        }
    }
    let mut anchor_pixels: Vec<(usize, usize)> = Vec::new();
    if args.draw_anchors {
        for s in mgr.all() {
            let (ax, ay, _) = s.anchor_pose();
            if let Some((i, j)) = global.world_to_idx(ax, ay) {
                anchor_pixels.push((i, j));
            }
        }
    }

    if let Err(e) = save_pgm(&global, &path_pixels, &anchor_pixels, &args.output) {
        eprintln!("save: {e}");
        return ExitCode::FAILURE;
    }

    if let Some(csv_path) = &args.trajectory_csv {
        if let Err(e) = (|| -> std::io::Result<()> {
            let mut f = BufWriter::new(File::create(csv_path)?);
            writeln!(f, "elapsed_s,tracked_x,tracked_y,tracked_yaw,\
                          odom_x,odom_y,odom_yaw")?;
            for (tx, ty, tyaw, ox, oy, oyaw, t) in &path_world {
                writeln!(f, "{:.6},{:.4},{:.4},{:.5},{:.4},{:.4},{:.5}",
                         t, tx, ty, tyaw, ox, oy, oyaw)?;
            }
            Ok(())
        })() {
            eprintln!("csv: {e}");
            return ExitCode::FAILURE;
        }
    }

    println!("processed {} ToF frames", tof_count);
    println!("submaps: {} total ({} frozen + {} active)",
             mgr.n_total(), mgr.n_frozen(),
             if mgr.current().is_some() { 1 } else { 0 });
    for (idx, s) in mgr.all().enumerate() {
        let (ax, ay, ayaw) = s.anchor_pose();
        let n_known = s.grid().log_raw().iter().filter(|&&v| v != 0).count();
        let kind = if idx < mgr.n_frozen() { "frozen" } else { "active" };
        println!("  [{idx}] {kind:>6}: anchor=({:+.2}, {:+.2}, {:+.1}°)  \
                  cells_known={}",
                 ax, ay, ayaw.to_degrees(), n_known);
    }
    println!("global render → {} ({} × {} cells)",
             args.output.display(), global.width(), global.height());
    ExitCode::SUCCESS
}
