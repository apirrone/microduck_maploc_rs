//! Offline single-submap tracker — Phase 3 acceptance.
//!
//! Reads a recorded `.mdlg`, advances a tracked pose by odom deltas,
//! runs Hector against a single submap on each ToF frame, integrates
//! the corrected scan, and dumps the resulting submap grid as a PGM
//! image (open with any image viewer or `convert grid.pgm grid.png`).
//!
//! Usage:
//!
//!     cargo run --release --example track_session -- \
//!         sessions/test_room_loop1.mdlg \
//!         --pitch-deg 3.36 --yaw-deg 1.24 --height-m 0.109 \
//!         -o /tmp/submap.pgm

use std::env;
use std::f32::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use microduck_maploc::grid::GridConfig;
use microduck_maploc::mount::{precompute_zone_lookups, project_frame, TofMount};
use microduck_maploc::replay::{Record, SessionReplayer};
use microduck_maploc::scan_matcher::{match_scan, ScanMatchConfig};
use microduck_maploc::submap::Submap;

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
    grid_half_m: f32,
    cell_m: f32,
    sm_prior_xy: f32,
    sm_prior_yaw: f32,
    max_seconds: Option<f32>,
    scan_match: bool,
    trajectory_csv: Option<PathBuf>,
    draw_path: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = env::args().skip(1);
    let input = match args.next() {
        Some(s) => PathBuf::from(s),
        None => return Err("missing <input.mdlg>".into()),
    };
    let mut output = PathBuf::from("/tmp/submap.pgm");
    let mut pitch_deg = 0.0_f32;
    let mut yaw_deg = 0.0_f32;
    let mut height_m = 0.0_f32;
    let mut floor_safety = 0.85_f32;
    let mut min_range_m = 0.10_f32;
    let mut grid_half_m = 4.0_f32;
    let mut cell_m = 0.05_f32;
    let mut sm_prior_xy = 0.30_f32;
    let mut sm_prior_yaw = 0.20_f32;
    let mut max_seconds: Option<f32> = None;
    // Default OFF: per-frame Hector drifts more than odom on this duck
    // (see docs/DESIGN.md / docs/PLAN.md). Kept as `--scan-match` for
    // experimentation.
    let mut scan_match = false;
    let mut trajectory_csv: Option<PathBuf> = None;
    let mut draw_path = false;
    while let Some(flag) = args.next() {
        let mut val = || args.next().ok_or(format!("missing value for {flag}"));
        match flag.as_str() {
            "-o" | "--output"        => output = PathBuf::from(val()?),
            "--pitch-deg"            => pitch_deg = val()?.parse().map_err(|e| format!("{e}"))?,
            "--yaw-deg"              => yaw_deg = val()?.parse().map_err(|e| format!("{e}"))?,
            "--height-m"             => height_m = val()?.parse().map_err(|e| format!("{e}"))?,
            "--floor-safety"         => floor_safety = val()?.parse().map_err(|e| format!("{e}"))?,
            "--min-range-m"          => min_range_m = val()?.parse().map_err(|e| format!("{e}"))?,
            "--grid-half-m"          => grid_half_m = val()?.parse().map_err(|e| format!("{e}"))?,
            "--cell-m"               => cell_m = val()?.parse().map_err(|e| format!("{e}"))?,
            "--sm-prior-xy"          => sm_prior_xy = val()?.parse().map_err(|e| format!("{e}"))?,
            "--sm-prior-yaw"         => sm_prior_yaw = val()?.parse().map_err(|e| format!("{e}"))?,
            "--max-seconds"          => max_seconds = Some(val()?.parse().map_err(|e| format!("{e}"))?),
            "--scan-match"           => scan_match = true,
            "--no-scan-match"        => scan_match = false,
            "--draw-path"            => draw_path = true,
            "--trajectory-csv"       => trajectory_csv = Some(PathBuf::from(val()?)),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(Args {
        input, output, pitch_deg, yaw_deg, height_m, floor_safety,
        min_range_m, grid_half_m, cell_m, sm_prior_xy, sm_prior_yaw,
        max_seconds, scan_match, trajectory_csv, draw_path,
    })
}

fn save_pgm(grid: &microduck_maploc::grid::OccupancyGrid,
            path_pixels: &[(usize, usize)],
            path: &std::path::Path) -> std::io::Result<()> {
    let w = grid.width();
    let h = grid.height();
    // Build a (h × w) byte buffer.
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
    // Overlay the path as mid-dark pixels so it stands out from walls
    // (0) and free space (235). Skip if `path_pixels` is empty.
    for &(i, j) in path_pixels {
        if i < h && j < w {
            buf[i * w + j] = 60;  // distinct from wall (0) and unknown (128)
        }
    }
    let f = File::create(path)?;
    let mut bw = BufWriter::new(f);
    writeln!(bw, "P5")?;
    writeln!(bw, "{} {}", w, h)?;
    writeln!(bw, "255")?;
    // Write upside-down so y-up world maps to standard image-y-down.
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
            eprintln!();
            eprintln!("usage: track_session <input.mdlg> [-o out.pgm] \\");
            eprintln!("       --pitch-deg P --yaw-deg Y --height-m H \\");
            eprintln!("       [--floor-safety 0.85] [--min-range-m 0.10] \\");
            eprintln!("       [--grid-half-m 4.0] [--cell-m 0.05] \\");
            eprintln!("       [--sm-prior-xy 0.30] [--sm-prior-yaw 0.20]");
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

    let grid_cfg = GridConfig {
        x_range: (-args.grid_half_m, args.grid_half_m),
        y_range: (-args.grid_half_m, args.grid_half_m),
        cell:    args.cell_m,
    };
    // Anchor at world origin — the tracked pose starts there.
    let mut submap = Submap::new_at((0.0, 0.0, 0.0), grid_cfg);

    let mut sm_cfg = ScanMatchConfig::default();
    sm_cfg.prior_sigma_xy  = args.sm_prior_xy;
    sm_cfg.prior_sigma_yaw = args.sm_prior_yaw;
    sm_cfg.sigma_m         = 0.30;

    let mut tracked = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut odom_track = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut last_odom: Option<(f32, f32, f32)> = None;
    let mut session_start_us: Option<u64> = None;

    // Diagnostics.
    let mut n_tof = 0u64;
    let mut sum_resid = 0.0_f64;
    let mut sum_corr  = 0.0_f64;
    let mut sum_iters = 0.0_f64;
    let mut n_used_sum = 0.0_f64;
    let mut max_corr  = 0.0_f32;

    // Path samples (one per ToF frame).
    let mut tracked_path: Vec<(f32, f32, f32, f32, f32, f32, f64, f32, f32)>
        = Vec::new();
    let mut path_pixels: Vec<(usize, usize)> = Vec::new();

    let replayer = match SessionReplayer::open(&args.input) {
        Ok(r) => r,
        Err(e) => { eprintln!("open: {e}"); return ExitCode::FAILURE; }
    };

    let grid_x0 = submap.grid().cfg().x_range.0;
    let grid_y0 = submap.grid().cfg().y_range.0;
    let cell    = args.cell_m;

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
                    // Pure-odom track (no scan correction, runs in
                    // parallel for diagnostics).
                    let oc = odom_track.2.cos(); let os = odom_track.2.sin();
                    odom_track.0 += oc * dx_b - os * dy_b;
                    odom_track.1 += os * dx_b + oc * dy_b;
                    odom_track.2  = wrap_pi(odom_track.2 + dyaw);
                    // Scan-matched track (applied here too; scan match
                    // refines on each ToF frame below).
                    let cy = tracked.2.cos(); let sy = tracked.2.sin();
                    tracked.0 += cy * dx_b - sy * dy_b;
                    tracked.1 += sy * dx_b + cy * dy_b;
                    tracked.2  = wrap_pi(tracked.2 + dyaw);
                }
                last_odom = Some(odom);
            }
            Record::Tof(tof) => {
                let (angles, ranges) = project_frame(&tof.ranges_m, &lut, &mount);
                let (corrected_pose, residual_m, iters, n_used) =
                    if args.scan_match {
                        let result = match_scan(
                            submap.grid_mut(),
                            &angles, &ranges,
                            tracked,
                            Some(tracked),
                            &sm_cfg,
                        );
                        (result.pose, result.residual_m,
                         result.iterations, result.n_beams_used)
                    } else {
                        (tracked, f32::NAN, 0u32, 0u32)
                    };
                let dx = corrected_pose.0 - tracked.0;
                let dy = corrected_pose.1 - tracked.1;
                let dist = (dx * dx + dy * dy).sqrt();
                tracked = corrected_pose;
                submap.integrate_scan(tracked, &angles, &ranges);

                n_tof += 1;
                if residual_m.is_finite() {
                    sum_resid  += residual_m as f64;
                    n_used_sum += n_used as f64;
                    sum_iters  += iters as f64;
                }
                sum_corr  += dist as f64;
                if dist > max_corr { max_corr = dist; }

                // Path bookkeeping for visualization + CSV.
                tracked_path.push((tracked.0, tracked.1, tracked.2,
                                   odom_track.0, odom_track.1, odom_track.2,
                                   elapsed_s, residual_m, dist));
                if args.draw_path {
                    let j = ((tracked.0 - grid_x0) / cell) as i32;
                    let i = ((tracked.1 - grid_y0) / cell) as i32;
                    if i >= 0 && j >= 0
                       && (i as usize) < submap.grid().height()
                       && (j as usize) < submap.grid().width()
                    {
                        path_pixels.push((i as usize, j as usize));
                    }
                }
            }
        }
    }

    if let Err(e) = save_pgm(submap.grid(), &path_pixels, &args.output) {
        eprintln!("save: {e}");
        return ExitCode::FAILURE;
    }
    if let Some(csv_path) = &args.trajectory_csv {
        if let Err(e) = (|| -> std::io::Result<()> {
            let mut f = BufWriter::new(File::create(csv_path)?);
            writeln!(f, "elapsed_s,tracked_x,tracked_y,tracked_yaw,\
                          odom_x,odom_y,odom_yaw,residual_m,correction_m")?;
            for (tx, ty, tyaw, ox, oy, oyaw, t, res, corr) in &tracked_path {
                writeln!(f, "{:.6},{:.4},{:.4},{:.5},{:.4},{:.4},{:.5},{:.4},{:.5}",
                         t, tx, ty, tyaw, ox, oy, oyaw, res, corr)?;
            }
            Ok(())
        })() {
            eprintln!("csv: {e}");
            return ExitCode::FAILURE;
        }
    }

    let n = n_tof.max(1) as f64;
    let n_finite = (n_used_sum > 0.0) as u8 as f64;
    let n_residual = if sum_resid > 0.0 { (n - 1.0).max(1.0) } else { 1.0 };
    let _ = n_finite;
    println!("processed {} ToF frames", n_tof);
    println!("  mean residual:    {:.3} m  (excluding empty-grid frames)",
             sum_resid / n_residual);
    println!("  mean correction:  {:.3} m  (max {:.3} m)", sum_corr / n, max_corr);
    println!("  mean iters:       {:.1}", sum_iters / n_residual);
    println!("  mean beams used:  {:.1}", n_used_sum / n_residual);
    println!("  final tracked:    x={:+.3}  y={:+.3}  yaw={:+.1}°",
             tracked.0, tracked.1, tracked.2.to_degrees());
    println!("  final pure-odom:  x={:+.3}  y={:+.3}  yaw={:+.1}°",
             odom_track.0, odom_track.1, odom_track.2.to_degrees());
    println!("  → {}", args.output.display());
    ExitCode::SUCCESS
}
