//! Phase 5 offline tracker — submapping + pose graph + loop closure.
//!
//! Builds on `track_submaps.rs` by:
//!   * Maintaining a `PoseGraph` of submap anchors.
//!   * Adding an odometry edge between consecutive submaps at switch.
//!   * On each submap close, running `detect_loops` against older
//!     submaps; appending detected loop edges; running the optimizer
//!     to redistribute drift across the graph; updating each submap's
//!     anchor with the optimized pose.
//!
//! Renders the corrected global composite at the end.
//!
//! Usage:
//!
//!     cargo run --release --example track_loop_closure -- \
//!         sessions/test_room_loop1.mdlg \
//!         --pitch-deg 3.36 --yaw-deg 1.24 --height-m 0.109 \
//!         -o /tmp/loop.pgm

use std::env;
use std::f32::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use microduck_maploc::global_render::{render_global, GlobalRenderConfig};
use microduck_maploc::grid::GridConfig;
use microduck_maploc::loop_closer::{detect_loops, LoopCloserConfig};
use microduck_maploc::mount::{precompute_zone_lookups, project_frame, TofMount};
use microduck_maploc::optimizer::{optimize, OptimizerConfig};
use microduck_maploc::pose_graph::{between, information_from_sigmas,
                                    PoseEdge, PoseGraph};
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
    odom_sigma_xy:  f32,
    odom_sigma_yaw: f32,
    loop_radius_m:  f32,
    loop_max_residual_m: f32,
    loop_min_beams_used: u32,
    loop_min_index_gap: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut a = env::args().skip(1);
    let input = match a.next() {
        Some(s) => PathBuf::from(s),
        None => return Err("missing <input.mdlg>".into()),
    };
    let mut out = PathBuf::from("/tmp/loop.pgm");
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
    let mut odom_sigma_xy  = 0.10_f32;
    let mut odom_sigma_yaw = 0.05_f32;
    let mut loop_radius_m  = 1.5_f32;
    let mut loop_max_residual_m = 0.10_f32;
    let mut loop_min_beams_used = 16u32;
    let mut loop_min_index_gap = 1usize;
    while let Some(flag) = a.next() {
        let mut val = || a.next().ok_or(format!("missing value for {flag}"));
        match flag.as_str() {
            "-o" | "--output"   => out = PathBuf::from(val()?),
            "--pitch-deg"       => pitch_deg = val()?.parse().map_err(|e| format!("{e}"))?,
            "--yaw-deg"         => yaw_deg   = val()?.parse().map_err(|e| format!("{e}"))?,
            "--height-m"        => height_m  = val()?.parse().map_err(|e| format!("{e}"))?,
            "--floor-safety"    => floor_safety = val()?.parse().map_err(|e| format!("{e}"))?,
            "--min-range-m"     => min_range_m  = val()?.parse().map_err(|e| format!("{e}"))?,
            "--submap-half-m"   => submap_half_m = val()?.parse().map_err(|e| format!("{e}"))?,
            "--cell-m"          => cell_m      = val()?.parse().map_err(|e| format!("{e}"))?,
            "--max-age-s"       => max_age_s   = val()?.parse().map_err(|e| format!("{e}"))?,
            "--max-travel-m"    => max_travel_m = val()?.parse().map_err(|e| format!("{e}"))?,
            "--margin-m"        => margin_m    = val()?.parse().map_err(|e| format!("{e}"))?,
            "--max-seconds"     => max_seconds = Some(val()?.parse().map_err(|e| format!("{e}"))?),
            "--draw-path"       => draw_path = true,
            "--draw-anchors"    => draw_anchors = true,
            "--odom-sigma-xy"   => odom_sigma_xy  = val()?.parse().map_err(|e| format!("{e}"))?,
            "--odom-sigma-yaw"  => odom_sigma_yaw = val()?.parse().map_err(|e| format!("{e}"))?,
            "--loop-radius-m"   => loop_radius_m  = val()?.parse().map_err(|e| format!("{e}"))?,
            "--loop-max-resid"  => loop_max_residual_m = val()?.parse().map_err(|e| format!("{e}"))?,
            "--loop-min-beams"  => loop_min_beams_used = val()?.parse().map_err(|e| format!("{e}"))?,
            "--loop-min-gap"    => loop_min_index_gap  = val()?.parse().map_err(|e| format!("{e}"))?,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(Args {
        input, output: out, pitch_deg, yaw_deg, height_m, floor_safety,
        min_range_m, submap_half_m, cell_m, max_age_s, max_travel_m,
        margin_m, max_seconds, draw_path, draw_anchors,
        odom_sigma_xy, odom_sigma_yaw,
        loop_radius_m, loop_max_residual_m, loop_min_beams_used,
        loop_min_index_gap,
    })
}

fn save_pgm(grid: &microduck_maploc::grid::OccupancyGrid,
            path_pixels: &[(usize, usize)],
            anchor_pixels: &[(usize, usize)],
            out_path: &std::path::Path) -> std::io::Result<()> {
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
    for &(i, j) in anchor_pixels {
        for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)] {
            let ii = i as i32 + di;
            let jj = j as i32 + dj;
            if ii >= 0 && jj >= 0 && (ii as usize) < h && (jj as usize) < w {
                buf[(ii as usize) * w + (jj as usize)] = 30;
            }
        }
    }
    let f = File::create(out_path)?;
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
        Err(e) => { eprintln!("error: {e}"); return ExitCode::FAILURE; }
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
        grid: submap_grid,
        max_age_s: args.max_age_s,
        max_travel_m: args.max_travel_m,
    };
    let mut mgr = SubmapManager::new(mgr_cfg);

    let mut graph = PoseGraph::new();
    let mut node_for_submap: Vec<usize> = Vec::new();
    let odom_info = information_from_sigmas(args.odom_sigma_xy,
                                            args.odom_sigma_yaw);
    let mut loop_cfg = LoopCloserConfig::default();
    loop_cfg.radius_m       = args.loop_radius_m;
    loop_cfg.min_index_gap  = args.loop_min_index_gap;
    loop_cfg.max_residual_m = args.loop_max_residual_m;
    loop_cfg.min_beams_used = args.loop_min_beams_used;
    loop_cfg.verbose = true;
    let opt_cfg = OptimizerConfig::default();

    let mut tracked = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut last_odom: Option<(f32, f32, f32)> = None;
    let mut session_start_us: Option<u64> = None;

    let mut tof_count = 0u64;
    let mut path_world: Vec<(f32, f32)> = Vec::new();
    let mut n_loops_total = 0u32;

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

        // Manager tick — may open / switch submap.
        let prev_n_total = mgr.n_total();
        let prev_n_frozen = mgr.n_frozen();
        let opened = mgr.tick(elapsed_s as f32, tracked);
        if opened {
            // A new submap was just created (mgr.current()). Add a node
            // for it with its current anchor.
            let new_idx = mgr.n_total() - 1; // index in mgr.all()
            let anchor = mgr.current().unwrap().anchor_pose();
            let node = graph.add_node(anchor, new_idx);
            node_for_submap.push(node);

            // If this opening was a *switch* (we just froze one),
            // run loop closure on the now-frozen submap.
            let did_switch = prev_n_total > 0 && mgr.n_frozen() > prev_n_frozen;
            if did_switch {
                let frozen_idx = mgr.n_frozen() - 1; // the one we just froze

                // Add an odometry edge between the previous submap and
                // the freshly-frozen one (just before this open).
                if frozen_idx >= 1 {
                    let prev_anchor = mgr.frozen()[frozen_idx - 1].anchor_pose();
                    let cur_anchor  = mgr.frozen()[frozen_idx].anchor_pose();
                    let z = between(prev_anchor, cur_anchor);
                    graph.add_edge(PoseEdge {
                        from: node_for_submap[frozen_idx - 1],
                        to:   node_for_submap[frozen_idx],
                        measurement: z,
                        information: odom_info,
                    });
                }

                // Loop closure search on the just-frozen submap.
                let loops = {
                    // We need a `&mut [Submap]` over frozen submaps.
                    // The `current` submap is excluded (it can't have
                    // loop closures yet).
                    let frozen = mgr.frozen_mut();
                    detect_loops(frozen, frozen_idx, &loop_cfg)
                };
                for lc in &loops {
                    println!("[loop] {} → {}  measurement=({:+.2}, {:+.2}, {:+.1}°)  \
                              residual={:.3} m  beams={}",
                             lc.from_idx, lc.to_idx,
                             lc.measurement.0, lc.measurement.1,
                             lc.measurement.2.to_degrees(),
                             lc.residual_m, lc.n_beams_used);
                    let info = information_from_sigmas(loop_cfg.edge_sigma_xy,
                                                      loop_cfg.edge_sigma_yaw);
                    graph.add_edge(PoseEdge {
                        from: node_for_submap[lc.from_idx],
                        to:   node_for_submap[lc.to_idx],
                        measurement: lc.measurement,
                        information: info,
                    });
                    n_loops_total += 1;
                }
                if !loops.is_empty() {
                    // Optimize and write the corrected anchors back.
                    let r = optimize(&mut graph, &opt_cfg);
                    println!("[graph] iters={} converged={} cost={:.4}",
                             r.iterations, r.converged, r.final_cost);
                    apply_graph_anchors(&graph, &node_for_submap, &mut mgr);
                }
            }
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
                path_world.push((tracked.0, tracked.1));
            }
        }
    }

    // Render global composite.
    let render_cfg = GlobalRenderConfig {
        cell_m: args.cell_m, margin_m: args.margin_m,
    };
    let global = match render_global(mgr.all(), &render_cfg) {
        Some(g) => g,
        None => { eprintln!("no submaps"); return ExitCode::FAILURE; }
    };

    let mut path_pixels = Vec::new();
    if args.draw_path {
        for (tx, ty) in &path_world {
            if let Some((i, j)) = global.world_to_idx(*tx, *ty) {
                path_pixels.push((i, j));
            }
        }
    }
    let mut anchor_pixels = Vec::new();
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

    println!();
    println!("processed {} ToF frames", tof_count);
    println!("submaps: {} total ({} frozen + {} active)",
             mgr.n_total(), mgr.n_frozen(),
             if mgr.current().is_some() { 1 } else { 0 });
    println!("loop closures detected: {}", n_loops_total);
    for (idx, s) in mgr.all().enumerate() {
        let (ax, ay, ayaw) = s.anchor_pose();
        println!("  [{idx}] anchor=({:+.2}, {:+.2}, {:+.1}°)",
                 ax, ay, ayaw.to_degrees());
    }
    println!("→ {}", args.output.display());
    ExitCode::SUCCESS
}

/// Copy optimized node poses back into the manager's submap anchors.
fn apply_graph_anchors(
    graph: &PoseGraph,
    node_for_submap: &[usize],
    mgr: &mut SubmapManager,
) {
    let n_total = mgr.n_total();
    for (sm_idx, &node_idx) in node_for_submap.iter().enumerate() {
        if sm_idx >= n_total { break; }
        let new_anchor = graph.nodes()[node_idx].pose;
        // Frozen come first, then current.
        if sm_idx < mgr.n_frozen() {
            mgr.frozen_mut()[sm_idx].set_anchor_pose(new_anchor);
        } else if let Some(cur) = mgr.current_mut() {
            cur.set_anchor_pose(new_anchor);
        }
    }
}
