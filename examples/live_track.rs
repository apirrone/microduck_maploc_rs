//! Live SLAM viewer — runs the v2 pipeline against the Pi's TCP streams.
//!
//! Walks the duck → laptop builds the map → image file updates ~1 Hz so
//! you can watch it grow with `feh -R 0.5` / `eog` auto-reload. Same
//! algorithm as `track_loop_closure.rs`; difference is the input source
//! (TCP) and the periodic on-disk render.
//!
//! Setup on the Pi:
//!     python3 tof_streamer.py
//!     microduck_runtime --stream         # exposes digital twin on 9870
//!
//! On the laptop:
//!     cargo run --release --example live_track -- --host <pi-ip> \
//!         --pitch-deg 3.36 --yaw-deg 1.24 --height-m 0.109 \
//!         -o /tmp/live_map.pgm
//!     feh -R 0.5 /tmp/live_map.pgm        # in another terminal
//!
//! Ctrl-C to stop.

use std::env;
use std::f32::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

use microduck_maploc::global_render::{render_global, GlobalRenderConfig};
use microduck_maploc::grid::{GridConfig, OccupancyGrid};
use microduck_maploc::loop_closer::{detect_loops, LoopCloserConfig};
use microduck_maploc::mount::{precompute_zone_lookups, project_frame, TofMount};
use microduck_maploc::optimizer::{optimize, OptimizerConfig};
use microduck_maploc::pose_graph::{between, information_from_sigmas,
                                    PoseEdge, PoseGraph};
use microduck_maploc::submap_manager::{SubmapManager, SubmapManagerConfig};

#[inline]
fn wrap_pi(a: f32) -> f32 {
    let two_pi = 2.0 * PI;
    let mut y = (a + PI).rem_euclid(two_pi) - PI;
    if y == PI { y = -PI; }
    y
}

// ── Wire decoding (mirrors `replay::decode_*`, but for live TCP streams)

const TOF_ROWS: usize = 8;
const TOF_COLS: usize = 8;

#[derive(Debug, Clone)]
struct TofMsg {
    ranges_m: [[f32; TOF_COLS]; TOF_ROWS],
}
#[derive(Debug, Clone, Copy)]
struct TwinMsg {
    odom_x: f32,
    odom_y: f32,
    odom_yaw: f32,
}

enum LiveMsg { Tof(TofMsg), Twin(TwinMsg) }

fn read_exact(s: &mut TcpStream, buf: &mut [u8]) -> std::io::Result<()> {
    let mut filled = 0;
    while filled < buf.len() {
        let n = s.read(&mut buf[filled..])?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof, "stream closed"));
        }
        filled += n;
    }
    Ok(())
}

fn read_u32_le(s: &mut TcpStream) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    read_exact(s, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn tof_reader_loop(host: String, port: u16, tx: Sender<LiveMsg>) {
    loop {
        eprintln!("[tof] connecting to {host}:{port} ...");
        let mut s = match TcpStream::connect((host.as_str(), port)) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[tof] connect failed: {e}, retrying in 1 s");
                thread::sleep(Duration::from_secs(1));
                continue;
            }
        };
        let _ = s.set_nodelay(true);
        eprintln!("[tof] connected.");
        loop {
            let size = match read_u32_le(&mut s) { Ok(v) => v, Err(_) => break };
            let mut payload = vec![0u8; size as usize];
            if read_exact(&mut s, &mut payload).is_err() { break; }
            // Header: f64 ts | u8 rows | u8 cols | u8[2] reserved
            if payload.len() < 12 { continue; }
            let rows = payload[8] as usize;
            let cols = payload[9] as usize;
            if rows != TOF_ROWS || cols != TOF_COLS { continue; }
            let n = rows * cols;
            let dist_off = 12;
            if payload.len() < dist_off + 4 * n { continue; }
            let mut ranges_m = [[0.0_f32; TOF_COLS]; TOF_ROWS];
            for r in 0..TOF_ROWS {
                for c in 0..TOF_COLS {
                    let off = dist_off + (r * TOF_COLS + c) * 4;
                    ranges_m[r][c] = f32::from_le_bytes(
                        payload[off..off + 4].try_into().unwrap());
                }
            }
            if tx.send(LiveMsg::Tof(TofMsg { ranges_m })).is_err() { return; }
        }
        eprintln!("[tof] disconnected, will reconnect ...");
        thread::sleep(Duration::from_millis(500));
    }
}

const TWIN_PACKET_SIZE: usize = 8 + 41 * 4;

fn twin_reader_loop(host: String, port: u16, tx: Sender<LiveMsg>) {
    loop {
        eprintln!("[twin] connecting to {host}:{port} ...");
        let mut s = match TcpStream::connect((host.as_str(), port)) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[twin] connect failed: {e}, retrying in 1 s");
                thread::sleep(Duration::from_secs(1));
                continue;
            }
        };
        let _ = s.set_nodelay(true);
        eprintln!("[twin] connected.");
        loop {
            let mut payload = vec![0u8; TWIN_PACKET_SIZE];
            if read_exact(&mut s, &mut payload).is_err() { break; }
            // Layout: f64 ts (8 B) | f32[41] floats. odom_x, odom_y, odom_yaw
            // are at indices 34, 35, 40 in the float array.
            let f = |idx: usize| -> f32 {
                let off = 8 + idx * 4;
                f32::from_le_bytes(payload[off..off + 4].try_into().unwrap())
            };
            let m = TwinMsg { odom_x: f(34), odom_y: f(35), odom_yaw: f(40) };
            if tx.send(LiveMsg::Twin(m)).is_err() { return; }
        }
        eprintln!("[twin] disconnected, will reconnect ...");
        thread::sleep(Duration::from_millis(500));
    }
}

// ── Image dump (PGM, walls=0 / free=235 / unknown=128, path overlay=60)

fn save_pgm(grid: &OccupancyGrid,
            path_pixels: &[(usize, usize)],
            anchor_pixels: &[(usize, usize)],
            out_path: &std::path::Path) -> std::io::Result<()> {
    let w = grid.width(); let h = grid.height();
    let mut buf = vec![0u8; w * h];
    for i in 0..h {
        for j in 0..w {
            buf[i * w + j] = if grid.is_occupied(i, j) { 0 }
                else if grid.is_known_free(i, j) { 235 }
                else { 128 };
        }
    }
    for &(i, j) in path_pixels {
        if i < h && j < w { buf[i * w + j] = 60; }
    }
    for &(i, j) in anchor_pixels {
        for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)] {
            let ii = i as i32 + di; let jj = j as i32 + dj;
            if ii >= 0 && jj >= 0 && (ii as usize) < h && (jj as usize) < w {
                buf[(ii as usize) * w + (jj as usize)] = 30;
            }
        }
    }
    // Write atomically: write to <path>.tmp, then rename.
    let tmp = out_path.with_extension("pgm.tmp");
    {
        let f = File::create(&tmp)?;
        let mut bw = BufWriter::new(f);
        writeln!(bw, "P5")?;
        writeln!(bw, "{} {}", w, h)?;
        writeln!(bw, "255")?;
        for i in (0..h).rev() {
            bw.write_all(&buf[i * w..(i + 1) * w])?;
        }
        bw.flush()?;
    }
    std::fs::rename(tmp, out_path)?;
    Ok(())
}

// ── Args + main ─────────────────────────────────────────────────────────────

struct Args {
    host: String,
    tof_port: u16,
    twin_port: u16,
    output: PathBuf,
    pitch_deg: f32, yaw_deg: f32, height_m: f32,
    floor_safety: f32, min_range_m: f32,
    submap_half_m: f32, cell_m: f32, margin_m: f32,
    max_age_s: f32, max_travel_m: f32,
    odom_sigma_xy: f32, odom_sigma_yaw: f32,
    loop_radius_m: f32, loop_max_residual_m: f32,
    loop_min_beams_used: u32, loop_min_index_gap: usize,
    dump_every_s: f32,
    draw_path: bool, draw_anchors: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = env::args().skip(1);
    let mut host: Option<String> = None;
    let mut tof_port: u16 = 9872;
    let mut twin_port: u16 = 9870;
    let mut output = PathBuf::from("/tmp/live_map.pgm");
    let mut pitch_deg = 0.0_f32; let mut yaw_deg = 0.0_f32;
    let mut height_m = 0.0_f32;
    let mut floor_safety = 0.85_f32; let mut min_range_m = 0.10_f32;
    let mut submap_half_m = 2.0_f32; let mut cell_m = 0.05_f32;
    let mut margin_m = 0.5_f32;
    let mut max_age_s = 8.0_f32; let mut max_travel_m = 1.0_f32;
    let mut odom_sigma_xy = 0.10_f32; let mut odom_sigma_yaw = 0.05_f32;
    let mut loop_radius_m = 1.5_f32; let mut loop_max_residual_m = 0.10_f32;
    let mut loop_min_beams_used = 16u32; let mut loop_min_index_gap = 1usize;
    let mut dump_every_s = 1.0_f32;
    let mut draw_path = true; let mut draw_anchors = true;
    while let Some(flag) = a.next() {
        let mut val = || a.next().ok_or(format!("missing value for {flag}"));
        match flag.as_str() {
            "--host"            => host = Some(val()?),
            "--tof-port"        => tof_port = val()?.parse().map_err(|e| format!("{e}"))?,
            "--twin-port"       => twin_port = val()?.parse().map_err(|e| format!("{e}"))?,
            "-o" | "--output"   => output = PathBuf::from(val()?),
            "--pitch-deg"       => pitch_deg = val()?.parse().map_err(|e| format!("{e}"))?,
            "--yaw-deg"         => yaw_deg   = val()?.parse().map_err(|e| format!("{e}"))?,
            "--height-m"        => height_m  = val()?.parse().map_err(|e| format!("{e}"))?,
            "--floor-safety"    => floor_safety = val()?.parse().map_err(|e| format!("{e}"))?,
            "--min-range-m"     => min_range_m  = val()?.parse().map_err(|e| format!("{e}"))?,
            "--submap-half-m"   => submap_half_m = val()?.parse().map_err(|e| format!("{e}"))?,
            "--cell-m"          => cell_m      = val()?.parse().map_err(|e| format!("{e}"))?,
            "--margin-m"        => margin_m    = val()?.parse().map_err(|e| format!("{e}"))?,
            "--max-age-s"       => max_age_s   = val()?.parse().map_err(|e| format!("{e}"))?,
            "--max-travel-m"    => max_travel_m = val()?.parse().map_err(|e| format!("{e}"))?,
            "--odom-sigma-xy"   => odom_sigma_xy  = val()?.parse().map_err(|e| format!("{e}"))?,
            "--odom-sigma-yaw"  => odom_sigma_yaw = val()?.parse().map_err(|e| format!("{e}"))?,
            "--loop-radius-m"   => loop_radius_m  = val()?.parse().map_err(|e| format!("{e}"))?,
            "--loop-max-resid"  => loop_max_residual_m = val()?.parse().map_err(|e| format!("{e}"))?,
            "--loop-min-beams"  => loop_min_beams_used = val()?.parse().map_err(|e| format!("{e}"))?,
            "--loop-min-gap"    => loop_min_index_gap  = val()?.parse().map_err(|e| format!("{e}"))?,
            "--dump-every-s"    => dump_every_s = val()?.parse().map_err(|e| format!("{e}"))?,
            "--no-path"         => draw_path = false,
            "--no-anchors"      => draw_anchors = false,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    let host = host.ok_or("--host is required")?;
    Ok(Args {
        host, tof_port, twin_port, output,
        pitch_deg, yaw_deg, height_m, floor_safety, min_range_m,
        submap_half_m, cell_m, margin_m, max_age_s, max_travel_m,
        odom_sigma_xy, odom_sigma_yaw,
        loop_radius_m, loop_max_residual_m, loop_min_beams_used, loop_min_index_gap,
        dump_every_s, draw_path, draw_anchors,
    })
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("usage: live_track --host <pi-ip> [--pitch-deg P] \
                       [--yaw-deg Y] [--height-m H] [-o /tmp/live_map.pgm]");
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
        grid: submap_grid,
        max_age_s: args.max_age_s,
        max_travel_m: args.max_travel_m,
    };
    let mut mgr = SubmapManager::new(mgr_cfg);
    let mut graph = PoseGraph::new();
    let mut node_for_submap: Vec<usize> = Vec::new();
    let odom_info = information_from_sigmas(args.odom_sigma_xy, args.odom_sigma_yaw);

    let mut loop_cfg = LoopCloserConfig::default();
    loop_cfg.radius_m       = args.loop_radius_m;
    loop_cfg.min_index_gap  = args.loop_min_index_gap;
    loop_cfg.max_residual_m = args.loop_max_residual_m;
    loop_cfg.min_beams_used = args.loop_min_beams_used;
    let opt_cfg = OptimizerConfig::default();

    let render_cfg = GlobalRenderConfig { cell_m: args.cell_m, margin_m: args.margin_m };

    let mut tracked = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut last_odom: Option<(f32, f32, f32)> = None;
    let session_start = Instant::now();
    let mut path_world: Vec<(f32, f32)> = Vec::new();
    let mut last_dump = Instant::now();
    let mut tof_count = 0u64;

    // Spawn reader threads.
    let (tx, rx): (Sender<LiveMsg>, Receiver<LiveMsg>) = channel();
    {
        let host = args.host.clone(); let tx = tx.clone();
        thread::spawn(move || tof_reader_loop(host, args.tof_port, tx));
    }
    {
        let host = args.host.clone(); let tx = tx.clone();
        thread::spawn(move || twin_reader_loop(host, args.twin_port, tx));
    }
    drop(tx);  // drop our copy so rx closes when threads do

    eprintln!("running. ctrl-c to stop. dumping {} every {:.2} s",
              args.output.display(), args.dump_every_s);

    loop {
        // Drain everything available, then dump if it's time.
        loop {
            match rx.try_recv() {
                Ok(msg) => process_msg(
                    msg, &mount, &lut,
                    &mut tracked, &mut last_odom,
                    &session_start,
                    &mut mgr, &mut graph, &mut node_for_submap,
                    odom_info, &loop_cfg, &opt_cfg,
                    &mut path_world, &mut tof_count,
                ),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    eprintln!("all readers disconnected, exiting.");
                    return ExitCode::SUCCESS;
                }
            }
        }
        if last_dump.elapsed().as_secs_f32() >= args.dump_every_s {
            if let Some(global) = render_global(mgr.all(), &render_cfg) {
                let mut path_pixels = Vec::new();
                if args.draw_path {
                    for (tx_, ty_) in &path_world {
                        if let Some((i, j)) = global.world_to_idx(*tx_, *ty_) {
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
                if let Err(e) = save_pgm(&global, &path_pixels, &anchor_pixels,
                                         &args.output) {
                    eprintln!("save failed: {e}");
                } else {
                    eprintln!("[dump] {} ToF, {} submaps, {} edges  → {}",
                             tof_count, mgr.n_total(), graph.edges().len(),
                             args.output.display());
                }
            }
            last_dump = Instant::now();
        }
        thread::sleep(Duration::from_millis(20));
    }
}

#[allow(clippy::too_many_arguments)]
fn process_msg(
    msg: LiveMsg,
    mount: &TofMount,
    lut: &microduck_maploc::mount::ZoneLut,
    tracked: &mut (f32, f32, f32),
    last_odom: &mut Option<(f32, f32, f32)>,
    session_start: &Instant,
    mgr: &mut SubmapManager,
    graph: &mut PoseGraph,
    node_for_submap: &mut Vec<usize>,
    odom_info: [[f32; 3]; 3],
    loop_cfg: &LoopCloserConfig,
    opt_cfg: &OptimizerConfig,
    path_world: &mut Vec<(f32, f32)>,
    tof_count: &mut u64,
) {
    let elapsed_s = session_start.elapsed().as_secs_f32();
    let prev_n_total = mgr.n_total();
    let prev_n_frozen = mgr.n_frozen();
    let opened = mgr.tick(elapsed_s, *tracked);
    if opened {
        let new_idx = mgr.n_total() - 1;
        let anchor = mgr.current().unwrap().anchor_pose();
        let node = graph.add_node(anchor, new_idx);
        node_for_submap.push(node);
        let did_switch = prev_n_total > 0 && mgr.n_frozen() > prev_n_frozen;
        if did_switch {
            let frozen_idx = mgr.n_frozen() - 1;
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
            let loops = {
                let frozen = mgr.frozen_mut();
                detect_loops(frozen, frozen_idx, loop_cfg)
            };
            for lc in &loops {
                eprintln!("[loop] {} → {}  resid={:.3} beams={}",
                          lc.from_idx, lc.to_idx,
                          lc.residual_m, lc.n_beams_used);
                let info = information_from_sigmas(loop_cfg.edge_sigma_xy,
                                                  loop_cfg.edge_sigma_yaw);
                graph.add_edge(PoseEdge {
                    from: node_for_submap[lc.from_idx],
                    to:   node_for_submap[lc.to_idx],
                    measurement: lc.measurement,
                    information: info,
                });
            }
            if !loops.is_empty() {
                let _ = optimize(graph, opt_cfg);
                // Push optimized anchors back to the manager.
                for (sm_idx, &node_idx) in node_for_submap.iter().enumerate() {
                    if sm_idx >= mgr.n_total() { break; }
                    let new_anchor = graph.nodes()[node_idx].pose;
                    if sm_idx < mgr.n_frozen() {
                        mgr.frozen_mut()[sm_idx].set_anchor_pose(new_anchor);
                    } else if let Some(cur) = mgr.current_mut() {
                        cur.set_anchor_pose(new_anchor);
                    }
                }
            }
        }
    }

    match msg {
        LiveMsg::Twin(t) => {
            let odom = (t.odom_x, t.odom_y, t.odom_yaw);
            if let Some((px, py, pyaw)) = *last_odom {
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
            *last_odom = Some(odom);
        }
        LiveMsg::Tof(t) => {
            let (angles, ranges) = project_frame(&t.ranges_m, lut, mount);
            if let Some(cur) = mgr.current_mut() {
                cur.integrate_scan(*tracked, &angles, &ranges);
            }
            *tof_count += 1;
            path_world.push((tracked.0, tracked.1));
        }
    }
}
